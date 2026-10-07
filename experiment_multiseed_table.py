"""Comparison table over several seeds: Ours, PLI, GSC (paper / our cost), SVM.

    python experiment_multiseed_table.py --sweep configs/sweeps/law_school/multiseed_table.yaml
        [--from-runs --seeds 42]

--from-runs rebuilds the table from the saved runs.csv (no retraining), using
only the --seeds given; the result goes to a subfolder seed_<seeds>/.

For every seed each method is retrained on the train split and evaluated on
the unseen test split in our environment (cheapest feasible single-feature
move, y' ~ Bernoulli(pi_Imp)); the seed also drives the sampled labels.
Reports mean +/- SD over seeds and saves every per-seed model.
"""

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.svm import LinearSVC

from experiment_gsc_baseline import linear_margin
from experiment_pli_comparison import context, ours_margin, pli_margin
from experiment_svm_baseline import evaluate
from pipeline.baselines_gsc import train_gsc
from pipeline.baselines_pli import train_pli
from pipeline.config import PROJECT_ROOT
from pipeline.data import file_sha256
from stage4_train_strat_imp import train_from_config
from validate_true_eta import true_eta_metrics

OURS, PLI = "Ours (STRAT-IMP-AWARE)", "PLI paper default (NN, FP 2, threshold 0.9)"
PLI_SEL = "PLI (NN, FP 2, threshold chosen on validation)"
GSC_NAMES = {"l2": "GSC strategic SVM, as in paper (L2 cost)", "ours": "GSC strategic SVM, our cost model"}
SVM_NAME = "Standard linear SVM"
BALANCED = ", class-balanced"


def method_order(sw):
    gsc_cw = sw["gsc"].get("class_weights", [None])
    svm_cw = sw["svm_class_weights"] if "svm_class_weights" in sw else [None]
    order = [OURS, PLI] + ([PLI_SEL] if sw["pli"].get("select_thresholds") else [])
    order += [GSC_NAMES[s] + (BALANCED if cw else "") for cw in gsc_cw for s in sw["gsc"]["shifts"]]
    order += [SVM_NAME + (BALANCED if cw else "") for cw in svm_cw]
    return order


def job(args):
    cfg_path, method, seed, sw, model_dir = args
    ctx = context(cfg_path)
    test, sel = sw["test_split"], sw["select_split"]
    u_tr, y_tr = ctx["data"][sw["train_split"]]
    e = ctx["env"]
    rows = []
    if method == "ours":
        scfg = copy.deepcopy(ctx["cfg"]["strat_imp_aware"])
        scfg["seed"] = seed
        run = train_from_config(ctx["cfg"], ctx["paths"], scfg, log=lambda *_: None)
        w, b = run["best"]["w"], run["best"]["b"]
        torch.save({"w": w, "b": b, "best_epoch": run["best"]["epoch"], "restart": run["best"].get("restart", 0),
                    "restart_seed": run["best"].get("restart_seed", seed)}, model_dir / f"ours_seed{seed}.pt")
        j = run["history"][run["best"]["epoch"] - 1]["j_star"]
        rows.append({"method": OURS, "detail": f"restart {run['best'].get('restart', 0)} of "
                     f"{run['hp']['restarts']}, epoch {run['best']['epoch']}, val {run['best']['val_loss']:.4f}, "
                     f"j*={run['features'][j]}",
                     **evaluate(ctx, ours_margin(w, b), test, seed), "_margin": ours_margin(w, b)})
    elif method == "pli":
        p = sw["pli"]
        net = train_pli(p["model_type"], p["loss"], u_tr.float(), torch.tensor(y_tr), seed,
                        p["epochs"], p["batch_size"], p["lr"])
        torch.save(net.state_dict(), model_dir / f"pli_seed{seed}.pt")
        rows.append({"method": PLI, "detail": f"threshold={p['threshold']:g}",
                     **evaluate(ctx, pli_margin(net, p["threshold"]), test, seed),
                     "_margin": pli_margin(net, p["threshold"])})
        if p.get("select_thresholds"):
            # Same trained network; threshold from PLI's grid with the lowest expected
            # improvement error on the selection split.
            val_err = {t: evaluate(ctx, pli_margin(net, t), sel, seed)["expected_improvement_error"]
                       for t in p["select_thresholds"]}
            tau = min(val_err, key=val_err.get)
            rows.append({"method": PLI_SEL, "detail": f"threshold={tau:g} (val error {val_err[tau]:.4f})",
                         **evaluate(ctx, pli_margin(net, tau), test, seed), "_margin": pli_margin(net, tau)})
    elif method == "gsc":
        g = sw["gsc"]
        for cw in g.get("class_weights", [None]):
            for shift in g["shifts"]:
                cands = []
                for lam in g["lambdas"]:
                    m = train_gsc(u_tr.float(), torch.tensor(y_tr), shift, float(lam), seed, e["beta"],
                                  e["alpha"].float(), g["epochs"], g["batch_size"], g["lr"], class_weight=cw)
                    w, c = m.fc.weight[0].detach().double(), float(m.fc.bias.detach()[0])
                    val = evaluate(ctx, linear_margin(w, c), sel, seed)["expected_improvement_error"]
                    cands.append((val, float(lam), w, c))
                val, lam, w, c = min(cands, key=lambda t: t[0])
                tag = f"gsc_{shift}" + ("_balanced" if cw else "")
                torch.save({"w": w, "c": c, "lambda": lam, "class_weight": cw}, model_dir / f"{tag}_seed{seed}.pt")
                rows.append({"method": GSC_NAMES[shift] + (BALANCED if cw else ""), "detail": f"lambda={lam:g}",
                             **evaluate(ctx, linear_margin(w, c), test, seed), "_margin": linear_margin(w, c)})
    elif method == "svm":
        for cw in sw.get("svm_class_weights", [None]):
            svm = LinearSVC(**sw["svm"], random_state=seed, class_weight=cw).fit(u_tr.numpy(), y_tr)
            joblib.dump(svm, model_dir / f"svm{'_balanced' if cw else ''}_seed{seed}.joblib")
            m_svm = linear_margin(svm.coef_.ravel(), svm.intercept_[0])
            rows.append({"method": SVM_NAME + (BALANCED if cw else ""), "detail": "",
                         **evaluate(ctx, m_svm, test, seed), "_margin": m_svm})
    for r in rows:
        r["seed"] = seed
        r["moves_along"] = json.dumps(r["moves_along"])
        if sw.get("true_eta"):  # synthetic data: also score improvement with the true eta
            r.update(true_eta_metrics(ctx, r.pop("_margin"), test, seed))
        r.pop("_margin", None)
    return rows


def fmt(mean, sd, digits):
    if np.isnan(mean):
        return "undefined"
    if np.isnan(sd):  # a single seed: no spread to report
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--from-runs", action="store_true", help="rebuild from saved runs.csv, no retraining")
    ap.add_argument("--seeds", type=int, nargs="+", help="with --from-runs: seeds to include")
    args = ap.parse_args()
    with open(args.sweep) as f:
        sw = yaml.safe_load(f)
    cfg_path = str(PROJECT_ROOT / sw["base_config"])
    ctx = context(cfg_path)
    out_dir = ctx["paths"]["root"] / "sweeps" / Path(args.sweep).stem
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.from_runs:
        runs = pd.read_csv(out_dir / "runs.csv")
        seeds = args.seeds or sw["seeds"]
        missing = sorted(set(seeds) - set(runs["seed"]))
        if missing:
            raise ValueError(f"seeds {missing} are not in {out_dir / 'runs.csv'}; train them first")
        runs = runs[runs["seed"].isin(seeds)]
        sw = dict(sw, seeds=seeds)
        out_dir = out_dir / ("seed_" + "_".join(map(str, seeds)))
        out_dir.mkdir(exist_ok=True)
        runs.to_csv(out_dir / "runs.csv", index=False)
        print(f"Rebuilt from saved runs for seeds {seeds} (models in {model_dir})")
    else:
        jobs = [(cfg_path, m, s, sw, model_dir) for s in sw["seeds"] for m in ("ours", "pli", "gsc", "svm")]
        print(f"{len(jobs)} jobs (4 methods x seeds {sw['seeds']}) on {sw.get('workers', os.cpu_count())} workers",
              flush=True)
        with ProcessPoolExecutor(max_workers=sw.get("workers", os.cpu_count())) as pool:
            runs = pd.DataFrame([r for rows in pool.map(job, jobs) for r in rows])
        runs.to_csv(out_dir / "runs.csv", index=False)

    metrics = ["predicted_positive_before", "manipulating", "manipulating_y0", "manipulating_y1", "improved",
               "expected_improved", "improvement_rate", "expected_improvement_rate", "improvement_error",
               "expected_improvement_error"]
    if sw.get("true_eta"):
        metrics += ["true_improved", "true_expected_improved", "true_improvement_rate",
                    "true_expected_improvement_rate", "true_improvement_error", "true_expected_improvement_error"]
    METHODS = method_order(sw)
    g = runs.groupby("method")[metrics]
    mean, sd = g.mean().reindex(METHODS), g.std(ddof=1).reindex(METHODS)
    summary = pd.concat({"mean": mean, "sd": sd}, axis=1)
    summary.to_csv(out_dir / "table_mean_sd.csv")

    n = len(sw["seeds"])
    table = pd.DataFrame({
        "Pred. positive before": [fmt(mean.loc[m, "predicted_positive_before"], sd.loc[m, "predicted_positive_before"], 1) for m in METHODS],
        "Manipulating": [fmt(mean.loc[m, "manipulating"], sd.loc[m, "manipulating"], 1) for m in METHODS],
        "Manip. y=0": [fmt(mean.loc[m, "manipulating_y0"], sd.loc[m, "manipulating_y0"], 1) for m in METHODS],
        "Improved (sampled)": [fmt(mean.loc[m, "improved"], sd.loc[m, "improved"], 1) for m in METHODS],
        "E[improved]": [fmt(mean.loc[m, "expected_improved"], sd.loc[m, "expected_improved"], 1) for m in METHODS],
        "Impr. rate (sampled)": [fmt(mean.loc[m, "improvement_rate"], sd.loc[m, "improvement_rate"], 3) for m in METHODS],
        "E[impr. rate]": [fmt(mean.loc[m, "expected_improvement_rate"], sd.loc[m, "expected_improvement_rate"], 3) for m in METHODS],
        "Impr. error (sampled)": [fmt(mean.loc[m, "improvement_error"], sd.loc[m, "improvement_error"], 4) for m in METHODS],
        "E[impr. error]": [fmt(mean.loc[m, "expected_improvement_error"], sd.loc[m, "expected_improvement_error"], 4) for m in METHODS],
    }, index=METHODS)
    if sw.get("true_eta"):
        for col, key, dg in [("Improved (sampled, true η)", "true_improved", 1),
                             ("E[improved] (true η)", "true_expected_improved", 1),
                             ("E[impr. rate] (true η)", "true_expected_improvement_rate", 3),
                             ("E[impr. error] (true η)", "true_expected_improvement_error", 4)]:
            table[col] = [fmt(mean.loc[m, key], sd.loc[m, key], dg) for m in METHODS]
    table.to_csv(out_dir / "table_formatted.csv")
    meta = {"seeds": sw["seeds"], "settings": {k: sw.get(k) for k in ("pli", "gsc", "svm", "svm_class_weights")},
            "environment_ours": {"beta": ctx["env"]["beta"], "alpha": dict(zip(ctx["feats"], ctx["env"]["alpha"].tolist())),
                                 "delta_g": ctx["env"]["delta_g"], "clip": "feasible (max of lr_train)"},
            "data": {"train": sw["train_split"], "select": sw["select_split"], "test": sw["test_split"],
                     "split_indices_sha256": file_sha256(ctx["paths"]["split_indices"])},
            "strat_imp_aware": ctx["cfg"]["strat_imp_aware"]}
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 20)
    u, y = ctx["data"][sw["test_split"]]
    print(f"\n== {ctx['cfg']['dataset']['name']}: our environment, {sw['test_split']} ({len(y):,} agents, "
          f"{int((np.asarray(y) == 0).sum())} with y=0); "
          + (f"seed {sw['seeds'][0]}" if n == 1 else f"mean ± SD over {n} seeds {sw['seeds']}"))
    print(table.to_string())
    print("\nPer-seed details:")
    print(runs[["method", "seed", "detail", "manipulating_y0", "improved", "expected_improvement_rate",
                "expected_improvement_error"]].sort_values(["method", "seed"]).round(4).to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
