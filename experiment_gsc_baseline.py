"""Strategic SVM (GSC, Levanon & Rosenfeld 2022) vs. ours and the other baselines.

    python experiment_gsc_baseline.py --sweep configs/sweeps/gsc_baseline.yaml [--reuse-grid]

--reuse-grid takes already-trained (shift, lambda) runs from gsc_grid.csv instead
of retraining (only the lambdas listed in the config are used).

Trains the GSC strategic SVM on the train split for each (shift, lambda),
picks lambda per shift by improvement error on the selection split (our
environment), then evaluates every method on the unseen test split with the
same agents (cheapest feasible single-feature move, y' ~ Bernoulli(pi_Imp)):
manipulating / improved counts (sampled with sample_seed and expected),
improvement rate and error, plus strategic error with labels held fixed.
"""

import argparse
import ast
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from experiment_pli_comparison import context, ours_margin, pli_margin
from experiment_svm_baseline import evaluate
from pipeline.baselines_gsc import train_gsc
from pipeline.baselines_pli import NNClassifier
from pipeline.best_response import cheapest_feasible_response
from pipeline.config import PROJECT_ROOT
from pipeline.data import file_sha256
from stage4_train_strat_imp import DTYPE


def linear_margin(w, c):
    w = torch.as_tensor(w, dtype=DTYPE)
    return lambda u: u @ w + float(c)


def train_job(args):
    cfg_path, shift, lam, g, splits, seed = args
    ctx = context(cfg_path)
    u_tr, y_tr = ctx["data"][g["train_split"]]
    e = ctx["env"]
    model = train_gsc(u_tr.float(), torch.tensor(y_tr), shift, lam, g["seed"], e["beta"], e["alpha"].float(),
                      g["epochs"], g["batch_size"], g["lr"])
    w, c = model.fc.weight[0].detach().double(), float(model.fc.bias.detach()[0])
    res = {s: evaluate(ctx, linear_margin(w, c), s, seed) for s in splits}
    return {"shift": shift, "lam": lam, "w": w.tolist(), "c": c, "results": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--reuse-grid", action="store_true", help="reuse trained runs from gsc_grid.csv")
    args = ap.parse_args()
    with open(args.sweep) as f:
        sw = yaml.safe_load(f)
    cfg_path = str(PROJECT_ROOT / sw["base_config"])
    ctx = context(cfg_path)
    paths, feats = ctx["paths"], ctx["feats"]
    out_dir = paths["root"] / "sweeps" / Path(args.sweep).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    g = {**sw["gsc"], "train_split": sw["train_split"]}
    sel, test, seed = sw["select_split"], sw["test_split"], sw["sample_seed"]

    if args.reuse_grid:
        old = pd.read_csv(out_dir / "gsc_grid.csv")
        runs = []
        for shift in g["shifts"]:
            for lam in g["lambdas"]:
                part = old[(old["shift"] == shift) & np.isclose(old["lambda"], float(lam))]
                if set(part["split"]) != {sel, test}:
                    raise KeyError(f"gsc_grid.csv has no complete run for shift={shift}, lambda={lam}; retrain")
                first = part.iloc[0]
                runs.append({"shift": shift, "lam": float(lam), "w": ast.literal_eval(first["w"]), "c": float(first["c"]),
                             "results": {r["split"]: {k: r[k] for k in part.columns
                                                      if k not in ("shift", "lambda", "split", "w", "c")}
                                         for _, r in part.iterrows()}})
        print(f"Reusing {len(runs)} trained GSC runs from gsc_grid.csv: shifts {g['shifts']} x lambda {g['lambdas']}")
    else:
        jobs = [(cfg_path, s, float(l), g, [sel, test], seed) for s in g["shifts"] for l in g["lambdas"]]
        print(f"Training {len(jobs)} GSC strategic SVMs on {sw['train_split']} "
              f"({len(ctx['data'][sw['train_split']][1]):,} rows): shifts {g['shifts']} x lambda {g['lambdas']}",
              flush=True)
        with ProcessPoolExecutor(max_workers=sw.get("workers", os.cpu_count())) as pool:
            runs = list(pool.map(train_job, jobs))

    grid = pd.DataFrame([{"shift": r["shift"], "lambda": r["lam"], "split": s, **{k: v for k, v in m.items()
                         if k != "moves_along"}, "w": r["w"], "c": r["c"]}
                         for r in runs for s, m in r["results"].items()])
    grid.to_csv(out_dir / "gsc_grid.csv", index=False)

    selected = {}
    for shift in g["shifts"]:
        cand = [r for r in runs if r["shift"] == shift]
        best = min(cand, key=lambda r: r["results"][sel]["expected_improvement_error"])
        selected[shift] = best
        torch.save({"w": torch.tensor(best["w"], dtype=DTYPE), "c": best["c"], "lambda": best["lam"],
                    "shift": shift, "features": feats}, out_dir / f"gsc_ssvm_{shift}.pt")

    # Everything evaluated the same way on the unseen test split.
    methods = {}
    ck = torch.load(paths["strat_model"])
    methods["Ours (STRAT-IMP-AWARE)"] = ours_margin(ck["w"], ck["b"])
    labels = {"l2": "GSC strategic SVM, as in paper (L2 cost, budget 2)",
              "ours": "GSC strategic SVM, our cost model"}
    for shift, r in selected.items():
        methods[f"{labels[shift]}, lambda={r['lam']:g}"] = linear_margin(r["w"], r["c"])
    sb = sw.get("saved_baselines", {})
    svm_path = PROJECT_ROOT / sb.get("svm", "")
    if sb.get("svm") and svm_path.exists():
        svm = joblib.load(svm_path)
        methods["Standard linear SVM (hinge, C=1)"] = linear_margin(svm.coef_.ravel(), svm.intercept_[0])
    pli_path = PROJECT_ROOT / sb.get("pli", "")
    if sb.get("pli") and pli_path.exists():
        net = NNClassifier(len(feats))
        net.load_state_dict(torch.load(pli_path))
        net.eval()
        methods["PLI paper default (NN, FP weight 2, threshold 0.9)"] = pli_margin(net, 0.9)

    res = {name: evaluate(ctx, m, test, seed) for name, m in methods.items()}
    u, y = ctx["data"][test]
    e = ctx["env"]
    for name, m in methods.items():  # GSC's own target: error after the response with labels held fixed
        _, moved, _ = cheapest_feasible_response(m, u, e["alpha"], e["beta"], e["upper"])
        y_hat = moved.numpy() | (m(u) >= 0).numpy()
        res[name]["strategic_error_labels_fixed"] = float((y_hat != np.asarray(y).astype(int)).mean())

    tbl = pd.DataFrame(res).T
    tbl.drop(columns=["moves_along"]).to_csv(out_dir / "comparison_env_ours_test.csv")
    meta = {
        "selected": {s: {"lambda": r["lam"], "w_standardized": dict(zip(feats, r["w"])), "c": r["c"],
                         f"{sel}_expected_improvement_error": r["results"][sel]["expected_improvement_error"],
                         "file": f"gsc_ssvm_{s}.pt", "sha256": file_sha256(out_dir / f"gsc_ssvm_{s}.pt")}
                     for s, r in selected.items()},
        "training": g, "environment_ours": {"beta": ctx["env"]["beta"],
                                            "alpha": dict(zip(feats, ctx["env"]["alpha"].tolist())),
                                            "delta_g": ctx["env"]["delta_g"], "clip": "feasible (max of lr_train)"},
        "split_indices_sha256": file_sha256(paths["split_indices"]), "sample_seed": seed,
        "results_test": {k: {c: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                             for c, v in r.items()} for k, r in res.items()},
    }
    with open(out_dir / "gsc_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nlambda selection on {sel} (expected improvement error in our environment):")
    for _, row in grid[grid["split"] == sel].iterrows():
        print(f"  shift={row['shift']:<5} lambda={row['lambda']:<5g} error={row['expected_improvement_error']:.4f} "
              f"rate={row['expected_improvement_rate']:.4f}  w={np.round(row['w'], 3).tolist()} c={row['c']:.3f}")
    cols = {"predicted_positive_before": "pos. before", "manipulating": "manipulating",
            "manipulating_y0": "manip. y=0", "manipulating_y1": "manip. y=1", "improved": f"improved (seed {seed})",
            "improvement_rate": "impr. rate", "expected_improvement_rate": "E[rate]",
            "improvement_error": "impr. error", "expected_improvement_error": "E[error]",
            "strategic_error_labels_fixed": "strat. error (labels fixed)", "error_no_response": "error w/o response"}
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    out = tbl[list(cols)].rename(columns=cols)
    for c in out.columns:
        out[c] = out[c].map(lambda v: f"{v:.4f}" if isinstance(v, float) and not float(v).is_integer() else f"{v:.0f}"
                            if isinstance(v, (int, float)) else v)
    print(f"\n== Our environment, {test} ({len(y):,} unseen agents)")
    print(out.to_string())
    print("\nmoves along:", {k: v for k, v in tbl["moves_along"].items()})
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
