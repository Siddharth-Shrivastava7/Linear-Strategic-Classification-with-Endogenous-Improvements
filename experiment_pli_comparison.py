"""Compare STRAT-IMP-AWARE with PAC Learning with Improvements (PLI).

    python experiment_pli_comparison.py --sweep configs/sweeps/pli_comparison.yaml

Phase 1 (train): our Algorithm 1 (base config, seeds) and the PLI learners
(model type x loss x seed) on the train split. Each PLI model is scored at
every threshold in environment "ours" on the selection and test splits.
Selection: per PLI model type, the (loss, threshold) with the lowest mean
improvement error on the selection split.
Phase 2 (environment "PLI"): the selected methods and the standard baselines
under PLI's PGD response and decision-tree labels, over their radius grid.
Everything reported is on the unseen test split, mean +/- SD over seeds.
"""

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.tree import DecisionTreeClassifier

from pipeline.baselines_pli import MODELS, make_loss, train_pli
from pipeline.best_response import evaluate_env_ours, evaluate_env_pli
from pipeline.config import PROJECT_ROOT, load_config, output_paths
from pipeline.data import load_split
from pipeline.evaluation import improvement_error
from pipeline.plotting import GRID, TEXT, TEXT_MUTED, _style
from stage4_train_strat_imp import DTYPE, build_eta, build_sim_fn, train_from_config

METHOD_ORDER = ["Ours (STRAT-IMP-AWARE)", "PLI, NN (selected)", "PLI, linear (selected)",
                "Standard NN (BCE, 0.5)", "Standard LR (BCE, 0.5)"]
COLORS = dict(zip(METHOD_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))

_CTX = {}


def context(cfg_path):
    """Per-process cache: config, frozen LR (standardized space), all splits, f*."""
    if cfg_path not in _CTX:
        torch.set_num_threads(1)
        cfg = load_config(cfg_path)
        paths = output_paths(cfg)
        _, eta = build_eta(cfg, paths, "standard")
        feats, target = eta.features, cfg["dataset"]["target"]
        data = {}
        for split in cfg["split"]["fractions"]:
            df = load_split(paths, split)
            data[split] = (eta.to_model(torch.tensor(df[feats].to_numpy(), dtype=DTYPE)), df[target].to_numpy())
        x_max = load_split(paths, cfg["logistic_regression"]["train_split"])[feats].max().to_numpy()
        scfg = cfg["strat_imp_aware"]
        env = {"beta": float(scfg["beta"]), "delta_g": float(scfg["delta_g"]),
               "alpha": torch.tensor([float(scfg["alpha"][f]) for f in feats], dtype=DTYPE),
               "upper": eta.to_model(torch.tensor(x_max, dtype=DTYPE))}
        # PLI's ground truth: an unpruned decision tree on all data (as in their notebook).
        u_all = torch.cat([u for u, _ in data.values()]).numpy()
        y_all = np.concatenate([y for _, y in data.values()])
        fstar = DecisionTreeClassifier(random_state=42).fit(u_all, y_all)
        _CTX[cfg_path] = {"cfg": cfg, "paths": paths, "eta": eta, "feats": feats, "data": data, "env": env,
                          "fstar": fstar}
    return _CTX[cfg_path]


def env_ours(ctx, margin, split):
    u, y = ctx["data"][split]
    e = ctx["env"]
    return evaluate_env_ours(margin, u, y, ctx["eta"], e["beta"], e["alpha"], e["upper"], e["delta_g"])


def ours_margin(w, b):
    return lambda u: u @ w - b


def pli_margin(model, tau):
    return lambda u: model(u.float()).double().flatten() - tau


# ----------------------------------------------------------------------------- phase 1
def train_job(args):
    cfg_path, kind, spec, splits, pli_cfg = args
    ctx = context(cfg_path)
    out = []
    if kind == "ours":
        scfg = copy.deepcopy(ctx["cfg"]["strat_imp_aware"])
        scfg["seed"] = spec["seed"]
        run = train_from_config(ctx["cfg"], ctx["paths"], scfg, log=lambda *_: None)
        w, b = run["best"]["w"], run["best"]["b"]
        for split in splits:
            m = env_ours(ctx, ours_margin(w, b), split)
            # Reference: Algorithm 2 (feasible) exactly as used in training.
            u, y = ctx["data"][split]
            alg2 = improvement_error(u, y, w, b, run["hp"], ctx["eta"], run["sim_fn"])
            m["alg2_improvement_error"] = alg2["expected_improvement_error"]
            m["alg2_improvement_rate"] = alg2["expected_improvement_rate"]
            out.append({"method": "ours", "seed": spec["seed"], "split": split, **m})
        return {"rows": out, "params": {"w": w.tolist(), "b": float(b), "best_epoch": run["best"]["epoch"]}}

    u_tr, y_tr = ctx["data"][pli_cfg["train_split"]]
    model = train_pli(spec["model_type"], spec["loss"], u_tr.float(), torch.tensor(y_tr), spec["seed"],
                      pli_cfg["epochs"], pli_cfg["batch_size"], pli_cfg["lr"])
    for tau in pli_cfg["thresholds"]:
        for split in splits:
            m = env_ours(ctx, pli_margin(model, tau), split)
            out.append({"method": "pli", "model_type": spec["model_type"], "loss": spec["loss"], "threshold": tau,
                        "seed": spec["seed"], "split": split, **m})
    return {"rows": out, "params": {k: v.numpy() for k, v in model.state_dict().items()}}


# ----------------------------------------------------------------------------- phase 2
def env_pli_job(args):
    cfg_path, method, spec, params, test_split, epcfg = args
    ctx = context(cfg_path)
    feature_ids = [ctx["feats"].index(f) for f in epcfg["improvable_features"]]
    if spec["kind"] == "ours":
        w, b = torch.tensor(params["w"], dtype=DTYPE), torch.tensor(params["b"], dtype=DTYPE)
        h, tau, loss_fn = (lambda x: torch.sigmoid(x @ w - b)[:, None]), 0.5, make_loss("bce")
    else:
        model = MODELS[spec["model_type"]](len(ctx["feats"]))
        model.load_state_dict({k: torch.tensor(v) for k, v in params.items()})
        model.eval()
        h, tau, loss_fn = (lambda x: model(x.float()).double()), spec["threshold"], make_loss(spec["loss"])
    u, _ = ctx["data"][test_split]
    rows = []
    for r in epcfg["radii"]:
        m = evaluate_env_pli(h, tau, loss_fn, u, ctx["fstar"], feature_ids, float(r), epcfg["step"], epcfg["iters"])
        rows.append({"method": method, "seed": spec["seed"], "r": float(r), **m})
    return rows


# ----------------------------------------------------------------------------- plots
def plot_env_ours(summary, test_split, path_stem):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    methods = [m for m in METHOD_ORDER if m in summary.index]
    ypos = np.arange(len(methods))[::-1]
    for ax, (col, title) in zip(axes, [("improvement_rate", "Improvement rate  (higher is better)"),
                                       ("improvement_error", "Improvement error  (lower is better)")]):
        for y0, m in zip(ypos, methods):
            mean, sd = summary.loc[m, (col, "mean")], summary.loc[m, (col, "std")]
            ax.errorbar(mean, y0, xerr=None if np.isnan(sd) else sd, fmt="o", color=COLORS[m], markersize=8, capsize=3, linewidth=1.5)
            ax.annotate(f"{mean:.3f}", (mean, y0), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=8.5, color=TEXT)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left", fontweight="semibold")
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(methods) - 0.4)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(methods, fontsize=9.5, color=TEXT)
    fig.suptitle(f"Environment: ours (feasible cost-based response, y' ~ Bernoulli(pi_Imp)) - {test_split}, "
                 f"mean ± SD over seeds", color=TEXT_MUTED, fontsize=9.5, y=-0.02)
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_env_pli(df, radii, test_split, path_stem):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"wspace": 0.22})
    x = np.arange(len(radii))
    g = df.groupby(["method", "r"])
    for ax, (col, title) in zip(axes, [("improvement_error", "Improvement error vs f*  (lower is better)"),
                                       ("improvement_rate", "Improvement rate  (higher is better)")]):
        for m in [m for m in METHOD_ORDER if m in df["method"].unique()]:
            mean = np.array([g.get_group((m, r))[col].mean() for r in radii])
            sd = np.array([g.get_group((m, r))[col].std(ddof=1) for r in radii])
            ax.fill_between(x, mean - sd, mean + sd, color=COLORS[m], alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=COLORS[m], linewidth=2, marker="o", markersize=4, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}" for r in radii], fontsize=8.5)
        ax.set_xlabel("improvement budget r (L-inf radius, standardized units)", color=TEXT, fontsize=9.5)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left", fontweight="semibold")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.suptitle(f"Environment: PLI (PGD response, decision-tree labels) - {test_split}, mean ± SD over seeds",
                 color=TEXT_MUTED, fontsize=9.5, y=-0.04)
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    args = ap.parse_args()

    with open(args.sweep) as f:
        sw = yaml.safe_load(f)
    cfg_path = str(PROJECT_ROOT / sw["base_config"])
    cfg = load_config(cfg_path)
    paths = output_paths(cfg)
    out_dir = paths["root"] / "sweeps" / Path(args.sweep).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_split, test_split = sw["select_split"], sw["test_split"]
    scfg = cfg["strat_imp_aware"]
    if scfg["train_split"] != sw["train_split"] or test_split in (scfg["train_split"], scfg["val_split"]):
        raise ValueError("train split must match strat_imp_aware.train_split and the test split must be unseen")
    pli_cfg = {**sw["pli"], "train_split": sw["train_split"]}
    workers = sw.get("workers", os.cpu_count())
    sizes = {k: len(load_split(paths, k)) for k in (sw["train_split"], sel_split, test_split)}
    print(f"Data (identical for both methods): train {sw['train_split']} = {sizes[sw['train_split']]:,} rows, "
          f"selection {sel_split} = {sizes[sel_split]:,} rows (ours: checkpoint; PLI: loss + threshold), "
          f"test {test_split} = {sizes[test_split]:,} rows (unseen)")
    print(f"Ours: Algorithm 1, T={scfg['epochs']}, batch={scfg['batch_size']}, clip_mode={scfg.get('clip_mode')}, "
          f"beta={scfg['beta']}, alpha={scfg['alpha']}.  PLI: {pli_cfg['model_types']} x {pli_cfg['losses']}, "
          f"epochs={pli_cfg['epochs']}, batch={pli_cfg['batch_size']}, thresholds={pli_cfg['thresholds']}; "
          f"seeds={sw['seeds']}")

    # Phase 1 -------------------------------------------------------------------
    jobs, keys = [], []
    for s in sw["seeds"]:
        jobs.append((cfg_path, "ours", {"seed": s}, [sel_split, test_split], pli_cfg))
        keys.append(("ours", None, None, s))
    for mt in pli_cfg["model_types"]:
        for loss in pli_cfg["losses"]:
            for s in sw["seeds"]:
                jobs.append((cfg_path, "pli", {"model_type": mt, "loss": loss, "seed": s},
                             [sel_split, test_split], pli_cfg))
                keys.append(("pli", mt, loss, s))
    print(f"Phase 1: {len(jobs)} trainings on {workers} workers "
          f"(ours: {len(sw['seeds'])}, PLI: {len(jobs) - len(sw['seeds'])})", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(train_job, jobs))
    params = dict(zip(keys, [r["params"] for r in results]))
    rows = pd.DataFrame([row for r in results for row in r["rows"]])
    rows.drop(columns=["move_features"]).to_csv(out_dir / "env_ours_all_runs.csv", index=False)

    # Selection on the selection split (PLI only; ours is fixed) ------------------
    pli_sel = rows[(rows["method"] == "pli") & (rows["split"] == sel_split)]
    grid = pli_sel.groupby(["model_type", "loss", "threshold"])[["improvement_error", "improvement_rate"]].mean()
    grid.reset_index().to_csv(out_dir / f"pli_grid_{sel_split}.csv", index=False)
    selected = {}
    for mt in pli_cfg["model_types"]:
        best = grid.loc[mt]["improvement_error"].idxmin()
        selected[mt] = {"loss": best[0], "threshold": float(best[1]),
                        f"{sel_split}_improvement_error": float(grid.loc[(mt, *best), "improvement_error"])}
    with open(out_dir / "pli_selected.json", "w") as f:
        json.dump(selected, f, indent=2)

    std = sw["standard"]
    methods = {
        "Ours (STRAT-IMP-AWARE)": {"kind": "ours"},
        "PLI, NN (selected)": {"kind": "pli", "model_type": "nn", **selected["nn"]},
        "PLI, linear (selected)": {"kind": "pli", "model_type": "linear", **selected["linear"]},
        "Standard NN (BCE, 0.5)": {"kind": "pli", "model_type": "nn", "loss": std["loss"], "threshold": std["threshold"]},
        "Standard LR (BCE, 0.5)": {"kind": "pli", "model_type": "linear", "loss": std["loss"],
                                   "threshold": std["threshold"]},
    }

    def method_rows(spec):
        t = rows[rows["split"] == test_split]
        if spec["kind"] == "ours":
            return t[t["method"] == "ours"]
        return t[(t["method"] == "pli") & (t["model_type"] == spec["model_type"]) & (t["loss"] == spec["loss"])
                 & np.isclose(t["threshold"], spec["threshold"])]

    test_rows = pd.concat([method_rows(spec).assign(label=name) for name, spec in methods.items()])
    metrics = ["improvement_rate", "improvement_error", "error_no_response", "neg_movers", "movers",
               "predicted_positive_before"]
    summary = test_rows.groupby("label")[metrics].agg(["mean", "std"]).reindex(METHOD_ORDER)
    summary.to_csv(out_dir / "env_ours_test_summary.csv")
    plot_env_ours(summary, test_split, out_dir / "env_ours_test")

    # Phase 2: PLI environment ---------------------------------------------------
    epcfg = sw["env_pli"]
    jobs2 = []
    for name, spec in methods.items():
        for s in sw["seeds"]:
            key = ("ours", None, None, s) if spec["kind"] == "ours" else ("pli", spec["model_type"], spec["loss"], s)
            jobs2.append((cfg_path, name, {**spec, "seed": s}, params[key], test_split, epcfg))
    print(f"Phase 2: PLI environment, {len(jobs2)} runs x {len(epcfg['radii'])} radii", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        env_pli = pd.DataFrame([row for r in pool.map(env_pli_job, jobs2) for row in r])
    env_pli.to_csv(out_dir / "env_pli_test_runs.csv", index=False)
    plot_env_pli(env_pli, [float(r) for r in epcfg["radii"]], test_split, out_dir / "env_pli_test")

    # Report ---------------------------------------------------------------------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"\nPLI selection on {sel_split} (lowest mean improvement error):")
    for mt, s in selected.items():
        print(f"  {mt:<7} loss={s['loss']:<7} threshold={s['threshold']}  ({sel_split} error "
              f"{s[f'{sel_split}_improvement_error']:.4f})")
    ours_test = rows[(rows["method"] == "ours") & (rows["split"] == test_split)]
    print(f"\nOurs: generic best response vs Algorithm 2 (feasible) on {test_split}: error "
          f"{ours_test['improvement_error'].mean():.4f} vs {ours_test['alg2_improvement_error'].mean():.4f}, "
          f"rate {ours_test['improvement_rate'].mean():.4f} vs {ours_test['alg2_improvement_rate'].mean():.4f}")
    print(f"\n== Environment ours, {test_split} (mean ± SD over {len(sw['seeds'])} seeds)")
    def pm(m, col):
        mean, sd = summary.loc[m, (col, "mean")], summary.loc[m, (col, "std")]
        return f"{mean:.4f}" if np.isnan(sd) else f"{mean:.4f} ± {sd:.4f}"

    tbl = pd.DataFrame({
        "improvement rate": [pm(m, "improvement_rate") for m in METHOD_ORDER],
        "improvement error": [pm(m, "improvement_error") for m in METHOD_ORDER],
        "error w/o response": [f"{summary.loc[m, ('error_no_response', 'mean')]:.4f}" for m in METHOD_ORDER],
        "neg movers": [f"{summary.loc[m, ('neg_movers', 'mean')]:.0f}" for m in METHOD_ORDER],
        "movers": [f"{summary.loc[m, ('movers', 'mean')]:.0f}" for m in METHOD_ORDER],
        "pred. positive before": [f"{summary.loc[m, ('predicted_positive_before', 'mean')]:.3f}" for m in METHOD_ORDER],
    }, index=METHOD_ORDER)
    print(tbl.to_string())
    print(f"\n== Environment PLI, {test_split}: improvement error (mean over seeds) by radius r")
    print(env_pli.pivot_table(index="method", columns="r", values="improvement_error", aggfunc="mean")
          .reindex(METHOD_ORDER).round(4).to_string())
    print(f"\n== Environment PLI, {test_split}: improvement rate (mean over seeds) by radius r")
    print(env_pli.pivot_table(index="method", columns="r", values="improvement_rate", aggfunc="mean")
          .reindex(METHOD_ORDER).round(4).to_string())
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
