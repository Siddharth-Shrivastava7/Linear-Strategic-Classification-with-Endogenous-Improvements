"""Strategic improvement error vs. training-set size, all datasets (paper Figure 1).

    python experiment_error_vs_n.py --sweep configs/sweeps/error_vs_n.yaml
    python experiment_error_vs_n.py --sweep ... --plot-only          # redraw from runs.csv
    python experiment_error_vs_n.py --sweep ... --sizes 6000 --repeats 1 --out-suffix _pilot

Each (dataset, n, repeat r) trains SVM, SERM, Attias et al. (PLI) and
Strat-Imp-Aware on the first n rows of permutation r of algo_train and
evaluates them on algo_test in our environment (pipeline/best_response.py).
Writes runs.csv, summary.csv and error_vs_n.png/.pdf.
"""

import argparse
import copy
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
from sklearn.svm import LinearSVC

from experiment_gsc_baseline import linear_margin
from experiment_pli_comparison import context, ours_margin, pli_margin
from experiment_svm_baseline import evaluate
from pipeline.baselines_gsc import train_gsc
from pipeline.baselines_pli import train_pli
from pipeline.config import PROJECT_ROOT
from pipeline.data import load_split
from pipeline.plotting import GRID, TEXT, TEXT_MUTED, _style
from stage4_train_strat_imp import train_from_config

METHODS = ["SVM", "SERM", "Attias", "Ours"]
LABELS = {"SVM": "Non-strategic Classifier (SVM)", "SERM": "Strategic Classifier (SERM)",
          "Attias": "Attias et al.", "Ours": "Strat-Imp-Aware"}
STYLE = {"SVM": ("#2a78d6", "o"), "SERM": ("#eb6834", "s"), "Attias": ("#e34948", "D"), "Ours": ("#008300", "^")}


def subset_ids(cfg_path, train_split, repeat, n):
    ctx = context(cfg_path)
    ids = load_split(ctx["paths"], train_split).index.to_numpy()
    return np.random.default_rng(repeat).permutation(ids)[:n]


def run_job(args):
    ds, method, n, r, sw = args
    cfg_path = str(PROJECT_ROOT / ds["config"])
    ctx = context(cfg_path)
    scfg = ctx["cfg"]["strat_imp_aware"]
    rows = subset_ids(cfg_path, scfg["train_split"], r, n)
    sel, test, e = sw["select_split"], sw["test_split"], ctx["env"]
    df_tr = load_split(ctx["paths"], scfg["train_split"]).loc[rows]
    u_all, _ = ctx["data"][scfg["train_split"]]
    pos = load_split(ctx["paths"], scfg["train_split"]).index.get_indexer(rows)
    X, y = u_all[pos], df_tr[ctx["cfg"]["dataset"]["target"]].to_numpy()
    detail = ""

    if method == "Ours":
        s = copy.deepcopy(scfg)
        s["seed"] = r
        run = train_from_config(ctx["cfg"], ctx["paths"], s, log=lambda *_: None, train_rows=rows.tolist())
        margin = ours_margin(run["best"]["w"], run["best"]["b"])
        detail = f"epoch {run['best']['epoch']}"
    elif method == "Attias":
        p = sw["pli"]
        net = train_pli(p["model_type"], p["loss"], X.float(), torch.tensor(y), r, p["epochs"], p["batch_size"], p["lr"])
        if ds["pli_threshold"] == "select":
            val = {t: evaluate(ctx, pli_margin(net, t), sel, r)["expected_improvement_error"] for t in p["thresholds"]}
            tau = min(val, key=val.get)
        else:
            tau = float(ds["pli_threshold"])
        margin, detail = pli_margin(net, tau), f"tau={tau:g}"
    elif method == "SERM":
        g = sw["serm"]
        cands = []
        for lam in g["lambdas"]:
            m = train_gsc(X.float(), torch.tensor(y), g["shift"], float(lam), r, e["beta"], e["alpha"].float(),
                          g["epochs"], g["batch_size"], g["lr"], class_weight=ds["class_weight"])
            w, c = m.fc.weight[0].detach().double(), float(m.fc.bias.detach()[0])
            cands.append((evaluate(ctx, linear_margin(w, c), sel, r)["expected_improvement_error"], lam, w, c))
        _, lam, w, c = min(cands, key=lambda t: t[0])
        margin, detail = linear_margin(w, c), f"lambda={lam:g}"
    else:  # SVM
        if len(np.unique(y)) < 2:
            raise ValueError("subset contains one class only")
        svm = LinearSVC(**sw["svm"], random_state=r, class_weight=ds["class_weight"]).fit(X.numpy(), y)
        margin = linear_margin(svm.coef_.ravel(), svm.intercept_[0])

    m = evaluate(ctx, margin, test, r)
    return {"dataset": ds["name"], "method": method, "n": n, "repeat": r, "detail": detail,
            "improvement_error": m["improvement_error"], "expected_improvement_error": m["expected_improvement_error"],
            "manipulating_y0": m["manipulating_y0"], "improved": m["improved"],
            "predicted_positive_before": m["predicted_positive_before"]}


def plot(summary, datasets, path_stem):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.1), squeeze=False)
    for k, (ax, name) in enumerate(zip(axes[0], datasets)):
        part = summary[summary["dataset"] == name]
        for meth in METHODS:
            s = part[part["method"] == meth].sort_values("n")
            if s.empty:
                continue
            color, marker = STYLE[meth]
            ax.fill_between(s["n"], s["err_min"], s["err_max"], color=color, alpha=0.15, linewidth=0)
            ax.plot(s["n"], s["err_mean"], color=color, marker=marker, markersize=4, linewidth=1.6, label=LABELS[meth])
        ax.set_title(f"({chr(97 + k)}) {name}", color=TEXT, fontsize=11, loc="center", y=-0.32)
        ax.set_xlabel("")
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="--")
        ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--")
        ax.set_xticks([0, 3000, 6000])
    axes[0][0].set_ylabel("strategic improvement error", color=TEXT, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 1.10),
               labelcolor=TEXT, edgecolor=GRID)
    fig.text(0.5, -0.02, "number of algo_train samples", ha="center", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize(runs):
    g = runs.groupby(["dataset", "method", "n"])
    return pd.DataFrame({
        "err_mean": g["improvement_error"].mean(), "err_min": g["improvement_error"].min(),
        "err_max": g["improvement_error"].max(), "err_sd": g["improvement_error"].std(ddof=1),
        "expected_err_mean": g["expected_improvement_error"].mean(),
        "manip_y0_mean": g["manipulating_y0"].mean(), "improved_mean": g["improved"].mean(),
        "runs": g.size(),
    }).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()
    with open(args.sweep) as f:
        sw = yaml.safe_load(f)
    sizes, repeats = args.sizes or sw["sizes"], args.repeats or sw["repeats"]
    datasets = [d for d in sw["datasets"] if not args.datasets or d["name"] in args.datasets]
    out_dir = PROJECT_ROOT / (sw["out_dir"] + args.out_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        runs = pd.read_csv(out_dir / "runs.csv")
    else:
        jobs = [(d, m, n, r, sw) for d in datasets for r in range(repeats) for n in sizes for m in reversed(METHODS)]
        print(f"{len(jobs)} trainings ({len(datasets)} datasets x {len(sizes)} sizes x {repeats} repeats x 4 methods) "
              f"on {sw.get('workers', os.cpu_count())} workers", flush=True)
        with ProcessPoolExecutor(max_workers=sw.get("workers", os.cpu_count())) as pool:
            runs = pd.DataFrame(list(pool.map(run_job, jobs, chunksize=1)))
        runs.to_csv(out_dir / "runs.csv", index=False)

    summary = summarize(runs)
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot(summary, [d["name"] for d in datasets], out_dir / "error_vs_n")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(summary.pivot_table(index=["dataset", "n"], columns="method", values="err_mean")
              .reindex(columns=METHODS).round(4).to_string())
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
