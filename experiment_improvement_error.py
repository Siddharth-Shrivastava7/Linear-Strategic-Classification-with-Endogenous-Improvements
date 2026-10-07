"""Improvement error of Algorithm 1 vs. number of training samples.

    python experiment_improvement_error.py --sweep configs/sweeps/improvement_error.yaml

For each training size n and repeat r: train Algorithm 1 on the first n rows
of a random permutation (seed r) of the train split, choosing the checkpoint
by validation loss as usual, and measure on the unseen eval split:
    improvement error = share of agents with y_hat(after best response) != y'
(exact expectation over the sampled labels, plus one sampled draw).
Writes runs.csv, summary.csv and improvement_error_vs_n.png/.pdf.
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

from pipeline.config import PROJECT_ROOT, load_config, output_paths
from pipeline.data import load_split
from pipeline.evaluation import improvement_error
from pipeline.plotting import LR_COLOR, TEXT, TEXT_MUTED, _style
from stage4_train_strat_imp import DTYPE, train_from_config


def run_one(args):
    cfg_path, n_label, repeat, rows, eval_split, sample_seed = args
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    paths = output_paths(cfg)
    scfg = copy.deepcopy(cfg["strat_imp_aware"])
    scfg["seed"] = repeat
    run = train_from_config(cfg, paths, scfg, log=lambda *_: None, train_rows=rows)
    best, eta, hp, features = run["best"], run["eta"], run["hp"], run["features"]

    df = load_split(paths, eval_split)
    u = eta.to_model(torch.tensor(df[features].to_numpy(), dtype=DTYPE))
    m = improvement_error(u, df[cfg["dataset"]["target"]].to_numpy(), best["w"], best["b"], hp, eta,
                          run["sim_fn"], np.random.default_rng(sample_seed))
    return {"n": len(rows), "n_label": n_label, "repeat": repeat, "best_epoch": best["epoch"],
            "epochs": hp["epochs"], "best_val_loss": best["val_loss"],
            "w": [round(v, 5) for v in best["w"].tolist()], "b": round(float(best["b"]), 5),
            "j_star": features[m.pop("j_star")], **m}


def plot(summary, runs, eval_split, path_stem):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = summary["n"].to_numpy()
    mean, sd = summary["err_mean"].to_numpy(), summary["err_sd"].to_numpy()
    ax.scatter(runs["n"], runs["expected_improvement_error"], s=10, color=LR_COLOR, alpha=0.25, linewidth=0,
               label="single run")
    ax.fill_between(x, mean - sd, mean + sd, color=LR_COLOR, alpha=0.15, linewidth=0, label="±1 SD over repeats")
    ax.plot(x, mean, color=LR_COLOR, linewidth=2, marker="o", markersize=5, label="mean over repeats")
    for xi, mi in zip(x, mean):
        ax.annotate(f"{mi:.3f}", (xi, mi), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=TEXT_MUTED)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:,}" if lbl != "all" else f"all\n{v:,}" for v, lbl in zip(x, summary["n_label"])],
                       fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel("training samples from algo_train (log scale)", color=TEXT, fontsize=10)
    ax.set_ylabel(f"improvement error on {eval_split}", color=TEXT, fontsize=10)
    ax.set_title("Improvement error vs. training size", color=TEXT, fontsize=12, loc="left", fontweight="semibold")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    args = ap.parse_args()

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    cfg_path = str(PROJECT_ROOT / sweep["base_config"])
    cfg = load_config(cfg_path)
    paths = output_paths(cfg)
    scfg = cfg["strat_imp_aware"]
    out_dir = paths["root"] / "sweeps" / Path(args.sweep).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_split = sweep["eval_split"]
    if eval_split in (scfg["train_split"], scfg["val_split"]):
        raise ValueError(f"eval_split {eval_split} is used for training/selection; pick an unseen split")

    train_ids = load_split(paths, scfg["train_split"]).index.to_numpy()
    jobs = []
    for r in range(sweep["repeats"]):
        perm = np.random.default_rng(r).permutation(train_ids)
        for size in sweep["train_sizes"]:
            n = len(perm) if size == "all" else int(size)
            jobs.append((cfg_path, str(size), r, perm[:n].tolist(), eval_split, sweep["sample_seed"]))

    workers = sweep.get("workers", os.cpu_count())
    print(f"{len(jobs)} trainings ({len(sweep['train_sizes'])} sizes x {sweep['repeats']} repeats) on {workers} workers; "
          f"clip_mode={scfg.get('clip_mode', 'none')}, beta={scfg['beta']}, alpha={scfg['alpha']}, "
          f"T={scfg['epochs']}, batch={scfg['batch_size']}; evaluated on {eval_split}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = pd.DataFrame(list(pool.map(run_one, jobs)))
    runs.to_csv(out_dir / "runs.csv", index=False)

    g = runs.groupby(["n", "n_label"], sort=True)
    summary = pd.DataFrame({
        "err_mean": g["expected_improvement_error"].mean(),
        "err_sd": g["expected_improvement_error"].std(ddof=1),
        "sampled_err_mean": g["sampled_improvement_error"].mean(),
        "err_no_response_mean": g["error_without_response"].mean(),
        "impr_rate_mean": g["expected_improvement_rate"].mean(),
        "impr_rate_sd": g["expected_improvement_rate"].std(ddof=1),
        "neg_movers_mean": g["manipulating_y0"].mean(),
        "pred_pos_mean": g["predicted_positive"].mean(),
        "val_loss_mean": g["best_val_loss"].mean(),
        "runs_best_at_last_epoch": g.apply(lambda d: int((d["best_epoch"] == d["epochs"]).sum()), include_groups=False),
        "j_star": g["j_star"].agg(lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().items())),
    }).reset_index()
    summary.to_csv(out_dir / "summary.csv", index=False)
    plot(summary, runs, eval_split, out_dir / "improvement_error_vs_n")

    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    diffs = np.diff(summary["err_mean"].to_numpy())
    print(f"\nMean improvement error monotonically decreasing: {bool(np.all(diffs < 0))}  "
          f"(steps: {', '.join(f'{d:+.4f}' for d in diffs)})")
    print(f"Saved -> {out_dir / 'runs.csv'}\n         {out_dir / 'summary.csv'}\n"
          f"         {out_dir / 'improvement_error_vs_n.png'} (.pdf)")


if __name__ == "__main__":
    main()
