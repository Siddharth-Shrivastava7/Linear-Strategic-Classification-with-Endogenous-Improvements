"""Improvement rate with bounded best responses, over beta x alpha.

    python experiment_clipped_improvement.py --sweep configs/sweeps/clipped_improvement.yaml

Trains Algorithm 1 for every (clip_mode, beta, alpha profile) cell (checkpoint
chosen by validation loss, as in the algorithm), then reports the exact
expected improvement rate (mean pi_Imp over manipulating agents with y = 0).
Writes results.csv, best_settings.csv and improvement_rate_vs_beta.png/.pdf.
Nothing here overwrites the stage 4 model.
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pipeline.config import PROJECT_ROOT, load_config, output_paths
from pipeline.plotting import GRID, TEXT, TEXT_MUTED, _style
from sweep_improvement_rate import resolve_variant, run_variant

# Categorical slots in fixed order (one per alpha profile).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODE_TITLES = {"none": "No bound (Algorithm 2 as written)",
               "clip": "clip: x^f capped at observed max",
               "feasible": "feasible: move only if boundary reachable"}


def plot_rates(df, profiles, rep, path_stem, features):
    modes = list(dict.fromkeys(df["clip_mode"]))
    fig, axes = plt.subplots(3, len(modes), figsize=(5.0 * len(modes), 9.6), sharex=True, sharey="row",
                             squeeze=False, gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    panels = [("rep_expected_improvement_rate", "Improvement rate"),
              ("rep_frac_negatives_manipulating", "Share of negatives moving"),
              ("rep_frac_movers_beyond_observed_max", "Movers whose target\nexceeds observed max")]
    for c, mode in enumerate(modes):
        part = df[df["clip_mode"] == mode]
        for r, (col, ylabel) in enumerate(panels):
            ax = axes[r, c]
            for i, prof in enumerate(profiles):
                s = part[part["profile"] == prof].sort_values("beta")
                ax.plot(s["beta"], s[col], color=SERIES[i], linewidth=2, marker="o", markersize=4,
                        label=prof if (r == 0 and c == 0) else None)
            ax.set_xscale("log", base=2)
            ax.set_ylim(-0.03, 1.03)
            _style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            if c == 0:
                ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
            if r == 0:
                ax.set_title(MODE_TITLES.get(mode, mode), color=TEXT, fontsize=11, loc="left", fontweight="semibold")
            if r == len(panels) - 1:
                ax.set_xlabel("utility β (log scale)", color=TEXT, fontsize=10)
                ax.set_xticks(sorted(df["beta"].unique()))
                ax.set_xticklabels([f"{b:g}" for b in sorted(df["beta"].unique())])
    fig.legend(title=f"cost profile α ({', '.join(features)})", loc="upper center", ncol=4, frameon=False,
               fontsize=8.5, title_fontsize=9, bbox_to_anchor=(0.5, 1.02), labelcolor=TEXT)
    fig.suptitle(f"Expected improvement rate on {rep}", y=1.07, color=TEXT, fontsize=13, fontweight="semibold")
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
    out_dir = paths["root"] / "sweeps" / Path(args.sweep).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    sel, rep = sweep["selection_split"], sweep["report_split"]
    grid = sweep["grid"]
    profiles = list(grid["alpha_profiles"])

    cells, jobs = [], []
    for mode in grid["clip_mode"]:
        for prof, alpha in grid["alpha_profiles"].items():
            for beta in grid["beta"]:
                variant = {"clip_mode": mode, "beta": float(beta), "alpha": {k: float(v) for k, v in alpha.items()}}
                cells.append({"clip_mode": mode, "profile": prof, "beta": float(beta)})
                jobs.append((cfg_path, resolve_variant(cfg, paths, variant), sel, rep))

    workers = sweep.get("workers", os.cpu_count())
    print(f"{len(jobs)} trainings on {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_variant, jobs))

    df = pd.DataFrame([{**c, **r} for c, r in zip(cells, results)])
    failed = df[df["error"].notna()] if "error" in df else df.iloc[:0]
    for _, r in failed.iterrows():
        print(f"FAILED {r['clip_mode']} / {r['profile']} / beta={r['beta']}: {r['error']}")
    df = df.drop(index=failed.index)
    df.to_csv(out_dir / "results.csv", index=False)
    plot_rates(df, profiles, rep, out_dir / "improvement_rate_vs_beta", cfg["logistic_regression"]["features"])

    view = {
        "rep_expected_improvement_rate": "rate",
        "sel_expected_improvement_rate": f"rate({sel})",
        "rep_manipulating_y0": "neg movers",
        "rep_expected_improved": "E[improved]",
        "rep_frac_movers_beyond_observed_max": "target>max",
        "sel_j_star": "j*",
        "sel_S_w": "S(w)",
        "best_val_loss": "val loss",
        "best_epoch": "epoch",
    }
    fmt = {"display.width": 250, "display.max_columns": 40, "display.float_format": "{:.3f}".format}
    with pd.option_context(*[x for kv in fmt.items() for x in kv]):
        for mode in grid["clip_mode"]:
            part = df[df["clip_mode"] == mode]
            print(f"\n== clip_mode = {mode}: rate on {rep} (rows: alpha profile, columns: beta)")
            print(part.pivot(index="profile", columns="beta", values="rep_expected_improvement_rate")
                  .reindex(profiles).to_string())
            print(f"   direction j* (rows: alpha profile, columns: beta)")
            print(part.pivot(index="profile", columns="beta", values="sel_j_star").reindex(profiles).to_string())

        # Best settings. Default: highest improvement rate on the selection split among
        # cells with enough negative movers. select_by: {metric: improvement_error,
        # split: report} instead picks the lowest expected improvement error there.
        on = "sel" if sweep.get("min_negative_movers_on", "report") == "selection" else "rep"
        ok = df[df[f"{on}_manipulating_y0"] >= sweep.get("min_negative_movers", 0)]
        # Convergence guard: skip cells whose kept checkpoint is from the first few
        # epochs (a collapsing run caught early, as seen at large beta).
        if sweep.get("min_best_epoch"):
            skipped = ok[ok["best_epoch"] < sweep["min_best_epoch"]]
            for _, r in skipped.iterrows():
                print(f"  excluded (best epoch {r['best_epoch']} < {sweep['min_best_epoch']}): "
                      f"{r['profile']}, beta={r['beta']:g}")
            ok = ok[ok["best_epoch"] >= sweep["min_best_epoch"]]
        sb = sweep.get("select_by", {"metric": "improvement_rate", "split": "selection"})
        key = ("sel_" if sb["split"] == "selection" else "rep_") + "expected_" + sb["metric"]
        best = (ok.sort_values(key, ascending=sb["metric"] == "improvement_error")
                  .groupby("clip_mode", sort=False).head(5))
        view = {key: f"SELECT: {sb['metric']} ({sel if sb['split'] == 'selection' else rep})", **view}
        if "rep_expected_improvement_error" in df:
            view["rep_expected_improvement_error"] = f"error ({rep})"
            print(f"\n== expected improvement error on {rep} (rows: alpha profile, columns: beta)")
            for mode in grid["clip_mode"]:
                part = df[df["clip_mode"] == mode]
                print(part.pivot(index="profile", columns="beta", values="rep_expected_improvement_error")
                      .reindex(profiles).to_string())
        cols = ["clip_mode", "profile", "beta"] + list(dict.fromkeys(view))
        best[cols].rename(columns=view).to_csv(out_dir / "best_settings.csv", index=False)
        print(f"\n== Top settings per clip_mode (by {key}; >= {sweep.get('min_negative_movers', 0)} negative movers on "
              f"{sel if on == 'sel' else rep})")
        print(best[cols].rename(columns=view).to_string(index=False))

    print(f"\nSaved -> {out_dir / 'results.csv'}\n         {out_dir / 'best_settings.csv'}\n"
          f"         {out_dir / 'improvement_rate_vs_beta.png'} (.pdf)")


if __name__ == "__main__":
    main()
