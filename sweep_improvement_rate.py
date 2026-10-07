"""Sweep STRAT-IMP-AWARE settings and measure the improvement rate.

    python sweep_improvement_rate.py --sweep configs/sweeps/improvement_rate.yaml

For every variant: train Algorithm 1 exactly as stage 4 (checkpoint chosen by
validation loss), then compute the exact expected improvement rate
(mean pi_Imp over manipulating agents with y = 0) on the selection split and
on the report split. Nothing here overwrites the stage 4 model.
"""

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import yaml

from pipeline.config import PROJECT_ROOT, load_config, output_paths
from pipeline.data import load_split
from pipeline.evaluation import improvement_error, improvement_stats
from stage4_train_strat_imp import DTYPE, build_eta, train_from_config


def resolve_variant(cfg, paths, variant):
    """Base `strat_imp_aware` section with the variant's overrides applied.
    alpha: EQUIV keeps the agents' costs equal to the base alpha under standard
    scaling: one model-space unit of feature j is scale_j / sd_j standard
    deviations, so alpha_j = base_alpha_j * scale_j / sd_j."""
    scfg = copy.deepcopy(cfg["strat_imp_aware"])
    scfg.update({k: v for k, v in variant.items() if k != "name"})
    if scfg["alpha"] == "EQUIV":
        _, eta = build_eta(cfg, paths, scfg.get("feature_scaling", "standard"))
        base = cfg["strat_imp_aware"]["alpha"]
        scfg["alpha"] = {f: float(base[f] * eta.scale[i] / eta.lr_scale[i]) for i, f in enumerate(eta.features)}
    return scfg


def run_variant(args):
    cfg_path, scfg, sel_split, rep_split = args
    torch.set_num_threads(1)
    cfg = load_config(cfg_path)
    paths = output_paths(cfg)
    target = cfg["dataset"]["target"]

    try:
        run = train_from_config(cfg, paths, scfg, log=lambda *_: None)
    except Exception as e:  # one bad variant should not sink the whole sweep
        return {"error": f"{type(e).__name__}: {e}"}
    best, eta, hp, features = run["best"], run["eta"], run["hp"], run["features"]
    x_max = load_split(paths, cfg["logistic_regression"]["train_split"])[features].max().to_numpy()

    row = {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
           "w": [round(v, 5) for v in best["w"].tolist()], "b": round(float(best["b"]), 5),
           "alpha_used": {f: round(v, 5) for f, v in scfg["alpha"].items()}}
    for tag, split in (("sel", sel_split), ("rep", rep_split)):
        df = load_split(paths, split)
        u = eta.to_model(torch.tensor(df[features].to_numpy(), dtype=DTYPE))
        _, _, _, s = improvement_stats(u, df[target].to_numpy(), best["w"], best["b"], hp, eta, x_max, run["sim_fn"])
        s["j_star"] = features[s["j_star"]]
        s["expected_improvement_error"] = improvement_error(u, df[target].to_numpy(), best["w"], best["b"], hp, eta,
                                                            run["sim_fn"])["expected_improvement_error"]
        row.update({f"{tag}_{k}": v for k, v in s.items()})
    return row


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

    # Identical variants (e.g. the default in several groups) are trained once.
    jobs, keys, entries = {}, [], []
    for group, variants in sweep["groups"].items():
        for v in variants:
            scfg = resolve_variant(cfg, paths, v)
            key = json.dumps(scfg, sort_keys=True)
            jobs.setdefault(key, (cfg_path, scfg, sweep["selection_split"], sweep["report_split"]))
            entries.append((group, v["name"], key))

    print(f"{len(entries)} variants ({len(jobs)} unique trainings) on {sweep.get('workers', os.cpu_count())} workers")
    with ProcessPoolExecutor(max_workers=sweep.get("workers", os.cpu_count())) as pool:
        results = dict(zip(jobs, pool.map(run_variant, jobs.values())))

    rows = [{"group": g, "variant": n, **results[k]} for g, n, k in entries]
    df = pd.DataFrame(rows)
    if "error" in df:
        for _, r in df[df["error"].notna()].iterrows():
            print(f"FAILED {r['group']} / {r['variant']}: {r['error']}")
        df = df[df["error"].isna()]
    df.to_csv(out_dir / "results.csv", index=False)

    sel, rep = sweep["selection_split"], sweep["report_split"]
    cols = {
        "sel_expected_improvement_rate": f"rate ({sel})",
        "rep_expected_improvement_rate": f"rate ({rep})",
        "rep_manipulating_y0": f"neg movers ({rep})",
        "rep_expected_improved": f"E[improved] ({rep})",
        "rep_frac_negatives_manipulating": "neg. moving",
        "rep_frac_movers_beyond_observed_max": "x^f > max",
        "sel_j_star": "j*",
        "sel_S_w": "S(w)",
        "best_val_loss": "val loss",
        "best_epoch": "epoch",
    }
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        for group, part in df.groupby("group", sort=False):
            print(f"\n== {group}")
            print(part.set_index("variant")[list(cols)].rename(columns=cols).to_string())
    print(f"\nSaved -> {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
