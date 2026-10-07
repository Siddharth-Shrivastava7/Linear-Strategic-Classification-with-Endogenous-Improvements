"""Stage 1: type every feature, drop removed types and missing rows, and make
the fixed random 45/45/10 split.

    python stage1_split_and_profile.py --config configs/retiring_adult.yaml

The split is written once and then kept fixed: re-running refuses to
overwrite it unless --force is given.
"""

import argparse
import json

import pandas as pd

from pipeline.config import load_config, output_paths
from pipeline.data import kept_features, load_raw, prepare_dataset, random_split, resolve_feature_types, save_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = output_paths(cfg)

    raw = load_raw(cfg)
    info = raw.attrs.get("load_info", {})
    print(f"Loaded {cfg['dataset']['path']}: {info.get('rows_file', raw.shape[0])} rows x {raw.shape[1]} columns")
    if cfg["dataset"].get("missing_values"):
        print(f"  codes {cfg['dataset']['missing_values']} treated as missing")
    if "rows_dropped_all_missing" in info:
        print(f"  dropped {info['rows_dropped_all_missing']} rows with every feature missing -> {raw.shape[0]} rows")
    if cfg["dataset"].get("negate_features"):
        print(f"  negated (higher = better): {cfg['dataset']['negate_features']}")
    if cfg["dataset"].get("use_features"):
        print(f"  features used (rows with missing values dropped only for these): {cfg['dataset']['use_features']}")
    print(f"Target: {cfg['dataset']['target']}  ({cfg['dataset']['label_description']})\n")

    types = resolve_feature_types(raw, cfg)
    types.to_csv(paths["feature_types"])
    kept = kept_features(types, cfg)
    with open(paths["feature_types_json"], "w") as f:
        json.dump({"types": types["final_type"].to_dict(), "kept_features": kept,
                   "dropped_types": cfg["feature_types"].get("drop_types") or []}, f, indent=2)

    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print("Feature types")
        print(types[["dtype", "n_unique", "n_missing", "min", "max", "detected_type", "final_type", "source"]].to_string())
        mismatched = types[types["detected_type"] != types["final_type"]]
        if len(mismatched):
            print("\nOverrides that differ from auto-detection:")
            for f, r in mismatched.iterrows():
                print(f"  {f}: detected {r.detected_type} ({r.detected_reason}) -> {r.final_type}")
    print(f"\nKept features: {kept}")
    print(f"Dropped feature types: {cfg['feature_types'].get('drop_types')}\n")

    if paths["split_indices"].exists() and not args.force:
        with open(paths["split_summary"]) as f:
            summary = json.load(f)
        print(f"Split already exists at {paths['splits']} and is kept fixed (use --force to redo it).")
    else:
        df, clean_info = prepare_dataset(raw, types, cfg)
        print(f"Dropped {clean_info['rows_dropped_missing']} rows with missing values -> {clean_info['rows_clean']} rows")
        target = cfg["dataset"]["target"]
        strat = df[target].to_numpy() if cfg["split"].get("stratify", False) else None
        splits = random_split(df.index, cfg["split"]["fractions"], cfg["split"]["seed"], labels=strat)
        summary = save_split(df, splits, paths, cfg, clean_info)
        print(f"Saved split to {paths['splits']}")

    print(f"\nRandom split (seed {summary['seed']}, "
          f"{'stratified by ' + cfg['dataset']['target'] if summary.get('stratified') else 'not stratified'})")
    for name, s in summary["splits"].items():
        print(f"  {name:<11} {s['rows']:>7} rows  ({s['fraction']:.2%})   P(y=1) = {s['positive_rate']:.4f}")


if __name__ == "__main__":
    main()
