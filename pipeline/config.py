"""Config loading and the on-disk layout of an experiment's outputs."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["output_dir"])
    cfg["output_dir"] = out if out.is_absolute() else PROJECT_ROOT / out
    return cfg


def output_paths(cfg):
    """All artifact locations for one experiment, created on first use."""
    root = Path(cfg["output_dir"])
    paths = {
        "root": root,
        "splits": root / "splits",
        "models": root / "models",
        "plots": root / "plots",
        "evaluation": root / "evaluation",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    paths.update(
        split_indices=paths["splits"] / "split_indices.json",
        split_summary=paths["splits"] / "split_summary.json",
        feature_types=root / "feature_types.csv",
        feature_types_json=root / "feature_types.json",
        lr_model=paths["models"] / "logistic_regression.joblib",
        lr_meta=paths["models"] / "logistic_regression_meta.json",
        strat_model=paths["models"] / "strat_imp_aware.pt",
        strat_meta=paths["models"] / "strat_imp_aware.json",
        strat_history=paths["models"] / "strat_imp_aware_history.csv",
    )
    return paths
