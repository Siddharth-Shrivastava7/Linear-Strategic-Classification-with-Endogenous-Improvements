"""Loading, feature typing, cleaning and the fixed random split."""

import hashlib
import json

import numpy as np
import pandas as pd

FEATURE_TYPES = ("continuous", "ordinal", "nominal")
ROW_ID = "row_id"  # 0-based row position in the raw CSV, kept through every stage


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


NEG_PREFIX = "neg_"


def load_raw(cfg):
    """Read the CSV and apply the optional dataset-level options:
        missing_values:          codes that mean "missing" (set to NaN)
        drop_rows_all_missing:   drop rows whose features are all missing
        negate_features:         replace x by -x (renamed neg_<x>) so that higher = better
    Row ids stay those of the raw CSV. Counts are kept in df.attrs["load_info"]."""
    dcfg = cfg["dataset"]
    df = pd.read_csv(dcfg["path"])
    df.index.name = ROW_ID
    target = dcfg["target"]
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in dataset columns {list(df.columns)}")
    info = {"rows_file": len(df)}
    features = [c for c in df.columns if c != target]
    codes = dcfg.get("missing_values") or []
    if codes:
        df[features] = df[features].mask(df[features].isin(codes))
    if dcfg.get("drop_rows_all_missing", False):
        all_missing = df[features].isna().all(axis=1)
        info["rows_dropped_all_missing"] = int(all_missing.sum())
        df = df[~all_missing]
    for f in dcfg.get("negate_features") or []:
        if f not in features:
            raise KeyError(f"negate_features: {f!r} is not a feature")
        df[NEG_PREFIX + f] = -df[f]
        df = df.drop(columns=f)
    df.attrs["load_info"] = info
    return df


# --------------------------------------------------------------------------- #
# Feature typing
# --------------------------------------------------------------------------- #
def infer_feature_type(s, max_discrete_unique):
    """Heuristic type of one column -> (type, reason).

    Numeric codes for categories cannot be told apart from ordinal levels by
    looking at the data alone, so every result here should be confirmed (and
    overridden in the config where needed).
    """
    values = s.dropna()
    n_unique = values.nunique()
    if pd.api.types.is_bool_dtype(s) or n_unique <= 2:
        return "nominal", "binary"
    if not pd.api.types.is_numeric_dtype(s):
        return "nominal", "non-numeric values"
    if not np.all(np.isclose(values, np.round(values))):
        return "continuous", "non-integer numeric values"
    if n_unique > max_discrete_unique:
        return "continuous", f"integer-valued, {n_unique} > {max_discrete_unique} unique values"
    return "ordinal", f"integer-valued, {n_unique} <= {max_discrete_unique} unique values (may be nominal codes)"


def resolve_feature_types(df, cfg):
    """Per-feature report: auto-detected type, config override, and final type."""
    ft_cfg = cfg["feature_types"]
    target = cfg["dataset"]["target"]
    features = [c for c in df.columns if c != target]

    overrides = {}
    for ftype, names in (ft_cfg.get("overrides") or {}).items():
        if ftype not in FEATURE_TYPES:
            raise ValueError(f"unknown feature type {ftype!r}; use one of {FEATURE_TYPES}")
        for name in names or []:
            if name not in features:
                raise KeyError(f"override for {name!r}: not a feature of this dataset")
            if name in overrides:
                raise ValueError(f"{name!r} is listed under two feature types")
            overrides[name] = ftype

    rows = []
    for f in features:
        s = df[f]
        detected, reason = infer_feature_type(s, ft_cfg["max_discrete_unique"])
        numeric = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "feature": f,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique()),
                "n_missing": int(s.isna().sum()),
                "min": s.min() if numeric else None,
                "max": s.max() if numeric else None,
                "detected_type": detected,
                "detected_reason": reason,
                "final_type": overrides.get(f, detected),
                "source": "config override" if f in overrides else "auto-detected",
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def kept_features(types_report, cfg):
    drop = set(cfg["feature_types"].get("drop_types") or [])
    return [f for f, t in types_report["final_type"].items() if t not in drop]


# --------------------------------------------------------------------------- #
# Cleaning + split
# --------------------------------------------------------------------------- #
def binarize_target(df, cfg):
    target, pos = cfg["dataset"]["target"], cfg["dataset"]["positive_class"]
    values = df[target].unique()
    if len(values) != 2:
        raise ValueError(f"target {target!r} must be binary; found values {sorted(values)}")
    if pos not in values:
        raise ValueError(f"positive_class {pos!r} not among target values {sorted(values)}")
    df[target] = (df[target] == pos).astype(int)
    return df


def prepare_dataset(raw, types_report, cfg):
    """Drop removed feature types (and, if dataset.use_features is given, every
    feature not listed), drop rows with missing values, binarize y."""
    target = cfg["dataset"]["target"]
    features = kept_features(types_report, cfg)
    use = cfg["dataset"].get("use_features")
    if use:
        bad = [f for f in use if f not in features]
        if bad:
            raise ValueError(f"use_features {bad} are not kept features ({features})")
        features = list(use)
    df = raw[features + [target]]
    n_before = len(df)
    df = df.dropna().copy()
    df = binarize_target(df, cfg)
    return df, {"rows_raw": n_before, "rows_dropped_missing": n_before - len(df), "rows_clean": len(df)}


def _partition(ids, fractions, rng):
    perm = rng.permutation(np.asarray(ids))
    names = list(fractions)
    sizes = [int(np.floor(fractions[n] * len(perm))) for n in names[:-1]]
    sizes.append(len(perm) - sum(sizes))
    bounds = np.cumsum([0] + sizes)
    return {n: perm[a:b] for n, a, b in zip(names, bounds[:-1], bounds[1:])}


def random_split(row_ids, fractions, seed, labels=None):
    """Random partition of `row_ids` by `fractions`. With `labels`, the split is
    stratified: each class is partitioned separately with the same fractions,
    so every split keeps the overall class balance."""
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")
    rng = np.random.default_rng(seed)
    if labels is None:
        parts = [_partition(row_ids, fractions, rng)]
    else:
        row_ids, labels = np.asarray(row_ids), np.asarray(labels)
        parts = [_partition(row_ids[labels == c], fractions, rng) for c in np.unique(labels)]
    return {n: np.sort(np.concatenate([p[n] for p in parts])) for n in fractions}


def save_split(df, splits, paths, cfg, clean_info):
    target = cfg["dataset"]["target"]
    with open(paths["split_indices"], "w") as f:
        json.dump({name: ids.tolist() for name, ids in splits.items()}, f)
    summary = {
        "dataset": cfg["dataset"]["path"],
        "dataset_sha256": file_sha256(cfg["dataset"]["path"]),
        "seed": cfg["split"]["seed"],
        "stratified": bool(cfg["split"].get("stratify", False)),
        "fractions": cfg["split"]["fractions"],
        "features": [c for c in df.columns if c != target],
        "target": target,
        **clean_info,
        "splits": {},
    }
    for name, ids in splits.items():
        part = df.loc[ids]
        part.to_csv(paths["splits"] / f"{name}.csv")
        summary["splits"][name] = {
            "rows": len(part),
            "fraction": round(len(part) / len(df), 6),
            "positive_rate": round(float(part[target].mean()), 6),
        }
    with open(paths["split_summary"], "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_split(paths, name):
    """Load one saved split (index = row_id in the raw CSV)."""
    return pd.read_csv(paths["splits"] / f"{name}.csv", index_col=ROW_ID)


def load_feature_types(paths):
    with open(paths["feature_types_json"]) as f:
        return json.load(f)
