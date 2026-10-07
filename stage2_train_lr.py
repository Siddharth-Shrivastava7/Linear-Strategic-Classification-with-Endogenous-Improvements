"""Stage 2: train the logistic regression on the selected features using the
`lr_train` split, and save it.

    python stage2_train_lr.py --config configs/retiring_adult.yaml
"""

import argparse

from pipeline.config import load_config, output_paths
from pipeline.data import load_feature_types, load_split
from pipeline.lr_model import build_lr_pipeline, build_meta, evaluate, save_lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = output_paths(cfg)
    lr_cfg = cfg["logistic_regression"]
    target = cfg["dataset"]["target"]
    features = lr_cfg["features"]

    ft = load_feature_types(paths)
    missing = [f for f in features if f not in ft["kept_features"]]
    if missing:
        raise ValueError(f"selected features {missing} are not in the kept dataset ({ft['kept_features']})")
    types = ft["types"]

    train = load_split(paths, lr_cfg["train_split"])
    X, y = train[features], train[target]
    print(f"Training LR on {lr_cfg['train_split']} ({len(train)} rows) with features "
          + ", ".join(f"{f} [{types[f]}]" for f in features))

    model = build_lr_pipeline(features, types, cfg, X)
    model.fit(X, y)

    # Train metrics, plus a held-out diagnostic on the other splits (nothing is
    # fitted on them here).
    metrics = {}
    for name in cfg["split"]["fractions"]:
        part = train if name == lr_cfg["train_split"] else load_split(paths, name)
        metrics[name] = evaluate(model, part[features], part[target])

    meta = build_meta(model, features, types, metrics, cfg)
    save_lr(model, meta, paths)

    print("\nCoefficients (standardized features):")
    print(f"  intercept   {meta['standardized_space']['intercept']:+.4f}")
    for f, w in meta["standardized_space"]["coef"].items():
        print(f"  {f:<11} {w:+.4f}")
    if meta["raw_space"]:
        print("Coefficients (original feature scale):")
        print(f"  intercept   {meta['raw_space']['intercept']:+.6f}")
        for f, w in meta["raw_space"]["coef"].items():
            print(f"  {f:<11} {w:+.6f}")

    print("\nMetrics" + " " * 9 + "accuracy  log_loss  roc_auc   brier  mean P(y=1)  true P(y=1)")
    for name, m in metrics.items():
        tag = "(train)" if name == lr_cfg["train_split"] else "(held out)"
        print(f"  {name:<10}{tag:<11}{m['accuracy']:.4f}    {m['log_loss']:.4f}   {m['roc_auc']:.4f}  "
              f"{m['brier']:.4f}     {m['mean_predicted_proba']:.4f}       {m['positive_rate_true']:.4f}")
    print(f"\nSaved model -> {paths['lr_model']}\nSaved metadata -> {paths['lr_meta']}")


if __name__ == "__main__":
    main()
