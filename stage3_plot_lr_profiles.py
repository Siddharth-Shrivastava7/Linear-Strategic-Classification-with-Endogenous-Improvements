"""Stage 3: for each selected continuous/ordinal feature, plot the saved LR's
P(y=1 | x) against the feature's observed values (mean +/- spread over the rows
that share that value), with the empirical rate of y = 1 overlaid.

    python stage3_plot_lr_profiles.py --config configs/retiring_adult.yaml

Nominal features are skipped.
"""

import argparse
import json

import matplotlib.pyplot as plt

from pipeline.config import load_config, output_paths
from pipeline.data import load_split
from pipeline.lr_model import load_lr, positive_proba
from pipeline.plotting import draw_profile, feature_profile, new_profile_axes, save_figure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = output_paths(cfg)
    pcfg = cfg["plots"]
    target = cfg["dataset"]["target"]
    labels = cfg.get("feature_labels") or {}

    model = load_lr(paths)
    with open(paths["lr_meta"]) as f:
        meta = json.load(f)
    features = meta["features"]
    plotted = [f for f in features if meta["feature_types"][f] in ("continuous", "ordinal")]
    skipped = [f for f in features if f not in plotted]

    data = load_split(paths, pcfg["data_split"])
    p = positive_proba(model, data[features])
    y = data[target].to_numpy()
    print(f"Profiles on {pcfg['data_split']} ({len(data)} rows); spread = {pcfg['spread']}")
    if skipped:
        print(f"Skipping nominal features: {skipped}")

    profiles = {}
    for f in plotted:
        prof, binned = feature_profile(data[f], p, y, pcfg["max_points"], pcfg["n_bins"])
        profiles[f] = (prof, binned)
        prof.to_csv(paths["plots"] / f"lr_profile_{f}.csv", index=False)

        fig, axes = new_profile_axes(1)
        draw_profile(axes[0, 0], axes[1, 0], prof, f, labels.get(f, f), pcfg["spread"], pcfg["low_count"], binned)
        save_figure(fig, paths["plots"] / f"lr_profile_{f}")
        print(f"  {f}: {len(prof)} {'bins' if binned else 'observed values'} -> lr_profile_{f}.png/.pdf/.csv")

    if len(plotted) > 1:
        fig, axes = new_profile_axes(len(plotted))
        for i, f in enumerate(plotted):
            prof, binned = profiles[f]
            draw_profile(axes[0, i], axes[1, i], prof, f, labels.get(f, f), pcfg["spread"], pcfg["low_count"],
                         binned, legend=(i == 0))
            if i:
                axes[0, i].set_ylabel("")
        fig.subplots_adjust(wspace=0.18)
        save_figure(fig, paths["plots"] / "lr_profile_all")
        print("  all features -> lr_profile_all.png/.pdf")
    plt.close("all")


if __name__ == "__main__":
    main()
