"""P(y=1 | x_j) profiles of the fitted LR along each observed feature."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LR_COLOR = "#2a78d6"         # model: mean predicted P(y=1) + spread
EMPIRICAL_COLOR = "#eb6834"  # data: observed rate of y = 1
COUNT_COLOR = "#b7b6b0"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e0"

SPREAD_LABELS = {"std": "±1 SD", "iqr": "IQR (25–75%)", "p10_p90": "10–90%"}


def feature_profile(x, p, y, max_points, n_bins):
    """Group rows by observed value of x (or quantile bins of x when there are
    too many unique values) and summarise the LR's P(y=1) and the empirical
    rate within each group. No other feature is held fixed: each group's mean
    averages over whatever the other features are in the data."""
    df = pd.DataFrame({"x": np.asarray(x, dtype=float), "p": p, "y": y})
    if df["x"].nunique() > max_points:
        df["group"] = pd.qcut(df["x"], q=n_bins, duplicates="drop")
        binned = True
    else:
        df["group"] = df["x"]
        binned = False
    g = df.groupby("group", observed=True)
    prof = pd.DataFrame(
        {
            "x": g["x"].mean(),
            "n": g.size(),
            "p_mean": g["p"].mean(),
            "p_std": g["p"].std(ddof=0),
            "p_q10": g["p"].quantile(0.10),
            "p_q25": g["p"].quantile(0.25),
            "p_q75": g["p"].quantile(0.75),
            "p_q90": g["p"].quantile(0.90),
            "empirical_rate": g["y"].mean(),
        }
    ).reset_index(drop=not binned)
    if binned:
        prof = prof.rename(columns={"group": "bin"})
    return prof.sort_values("x").reset_index(drop=True), binned


def spread_bounds(prof, spread):
    if spread == "std":
        lo, hi = prof["p_mean"] - prof["p_std"], prof["p_mean"] + prof["p_std"]
    elif spread == "iqr":
        lo, hi = prof["p_q25"], prof["p_q75"]
    elif spread == "p10_p90":
        lo, hi = prof["p_q10"], prof["p_q90"]
    else:
        raise ValueError(f"unknown spread {spread!r}; use one of {list(SPREAD_LABELS)}")
    return lo.clip(0, 1), hi.clip(0, 1)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_profile(ax, ax_count, prof, feature, label, spread, low_count, binned, legend=True):
    lo, hi = spread_bounds(prof, spread)
    ax.fill_between(prof["x"], lo, hi, color=LR_COLOR, alpha=0.18, linewidth=0,
                    label=f"LR P(y=1) spread ({SPREAD_LABELS[spread]})")
    ax.plot(prof["x"], prof["p_mean"], color=LR_COLOR, linewidth=2, marker="o", markersize=3.5,
            label="LR mean P(y=1 | x)")

    enough = prof["n"] >= low_count
    ax.scatter(prof.loc[enough, "x"], prof.loc[enough, "empirical_rate"], s=26, color=EMPIRICAL_COLOR,
               edgecolor="white", linewidth=0.6, zorder=3, label="Empirical rate of y = 1")
    if (~enough).any():
        ax.scatter(prof.loc[~enough, "x"], prof.loc[~enough, "empirical_rate"], s=26, facecolor="none",
                   edgecolor=EMPIRICAL_COLOR, linewidth=1, zorder=3, label=f"Empirical rate (n < {low_count})")

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(y = 1 | x)", color=TEXT, fontsize=10)
    ax.set_title(feature, color=TEXT, fontsize=12, loc="left", fontweight="semibold")
    _style(ax)
    if legend:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=TEXT)

    gaps = np.diff(prof["x"].to_numpy())
    width = 0.8 * (gaps.min() if len(gaps) else 1.0)
    ax_count.bar(prof["x"], prof["n"], width=width, color=COUNT_COLOR, linewidth=0)
    ax_count.set_ylabel("rows" + (" / bin" if binned else ""), color=TEXT_MUTED, fontsize=9)
    ax_count.set_xlabel(label, color=TEXT, fontsize=10)
    _style(ax_count)
    if not binned and len(prof) <= 30:
        step = 1 if len(prof) <= 15 else 2  # avoid overlapping labels
        ticks = prof["x"].iloc[::step]
        ax_count.set_xticks(ticks)
        ax_count.set_xticklabels([f"{v:g}" for v in ticks], fontsize=8)


def new_profile_axes(n_features=1):
    fig, axes = plt.subplots(2, n_features, figsize=(6.4 * n_features, 5.6), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, squeeze=False)
    return fig, axes


def save_figure(fig, stem):
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
