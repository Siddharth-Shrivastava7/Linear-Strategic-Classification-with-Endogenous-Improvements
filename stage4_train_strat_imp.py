"""Stage 4: train STRAT-IMP-AWARE (Algorithm 1) on `algo_train`, select on
`algo_val`, with the saved LR (frozen) as eta_hat. Saves the best (w, b).

    python stage4_train_strat_imp.py --config configs/retiring_adult.yaml
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pipeline.config import load_config, output_paths
from pipeline.data import load_split
from pipeline.lr_model import load_lr, positive_proba
from pipeline.plotting import EMPIRICAL_COLOR, LR_COLOR, TEXT, TEXT_MUTED, _style
from pipeline.strat_imp import FrozenLR, model_space, to_raw_space, train_strat_imp_aware
from pipeline.strat_imp_clipped import make_sim_imp_prob

DTYPE = torch.float64


def hyperparameters(scfg, features):
    return {
        "epochs": scfg["epochs"], "lr": scfg["lr"], "batch_size": scfg["batch_size"],
        "beta": scfg["beta"], "delta_g": scfg["delta_g"], "seed": scfg["seed"],
        "init": scfg.get("init", "uniform"), "feature_scaling": scfg.get("feature_scaling", "standard"),
        "clip_mode": scfg.get("clip_mode", "none"), "restarts": int(scfg.get("restarts", 1)),
        "class_weight": scfg.get("class_weight"),
        "alpha": torch.tensor([float(scfg["alpha"][f]) for f in features], dtype=DTYPE),
    }


def build_eta(cfg, paths, scaling):
    """Frozen LR taking inputs in the model space chosen by `scaling`."""
    sk_lr = load_lr(paths)
    base = FrozenLR(sk_lr, dtype=DTYPE)
    x_ref = load_split(paths, cfg["logistic_regression"]["train_split"])[base.features].to_numpy()
    shift, scale = model_space(scaling, base.lr_mean.numpy(), base.lr_scale.numpy(), x_ref)
    return sk_lr, FrozenLR(sk_lr, shift, scale, DTYPE)


def build_sim_fn(cfg, paths, eta, clip_mode):
    """Algorithm 2 as written, or bounded by the largest value of each feature
    observed in the LR's training split (eta_hat's data support)."""
    if clip_mode == "none":
        return make_sim_imp_prob("none")
    x_max = load_split(paths, cfg["logistic_regression"]["train_split"])[eta.features].max().to_numpy()
    return make_sim_imp_prob(clip_mode, eta.to_model(torch.tensor(x_max, dtype=DTYPE)))


def load_model_space(paths, split, features, target, eta):
    df = load_split(paths, split)
    x = torch.tensor(df[features].to_numpy(), dtype=DTYPE)
    y = torch.tensor(df[target].to_numpy(), dtype=DTYPE)
    return df, eta.to_model(x), y


def train_from_config(cfg, paths, scfg, log=print, train_rows=None):
    """Everything stage 4 does except saving; reused by the experiments.
    train_rows: optional row_ids to train on (a subset of the train split)."""
    target = cfg["dataset"]["target"]
    sk_lr, eta = build_eta(cfg, paths, scfg.get("feature_scaling", "standard"))
    features = eta.features
    missing_alpha = [f for f in features if f not in scfg["alpha"]]
    if missing_alpha:
        raise KeyError(f"strat_imp_aware.alpha is missing {missing_alpha}")

    df_tr, u_tr, y_tr = load_model_space(paths, scfg["train_split"], features, target, eta)
    if train_rows is not None:
        pos = df_tr.index.get_indexer(train_rows)
        if (pos < 0).any():
            raise KeyError(f"{int((pos < 0).sum())} train_rows are not in {scfg['train_split']}")
        u_tr, y_tr = u_tr[pos], y_tr[pos]
    df_val, u_val, y_val = load_model_space(paths, scfg["val_split"], features, target, eta)

    # The torch copy of eta_hat must reproduce the saved sklearn LR.
    with torch.no_grad():
        diff = np.max(np.abs(eta(u_val).numpy() - positive_proba(sk_lr, df_val[features])))
    if diff > 1e-9:
        raise RuntimeError(f"FrozenLR disagrees with the saved LR (max |diff| = {diff:.2e})")

    hp = hyperparameters(scfg, features)
    log(f"Features: {features}   feature_scaling: {hp['feature_scaling']}   init: {hp['init']}   "
        f"clip_mode: {hp['clip_mode']}")
    log(f"Train {scfg['train_split']}: {len(u_tr)} rows   Val {scfg['val_split']}: {len(u_val)} rows")
    log(f"FrozenLR matches sklearn LR (max |diff| = {diff:.1e}); its parameters are buffers, not trained")
    log("Hyperparameters: " + ", ".join(f"{k}={v if k != 'alpha' else dict(zip(features, v.tolist()))}"
                                        for k, v in hp.items()) + "\n")
    torch.manual_seed(scfg["seed"])
    sim_fn = build_sim_fn(cfg, paths, eta, hp["clip_mode"])
    # Restarts: Algorithm 1 is non-convex (j* is a hard choice), so a random start
    # can settle along a worse direction. Train from `restarts` random starts
    # (seeds seed, seed + 1000, ...) and keep the one with the lowest validation
    # loss, the same criterion Algorithm 1 uses to pick its checkpoint.
    best, history = None, None
    for k in range(hp["restarts"]):
        hp_k = dict(hp, seed=hp["seed"] + 1000 * k)
        torch.manual_seed(hp_k["seed"])
        quiet = hp["restarts"] > 1
        b_k, h_k = train_strat_imp_aware(u_tr, y_tr, u_val, y_val, eta, hp_k,
                                         log=(lambda *_: None) if quiet else log, sim_fn=sim_fn)
        if quiet:
            j = h_k[b_k["epoch"] - 1]["j_star"]
            log(f"restart {k} (seed {hp_k['seed']}): best epoch {b_k['epoch']}, val loss {b_k['val_loss']:.5f}, "
                f"j* = {features[j]}")
        if best is None or b_k["val_loss"] < best["val_loss"]:
            best, history = dict(b_k, restart=k, restart_seed=hp_k["seed"]), h_k
    if hp["restarts"] > 1:
        log(f"-> kept restart {best['restart']} (seed {best['restart_seed']}), val loss {best['val_loss']:.5f}")
    return {"best": best, "history": history, "eta": eta, "hp": hp, "features": features, "sim_fn": sim_fn,
            "train": (u_tr, y_tr), "val": (u_val, y_val)}


def plot_history(hist, best_epoch, path_stem):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(hist["epoch"], hist["train_loss"], color=LR_COLOR, linewidth=2, label="Train loss (algo_train)")
    ax.plot(hist["epoch"], hist["val_loss"], color=EMPIRICAL_COLOR, linewidth=2, label="Validation loss (algo_val)")
    ax.axvline(best_epoch, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.annotate(f"best epoch {best_epoch}", (best_epoch, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=TEXT_MUTED, fontsize=8.5)
    ax.set_xlabel("epoch", color=TEXT, fontsize=10)
    ax.set_ylabel("STRAT-IMP-LOSS (mean)", color=TEXT, fontsize=10)
    ax.set_title("STRAT-IMP-AWARE training", color=TEXT, fontsize=12, loc="left", fontweight="semibold")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXT)
    fig.savefig(f"{path_stem}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = output_paths(cfg)
    scfg = cfg["strat_imp_aware"]

    run = train_from_config(cfg, paths, scfg)
    best, history, eta, hp, features = run["best"], run["history"], run["eta"], run["hp"], run["features"]

    hist = pd.DataFrame(history).rename(columns={f"w_{i}": f"w_{f}" for i, f in enumerate(features)})
    hist.to_csv(paths["strat_history"], index=False)
    plot_history(hist, best["epoch"], paths["models"] / "strat_imp_aware_loss")

    w_m, b_m = best["w"].numpy(), float(best["b"])
    w_raw, b_raw = to_raw_space(w_m, b_m, eta.shift.numpy(), eta.scale.numpy())
    torch.save({"w": best["w"], "b": best["b"], "features": features,
                "shift": eta.shift, "scale": eta.scale, "feature_scaling": hp["feature_scaling"]},
               paths["strat_model"])
    meta = {
        "decision_rule": "positive iff w . u - b >= 0, u = (x - shift) / scale  (equivalently w_raw . x - b_raw >= 0)",
        "features": features,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "restart": {"kept": best.get("restart", 0), "seed": best.get("restart_seed", hp["seed"]),
                    "restarts": hp["restarts"]},
        "model_space": {"w": dict(zip(features, map(float, w_m))), "b": b_m},
        "raw_space": {"w": dict(zip(features, map(float, w_raw))), "b": float(b_raw)},
        "feature_scaling": {"kind": hp["feature_scaling"],
                            "shift": dict(zip(features, eta.shift.tolist())),
                            "scale": dict(zip(features, eta.scale.tolist())),
                            "stats_from": cfg["logistic_regression"]["train_split"]},
        "hyperparameters": {**{k: v for k, v in hp.items() if k != "alpha"}, "alpha": scfg["alpha"]},
        "train_split": scfg["train_split"],
        "val_split": scfg["val_split"],
    }
    with open(paths["strat_meta"], "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nBest epoch {best['epoch']}: val loss {best['val_loss']:.5f}")
    print(f"  model space ({hp['feature_scaling']}): w = " + ", ".join(f"{f} {v:.5f}" for f, v in zip(features, w_m))
          + f";  b = {b_m:.5f}")
    print("  raw units:    w = " + ", ".join(f"{f} {v:.6f}" for f, v in zip(features, w_raw)) + f";  b = {b_raw:.5f}")
    print(f"Saved -> {paths['strat_model']}\n         {paths['strat_meta']}\n         {paths['strat_history']}")


if __name__ == "__main__":
    main()
