"""Stage 5a: improvement rate of the saved STRAT-IMP-AWARE model.

    python stage5_improvement_rate.py --config configs/retiring_adult.yaml

On the evaluation split, each agent best-responds to the saved (w, b) as in
Algorithm 2. Manipulating agents are those that move (0 < b - w.u <= S(w)).
Each mover's post-response label is sampled:
    original y = 1 -> stays 1
    original y = 0 -> y' ~ Bernoulli(pi_Imp),  pi_Imp from Algorithm 2 with the frozen LR
An agent is improved if it moves, y = 0 and y' = 1.
Improvement rate = improved / manipulating agents that start negative (y = 0).
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from pipeline.config import load_config, output_paths
from pipeline.data import load_split
from pipeline.evaluation import improvement_stats, sample_post_labels
from pipeline.lr_model import load_lr
from pipeline.strat_imp import FrozenLR
from stage4_train_strat_imp import build_sim_fn

DTYPE = torch.float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", help="evaluate on this split instead of evaluation.split")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = output_paths(cfg)
    ecfg = dict(cfg["evaluation"])
    if args.split:
        ecfg["split"] = args.split
    if ecfg["split"] in (cfg["strat_imp_aware"]["train_split"], cfg["strat_imp_aware"]["val_split"]):
        print(f"note: {ecfg['split']} was used to train / select the model, so this is not an unseen-data estimate")
    target = cfg["dataset"]["target"]

    ckpt = torch.load(paths["strat_model"])
    with open(paths["strat_meta"]) as f:
        meta = json.load(f)
    features = meta["features"]
    eta = FrozenLR(load_lr(paths), ckpt["shift"], ckpt["scale"], DTYPE)
    if features != eta.features:
        raise ValueError(f"model features {features} differ from LR features {eta.features}")
    hp = dict(meta["hyperparameters"])
    hp["alpha"] = torch.tensor([hp["alpha"][f] for f in features], dtype=DTYPE)
    w, b = ckpt["w"].to(DTYPE), ckpt["b"].to(DTYPE)

    df = load_split(paths, ecfg["split"])
    u = eta.to_model(torch.tensor(df[features].to_numpy(), dtype=DTYPE))
    y = df[target].to_numpy().astype(int)
    x_max = load_split(paths, cfg["logistic_regression"]["train_split"])[features].max().to_numpy()
    hp.setdefault("clip_mode", "none")
    sim_fn = build_sim_fn(cfg, paths, eta, hp["clip_mode"])
    pi_hat, moved, cand, s = improvement_stats(u, y, w, b, hp, eta, x_max, sim_fn)
    n_cand = s["manipulating_y0"]

    # Headline: one draw with the configured seed; then further independent draws.
    rng = np.random.default_rng(ecfg["seed"])
    y_post = sample_post_labels(y, pi_hat, cand, rng)
    improved = cand & (y_post == 1)
    n_improved = int(improved.sum())
    repeats = np.array([int(sample_post_labels(y, pi_hat, cand, rng)[cand].sum()) for _ in range(ecfg["n_repeats"])])

    rate = n_improved / n_cand if n_cand else float("nan")
    result = {
        "split": ecfg["split"],
        "model": str(paths["strat_model"]),
        "model_best_epoch": meta["best_epoch"],
        "feature_scaling": hp["feature_scaling"],
        "clip_mode": hp["clip_mode"],
        "seed": ecfg["seed"],
        "n_agents": s["n_agents"],
        "j_star": features[s["j_star"]],
        "S_w": s["S_w"],
        "manipulating_agents": s["manipulating"],
        "manipulating_by_original_label": {"y=0": n_cand, "y=1": s["manipulating_y1"]},
        "improved_agents": n_improved,
        "improvement_rate": rate,
        "improvement_rate_definition": "improved / manipulating agents with original y = 0",
        "fraction_of_agents_manipulating": s["manipulating"] / s["n_agents"],
        "frac_movers_beyond_observed_max": s.get("frac_movers_beyond_observed_max"),
        "expected_improved_agents": s["expected_improved"],
        "expected_improvement_rate": s["expected_improvement_rate"],
        f"repeats_{ecfg['n_repeats']}": {
            "improved_mean": float(repeats.mean()),
            "improved_std": float(repeats.std(ddof=1)),
            "improvement_rate_mean": float(repeats.mean() / n_cand),
            "improvement_rate_std": float(repeats.std(ddof=1) / n_cand),
        },
        "pi_imp_among_negative_movers": {
            "mean": float(pi_hat[cand].mean()),
            "median": float(np.median(pi_hat[cand])),
            "min": float(pi_hat[cand].min()),
            "max": float(pi_hat[cand].max()),
        },
    }
    out = paths["evaluation"]
    with open(out / f"improvement_rate_{ecfg['split']}.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(
        {"y": y, "moved": moved, "pi_imp": pi_hat, "y_post": y_post, "improved": improved}, index=df.index
    ).to_csv(out / f"improvement_agents_{ecfg['split']}.csv")

    print(f"Model: {paths['strat_model'].name} (epoch {meta['best_epoch']}, {hp['feature_scaling']} scaling)   "
          f"Split: {ecfg['split']} ({s['n_agents']} agents)")
    print(f"Improvement direction j* = {result['j_star']},  reach S(w) = {s['S_w']:.4f}\n")
    print(f"Manipulating agents        {s['manipulating']:>6}  ({result['fraction_of_agents_manipulating']:.2%} of agents)")
    print(f"  original label y = 0     {n_cand:>6}   <- denominator")
    print(f"  original label y = 1     {s['manipulating_y1']:>6}")
    print(f"Improved agents            {n_improved:>6}   (seed {ecfg['seed']})")
    print(f"Improvement rate           {rate:.4f}   = {n_improved} / {n_cand}")
    r = result[f"repeats_{ecfg['n_repeats']}"]
    print(f"\nOver {ecfg['n_repeats']} draws: improved {r['improved_mean']:.1f} ± {r['improved_std']:.1f}, "
          f"rate {r['improvement_rate_mean']:.4f} ± {r['improvement_rate_std']:.4f}")
    print(f"Expectation (sum of pi_Imp): improved {s['expected_improved']:.1f}, rate {s['expected_improvement_rate']:.4f}")
    print(f"clip_mode {hp['clip_mode']}: movers whose boundary point lies past the largest observed "
          f"{result['j_star']}: {result['frac_movers_beyond_observed_max']:.2%}")
    split = ecfg["split"]
    print(f"\nSaved -> {out / f'improvement_rate_{split}.json'}\n         {out / f'improvement_agents_{split}.csv'}")


if __name__ == "__main__":
    main()
