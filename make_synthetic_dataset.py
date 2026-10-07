"""Synthetic dataset whose labels follow a known logistic regression.

    python make_synthetic_dataset.py            # defaults below
    python make_synthetic_dataset.py --n 20000 --weights 1.5 1.2 0.9 0.6 0.3 --bias -0.5 --seed 42

    x_j ~ N(0, 1) independent (j = 1..d),  eta(x) = sigmoid(w* . x + b*),  y ~ Bernoulli(eta(x)).

Writes data/synthetic_logistic.csv (features x1..xd, eta_true, label) and
data/synthetic_logistic_params.json (true parameters and summary). eta_true is
the true P(y = 1 | x); it is not a feature (configs list the features explicitly).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--weights", type=float, nargs="+", default=[1.5, 1.2, 0.9, 0.6, 0.3])
    ap.add_argument("--bias", type=float, default=-0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_logistic.csv")
    args = ap.parse_args()

    w = np.asarray(args.weights)
    if np.any(w <= 0):
        raise ValueError("all true weights must be positive")
    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, len(w)))
    eta = 1.0 / (1.0 + np.exp(-(X @ w + args.bias)))
    y = (rng.random(args.n) < eta).astype(int)

    cols = [f"x{j + 1}" for j in range(len(w))]
    df = pd.DataFrame(X, columns=cols)
    df["eta_true"] = eta
    df["label"] = y
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    params = {
        "n": args.n, "seed": args.seed, "features": cols,
        "feature_distribution": "independent N(0, 1)",
        "true_weights": dict(zip(cols, w.tolist())), "true_bias": args.bias,
        "label_model": "y ~ Bernoulli(sigmoid(w . x + b))",
        "positive_rate": float(y.mean()), "mean_eta_true": float(eta.mean()),
        "bayes_accuracy": float(np.mean((eta >= 0.5) == y)),
    }
    with open(out.with_name(out.stem + "_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps(params, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
