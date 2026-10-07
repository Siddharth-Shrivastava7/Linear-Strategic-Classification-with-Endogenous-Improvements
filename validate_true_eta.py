"""Synthetic data only: compare improvement measured with the estimated eta_hat
(the frozen LR, as in Algorithms 1-2) against the TRUE eta of the generator.

    python validate_true_eta.py --config configs/synthetic.yaml [--split algo_test]

Agents best-respond to the saved stage 4 model (cheapest feasible single-feature
move, as in the comparison tables). For each mover with y = 0,
    pi = clamp((eta(x^f) - eta(x)) / max(1 - eta(x), delta_g), 0, 1)
with eta = eta_hat (LR) or eta_true = sigmoid(w* . x + b*). Reports expected
improvement rate / error under both, one sampled draw under the true eta, and
how closely pi_hat tracks pi_true.
"""

import argparse
import json

import numpy as np
import torch

from experiment_pli_comparison import context, ours_margin
from pipeline.best_response import cheapest_feasible_response
from pipeline.evaluation import sample_post_labels


def true_eta_fn(ctx):
    """True P(y=1|x) of the synthetic generator, taking model-space inputs u."""
    with open(ctx["cfg"]["dataset"]["true_model_params"]) as f:
        tp = json.load(f)
    eta = ctx["eta"]
    w_true = torch.tensor([tp["true_weights"][f] for f in ctx["feats"]], dtype=torch.float64)
    b_true = float(tp["true_bias"])
    return lambda u: torch.sigmoid((u * eta.scale + eta.shift) @ w_true + b_true)


def true_eta_metrics(ctx, margin, split, seed):
    """Same agents as `evaluate` (cheapest feasible move), but improvement scored
    with the TRUE eta: expected rate / error and one sampled draw (seed)."""
    e, eta_t = ctx["env"], true_eta_fn(ctx)
    u, y = ctx["data"][split]
    y = np.asarray(y).astype(int)
    u_f, moved, _ = cheapest_feasible_response(margin, u, e["alpha"], e["beta"], e["upper"])
    moved = moved.numpy()
    cand = moved & (y == 0)
    with torch.no_grad():
        y_hat = moved | (margin(u).numpy() >= 0)
        a, b = eta_t(u), eta_t(u_f)
        pi = torch.clamp((b - a) / torch.clamp(1 - a, min=e["delta_g"]), 0, 1).numpy()
    y_post = sample_post_labels(y, pi, cand, np.random.default_rng(seed))
    n_cand = int(cand.sum())
    return {
        "true_expected_improvement_rate": float(pi[cand].mean()) if n_cand else float("nan"),
        "true_expected_improvement_error": float(((y_hat[~cand] != y[~cand]).sum() + (1 - pi[cand]).sum()) / len(y)),
        "true_expected_improved": float(pi[cand].sum()),
        "true_improved": int((cand & (y_post == 1)).sum()),
        "true_improvement_rate": int((cand & (y_post == 1)).sum()) / n_cand if n_cand else float("nan"),
        "true_improvement_error": float((y_hat != y_post).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None)
    args = ap.parse_args()

    ctx = context(args.config)
    cfg, paths, eta, e = ctx["cfg"], ctx["paths"], ctx["eta"], ctx["env"]
    split = args.split or cfg["evaluation"]["split"]
    with open(cfg["dataset"]["true_model_params"]) as f:
        tp = json.load(f)
    w_true = torch.tensor([tp["true_weights"][f] for f in ctx["feats"]], dtype=torch.float64)
    b_true = float(tp["true_bias"])

    def eta_true(u):  # u in the model space -> raw x -> true probability
        return torch.sigmoid((u * eta.scale + eta.shift) @ w_true + b_true)

    ck = torch.load(paths["strat_model"])
    margin = ours_margin(ck["w"], ck["b"])
    u, y = ctx["data"][split]
    y = np.asarray(y).astype(int)
    u_f, moved, _ = cheapest_feasible_response(margin, u, e["alpha"], e["beta"], e["upper"])
    moved = moved.numpy()
    cand = moved & (y == 0)
    with torch.no_grad():
        y_hat = moved | (margin(u).numpy() >= 0)

    def pi_of(fn):
        with torch.no_grad():
            a, b = fn(u), fn(u_f)
            return torch.clamp((b - a) / torch.clamp(1 - a, min=e["delta_g"]), 0, 1).numpy()

    pi_hat, pi_star = pi_of(eta), pi_of(eta_true)
    fixed_err = (y_hat[~cand] != y[~cand]).sum()

    def expected(pi):
        return pi[cand].mean(), (fixed_err + (1 - pi[cand]).sum()) / len(y), pi[cand].sum()

    rh, eh, ih = expected(pi_hat)
    rt, et, it = expected(pi_star)
    y_post = sample_post_labels(y, pi_star, cand, np.random.default_rng(cfg["evaluation"]["seed"]))
    improved_true = int((cand & (y_post == 1)).sum())
    with torch.no_grad():
        eta_err = (eta(u) - eta_true(u)).abs().numpy()

    out = {
        "split": split, "agents": len(y), "manipulating": int(moved.sum()), "manipulating_y0": int(cand.sum()),
        "eta_hat_vs_true_mean_abs": float(eta_err.mean()),
        "pi_hat_vs_true": {"mean_abs_diff": float(np.abs(pi_hat - pi_star)[cand].mean()),
                           "max_abs_diff": float(np.abs(pi_hat - pi_star)[cand].max()),
                           "corr": float(np.corrcoef(pi_hat[cand], pi_star[cand])[0, 1])},
        "estimated_eta": {"expected_improvement_rate": float(rh), "expected_improvement_error": float(eh),
                          "expected_improved": float(ih)},
        "true_eta": {"expected_improvement_rate": float(rt), "expected_improvement_error": float(et),
                     "expected_improved": float(it), "sampled_improved_seed": improved_true,
                     "sampled_improvement_rate": improved_true / int(cand.sum())},
    }
    with open(paths["evaluation"] / f"true_eta_validation_{split}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
