"""Improvement statistics of a trained STRAT-IMP-AWARE model."""

import numpy as np
import torch

from pipeline.strat_imp import improvement_reach, sim_imp_prob


@torch.no_grad()
def improvement_stats(u, y, w, b, hp, eta, x_max_raw=None, sim_fn=sim_imp_prob):
    """Best responses to (w, b) on model-space features u (Algorithm 2).

    Returns per-agent arrays (pi_imp, moved, candidates = moved & y == 0) and a
    summary with the exact expected improvement rate
        E[improved] / #candidates = mean of pi_Imp over candidates.
    If x_max_raw is given, also the share of movers whose boundary point lies
    past the largest observed raw value of the feature they move along (under
    clip_mode "clip" these are the movers that get capped)."""
    alpha = hp["alpha"]
    pi_hat, moved = sim_fn(u, w, b, hp["beta"], alpha, eta, hp["delta_g"])
    j_star, S = improvement_reach(w, alpha, hp["beta"])
    y = np.asarray(y).astype(int)
    pi_hat, moved_np = pi_hat.numpy(), moved.numpy()
    cand = moved_np & (y == 0)
    n_cand = int(cand.sum())
    summary = {
        "n_agents": len(y),
        "j_star": int(j_star),
        "S_w": float(S),
        "manipulating": int(moved_np.sum()),
        "manipulating_y0": n_cand,
        "manipulating_y1": int((moved_np & (y == 1)).sum()),
        "expected_improved": float(pi_hat[cand].sum()),
        "expected_improvement_rate": float(pi_hat[cand].mean()) if n_cand else float("nan"),
        "frac_negatives_manipulating": n_cand / max(int((y == 0).sum()), 1),
    }
    if x_max_raw is not None and moved_np.any():
        j = int(j_star)
        gap = b - u[moved] @ w
        x_f_raw = (u[moved, j] + gap / w[j]) * eta.scale[j] + eta.shift[j]
        summary["frac_movers_beyond_observed_max"] = float((x_f_raw > float(x_max_raw[j]) + 1e-9).double().mean())
    return pi_hat, moved_np, cand, summary


def sample_post_labels(y, pi_hat, cand, rng):
    """Post-response labels: candidates (movers with y = 0) become 1 with
    probability pi_Imp; everyone else keeps their label."""
    y_post = np.asarray(y).astype(int).copy()
    y_post[cand] = (rng.random(int(cand.sum())) < pi_hat[cand]).astype(int)
    return y_post


@torch.no_grad()
def improvement_error(u, y, w, b, hp, eta, sim_fn, rng=None):
    """Improvement error: share of agents whose post-deployment label y' differs
    from the prediction after best responding.

    prediction: y_hat = 1 if w.u^f - b >= 0. Movers land exactly on the
                boundary, so they are predicted 1 (set explicitly, so floating
                point cannot flip them).
    label:      movers with y = 0 get y' ~ Bernoulli(pi_Imp); everyone else keeps y.
    Returns the exact expected error and, if rng is given, one sampled error."""
    pi_hat, moved, cand, summary = improvement_stats(u, y, w, b, hp, eta, sim_fn=sim_fn)
    y = np.asarray(y).astype(int)
    y_hat = moved | ((u @ w - b).numpy() >= 0)
    fixed = ~cand                                  # label known: y' = y
    errors_fixed = (y_hat[fixed] != y[fixed]).sum()
    expected = (errors_fixed + (1.0 - pi_hat[cand]).sum()) / len(y)  # candidates: y_hat = 1, wrong iff y' = 0
    out = {"expected_improvement_error": float(expected),
           "predicted_positive": float(y_hat.mean()),
           "error_without_response": float((((u @ w - b).numpy() >= 0) != y).mean()),
           **summary}
    if rng is not None:
        y_post = sample_post_labels(y, pi_hat, cand, rng)
        out["sampled_improvement_error"] = float((y_hat != y_post).mean())
    return out
