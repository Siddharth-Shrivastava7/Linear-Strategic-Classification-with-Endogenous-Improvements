"""SIM-IMP-PROB (Algorithm 2) with the improved point kept inside the
observed feature range, so eta_hat is never evaluated outside its data.

clip_mode:
    none      -> Algorithm 2 as written (no bound)
    clip      -> x^f_j* = min(boundary point, upper_j*): an agent whose
                 boundary point lies beyond the largest observed value stops
                 at that value (it improves but stays negatively classified)
    feasible  -> an agent moves only if it can reach the boundary within the
                 observed range; otherwise it does not move at all

Agents only move upward along e_j* (gap > 0 and w_j* > 0), so only the upper
bound can bind. Bounds are in the model space u, like everything else.
"""

import torch

from pipeline.strat_imp import improvement_reach, sim_imp_prob

CLIP_MODES = ("none", "clip", "feasible")


def make_sim_imp_prob(clip_mode, upper=None):
    """Return a function with the signature of `sim_imp_prob`."""
    if clip_mode == "none":
        return sim_imp_prob
    if clip_mode not in CLIP_MODES:
        raise ValueError(f"unknown clip_mode {clip_mode!r}; use one of {CLIP_MODES}")
    if upper is None:
        raise ValueError(f"clip_mode {clip_mode!r} needs upper bounds")

    def sim_imp_prob_bounded(z, w, b, beta, alpha, eta, delta_g):
        j_star, S = improvement_reach(w, alpha, beta)
        gap = b - z @ w
        moved = (gap > 0) & (gap <= S)
        w_j = w[j_star]
        w_j_safe = torch.where(w_j > 0, w_j, torch.ones_like(w_j))
        z_j = z[:, j_star]
        target = z_j + gap / w_j_safe          # j*-coordinate on the boundary
        cap = upper[j_star]
        if clip_mode == "feasible":
            moved = moved & (target <= cap)
            new_j = target
        else:
            new_j = torch.maximum(torch.minimum(target, cap), z_j)  # never move down
        new_j = torch.where(moved, new_j, z_j)
        e_j = torch.zeros_like(w)
        e_j[j_star] = 1.0
        z_f = z + (new_j - z_j)[:, None] * e_j

        eta_x = eta(z)
        eta_f = eta(z_f)
        denom = torch.clamp(1.0 - eta_x, min=delta_g)
        pi_hat = torch.clamp((eta_f - eta_x) / denom, 0.0, 1.0)
        return pi_hat, moved

    return sim_imp_prob_bounded
