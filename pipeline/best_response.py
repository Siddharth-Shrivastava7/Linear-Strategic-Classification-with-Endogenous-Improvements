"""Agent responses and metrics for comparing any classifier in two environments.

Environment "ours": cost-based best response, as in Algorithm 2 (feasible
mode) but for an arbitrary classifier. An agent predicted negative makes the
cheapest single-feature increase that makes it predicted positive, if that
costs alpha_j * delta <= beta and keeps the feature within its observed max;
otherwise it stays. For a linear classifier this is exactly Algorithm 2's
move (the cheapest feature is argmax_j w_j / alpha_j), except that an agent
whose cheapest feature is out of range may use another feasible feature.
Post-response labels: movers with y = 0 get y' ~ Bernoulli(pi_Imp) from the
frozen LR; everyone else keeps y.

Environment "PLI": the PLI repository's PGD response and decision-tree labels
(see pipeline/baselines_pli.py).

Classifiers are passed as `margin(u)` (>= 0 means predicted positive), with
inputs in the LR-standardized model space.
"""

import numpy as np
import torch

from pipeline.baselines_pli import pgd_improvement


@torch.no_grad()
def cheapest_feasible_response(margin, u, alpha, beta, upper, grid=64, bisect=40):
    """Returns (u_f, moved, feature). Grid search for the first crossing along
    each feature, then bisection; the returned point satisfies margin >= 0."""
    n, d = u.shape
    neg_idx = torch.where(margin(u) < 0)[0]
    best_cost = torch.full((len(neg_idx),), float("inf"), dtype=u.dtype)
    best_delta = torch.zeros(len(neg_idx), dtype=u.dtype)
    best_j = torch.full((len(neg_idx),), -1, dtype=torch.long)
    un = u[neg_idx]
    m = len(neg_idx)
    if m:
        frac = torch.arange(1, grid + 1, dtype=u.dtype) / grid
        for j in range(d):
            a = float(alpha[j])
            if not np.isfinite(a) or a <= 0:
                continue
            dmax = torch.clamp(torch.minimum(torch.full_like(un[:, j], beta / a), upper[j] - un[:, j]), min=0.0)
            deltas = dmax[:, None] * frac[None, :]                       # (m, grid)
            pts = un[:, None, :].expand(m, grid, d).clone()
            pts[:, :, j] += deltas
            ok = margin(pts.reshape(-1, d)).reshape(m, grid) >= 0
            has = ok.any(dim=1)
            first = ok.to(torch.int8).argmax(dim=1)
            hi = deltas.gather(1, first[:, None]).squeeze(1)
            lo = torch.where(first > 0, deltas.gather(1, (first - 1).clamp(min=0)[:, None]).squeeze(1),
                             torch.zeros_like(hi))
            for _ in range(bisect):
                mid = (lo + hi) / 2
                p = un.clone()
                p[:, j] += mid
                mid_ok = margin(p) >= 0
                hi = torch.where(mid_ok, mid, hi)
                lo = torch.where(mid_ok, lo, mid)
            cost = a * hi
            better = has & (cost < best_cost)
            best_cost = torch.where(better, cost, best_cost)
            best_delta = torch.where(better, hi, best_delta)
            best_j = torch.where(better, torch.full_like(best_j, j), best_j)

    moved = torch.zeros(n, dtype=torch.bool)
    feature = torch.full((n,), -1, dtype=torch.long)
    u_f = u.clone()
    sel = best_j >= 0
    rows = neg_idx[sel]
    moved[rows] = True
    feature[rows] = best_j[sel]
    u_f[rows, best_j[sel]] += best_delta[sel]
    return u_f, moved, feature


@torch.no_grad()
def evaluate_env_ours(margin, u, y, eta, beta, alpha, upper, delta_g):
    """Exact expected improvement rate / error under our environment."""
    u_f, moved, feature = cheapest_feasible_response(margin, u, alpha, beta, upper)
    eta_x, eta_f = eta(u), eta(u_f)
    pi = torch.clamp((eta_f - eta_x) / torch.clamp(1.0 - eta_x, min=delta_g), 0.0, 1.0).numpy()
    y = np.asarray(y).astype(int)
    moved = moved.numpy()
    y_hat = moved | (margin(u).numpy() >= 0)
    cand = moved & (y == 0)
    errors_fixed = (y_hat[~cand] != y[~cand]).sum()
    feats = feature.numpy()[moved]
    return {
        "improvement_rate": float(pi[cand].mean()) if cand.any() else float("nan"),
        "improvement_error": float((errors_fixed + (1.0 - pi[cand]).sum()) / len(y)),
        "error_no_response": float(((margin(u).numpy() >= 0) != y).mean()),
        "movers": int(moved.sum()),
        "neg_movers": int(cand.sum()),
        "expected_improved": float(pi[cand].sum()),
        "predicted_positive_before": float((margin(u).numpy() >= 0).mean()),
        "move_features": {int(k): int(v) for k, v in zip(*np.unique(feats, return_counts=True))},
    }


def evaluate_env_pli(h, tau, loss_fn, u, fstar, feature_ids, r, step=0.01, iters=500):
    """PLI environment: negatives (h(x) <= tau) run PGD within radius r and keep
    x' only if h(x') > tau. Error = mean(h(x_final) > tau != f*(x_final)).
    Improvement rate = among movers with f*(x) = 0, share with f*(x') = 1."""
    with torch.no_grad():
        pred = (h(u).flatten() > tau)
    neg_idx = torch.where(~pred)[0]
    u_fin = u.clone()
    moved = torch.zeros(len(u), dtype=torch.bool)
    if len(neg_idx) and r > 0:
        adv = pgd_improvement(h, loss_fn, u[neg_idx], feature_ids, r, step, iters)
        with torch.no_grad():
            flip = h(adv).flatten() > tau
        u_fin[neg_idx[flip]] = adv[flip]
        moved[neg_idx[flip]] = True
    with torch.no_grad():
        y_hat = (h(u_fin).flatten() > tau).numpy()
    f_x = fstar.predict(u.numpy())
    f_xf = fstar.predict(u_fin.numpy())
    moved = moved.numpy()
    cand = moved & (f_x == 0)
    return {
        "improvement_error": float((y_hat != f_xf).mean()),
        "improvement_rate": float((f_xf[cand] == 1).mean()) if cand.any() else float("nan"),
        "movers": int(moved.sum()),
        "neg_movers": int(cand.sum()),
    }
