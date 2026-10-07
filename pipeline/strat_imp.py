"""STRAT-IMP-AWARE (Algorithms 1-3) in PyTorch.

The linear classifier works in a "model space" u = (x - shift) / scale chosen
by `feature_scaling`:
    standard -> the LR's own standardization (mean/SD of its training split);
                alpha = cost per one standard deviation
    minmax   -> min/range of the LR's training split; alpha = cost per full range
    none     -> raw units; alpha = cost per raw unit (year, hour, level)
The frozen LR is evaluated on the same u (mapped back to raw x internally).

Linear classifier: positive iff w . u - b >= 0, with w >= 0.
"""

import numpy as np
import torch
import torch.nn as nn

FEATURE_SCALINGS = ("standard", "minmax", "none")


class FrozenLR(nn.Module):
    """Differentiable copy of the fitted sklearn LR pipeline, taking inputs in
    the model space x = u * scale + shift. All tensors are buffers (never
    trained), but gradients still flow to the input."""

    def __init__(self, sk_model, shift=None, scale=None, dtype=torch.float64):
        super().__init__()
        pre = sk_model.named_steps["preprocess"]
        names = [name for name, *_ in pre.transformers_ if name != "remainder"]
        if names != ["numeric"]:
            raise ValueError("FrozenLR needs every LR feature to be continuous/numeric-ordinal (standardized); "
                             f"got transformer groups {names}")
        scaler = pre.named_transformers_["numeric"]
        clf = sk_model.named_steps["clf"]
        self.features = list(pre.get_feature_names_out())
        self.register_buffer("lr_mean", torch.tensor(scaler.mean_, dtype=dtype))
        self.register_buffer("lr_scale", torch.tensor(scaler.scale_, dtype=dtype))
        self.register_buffer("coef", torch.tensor(clf.coef_.ravel(), dtype=dtype))
        self.register_buffer("intercept", torch.tensor(clf.intercept_[0], dtype=dtype))
        self.positive_index = list(clf.classes_).index(1)
        shift = self.lr_mean if shift is None else torch.as_tensor(shift, dtype=dtype)
        scale = self.lr_scale if scale is None else torch.as_tensor(scale, dtype=dtype)
        self.register_buffer("shift", shift.clone())
        self.register_buffer("scale", scale.clone())

    def to_model(self, x_raw):
        return (x_raw - self.shift) / self.scale

    def forward(self, u):
        """eta_hat(u) = P(y = 1 | x) for model-space inputs u of shape (n, d)."""
        z = (u * self.scale + self.shift - self.lr_mean) / self.lr_scale
        logit = z @ self.coef + self.intercept
        return torch.sigmoid(logit if self.positive_index == 1 else -logit)

    def init_wb(self):
        """The LR's own decision rule in model space (w clipped to >= 0):
        logit = w . u - b  with  w = coef_raw * scale."""
        coef_raw = self.coef / self.lr_scale
        int_raw = self.intercept - torch.sum(coef_raw * self.lr_mean)
        if self.positive_index != 1:
            coef_raw, int_raw = -coef_raw, -int_raw
        return (coef_raw * self.scale).clamp(min=0.0), -(int_raw + torch.sum(coef_raw * self.shift))


def model_space(kind, lr_mean, lr_scale, x_ref):
    """(shift, scale) of the model space; x_ref = raw features of the LR's
    training split, used for minmax."""
    if kind == "standard":
        return np.asarray(lr_mean), np.asarray(lr_scale)
    if kind == "minmax":
        lo, hi = x_ref.min(axis=0), x_ref.max(axis=0)
        return lo, np.where(hi > lo, hi - lo, 1.0)
    if kind == "none":
        return np.zeros(x_ref.shape[1]), np.ones(x_ref.shape[1])
    raise ValueError(f"unknown feature_scaling {kind!r}; use one of {FEATURE_SCALINGS}")


def improvement_reach(w, alpha, beta):
    """j* = min argmax_j w_j / alpha_j and S(w) = beta * w_j* / alpha_j*.
    torch.argmax returns the first maximal index, i.e. the min argmax."""
    ratio = w / alpha
    j_star = torch.argmax(ratio)
    return j_star, beta * ratio[j_star]


def sim_imp_prob(z, w, b, beta, alpha, eta, delta_g):
    """Algorithm 2, vectorized over the rows of z. Returns (pi_hat, moved).

    Rows with 0 < b - w.z <= S(w) move along e_j* exactly onto the boundary
    w.z = b. The branch test and the choice of j* are hard decisions (no
    gradient); the moved point x^f carries gradient to w and b."""
    j_star, S = improvement_reach(w, alpha, beta)
    gap = b - z @ w
    moved = (gap > 0) & (gap <= S)
    w_j = w[j_star]
    # When no row moves w_j* may be 0; dividing by a safe value keeps the
    # unused branch of torch.where from producing NaN gradients.
    w_j_safe = torch.where(w_j > 0, w_j, torch.ones_like(w_j))
    step = torch.where(moved, gap / w_j_safe, torch.zeros_like(gap))
    e_j = torch.zeros_like(w)
    e_j[j_star] = 1.0
    z_f = z + step[:, None] * e_j

    eta_x = eta(z)
    eta_f = eta(z_f)
    denom = torch.clamp(1.0 - eta_x, min=delta_g)  # max{1 - eta(x), delta_g}
    pi_hat = torch.clamp((eta_f - eta_x) / denom, 0.0, 1.0)
    return pi_hat, moved


def strat_imp_loss(z, y, pi_hat, w, b, beta, alpha):
    """Algorithm 3, vectorized: per-row losses (n,)."""
    q = y + (1.0 - y) * pi_hat
    _, S = improvement_reach(w, alpha, beta)
    score = z @ w - b + S
    return q * torch.relu(1.0 - score) + (1.0 - q) * torch.relu(1.0 + score)


def class_weight_factors(y, class_weight):
    """(weight for y = 0, weight for y = 1): ones, or n / (2 n_class) for 'balanced'."""
    if class_weight in (None, "none"):
        return 1.0, 1.0
    if class_weight != "balanced":
        raise ValueError(f"unknown class_weight {class_weight!r}")
    n1 = float(y.sum())
    n0 = float(len(y)) - n1
    return len(y) / (2.0 * n0), len(y) / (2.0 * n1)


def batch_loss(z, y, w, b, hp, eta, sim_fn=sim_imp_prob):
    pi_hat, moved = sim_fn(z, w, b, hp["beta"], hp["alpha"], eta, hp["delta_g"])
    losses = strat_imp_loss(z, y, pi_hat, w, b, hp["beta"], hp["alpha"])
    cw = hp.get("_class_factors")
    if cw is not None:  # optional class weighting (default: none, Algorithm 1 as written)
        losses = losses * torch.where(y > 0.5, cw[1], cw[0])
    return losses.mean(), pi_hat, moved


def to_raw_space(w, b, shift, scale):
    """w.u - b with u = (x - shift)/scale  ==  w_raw.x - b_raw."""
    w_raw = w / scale
    return w_raw, b + float(np.sum(w_raw * shift))


def train_strat_imp_aware(z_tr, y_tr, z_val, y_val, eta, hp, log=print, sim_fn=sim_imp_prob):
    """Algorithm 1: projected mini-batch gradient descent, keeping the
    (w, b) with the lowest validation loss. hp["init"]: "uniform" (w ~ U(0,1),
    b = 0) or "lr" (the frozen LR's own decision rule)."""
    gen = torch.Generator().manual_seed(hp["seed"])
    d, dtype = z_tr.shape[1], z_tr.dtype
    if hp.get("init", "uniform") == "uniform":
        w, b = torch.rand(d, generator=gen, dtype=dtype), torch.zeros((), dtype=dtype)
    elif hp["init"] == "lr":
        w, b = (t.detach().clone() for t in eta.init_wb())
    else:
        raise ValueError(f"unknown init {hp['init']!r}; use uniform or lr")
    if not torch.any(w > 0):  # w in R^d_{>=0} \ {0}
        raise RuntimeError("initial w is all zeros")
    w.requires_grad_(True)
    b.requires_grad_(True)
    if hp.get("class_weight") not in (None, "none"):
        hp = dict(hp, _class_factors=class_weight_factors(y_tr, hp["class_weight"]))
    lr, bs = hp["lr"], hp["batch_size"]

    best = {"val_loss": float("inf")}
    history = []
    for epoch in range(1, hp["epochs"] + 1):
        perm = torch.randperm(len(z_tr), generator=gen)
        train_losses = []
        for start in range(0, len(perm), bs):
            idx = perm[start:start + bs]
            loss, _, _ = batch_loss(z_tr[idx], y_tr[idx], w, b, hp, eta, sim_fn)
            w.grad, b.grad = None, None
            loss.backward()
            with torch.no_grad():
                w.sub_(lr * w.grad).clamp_(min=0.0)  # (w - gamma * grad)_+
                b.sub_(lr * b.grad)
            train_losses.append(loss.item())

        with torch.no_grad():
            val_losses, n_moved, pi_neg = [], 0, []
            for start in range(0, len(z_val), bs):
                zb, yb = z_val[start:start + bs], y_val[start:start + bs]
                loss, pi_hat, moved = batch_loss(zb, yb, w, b, hp, eta, sim_fn)
                val_losses.append(loss.item())
                n_moved += int(moved.sum())
                pi_neg.append(pi_hat[yb == 0])
            val_loss = float(np.mean(val_losses))  # L_val / m
            j_star, S = improvement_reach(w, hp["alpha"], hp["beta"])
            score = z_val @ w - b
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "b": b.item(),
                **{f"w_{i}": v for i, v in enumerate(w.tolist())},
                "j_star": int(j_star),
                "S": S.item(),
                "val_frac_positive": float((score >= 0).double().mean()),
                "val_frac_moved": n_moved / len(z_val),
                "val_mean_pi_y0": float(torch.cat(pi_neg).mean()),
            }
            history.append(row)

            improved = val_loss < best["val_loss"]
            if improved:
                best = {"val_loss": val_loss, "epoch": epoch, "w": w.detach().clone(), "b": b.detach().clone()}
            log(f"epoch {epoch:>3}  train {row['train_loss']:.5f}  val {val_loss:.5f}  "
                f"w={np.round(w.tolist(), 4)}  b={b.item():+.4f}  j*={int(j_star)}  S={S.item():.4f}  "
                f"moved={row['val_frac_moved']:.3f}{'  *best' if improved else ''}")
            if not torch.any(w > 0):
                log("  warning: projection made w all zeros")
    return best, history
