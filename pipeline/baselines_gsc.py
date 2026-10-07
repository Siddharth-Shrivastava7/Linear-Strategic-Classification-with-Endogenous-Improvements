"""Reimplementation of the strategic SVM of "Generalized Strategic
Classification and the Case of Aligned Incentives" (Levanon & Rosenfeld,
ICML 2022; arXiv 2202.04357; github.com/SagiLevanon1/GSC, generalization.ipynb).

    model:  f(x) = w . x + b (nn.Linear with bias), predict positive iff f(x) >= 0
    loss:   mean relu(1 - y f(x) - y * shift(w)) + lam * ||w||_2^2,  y in {-1, +1}
            (their "SSVM" s-hinge with r = +1: every user wants a positive prediction)
    shift:  "l2"   -> 2 ||w||_2: their cost c(x, x') = ||x - x'||_2 with budget 2
            "ours" -> beta * max_j |w_j| / alpha_j: the largest score gain under our
                      cost sum_j alpha_j |delta_j| with budget beta (the same reach as
                      S(w) in Algorithm 3; labels are never changed by the move)
    train:  Adam, shuffled mini-batches; the final epoch's model is returned.
    class_weight="balanced" (our addition for imbalanced data, as in sklearn):
            sample i's hinge term is multiplied by n / (2 n_{y_i}); the weights
            average to 1, so the loss stays on the unweighted scale.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pipeline.baselines_pli import set_random_seed


def strategic_shift(w, kind, beta=None, alpha=None):
    if kind == "l2":
        return 2.0 * torch.norm(w, p=2)
    if kind == "ours":
        return beta * torch.max(torch.abs(w) / alpha)
    raise ValueError(f"unknown shift {kind!r}; use l2 or ours")


class SSVM(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.fc = nn.Linear(x_dim, 1, bias=True)

    def forward(self, x):
        return torch.flatten(self.fc(x))


def class_weights(y01, class_weight):
    """Per-sample weights: ones, or n / (2 n_class) for class_weight='balanced'."""
    y01 = y01.long()
    if class_weight is None:
        return torch.ones(len(y01))
    if class_weight != "balanced":
        raise ValueError(f"unknown class_weight {class_weight!r}")
    counts = torch.bincount(y01, minlength=2).float()
    return (len(y01) / (2.0 * counts))[y01]


def train_gsc(X, y01, shift, lam, seed, beta=None, alpha=None, epochs=100, batch_size=128, lr=0.05,
              class_weight=None):
    """X: (n, d) float32 standardized features; y01: (n,) labels in {0, 1}."""
    set_random_seed(seed)
    model = SSVM(X.shape[1])
    y = 2.0 * y01.float() - 1.0
    sw = class_weights(y01, class_weight)
    a = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y, sw), batch_size=batch_size, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for xb, yb, wb in loader:
            opt.zero_grad()
            w = model.fc.weight[0]
            hinge = (wb * torch.relu(1 - yb * model(xb) - yb * strategic_shift(w, shift, beta, a))).mean()
            loss = hinge + lam * torch.norm(w, p=2) ** 2
            loss.backward()
            opt.step()
    return model.eval()
