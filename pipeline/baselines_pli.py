"""Reimplementation of the "PAC Learning with Improvements" (PLI) baseline.

Source: Attias et al., ICML 2025, github.com/ripl/PLI, notebook
improvement/oulad_improv.ipynb (cells 4-6, 14-15). Reproduced here so the
baseline runs on our splits and features; nothing is downloaded or executed
from the repository.

    learner:  NNClassifier (d -> 64 -> ReLU -> 1 -> sigmoid), Adam lr 1e-3,
              100 epochs, batch 64, no shuffling; loss BCE or weighted BCE
              (false-positive weight w, false-negative weight 1.33);
              predict positive iff h(x) > threshold.
              We add LinearClassifier (d -> 1 -> sigmoid) with the same
              losses, to separate the training objective from model capacity.
    response: agents predicted negative run 500 steps of signed-gradient
              ascent (step 0.01) on loss(h(x'), h(x)) over the improvable
              features, projected onto the L-inf ball of radius r around x;
              they keep x' only if their prediction flips.
    labels:   f* = DecisionTreeClassifier(random_state=42) fitted on all data.
"""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCELossx(nn.Module):
    def forward(self, predictions, targets):
        predictions = torch.clamp(predictions, 1e-6, 1 - 1e-6)
        return -torch.mean((1 - targets) * torch.log(1 - predictions) + targets * torch.log(predictions))


class WBCELossx(nn.Module):
    def __init__(self, false_positive_weight=2.0, false_negative_weight=1.33):
        super().__init__()
        self.fp_weight = false_positive_weight
        self.fn_weight = false_negative_weight

    def forward(self, predictions, targets):
        predictions = torch.clamp(predictions, 1e-6, 1 - 1e-6)
        fp = self.fp_weight * (1 - targets) * torch.log(1 - predictions)
        fn = self.fn_weight * targets * torch.log(predictions)
        return -torch.mean(fp + fn)


class NNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))


MODELS = {"nn": NNClassifier, "linear": LinearClassifier}


def make_loss(name):
    """'bce' or 'wbce_<false-positive weight>', e.g. 'wbce_2'."""
    if name == "bce":
        return BCELossx()
    if name.startswith("wbce_"):
        return WBCELossx(false_positive_weight=float(name.split("_", 1)[1]))
    raise ValueError(f"unknown PLI loss {name!r}")


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_pli(model_type, loss_name, X, y, seed, epochs=100, batch_size=64, lr=1e-3):
    """X: (n, d) float32 standardized features, y: (n,) {0, 1}."""
    set_random_seed(seed)
    model = MODELS[model_type](X.shape[1])
    criterion = make_loss(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X, y.float().view(-1, 1)), batch_size=batch_size, shuffle=False)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model.eval()


def pgd_improvement(h, loss_fn, x, feature_ids, eps, step=0.01, iters=500):
    """PLI's response (their `pgd_improvement`): signed-gradient ascent on
    loss(h(x'), h(x)) restricted to `feature_ids`, projected to |x' - x|_inf <= eps."""
    ori = x.detach()
    with torch.no_grad():
        ori_out = h(ori)
    adv = ori.clone()
    for _ in range(iters):
        adv.requires_grad_(True)
        cost = loss_fn(h(adv), ori_out)
        grad, = torch.autograd.grad(cost, adv)
        pert = torch.zeros_like(adv)
        pert[:, feature_ids] = step * grad[:, feature_ids].sign()
        adv = torch.clamp(adv.detach() + pert, ori - eps, ori + eps)
    return adv.detach()
