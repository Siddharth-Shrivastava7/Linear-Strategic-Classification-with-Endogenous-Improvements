from xml.parsers.expat import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from scipy.stats import zscore


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_preprocessing import *
from classifiers import *
from utils import *


# ═══════════════════════════════════════════════════════════
# 1.  DATASET
# ═══════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples=2500):
    X, y = make_classification(n_samples=n_samples, n_features=8, n_informative=8, n_redundant=0, n_classes=2, class_sep=2.0, random_state=42)
    mask = np.ones(len(y), dtype=bool)
    for c, keep_frac in zip([0, 1], [0.9, 0.8]):
        idx = np.where(y == c)[0]
        mz = np.abs(zscore(X[idx])).max(axis=1)
        cutoff = np.quantile(mz, keep_frac)
        mask[idx[mz > cutoff]] = False
    X, y = X[mask], y[mask]
    X, y = SMOTE(random_state=42).fit_resample(X, y)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# ═══════════════════════════════════════════════════════════
# 2.  GROUND-TRUTH LABELER  f*
# ═══════════════════════════════════════════════════════════

def train_fstar(X_train, y_train):
    clf = DecisionTreeClassifier(
        criterion='gini', min_samples_split=2,
        min_samples_leaf=1, random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


# ═══════════════════════════════════════════════════════════
# 3.  TWO-LAYER NEURAL NETWORK  h
# ═══════════════════════════════════════════════════════════

# class TwoLayerNN(nn.Module):
#     def __init__(self, input_dim, hidden=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, 1),         nn.Sigmoid()
#         )
#     def forward(self, x):
#         return self.net(x)

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden, 1)   # ← IMPORTANT
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return self.sigmoid(x)


def weighted_bce_loss(pred, target, w_fp, w_fn):
    """Eq. 5:  -1/n Σ [ wFP*(1-y)*log(1-ŷ) + wFN*y*log(ŷ) ]"""
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
    return (-(w_fn * target * torch.log(pred)
              + w_fp * (1 - target) * torch.log(1 - pred))).mean()

def train_h(X_train, y_train, w_fp=1.0, w_fn=1.0, epochs=250, bs=64):
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train).view(-1, 1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)
    model = TwoLayerNN(X_train.shape[1])
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            weighted_bce_loss(model(xb), yb, w_fp, w_fn).backward()
            opt.step()
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
# 4.  AGENT IMPROVEMENT  —  Projected Gradient Descent (Eq. 6)
# ═══════════════════════════════════════════════════════════

def improve_agent(x_orig, model, budget_r, alpha=0.05, max_iter=100, threshold=0.5):
    """
    Minimise L(h(x), 1) = -log(h(x))  subject to  ||x' - x_orig||_inf <= r.

    Correct update (gradient DESCENT on L):
        x'(t+1) = Proj[ x'(t)  -  alpha * sign( ∇_x L ) ]

    Why minus: ∇_x(-log h(x)) points in the direction that DECREASES h(x).
    Subtracting it moves x toward HIGHER h(x), i.e. positive classification.
    """
    model.eval()
    orig = torch.FloatTensor(x_orig)
    x    = orig.clone().detach().requires_grad_(True)
    target = torch.tensor([1.0])

    for _ in range(max_iter):
        if x.grad is not None:
            x.grad.zero_()

        pred = model(x.unsqueeze(0)).squeeze()  # scalar in [0,1]

        if pred.item() >= threshold:
            return x.detach().numpy()           # already positive — done

        loss = nn.BCELoss()(pred, target[0])
        loss.backward()

        with torch.no_grad():
            # ── CORRECT: MINUS sign (gradient descent) ──────────────────
            x_new = x - alpha * torch.sign(x.grad)
            # Project onto ℓ∞-ball of radius r
            delta = torch.clamp(x_new - orig, -budget_r, budget_r)
            x = (orig + delta).detach().requires_grad_(True)

    return x.detach().numpy()


# ═══════════════════════════════════════════════════════════
# 5.  LOSS FUNCTION  (Eq. 1 from paper)
# ═══════════════════════════════════════════════════════════

def compute_loss_and_outcome(x, h_model, fstar, budget_r, threshold=0.5):
    """
    Returns (loss, outcome_code) where outcome_code is one of:
        'TP'   – h=1, f*=1
        'FP'   – h=1, f*=0          (error)
        'TN'   – h=0, stayed, f*=0
        'FN'   – h=0, stayed, f*=1  (error at r=0; not error with improvements)
        'FN→TP'– improved, h(x')=1, f*(x')=1  (no error)
        'FN→FP'– improved, h(x')=1, f*(x')=0  (error)
        'TN→TP'– improved, h(x')=1, f*(x')=1  (no error, bonus improvement)
        'TN→FP'– improved, h(x')=1, f*(x')=0  (error)
        'no_move' – h=0, PGD found no positive point in budget
    """
    h_model.eval()
    x_t = torch.FloatTensor(x)
    h_orig = int(h_model(x_t.unsqueeze(0)).item() >= threshold)
    f_orig = int(fstar.predict(x.reshape(1, -1))[0])

    if h_orig == 1:
        err = int(f_orig == 0)
        return err, 'FP' if f_orig == 0 else 'TP'

    # h(x)=0: try to improve
    if budget_r > 0:
        x_imp = improve_agent(x, h_model, budget_r, threshold=threshold)
        h_imp = int(h_model(torch.FloatTensor(x_imp).unsqueeze(0)).item() >= threshold)
        if h_imp == 1:
            f_imp = int(fstar.predict(x_imp.reshape(1, -1))[0])
            if f_orig == 1:
                code = 'FN→TP' if f_imp == 1 else 'FN→FP'
            else:
                code = 'TN→TP' if f_imp == 1 else 'TN→FP'
            err = int(f_imp == 0)
            return err, code
        else:
            # PGD found nothing in budget
            err = int(f_orig == 1)   # FN still counts as error
            # return err, 'FN' if f_orig == 1 else 'TN'
            return err, 'no_move'

    # r=0: no improvement
    err = int(h_orig != f_orig)
    return err, ('FN' if f_orig == 1 else 'TN')


# ═══════════════════════════════════════════════════════════
# 6.  FULL EVALUATION
# ═══════════════════════════════════════════════════════════

OUTCOME_CODES = ['TP','FP','TN','FN','FN→TP','FN→FP','TN→TP','TN→FP','no_move']

def evaluate(X_test, fstar, models_dict, budgets, threshold=0.5):
    print("\n" + "="*70)
    print("Results  (↓ error = improvement working correctly)")
    print("="*70)

    for name, h_model in models_dict.items():
        print(f"\n{'─'*70}")
        print(f"  Model: {name}")
        print(f"{'─'*70}")
        header = (f"  {'Budget r':>9} │ {'Error%':>7} │ {'Errors':>6} │"
                  f" {'FN→TP':>6} │ {'FN→FP':>6} │ {'TN→TP':>6} │"
                  f" {'TN→FP':>6} │ {'no_move':>8}")
        print(header)
        print("  " + "─"*len(header.strip()))

        for r in budgets:
            counts = {c: 0 for c in OUTCOME_CODES}
            total_err = 0
            for i in range(len(X_test)):
                err, code = compute_loss_and_outcome(
                    X_test[i], h_model, fstar, r, threshold)
                total_err += err
                counts[code] = counts.get(code, 0) + 1

            n = len(X_test)
            print(
                f"  {r:>9.1f} │ {100*total_err/n:>6.2f}% │ {total_err:>6} │"
                f" {counts['FN→TP']:>6} │ {counts['FN→FP']:>6} │"
                f" {counts['TN→TP']:>6} │ {counts['TN→FP']:>6} │"
                f" {counts.get('no_move',0):>8}"
            )


# ═══════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("PAC Learning with Improvements")
    print("="*60)

    print("\n[1] Generating data ...")
    X_train, X_test, y_train, y_test = generate_synthetic_data()
    print(f"    Train: {len(y_train)}  Test: {len(y_test)}")
    print(f"    Train balance: {(y_train==1).sum()} pos / {(y_train==0).sum()} neg")

    print("\n[2] Training f* ...")
    fstar = train_fstar(X_train, y_train)
    print(f"    f* train acc = {accuracy_score(y_train, fstar.predict(X_train)):.4f}")
    print(f"    f* test  acc = {accuracy_score(y_test,  fstar.predict(X_test)):.4f}")

    # Use f* labels for training h (as per paper — h learns from f* labels)
    y_fstar_train = fstar.predict(X_train)

    print("\n[3] Training decision-maker models h ...")
    models = {}
    # configs = [
    #     ("BCE  (wFP=1, wFN=1)",   1.0, 1.0),
    #     ("wBCE (wFP=3, wFN=1)",   3.0, 1.0),
    #     ("wBCE (wFP=5, wFN=1)",   5.0, 1.0),
    #     ("wBCE (wFP=8, wFN=1)",   8.0, 1.0),
    # ]
    configs = [
        ("BCE  (wFP=1.0, wFN=1.0)",   1.0, 1.0),
        ("wBCE (wFP=3.0, wFN=0.009)", 3.0, 0.009),
        ("wBCE (wFP=5.0, wFN=0.009)", 5.0, 0.009),
        ("wBCE (wFP=8.0, wFN=0.009)", 8.0, 0.009),
    ]
    for label, wfp, wfn in configs:
        print(f"    Training {label} ...")
        m = train_h(X_train, y_fstar_train, w_fp=wfp, w_fn=wfn)
        preds = (m(torch.FloatTensor(X_test)).detach().numpy().flatten() >= 0.5).astype(int)
        print(f"      baseline acc (r=0): {accuracy_score(y_test, preds):.4f}  "
              f"  FP={((preds==1)&(y_test==0)).sum()}  "
              f"FN={((preds==0)&(y_test==1)).sum()}")
        models[label] = m

    budgets = [0.0, 0.5, 1.0, 2.0, 4.0]
    evaluate(X_test, fstar, models, budgets)



