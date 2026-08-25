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
from tqdm import tqdm 


# ═══════════════════════════════════════════════════════════
# 1.  DATASET
# ═══════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples=2500, class_sep=4.0, random_state=42):
    """
    8-dim synthetic binary classification dataset, following Appendix E.1.

    Paper specifies class_sep=4 (high separability). The previous version
    of this code used class_sep=2.0, which produces a noisier boundary
    and prevents the synthetic error from converging to ~0 as r grows
    (cf. Figure 3d in the paper).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_classes=2,
        class_sep=class_sep,
        random_state=random_state,
    )

    # Z-score outlier filtering, per Appendix E.1 (0.9 for class 0, 0.8 for class 1).
    mask = np.ones(len(y), dtype=bool)
    for c, keep_frac in zip([0, 1], [0.9, 0.8]):
        idx = np.where(y == c)[0]
        mz = np.abs(zscore(X[idx])).max(axis=1)
        cutoff = np.quantile(mz, keep_frac)
        mask[idx[mz > cutoff]] = False
    X, y = X[mask], y[mask]

    X, y = SMOTE(random_state=random_state).fit_resample(X, y)
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


# ═══════════════════════════════════════════════════════════
# 2.  GROUND-TRUTH LABELER  f*
# ═══════════════════════════════════════════════════════════

def train_fstar(X_train, y_train, random_state=42):
    """Decision tree (DTC2 in the paper, Appendix E.2.1)."""
    clf = DecisionTreeClassifier(
        criterion="gini",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)
    return clf


# ═══════════════════════════════════════════════════════════
# 3.  TWO-LAYER NEURAL NETWORK  h
# ═══════════════════════════════════════════════════════════

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.out(self.hidden(x)))


def weighted_bce_loss(pred, target, w_fp, w_fn):
    """Eq. (5):  L = -1/n  Σ [ w_FN * y * log(ŷ)  +  w_FP * (1-y) * log(1-ŷ) ]."""
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
    return (-(w_fn * target * torch.log(pred)
              + w_fp * (1 - target) * torch.log(1 - pred))).mean()


def train_h(X_train, y_train, w_fp=1.0, w_fn=1.0, epochs=80, bs=512,
            hidden=1, lr=1e-1, seed=42):
    torch.manual_seed(seed)
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train).view(-1, 1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)

    model = TwoLayerNN(X_train.shape[1], hidden=hidden)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in tqdm(range(epochs)):
        for xb, yb in loader:
            opt.zero_grad()
            weighted_bce_loss(model(xb), yb, w_fp, w_fn).backward()
            opt.step()
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════
# 4.  AGENT IMPROVEMENT  —  Projected Gradient Descent (Eq. 6)
# ═══════════════════════════════════════════════════════════

def improve_agent(
    x_orig,
    h_model,
    budget_r,
    fstar=None,
    improvable_features=None,
    alpha=0.05,
    max_iter=100,
    threshold=0.5,
    adversarial_tiebreak=False,
):
    """
    Solve  min_{x' in B_inf(x_orig, r)}  L(h(x'), 1)
    by projected gradient descent.

    Args:
        x_orig: 1-D numpy array, the agent's original features.
        h_model: trained TwoLayerNN.
        budget_r: ℓ_∞ improvement radius.
        fstar: ground-truth labeler. Required iff adversarial_tiebreak=True.
        improvable_features: list / array of feature indices the agent is
            allowed to modify. None means all features are improvable
            (correct for the synthetic dataset, where Table 1 says
            "all features are used"). For Adult / OULAD / Law school you
            MUST pass the index list from Table 1 — otherwise agents will
            silently move non-improvable coordinates and the reported
            improvement gain will be inflated.
        alpha: step size.
        max_iter: PGD iterations.
        threshold: classification threshold for h.
        adversarial_tiebreak: if True, implement Eq. (1) faithfully —
            scan the PGD trajectory and prefer an x' with h(x')=1 and
            f*(x')=0 (adversarial tie-breaking) over one with f*(x')=1.
            If False (default), return the first iterate with h(x')>=
            threshold, which matches the paper's empirical methodology
            (Appendix E.3) but only approximates Eq. (1).

    Returns:
        x' as a 1-D numpy array.
    """
    if adversarial_tiebreak and fstar is None:
        raise ValueError("adversarial_tiebreak=True requires fstar.")

    h_model.eval()
    d = x_orig.shape[0]

    # Build a coordinate mask that zeros out gradients on non-improvable features.
    if improvable_features is None:
        feat_mask = torch.ones(d)
    else:
        feat_mask = torch.zeros(d)
        feat_mask[list(improvable_features)] = 1.0

    orig = torch.FloatTensor(x_orig)
    x = orig.clone().detach().requires_grad_(True)
    target = torch.tensor(1.0)

    # Track candidates that h classifies positive, so we can pick adversarially.
    positive_iterates = []  # list of np.ndarray

    for _ in range(max_iter):
        if x.grad is not None:
            x.grad.zero_()

        pred = h_model(x.unsqueeze(0)).squeeze()

        if pred.item() >= threshold:
            x_np = x.detach().numpy().copy()
            if not adversarial_tiebreak:
                return x_np
            positive_iterates.append(x_np)

        loss = nn.BCELoss()(pred, target)
        loss.backward()

        with torch.no_grad():
            grad = x.grad * feat_mask          # mask non-improvable coords
            x_new = x - alpha * torch.sign(grad)
            delta = torch.clamp(x_new - orig, -budget_r, budget_r)
            # Make sure non-improvable coords stay exactly equal to the original
            # (sign(0) is 0 so they shouldn't drift, but be defensive).
            delta = delta * feat_mask
            x = (orig + delta).detach().requires_grad_(True)

    # ── adversarial tie-breaking among positive iterates ────────────────
    if adversarial_tiebreak and positive_iterates:
        # Prefer an x' that h calls positive AND f* calls negative.
        for cand in positive_iterates:
            if int(fstar.predict(cand.reshape(1, -1))[0]) == 0:
                return cand
        return positive_iterates[-1]           # all positives are also f*=1

    return x.detach().numpy()


# ═══════════════════════════════════════════════════════════
# 5.  LOSS FUNCTION  (Eq. 1 from paper)
# ═══════════════════════════════════════════════════════════

def compute_loss_and_outcome(
    x, h_model, fstar, budget_r,
    improvable_features=None,
    threshold=0.5,
    adversarial_tiebreak=False,
):
    """
    Returns (loss, outcome_code). See OUTCOME_CODES below.
    """
    h_model.eval()
    x_t = torch.FloatTensor(x)
    h_orig = int(h_model(x_t.unsqueeze(0)).item() >= threshold)
    f_orig = int(fstar.predict(x.reshape(1, -1))[0])

    if h_orig == 1:
        err = int(f_orig == 0)
        return err, ("FP" if f_orig == 0 else "TP")

    if budget_r > 0:
        x_imp = improve_agent(
            x, h_model, budget_r,
            fstar=fstar,
            improvable_features=improvable_features,
            threshold=threshold,
            adversarial_tiebreak=adversarial_tiebreak,
        )
        h_imp = int(h_model(torch.FloatTensor(x_imp).unsqueeze(0)).item() >= threshold)
        if h_imp == 1:
            f_imp = int(fstar.predict(x_imp.reshape(1, -1))[0])
            if f_orig == 1:
                code = "FN→TP" if f_imp == 1 else "FN→FP"
            else:
                code = "TN→TP" if f_imp == 1 else "TN→FP"
            err = int(f_imp == 0)
            return err, code
        else:
            err = int(f_orig == 1)             # FN still counts as error
            return err, "no_move"

    # r = 0: no improvement possible
    err = int(h_orig != f_orig)
    return err, ("FN" if f_orig == 1 else "TN")


# ═══════════════════════════════════════════════════════════
# 6.  FULL EVALUATION
# ═══════════════════════════════════════════════════════════

OUTCOME_CODES = ["TP", "FP", "TN", "FN", "FN→TP", "FN→FP", "TN→TP", "TN→FP", "no_move"]


def evaluate(
    X_test, fstar, models_dict, budgets,
    improvable_features=None,
    threshold=0.5,
    adversarial_tiebreak=False,
):
    print("\n" + "=" * 70)
    print(f"Results  (adversarial_tiebreak={adversarial_tiebreak})")
    print("=" * 70)

    for name, h_model in models_dict.items():
        print(f"\n{'─' * 70}")
        print(f"  Model: {name}")
        print(f"{'─' * 70}")
        header = (f"  {'Budget r':>9} │ {'Error%':>7} │ {'Errors':>6} │"
                  f" {'FN→TP':>6} │ {'FN→FP':>6} │ {'TN→TP':>6} │"
                  f" {'TN→FP':>6} │ {'no_move':>8}")
        print(header)
        print("  " + "─" * len(header.strip()))

        for r in budgets:
            counts = {c: 0 for c in OUTCOME_CODES}
            total_err = 0
            for i in range(len(X_test)):
                err, code = compute_loss_and_outcome(
                    X_test[i], h_model, fstar, r,
                    improvable_features=improvable_features,
                    threshold=threshold,
                    adversarial_tiebreak=adversarial_tiebreak,
                )
                total_err += err
                counts[code] = counts.get(code, 0) + 1

            n = len(X_test)
            print(
                f"  {r:>9.1f} │ {100 * total_err / n:>6.2f}% │ {total_err:>6} │"
                f" {counts['FN→TP']:>6} │ {counts['FN→FP']:>6} │"
                f" {counts['TN→TP']:>6} │ {counts['TN→FP']:>6} │"
                f" {counts.get('no_move', 0):>8}"
            )


# ═══════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PAC Learning with Improvements  —  corrected baseline")
    print("=" * 60)

    print("\n[1] Generating synthetic data (class_sep=4) ...")
    X_train, X_test, y_train, y_test = generate_synthetic_data(class_sep=4.0)
    print(f"    Train: {len(y_train)}   Test: {len(y_test)}")
    print(f"    Train balance: {(y_train==1).sum()} pos / {(y_train==0).sum()} neg")

    print("\n[2] Training f* (DTC2) ...")
    fstar = train_fstar(X_train, y_train)
    print(f"    f* train acc = {accuracy_score(y_train, fstar.predict(X_train)):.4f}")
    print(f"    f* test  acc = {accuracy_score(y_test,  fstar.predict(X_test)):.4f}")

    # Train h on f*'s labels (Section 6, paragraph "Classifiers").
    y_fstar_train = fstar.predict(X_train)

    print("\n[3] Training decision-maker models h ...")
    configs = [
        ("BCE  (wFP=1.0, wFN=1.0)",   1.0, 1.0),
        ("wBCE (wFP=3.0, wFN=0.009)", 3.0, 0.009),
        ("wBCE (wFP=5.0, wFN=0.009)", 5.0, 0.009),
        ("wBCE (wFP=8.0, wFN=0.009)", 8.0, 0.009),
    ]
    models = {}
    for label, wfp, wfn in configs:
        print(f"    Training {label} ...")
        m = train_h(X_train, y_fstar_train, w_fp=wfp, w_fn=wfn)
        preds = (m(torch.FloatTensor(X_test)).detach().numpy().flatten() >= 0.5).astype(int)
        print(f"      baseline acc (r=0): {accuracy_score(y_test, preds):.4f}   "
              f"FP={((preds==1) & (y_test==0)).sum()}   "
              f"FN={((preds==0) & (y_test==1)).sum()}")
        models[label] = m

    budgets = [0.0, 0.5, 1.0, 2.0, 4.0]

    # Synthetic dataset → all 8 features improvable (Table 1).
    improvable_features = list(range(X_train.shape[1]))

    # Paper's empirical methodology (single-shot PGD).
    evaluate(
        X_test, fstar, models, budgets,
        improvable_features=improvable_features,
        threshold=0.5,
        adversarial_tiebreak=False,
    )

    # Eq. (1)-faithful evaluation with adversarial tie-breaking.
    # Useful as a robustness check / additional baseline.
    evaluate(
        X_test, fstar, models, budgets,
        improvable_features=improvable_features,
        threshold=0.5,
        adversarial_tiebreak=True,
    )