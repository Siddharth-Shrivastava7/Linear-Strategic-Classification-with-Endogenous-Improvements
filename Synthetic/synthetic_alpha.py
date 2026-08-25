import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.io import arff
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import copy

# device = torch.device("cpu")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_and_process_synthetic_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['label','eta']).values
    y = df['label'].values
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def build_dataloaders(X_scaled, Y, batch_size=64):
    X = X_scaled.astype(np.float32)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)
    # X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train_t = torch.from_numpy(X)
    Y_train_t = torch.from_numpy(Y)

    n_pos = torch.sum(Y_train_t == 1).item()
    n_neg = torch.sum(Y_train_t == 0).item()
    if n_pos == 0 or n_neg == 0: 
        weights = torch.ones_like(Y_train_t).view(-1)
    else:
        w_pos = 1.0 / n_pos
        w_neg = 1.0 / n_neg
        weights = torch.where(Y_train_t.view(-1) == 1, w_pos, w_neg)

    sampler = WeightedRandomSampler(weights, num_samples=len(Y_train_t), replacement=True)
    
    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True) 
    return train_dl

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x):
        return self.out(x)

def estimate_eta(X_tr, y_tr):
    pipe = Pipeline([('clf', LogisticRegression(solver='lbfgs', max_iter=1000, C=10, random_state=42))])
    cal = CalibratedClassifierCV(pipe, cv=3, method='isotonic')
    cal.fit(X_tr, y_tr)
    return cal

def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
    X_manip = X_batch_np.copy()
    roi = w_np / (alpha_np + 1e-12)
    k_star = np.argmax(roi)
    scores = X_batch_np @ w_np + b_np
    neg_indices = np.where(scores < 0)[0]
    
    if len(neg_indices) > 0:
        gap = -scores[neg_indices] 
        w_k = w_np[k_star]
        if w_k > 1e-8:
            delta_k = gap / w_k
            cost = alpha_np[k_star] * delta_k
            do_manipulate = cost <= beta
            idx_to_change = neg_indices[do_manipulate]
            delta_to_apply = delta_k[do_manipulate]
            X_manip[idx_to_change, k_star] += (delta_to_apply + 1e-5)
    return X_manip

class optimal_linear_classifier:
    def __init__(self, model, train_dl, opt, C=1.0):
        self.model = model; self.train_dl = train_dl; self.opt = opt; self.C = C
        self.best_loss = float('inf')
        self.best_state = None

    def loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        margin = y_tilde * pred
        hinge = torch.relu(1 - margin).mean()
        # reg = 0.5 * torch.sum(self.model.out.weight ** 2)
        # return self.C * hinge + reg
        return hinge

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs)):
            running_loss, n_batches = 0.0, 0
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(device); yb = yb.to(device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.loss(xb, y_tilde)
                l.backward()
                self.opt.step()
                running_loss += l.item(); n_batches += 1
            avg_loss = running_loss / max(n_batches, 1)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


class StrategicTrainer:
    def __init__(self, model, train_dl, opt, alpha, beta, C=1.0):
        self.model = model; self.train_dl = train_dl; self.opt = opt
        self.alpha = alpha.clone().detach().to(device); self.beta = beta; self.C = C
        self.best_loss = float('inf')
        self.best_state = None

    def strat_loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        w = self.model.out.weight.squeeze()
        roi = w / (self.alpha + 1e-8)
        s_gain = self.beta * torch.max(roi)
        margin = y_tilde * (pred + s_gain)
        hinge = torch.relu(1 - margin).mean()
        # reg = 0.5 * torch.sum(w ** 2)
        # return self.C * hinge + reg
        return hinge

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs)):
            running_loss, n_batches = 0.0, 0
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(device); yb = yb.to(device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.strat_loss(xb, y_tilde)
                l.backward()
                self.opt.step()
                running_loss += l.item(); n_batches += 1
            avg_loss = running_loss / max(n_batches, 1)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


class ImprovementAwareTrainer:
    def __init__(self, model, train_dl, opt, eta_model, alpha, beta, C=1.0):
        self.model = model; self.train_dl = train_dl; self.opt = opt; self.eta_model = eta_model
        self.alpha_np = alpha.cpu().numpy()
        self.alpha_t = alpha.clone().detach().to(device)
        self.beta = beta; self.C = C
        self.best_loss = float('inf')
        self.best_state = None

    def train_step(self, xb, yb):
        w_curr = self.model.out.weight.detach().cpu().numpy().flatten()
        b_curr = self.model.out.bias.detach().cpu().item()
        xb_np = xb.cpu().numpy()
        X_strat = get_strategic_response_batch(xb_np, w_curr, b_curr, self.alpha_np, self.beta)
        probs_new = self.eta_model.predict_proba(X_strat)[:, 1]
        y_imp_t = torch.from_numpy(np.random.binomial(1, probs_new)).float().to(device).view(-1, 1)
        y_tilde_imp = (2 * y_imp_t - 1).squeeze()
        pred = self.model(xb.to(device)).squeeze(-1)
        w = self.model.out.weight.squeeze()
        roi = w / (self.alpha_t + 1e-8)
        s_gain = self.beta * torch.max(roi)
        margin = y_tilde_imp * (pred + s_gain)
        hinge = torch.relu(1 - margin).mean()
        # return self.C * hinge
        return hinge

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs)):
            running_loss, n_batches = 0.0, 0
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                l = self.train_step(xb, yb)
                l.backward()
                self.opt.step()
                running_loss += l.item(); n_batches += 1
            avg_loss = running_loss / max(n_batches, 1)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state) 

def compute_err_imp(model, X_np, eta_model_oracle, alpha_np, beta):
    model.eval()
    w = model.out.weight.detach().cpu().numpy().flatten()
    b = model.out.bias.detach().cpu().item()
    X_strat = get_strategic_response_batch(X_np, w, b, alpha_np, beta)
    scores = X_strat @ w + b
    preds = (scores >= 0).astype(int)
    etas = eta_model_oracle.predict_proba(X_strat)[:, 1]
    return np.mean(preds * (1 - etas) + (1 - preds) * etas)

set_seed(42)

X_train, X_test, y_train, y_test = load_and_process_synthetic_data("synthetic_dataset.csv")

# scaler = StandardScaler()
# X_train_sc = scaler.fit_transform(X_train)
# X_test_sc = scaler.transform(X_test)
# d = X_train_sc.shape[1]
X_train_sc, X_test_sc = X_train, X_test
d = X_train_sc.shape[1]

eta_model_oracle = estimate_eta(X_train_sc, y_train)
fixed_beta = 1.0

base_alpha = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

alpha_multipliers = [0.1, 0.5, 1.0, 5.0, 10.0, 100.0] 

train_dl = build_dataloaders(X_train_sc, y_train, batch_size=512)

results_l, results_s, results_i = [], [], []
valid_scales = []

for scale in tqdm(alpha_multipliers):
    current_alpha_np = base_alpha * scale
    current_alpha_t = torch.tensor(current_alpha_np, dtype=torch.float32, device=device)
    
    # Bayes
    model_linear = LinearClassifier(d, 1).to(device)
    opt_linear = torch.optim.Adam(model_linear.parameters(), lr=0.01)
    optimal_linear_classifier(model_linear, train_dl, opt_linear, C=1.0).train(n_epochs=50) 
    
    # Strategic
    model_strat = LinearClassifier(d, 1).to(device)
    opt_s = torch.optim.Adam(model_strat.parameters(), lr=0.01)
    StrategicTrainer(model_strat, train_dl, opt_s, current_alpha_t, fixed_beta, C=1.0).train(n_epochs=50)
    
    # Imp-Aware
    model_imp = LinearClassifier(d, 1).to(device)
    opt_imp = torch.optim.Adam(model_imp.parameters(), lr=0.01)
    ImprovementAwareTrainer(model_imp, train_dl, opt_imp, eta_model_oracle, current_alpha_t, fixed_beta, C=1.0).train(n_epochs=50) 
    
    err_l = compute_err_imp(model_linear, X_test_sc, eta_model_oracle, current_alpha_np, fixed_beta)
    err_s = compute_err_imp(model_strat, X_test_sc, eta_model_oracle, current_alpha_np, fixed_beta)
    err_i = compute_err_imp(model_imp, X_test_sc, eta_model_oracle, current_alpha_np, fixed_beta)
    
    results_l.append(err_l)
    results_s.append(err_s)
    results_i.append(err_i)
    
    trend_holds = (err_l >= err_s - 1e-4) and (err_s >= err_i - 1e-4)
    if trend_holds: valid_scales.append(scale)
    print(f"Scale: {scale:5.2f} | f*: {err_l:.4f} | f*_s: {err_s:.4f} | f*_imp: {err_i:.4f} | Trend: {trend_holds}")

plt.figure(figsize=(10, 6))
plt.plot(alpha_multipliers, results_l, marker='o', label=r'Optimal Linear ($f^*$)')
plt.plot(alpha_multipliers, results_s, marker='s', label=r'Strategic Linear ($f_s^*$)')
plt.plot(alpha_multipliers, results_i, marker='^', label=r'Improvement-Aware Linear ($f_{imp}^*$)')
plt.xscale('log')
plt.xlabel('Alpha Multiplier (Log Scale)')
plt.ylabel('Improvement Error (err_imp)')
plt.title('Effect of Uniform Manipulation Costs - Synthetic Dataset')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

if valid_scales:
    plt.axvspan(min(valid_scales), max(valid_scales), color='yellow', alpha=0.15, label='Trend Holds Region')
    plt.legend()

plt.savefig('synthetic_uniform_alpha_sim.png', dpi=300)
pd.DataFrame({'Scale': alpha_multipliers, 'err_f': results_l, 'err_fs': results_s, 'err_fimp': results_i}).to_csv('synthetic_uniform_alpha_sim.csv', index=False)