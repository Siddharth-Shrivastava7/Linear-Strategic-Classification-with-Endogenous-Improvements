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
import warnings, copy
warnings.filterwarnings('ignore')
from tqdm import tqdm

# device = torch.device("cpu")

# def set_seed(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_and_process_lawschool(datapath='law_school_clean.csv'):
    df = pd.read_csv('law_school_clean.csv')
    target = 'pass_bar'

    non_manipulable = ['fulltime', 'male', 'race']
    feature_cols = [c for c in df.columns if c not in [target] + non_manipulable]

    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_cols

def build_dataloaders(X_scaled, Y, batch_size=512):
    X = X_scaled.astype(np.float32)
    if Y.ndim == 1: Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)
    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    X_train_t = torch.from_numpy(X_train_np)
    Y_train_t = torch.from_numpy(Y_train_np)
    # X_val_t = torch.from_numpy(X_val_np)
    # Y_val_t = torch.from_numpy(Y_val_np)

    # n_pos = torch.sum(Y_train_t == 1).item()
    # n_neg = torch.sum(Y_train_t == 0).item()
    # if n_pos == 0 or n_neg == 0: 
    #     weights = torch.ones_like(Y_train_t).view(-1)
    # else:
    #     w_pos = 1.0 / n_pos
    #     w_neg = 1.0 / n_neg
    #     weights = torch.where(Y_train_t.view(-1) == 1, w_pos, w_neg)

    # sampler = WeightedRandomSampler(weights, num_samples=len(Y_train_t), replacement=True)
    # train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, sampler=sampler)
    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True) 
    return train_dl

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out = nn.Linear(in_dim, out_dim, bias=True)
    def forward(self, x):
        return self.out(x)

def estimate_eta(X_tr, y_tr):
    pipe = Pipeline([('clf', LogisticRegression(solver='lbfgs', max_iter=200, C=10, random_state=42))])
    cal = CalibratedClassifierCV(pipe, cv=3, method='isotonic')
    cal.fit(X_tr, y_tr)
    return cal

# def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
#     X_manip = X_batch_np.copy()
#     roi = w_np / (alpha_np + 1e-12)
#     k_star = np.argmax(roi)
#     scores = X_batch_np @ w_np + b_np
#     neg_indices = np.where(scores < 0)[0]
    
#     if len(neg_indices) > 0:
#         gap = -scores[neg_indices] 
#         w_k = w_np[k_star]
#         if w_k > 1e-8:
#             delta_k = gap / w_k
#             cost = alpha_np[k_star] * delta_k
#             do_manipulate = cost <= beta
#             idx_to_change = neg_indices[do_manipulate]
#             delta_to_apply = delta_k[do_manipulate]
#             X_manip[idx_to_change, k_star] += (delta_to_apply + 1e-5)
#     return X_manip

def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
    batch_size, d = X_batch_np.shape
    X_manip = X_batch_np.copy()
    
    # last two features are categorical
    categorical_indices = [d-2, d-1]
    
    roi = w_np / (alpha_np + 1e-12)
    k_star = np.argmax(roi)
    
    scores = X_batch_np @ w_np + b_np
    neg_indices = np.where(scores < 0)[0]

    manipulated_indices = []
    
    if len(neg_indices) > 0:
        
        # =========================
        # CASE 1: Numerical feature
        # =========================
        if k_star not in categorical_indices:
            
            gap = -scores[neg_indices] 
            w_k = w_np[k_star]
            
            if w_k > 1e-8:
                delta_k = gap / w_k
                cost = alpha_np[k_star] * delta_k
                
                do_manipulate = cost <= beta
                
                idx_to_change = neg_indices[do_manipulate]
                delta_to_apply = delta_k[do_manipulate]
                
                X_manip[idx_to_change, k_star] += (delta_to_apply + 1e-5)

                manipulated_indices = idx_to_change.tolist()
        
        # =========================
        # CASE 2: Categorical feature
        # =========================
        else:
            for i in neg_indices:
                
                current_val = X_batch_np[i, k_star]
                best_val = current_val
                best_gain = 0
                
                # corrected ranges
                if k_star == d-2:   # fam_inc: 1–5
                    possible_values = [1,2,3,4,5]
                else:              # tier: 1–6
                    possible_values = [1,2,3,4,5,6]
                
                for v in possible_values:
                    if v == current_val:
                        continue
                    
                    gain = w_np[k_star] * (v - current_val)
                    cost = alpha_np[k_star]
                    
                    if gain > best_gain and cost <= beta:
                        best_gain = gain
                        best_val = v
                
                if best_val != current_val:
                    X_manip[i, k_star] = best_val
                    manipulated_indices.append(i)

    return X_manip


class BayesTrainer:
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

X_train, X_test, y_train, y_test, feature_cols = load_and_process_lawschool('law_school_clean.csv')

# Subsample to speed up execution for the simulation sweep
# train_size = len(X_train)
# idx_tr = np.random.choice(len(X_train), train_size, replace=False)
# X_train, y_train = X_train[idx_tr], y_train[idx_tr]

# test_size = min(5000, len(X_test))
# idx_te = np.random.choice(len(X_test), test_size, replace=False)
# X_test, y_test = X_test[idx_te], y_test[idx_te]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
d = X_train_sc.shape[1]

eta_model_oracle = estimate_eta(X_train_sc, y_train)
fixed_beta = 1

# print(feature_cols) # ['decile1b', 'decile3', 'lsat', 'ugpa', 'zfygpa', 'zgpa', 'fam_inc', 'tier']

cost_map = {
    'decile1b': 1.0,  
    'decile3': 1.0,
    'lsat': 1.0,
    'ugpa': 1.0,
    'zfygpa': 1.0,
    'zgpa': 1.0,
    'fam_inc': 1.0,
    'tier': 1.0
}

alpha_multipliers = [0.01, 0.09, 0.1, 0.5, 0.7, 1.0, 2.0, 5.0]
base_alpha = np.array([cost_map.get(f, 100.0) for f in feature_cols])


train_dl = build_dataloaders(X_train_sc, y_train, batch_size=512)

results_b, results_s, results_i = [], [], []
valid_scales = []

for scale in tqdm(alpha_multipliers):
    current_alpha_np = base_alpha * scale
    current_alpha_t = torch.tensor(current_alpha_np, dtype=torch.float32, device=device)
    
    # Bayes
    model_bayes = LinearClassifier(d, 1).to(device)
    opt_b = torch.optim.Adam(model_bayes.parameters(), lr=0.01)
    BayesTrainer(model_bayes, train_dl, opt_b, C=1.0).train(n_epochs=50) 
    
    # Strategic
    model_strat = LinearClassifier(d, 1).to(device)
    opt_s = torch.optim.Adam(model_strat.parameters(), lr=0.01)
    StrategicTrainer(model_strat, train_dl, opt_s, current_alpha_t, fixed_beta, C=1.0).train(n_epochs=50)
    
    # Imp-Aware
    model_imp = LinearClassifier(d, 1).to(device)
    opt_imp = torch.optim.Adam(model_imp.parameters(), lr=0.01)
    ImprovementAwareTrainer(model_imp, train_dl, opt_imp, eta_model_oracle, current_alpha_t, fixed_beta, C=1.0).train(n_epochs=80) 
    
    err_b = compute_err_imp(model_bayes, X_test_sc, eta_model_oracle, current_alpha_np, fixed_beta)
    err_s = compute_err_imp(model_strat, X_test_sc, eta_model_oracle, current_alpha_np, fixed_beta)
    err_i = compute_err_imp(model_imp, X_test_sc, eta_model_oracle, current_alpha_np, fixed_beta)
    
    results_b.append(err_b)
    results_s.append(err_s)
    results_i.append(err_i)
    
    trend_holds = (err_b >= err_s - 1e-4) and (err_s >= err_i - 1e-4)
    if trend_holds: valid_scales.append(scale)
    print(f"Scale: {scale:5.2f} | f*: {err_b:.4f} | f*_s: {err_s:.4f} | f*_imp: {err_i:.4f} | Trend: {trend_holds}")

plt.figure(figsize=(10, 6))
plt.plot(alpha_multipliers, results_b, marker='o', label=r'Optimal Linear ($f^*$)')
plt.plot(alpha_multipliers, results_s, marker='s', label=r'Strategic Linear ($f_s^*$)')
plt.plot(alpha_multipliers, results_i, marker='^', label=r'Improvement-Aware Linear ($f_{imp}^*$)')
plt.xscale('log')
plt.xlabel('Alpha Multiplier (Log Scale)')
plt.ylabel('Improvement Error (err_imp)')
plt.title('Effect of Uniform Manipulation Costs - Law School Dataset')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

if valid_scales:
    plt.axvspan(min(valid_scales), max(valid_scales), color='yellow', alpha=0.15, label='Trend Holds Region')
    plt.legend()

plt.savefig('lawschool_uniform_alpha_sim_update.png', dpi=300)
pd.DataFrame({'Scale': alpha_multipliers, 'err_f': results_b, 'err_fs': results_s, 'err_fimp': results_i}).to_csv('lawschool_uniform_alpha_sim.csv', index=False)