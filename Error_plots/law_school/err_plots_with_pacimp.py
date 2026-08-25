import random
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import arff

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 0. REPRODUCIBILITY & UTILS
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
def save_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


# =============================================================================
# 1. DATA PREPARATION (LAW SCHOOL DATASET)
# =============================================================================

# def load_and_process_lawschool(datapath='law_school_clean.csv'):
def load_and_process_lawschool(datapath):
    df = pd.read_csv(datapath)
    target = 'pass_bar'

    non_manipulable = ['fulltime', 'male', 'race']
    feature_cols = [c for c in df.columns if c not in [target] + non_manipulable]

    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test, feature_cols


def build_dataloaders(X, Y, batch_size=64):
    
    X = X.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)
    
    X_train_t = torch.from_numpy(X)
    Y_train_t = torch.from_numpy(Y)
    
    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    
    return train_dl

# =============================================================================
# 2. MODELS & UTILS
# =============================================================================
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

# =============================================================================
# 2b. TWO-LAYER NN CLASSIFIER (ADDED)
# =============================================================================
class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.out(self.hidden(x)))


def weighted_bce_loss(pred, target, w_fp=1.0, w_fn=1.0):
    pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
    return (-(w_fn * target * torch.log(pred) +
              w_fp * (1 - target) * torch.log(1 - pred))).mean()


def train_two_layer_nn(X_train, y_train, w_fp=1.0, w_fn=1.0,
                       epochs=80, bs=512, hidden=64, lr=1e-1, seed=42):
    torch.manual_seed(seed)
    Xt = torch.FloatTensor(X_train)
    yt = torch.FloatTensor(y_train).view(-1, 1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)

    model = TwoLayerNN(X_train.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in tqdm(range(epochs), desc="Training Two-Layer NN", leave=False):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            weighted_bce_loss(model(xb), yb, w_fp, w_fn).backward()
            opt.step()

    model.eval()
    return model


def improve_agent_nn(
    x_orig,
    h_model,
    budget_r,
    alpha=0.05,
    max_iter=100,
    threshold=0.5,
    improvable_features=None,
):
    h_model.eval()
    d = x_orig.shape[0]

    if improvable_features is None:
        feat_mask = torch.ones(d, device=device)
    else:
        feat_mask = torch.zeros(d, device=device)
        feat_mask[list(improvable_features)] = 1.0

    orig = torch.FloatTensor(x_orig).to(device)
    x = orig.clone().detach().requires_grad_(True)
    target = torch.tensor([[1.0]], device=device)

    for _ in tqdm(range(max_iter), desc="Improving Agent (NN)", leave=False):
        if x.grad is not None:
            x.grad.zero_()

        pred = h_model(x.unsqueeze(0))
        if pred.item() >= threshold:
            return x.detach().cpu().numpy()

        loss = nn.BCELoss()(pred, target)
        loss.backward()

        with torch.no_grad():
            grad = x.grad * feat_mask
            x_new = x - alpha * torch.sign(grad)
            delta = torch.clamp(x_new - orig, -budget_r, budget_r)
            delta = delta * feat_mask
            x = (orig + delta).detach().requires_grad_(True)

    return x.detach().cpu().numpy()


def compute_err_imp_nn(
    model,
    X_np,
    eta_model_oracle,
    budget_r,
    alpha=0.05,
    max_iter=100,
    threshold=0.5,
    improvable_features=None,
):
    model.eval()
    errors = []

    for x in X_np:
        x_strat = improve_agent_nn(
            x,
            model,
            budget_r=budget_r,
            alpha=alpha,
            max_iter=max_iter,
            threshold=threshold,
            improvable_features=improvable_features,
        )

        pred = int(model(torch.FloatTensor(x_strat).unsqueeze(0).to(device)).item() >= threshold)
        
        
        # prob = eta_model_oracle.predict_proba(x_strat.reshape(1, -1))[:, 1]
        # y_oracle = np.random.binomial(1, prob)[0]

        # errors.append(int(pred != y_oracle))
     
        prob = eta_model_oracle.predict_proba(x_strat.reshape(1, -1))[:, 1][0]

        error = pred * (1 - prob) + (1 - pred) * prob

        errors.append(error)        
        

    return float(np.mean(errors))



# =============================================================================
# 3. STRATEGIC LOGIC (Helpers)
# =============================================================================
### Strategic response function for batch of samples

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

# =============================================================================
# 4. TRAINERS
# =============================================================================
class BayesTrainer:
    def __init__(self, model, train_dl, opt, C=1.0):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.C = C

    def loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        margin = y_tilde * pred
        hinge = torch.relu(1 - margin).mean()
        reg = 0.5 * torch.sum(self.model.out.weight ** 2)
        # return self.C * hinge + reg
        return hinge

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs), desc="Training Bayes Optimal (f*)", leave=False):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.loss(xb, y_tilde)
                l.backward()
                self.opt.step()

class StrategicTrainer:
    def __init__(self, model, train_dl, opt, alpha, beta, C=1.0):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.alpha = alpha.clone().detach().to(device)
        self.beta = beta
        self.C = C

    def strat_loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde * (pred + s_gain)
        hinge = torch.relu(1 - margin).mean()
        reg = 0.5 * torch.sum(w ** 2)
        # return self.C * hinge + reg
        return hinge

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs), desc="Training Strategic (f*_s)", leave=False):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(device)
                yb = yb.to(device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.strat_loss(xb, y_tilde)
                l.backward()
                self.opt.step()

class ImprovementAwareTrainer:
    def __init__(self, model, train_dl, opt, eta_model, alpha, beta, C=1.0):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.eta_model = eta_model
        self.alpha_np = alpha.cpu().numpy()
        self.alpha_t = alpha.clone().detach().to(device)
        self.beta = beta
        self.C = C

    def train_step(self, xb, yb):
        w_curr = self.model.out.weight.detach().cpu().numpy().flatten()
        b_curr = self.model.out.bias.detach().cpu().item()
        
        xb_np = xb.cpu().numpy()
        X_strat = get_strategic_response_batch(xb_np, w_curr, b_curr, self.alpha_np, self.beta)
        
        probs_new = self.eta_model.predict_proba(X_strat)[:, 1]
        y_imp_np = np.random.binomial(1, probs_new)
        
        y_imp_t = torch.from_numpy(y_imp_np).float().to(device).view(-1, 1)
        y_tilde_imp = (2 * y_imp_t - 1).squeeze()
        
        pred = self.model(xb.to(device)).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha_t + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde_imp * (pred + s_gain)
        hinge = torch.relu(1 - margin).mean()
        # reg = 0.5 * torch.sum(w ** 2)        
        # return self.C * hinge + reg
        
        return hinge

    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs), desc="Training Improvement-Aware (f*_imp)", leave=False):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                loss = self.train_step(xb, yb)
                loss.backward()
                self.opt.step()

# =============================================================================
# 5. EVALUATION METRIC
# =============================================================================
def compute_err_imp(model, X_np, eta_model_oracle, alpha_np, beta):
    model.eval()
    w = model.out.weight.detach().cpu().numpy().flatten()
    b = model.out.bias.detach().cpu().item()
    
    # 1. Test set reacts strategically to the model
    X_strat = get_strategic_response_batch(X_np, w, b, alpha_np, beta)
    
    # 2. Model makes predictions on the manipulated features
    scores = X_strat @ w + b
    preds = (scores >= 0).astype(int)
     
    # 3. Calculate expected error against the *Oracle* true probabilities
    etas = eta_model_oracle.predict_proba(X_strat)[:, 1]
    errors = preds * (1 - etas) + (1 - preds) * etas
    
    return np.mean(errors)

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    set_seed(42)
    
    try:
        # X_train, X_test, y_train, y_test, feature_cols = load_and_process_lawschool('law_dataset.arff')
        X_train, X_test, y_train, y_test, feature_cols = load_and_process_lawschool('NeurIPS_submission/Error_plots/law_school/law_school_clean.csv')
    except Exception as e:
        print(f"Data loading error: {e}.")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    d = X_train_sc.shape[1]
    
    # -------------------------------------------------------------------------
    # ORACLE SETUP
    # -------------------------------------------------------------------------
    print("\nTraining ORACLE Eta Model on full training set...")
    eta_model_oracle = estimate_eta(X_train_sc, y_train)
    
    # Experiment Constants
    # fixed_beta = 5.0
    fixed_beta = 1.0
    
    
    # DYNAMIC ALPHA SIZING for Law School
    # cost_map = {
    # 'decile1b': 1.0,  
    # 'decile3': 1.0,
    # 'lsat': 1.0,
    # 'ugpa': 1.0,
    # 'zfygpa': 1.0,
    # 'zgpa': 1.0,
    # 'fam_inc': 1.0,
    # 'tier': 1.0
    # }   
    
    cost_map = {
    'decile1b': 0.2,  
    'decile3': 0.2,
    'lsat': 0.2,
    'ugpa': 0.2,
    'zfygpa': 0.2,
    'zgpa': 0.1,
    'fam_inc': 0.1,
    'tier': 0.1
    }  
    
    
    # cost_map = {
    # 'decile1b': 1.0,  
    # 'decile3': 1.0,
    # 'lsat': 1.5,
    # 'ugpa': 2.0,
    # 'zfygpa': 10.0,
    # 'zgpa': 10,
    # 'fam_inc': 0.8,
    # 'tier': 0.8
    # } 
    
    # cost_map = {
    # 'decile1b': 1.0,  
    # 'decile3': 1.0,
    # 'lsat': 1.5,
    # 'ugpa': 2.0,
    # 'zfygpa': 10.0,
    # 'zgpa': 10,
    # 'fam_inc': 0.8,
    # 'tier': 0.8
    # } 
    
    # cost_map = {
    # 'decile1b': 0.125,  
    # 'decile3': 0.125,
    # 'lsat': 0.125,
    # 'ugpa': 0.125,
    # 'zfygpa': 0.125,
    # 'zgpa': 0.125,
    # 'fam_inc': 0.125,
    # 'tier': 0.125
    # } 
    
    alpha_vals = [cost_map[f] for f in cost_map]
    print(f"Assigned Alpha Costs: {dict(zip(feature_cols, alpha_vals))}")
    
    fixed_alpha = torch.tensor(alpha_vals, dtype=torch.float32, device=device)
    fixed_alpha_np = fixed_alpha.cpu().numpy()
    
    
    # train_sizes = [60, 200, 500, 1000, 2000, 4000, 6000] 
    # train_sizes = [100, 200, 400, 600, 1000, 2000, 4000, 6000, 10000, 12000, 14000] 
    # train_sizes = [60, 140, 300, 500, 1000, 4000, 6000, 8000, 10000, 12000, 14000] 
    train_sizes = [60, 140, 300, 500, 1000, 4000, 6000] 
    
    
    print("Training sizes on which it is being trained: ", train_sizes) ## updated training size
    
    # Define number of random splits per sample size
    num_runs = 10 # for now, to get it fast 
    
    # Tracking metrics
    mean_b, std_b = [], []
    mean_s, std_s = [], []
    mean_i, std_i = [], []
    mean_nn, std_nn = [], []
    
    
    print("\n" + "="*85)
    print(f"{'STARTING SAMPLE COMPLEXITY EXPERIMENT (AVERAGED OVER ' + str(num_runs) + ' RUNS)':^85}")
    print("="*85)
    
    for n in train_sizes:
        
        runs_err_b, runs_err_s, runs_err_i = [], [], []
        runs_err_nn = []
        
        for run in range(num_runs):
            print(f"  > Run {run + 1}/{num_runs}...")
            
            X_tr_sub, y_tr_sub = resample(
                X_train_sc, y_train, 
                n_samples=n, 
                replace=True, 
                random_state=42 + run
            )
            
            train_dl_sub = build_dataloaders(X_tr_sub, y_tr_sub)
            eta_model_sub = estimate_eta(X_tr_sub, y_tr_sub)
            
            model_bayes = LinearClassifier(d, 1).to(device)
            # opt_b = torch.optim.Adam(model_bayes.parameters(), lr=0.1)
            opt_b = torch.optim.SGD(model_bayes.parameters(), lr=0.1, momentum=0.9)
            trainer_b = BayesTrainer(model_bayes, train_dl_sub, opt_b, C=1.0)
            trainer_b.train(n_epochs=50) 
            
            model_strat = LinearClassifier(d, 1).to(device)
            model_strat.load_state_dict(model_bayes.state_dict())
            # opt_s = torch.optim.Adam(model_strat.parameters(), lr=0.1)
            opt_s = torch.optim.SGD(model_strat.parameters(), lr=0.1, momentum=0.9)
            trainer_s = StrategicTrainer(model_strat, train_dl_sub, opt_s, fixed_alpha, fixed_beta, C=1.0)
            trainer_s.train(n_epochs=65)
            
            model_imp = LinearClassifier(d, 1).to(device)
            model_imp.load_state_dict(model_bayes.state_dict())
            # opt_imp = torch.optim.Adam(model_imp.parameters(), lr=0.1)
            opt_imp = torch.optim.SGD(model_imp.parameters(), lr=0.1, momentum=0.9)
            trainer_imp = ImprovementAwareTrainer(model_imp, train_dl_sub, opt_imp, eta_model_sub, fixed_alpha, fixed_beta, C=1.0)
            trainer_imp.train(n_epochs=70) 
            
            err_b = compute_err_imp(model_bayes, X_test_sc, eta_model_oracle, fixed_alpha_np, fixed_beta)
            err_s = compute_err_imp(model_strat, X_test_sc, eta_model_oracle, fixed_alpha_np, fixed_beta)
            err_i = compute_err_imp(model_imp, X_test_sc, eta_model_oracle, fixed_alpha_np, fixed_beta)
            
            
            rng_state = save_rng_state()

            # 6b. Train Two-Layer NN classifier
            model_nn = train_two_layer_nn(
                X_tr_sub, y_tr_sub,
                w_fp=1.0, w_fn=1.0,
                epochs=80, bs=512, hidden=16, lr=1e-1, seed=42
            )

            # 7b. Evaluate Improvement Error for NN using the oracle
            err_nn = compute_err_imp_nn(
                model_nn, X_test_sc, eta_model_oracle,
                budget_r=fixed_beta, alpha=0.05, max_iter=10,
                threshold=0.5, improvable_features=list(range(d))
            )
            runs_err_nn.append(err_nn)

            load_rng_state(rng_state)
            
            runs_err_b.append(err_b)
            runs_err_s.append(err_s)
            runs_err_i.append(err_i)
            
        mean_b.append(np.mean(runs_err_b))
        std_b.append(np.std(runs_err_b))
        
        mean_s.append(np.mean(runs_err_s))
        std_s.append(np.std(runs_err_s))
        
        mean_i.append(np.mean(runs_err_i))
        std_i.append(np.std(runs_err_i))
        
        mean_nn.append(np.mean(runs_err_nn))
        std_nn.append(np.std(runs_err_nn))
        
        print(f"  [Aggregated Results for N={n} over {num_runs} runs]")
        print(f"    err_imp(f*)     = {mean_b[-1]:.4f} ± {std_b[-1]:.4f}")
        print(f"    err_imp(f*_s)   = {mean_s[-1]:.4f} ± {std_s[-1]:.4f}")
        print(f"    err_imp(f*_imp) = {mean_i[-1]:.4f} ± {std_i[-1]:.4f}")
        print(f"    err_imp(h)  = {mean_nn[-1]:.4f} ± {std_nn[-1]:.4f}")

    # =============================================================================
    # 7. PLOTTING THE RESULTS
    # =============================================================================
    print("\nGenerating sample complexity plots with variance bounds...")
    
    mean_b, std_b = np.array(mean_b), np.array(std_b)
    mean_s, std_s = np.array(mean_s), np.array(std_s)
    mean_i, std_i = np.array(mean_i), np.array(std_i)
    mean_nn, std_nn = np.array(mean_nn), np.array(std_nn)
    plt.figure(figsize=(10, 6))
     
    plt.plot(train_sizes, mean_b, marker='o', linestyle='-', linewidth=2, color='tab:blue', label=r'Optimal Linear ($f^*$)')
    plt.fill_between(train_sizes, mean_b - std_b, mean_b + std_b, color='tab:blue', alpha=0.2)
    
    plt.plot(train_sizes, mean_s, marker='s', linestyle='-', linewidth=2, color='tab:orange', label=r'Strategic Linear ($f_s^*$)')
    plt.fill_between(train_sizes, mean_s - std_s, mean_s + std_s, color='tab:orange', alpha=0.2)
    
    plt.plot(train_sizes, mean_i, marker='^', linestyle='-', linewidth=2, color='tab:green', label=r'Improvement-Aware Linear ($f_{imp}^*$)')
    plt.fill_between(train_sizes, mean_i - std_i, mean_i + std_i, color='tab:green', alpha=0.2)
    
    plt.plot(train_sizes, mean_nn, marker='d', linestyle='-', linewidth=2, color='tab:red', label=r'Pac baseline ($f_{imp}^{NN}$)')
    plt.fill_between(train_sizes, mean_nn - std_nn, mean_nn + std_nn, color='tab:red', alpha=0.2)
    
    plt.xlabel('Training Samples', fontsize=12)
    plt.ylabel('Improvement-Aware Strategic Error', fontsize=12)
    # plt.title(f'Sample Complexity vs Improvement Error', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # plt.savefig('sample_complexity_curves_with_variance.png', dpi=300)
    plt.savefig('law_school_with_pac.png', dpi=300)
    
    plt.show()
    # print("Plot saved as 'sample_complexity_curves_with_variance.png'.")
    print("Plot saved as 'law_school_with_pac.png'.")