import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, SequentialSampler, SubsetRandomSampler
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import os
import random


## Data loading and preprocessing


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def data_split(X, y, test_size, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



def load_and_process_lawschool(datapath):
    
    df = pd.read_csv(datapath)
    target = 'pass_bar'
    
    non_manipulable = ['fulltime', 'male', 'race']
    feature_cols = [c for c in df.columns if c not in [target] + non_manipulable]

    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    
    return X, y


def build_dataloaders(X, Y, batch_size=64):
    
    X = X.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)
    
    X_train_t = torch.from_numpy(X)
    Y_train_t = torch.from_numpy(Y)
    
    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
    
    return train_dl


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

    return X_manip, manipulated_indices

### Classifier definitions and training loops

class LinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out = nn.Linear(in_dim, out_dim, bias=True)
        
    def forward(self, x):
        return self.out(x)

def estimate_eta(X_tr, y_tr):
    pipe = Pipeline([('clf', LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=1000, C=10, random_state=42))])
    cal = CalibratedClassifierCV(pipe, cv=5, method='isotonic')
    cal.fit(X_tr, y_tr)
    return cal




class optimal_linear_classifier:
    def __init__(self, model, train_dl, opt, device):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.device = device            
        
    def loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        # print('>>>>>', self.device, pred.device, y_tilde.device)
        margin = y_tilde * pred
        
    
        w = self.model.out.weight.squeeze()
        
        reg = 0.5 * torch.sum(w ** 2)
        lambda_reg = 1e-3
        return torch.relu(1 - margin).mean() + lambda_reg * reg
        # return torch.relu(1 - margin).mean()
    
    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs)):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(self.device)
                yb = yb.to(self.device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.loss(xb, y_tilde)
                l.backward()
                self.opt.step()

class StrategicTrainer:
    
    def __init__(self, model, train_dl, opt, alpha, beta, device):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.device = device
        self.alpha = alpha.clone().detach().to(self.device)
        self.beta = beta; 
        
    def strat_loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde * (pred + s_gain)
                
        reg = 0.5 * torch.sum(w ** 2)
        lambda_reg = 1e-3   # try 1e-3 or 1e-2
        return torch.relu(1 - margin).mean() + lambda_reg * reg
    
        # return torch.relu(1 - margin).mean()
        
    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs)):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                xb = xb.to(self.device)
                yb = yb.to(self.device).view(-1).long()
                y_tilde = (2 * yb - 1).float()
                l = self.strat_loss(xb, y_tilde)
                l.backward()
                self.opt.step()

class ImprovementAwareTrainer:
    
    def __init__(self, model, train_dl, opt, eta_model, alpha, beta , device):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.eta_model = eta_model
        self.alpha_np = alpha.cpu().numpy()
        self.device = device
        self.alpha_t = alpha.clone().detach().to(self.device)
        self.beta = beta
        
    def train_step(self, xb, yb):
        w_curr = self.model.out.weight.detach().cpu().numpy().flatten()
        b_curr = self.model.out.bias.detach().cpu().item()
        
        xb_np = xb.cpu().numpy()
        X_strat, _ = get_strategic_response_batch(xb_np, w_curr, b_curr, self.alpha_np, self.beta)
        
        probs_new = self.eta_model.predict_proba(X_strat)[:, 1]
        y_imp_np = np.random.binomial(1, probs_new)
        
        y_imp_t = torch.from_numpy(y_imp_np).float().to(self.device).view(-1, 1)
        y_tilde_imp = (2 * y_imp_t - 1).squeeze()
        
        pred = self.model(xb.to(self.device)).squeeze(-1)
        w = self.model.out.weight.squeeze()
        
        roi = w / (self.alpha_t + 1e-8)
        s_gain = self.beta * torch.max(roi)
        
        margin = y_tilde_imp * (pred + s_gain)
        
        reg = 0.5 * torch.sum(w ** 2)
        lambda_reg = 1e-3
        return torch.relu(1 - margin).mean() + lambda_reg * reg
    
        # return torch.relu(1 - margin).mean()
    
    def train(self, n_epochs):
        self.model.train()
        for _ in tqdm(range(n_epochs)):
            for xb, yb in self.train_dl:
                self.opt.zero_grad()
                loss = self.train_step(xb, yb)
                loss.backward()
                self.opt.step()


### Evaluation of strategic manipulation and improvement rates

def compute_manipulated_stats(model, X_test, y_test, eta_model, alpha_np, beta):

    model.eval()

    w = model.out.weight.detach().cpu().numpy().flatten()
    b = model.out.bias.detach().cpu().item()

    # strategic features
    X_strat, manipulated_indices = get_strategic_response_batch(X_test, w, b, alpha_np, beta)
    
    num_manipulated = len(manipulated_indices)

    probs = eta_model.predict_proba(X_strat)[:,1]
    y_tilde = np.random.binomial(1, probs)

    improved_mask = (y_test == 0) & (y_tilde == 1)

    num_improved = np.sum(improved_mask[manipulated_indices])

    improvement_rate = (num_improved / num_manipulated if num_manipulated > 0 else 0)

    return w, b, num_manipulated, num_improved, improvement_rate

def compute_err_imp(model, X_np, eta_model_oracle, alpha_np, beta):
    model.eval()
    w = model.out.weight.detach().cpu().numpy().flatten()
    b = model.out.bias.detach().cpu().item()
    
    X_strat, _ = get_strategic_response_batch(X_np, w, b, alpha_np, beta)
    
    scores = X_strat @ w + b
    preds = (scores >= 0).astype(int)
    
    etas = eta_model_oracle.predict_proba(X_strat)[:, 1]
    erros = np.mean(preds * (1 - etas) + (1 - preds) * etas)
    
    return np.mean(erros)

## Main training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
set_seed(42)

X, y = load_and_process_lawschool('law_school_clean.csv')

X_train, X_test, y_train, y_test = data_split(X, y, 0.3)
train_dl = build_dataloaders(X_train, y_train, batch_size=64)
d = X_train.shape[1]  
eta_model = estimate_eta(X_train, y_train)


beta = 1.0

alpha = torch.tensor([1, 1, 1.5, 2.0, 10, 10, 0.8, 0.8], dtype=torch.float32, device=device)



# Train Optimal linear classifier (Independent of alpha and beta)

model_linear = LinearClassifier(d, 1).to(device)
opt_linear = torch.optim.SGD(model_linear.parameters(), lr=0.1, momentum=0.9)

trainer_linear = optimal_linear_classifier(model_linear, train_dl, opt_linear, device)
trainer_linear.train(n_epochs=50) 


# Train Strategic classifier (f*_s)

model_strat = LinearClassifier(d, 1).to(device)
model_strat.load_state_dict(model_linear.state_dict()) # Warm start
opt_s = torch.optim.SGD(model_strat.parameters(), lr=0.1, momentum=0.9)

trainer_s = StrategicTrainer(model_strat, train_dl, opt_s, alpha, beta, device)
trainer_s.train(n_epochs=65)


# Train Improvement-Aware classiifier (f*_imp)

model_imp_aware = LinearClassifier(d, 1).to(device)
model_imp_aware.load_state_dict(model_strat.state_dict()) # Warm start
opt_imp = torch.optim.SGD(model_imp_aware.parameters(), lr=0.1, momentum=0.9)

trainer_imp = ImprovementAwareTrainer(model_imp_aware, train_dl, opt_imp, eta_model, alpha, beta, device)
trainer_imp.train(n_epochs=70) 



# Computing error_imp for all classifiers

err_linear = compute_err_imp(model_linear, X_test, eta_model, alpha.cpu().numpy(), beta)
print("err_imp_linear:", err_linear)

err_strat = compute_err_imp(model_strat, X_test, eta_model, alpha.cpu().numpy(), beta)
print("err_imp_strat:", err_strat)

err_imp_aware = compute_err_imp(model_imp_aware, X_test, eta_model, alpha.cpu().numpy(), beta)
print("err_imp_imp_aware:", err_imp_aware)


alpha_np = alpha.cpu().numpy()
alpha_list = alpha_np.tolist()

results = []

models = {
    "Optimal Linear classifier (f*)": model_linear,
    "Strategic (f*_s)": model_strat,
    "Strat-Imp-Aware (f*_imp)": model_imp_aware
}




for name, model in models.items():
    
    w, b, n_manip, n_imp, rate = compute_manipulated_stats(
        model,
        X_test,
        y_test,
        eta_model,
        alpha_np,
        beta
    )   
    
    print(f"{name   }: Manipulated={n_manip}, Improved={n_imp}, Improvement Rate={rate:.4f}" )