import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import os
import time


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

def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
    batch_size, d = X_batch_np.shape
    X_manip = X_batch_np.copy()
    
    roi = w_np / (alpha_np + 1e-12)
    k_star = np.argmax(roi)
    
    scores = X_batch_np @ w_np + b_np
    neg_indices = np.where(scores < 0)[0]
    
    manipulated_indices = []

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
            manipulated_indices = idx_to_change.tolist()

    return X_manip, manipulated_indices



# def get_strategic_response_batch(X_batch_np, w_np, b_np, alpha_np, beta):
#     batch_size, d = X_batch_np.shape
#     X_manip = X_batch_np.copy()
    
#     # last two features are categorical
#     categorical_indices = [d-2, d-1]
    
#     roi = w_np / (alpha_np + 1e-12)
#     k_star = np.argmax(roi)
    
#     scores = X_batch_np @ w_np + b_np
#     neg_indices = np.where(scores < 0)[0]

#     manipulated_indices = []
    
#     if len(neg_indices) > 0:
        
#         # =========================
#         # CASE 1: Numerical feature
#         # =========================
#         if k_star not in categorical_indices:
            
#             gap = -scores[neg_indices] 
#             w_k = w_np[k_star]
            
#             if w_k > 1e-8:
#                 delta_k = gap / w_k
#                 cost = alpha_np[k_star] * delta_k
                
#                 do_manipulate = cost <= beta
                
#                 idx_to_change = neg_indices[do_manipulate]
#                 delta_to_apply = delta_k[do_manipulate]
                
#                 X_manip[idx_to_change, k_star] += (delta_to_apply + 1e-5)

#                 manipulated_indices = idx_to_change.tolist()
        
#         # =========================
#         # CASE 2: Categorical feature
#         # =========================
#         else:
#             for i in neg_indices:
                
#                 current_val = X_batch_np[i, k_star]
#                 best_val = current_val
#                 best_gain = 0
                
#                 # corrected ranges
#                 if k_star == d-2:   # fam_inc: 1–5
#                     possible_values = [1,2,3,4,5]
#                 else:              # tier: 1–6
#                     possible_values = [1,2,3,4,5,6]
                
#                 for v in possible_values:
#                     if v == current_val:
#                         continue
                    
#                     gain = w_np[k_star] * (v - current_val)
#                     cost = alpha_np[k_star]
                    
#                     if gain > best_gain and cost <= beta:
#                         best_gain = gain
#                         best_val = v
                
#                 if best_val != current_val:
#                     X_manip[i, k_star] = best_val
#                     manipulated_indices.append(i)

#     return X_manip, manipulated_indices


class optimal_linear_classifier:
    def __init__(self, model, train_dl, opt, device):
        self.model = model
        self.train_dl = train_dl
        self.opt = opt
        self.device = device            
        
    def loss(self, xb, y_tilde):
        pred = self.model(xb).squeeze(-1)
        margin = y_tilde * pred
        
        # return torch.relu(1 - margin).mean()
    
        w = self.model.out.weight.squeeze()
        reg = 0.5 * torch.sum(w ** 2)
        lambda_reg = 1e-3
        return torch.relu(1 - margin).mean() + lambda_reg * reg
    
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
        
        # return torch.relu(1 - margin).mean()
        
        reg = 0.5 * torch.sum(w ** 2)
        lambda_reg = 1e-3   # try 1e-3 or 1e-2
        return torch.relu(1 - margin).mean() + lambda_reg * reg
        
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