import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import os

from classifiers import *


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

def compute_err_imp_from_weights(w, b, X_np, eta_model_oracle, alpha_np, beta):

    w = np.array(w)

    # strategic response
    X_strat, _ = get_strategic_response_batch(X_np, w, b, alpha_np, beta)

    # classifier predictions
    scores = X_strat @ w + b
    preds = (scores >= 0).astype(int)

    # oracle probabilities
    etas = eta_model_oracle.predict_proba(X_strat)[:, 1]
    errors = preds * (1 - etas) + (1 - preds) * etas

    return np.mean(errors)


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



def compute_manipulated_stats_baseline(model, X_test, y_test, eta_model, alpha_np, beta):
    
    if isinstance(model, tuple):   # ← ADD THIS
        w, b = model
    else:
        w = model.coef_.flatten()
        b = model.intercept_[0]

    # strategic features
    X_strat, manipulated_indices = get_strategic_response_batch(X_test, w, b, alpha_np, beta)
    
    num_manipulated = len(manipulated_indices)

    probs = eta_model.predict_proba(X_strat)[:,1]
    y_tilde = np.random.binomial(1, probs)

    improved_mask = (y_test == 0) & (y_tilde == 1)

    num_improved = np.sum(improved_mask[manipulated_indices])

    improvement_rate = (num_improved / num_manipulated if num_manipulated > 0 else 0)

    return w, b, num_manipulated, num_improved, improvement_rate


def compute_stats_from_weights(w, b, X_test, y_test, eta_model, alpha_np, beta):

    w = np.array(w)

    # ROI direction
    roi = w / (alpha_np + 1e-12)
    k_star = np.argmax(roi)

    scores = X_test @ w + b
    neg_indices = np.where(scores < 0)[0]

    manipulated_indices = []

    if len(neg_indices) > 0:

        gap = -scores[neg_indices]
        w_k = w[k_star]

        if w_k > 1e-8:

            delta_k = gap / w_k
            cost = alpha_np[k_star] * delta_k

            manipulate_mask = cost <= beta
            manipulated_indices = neg_indices[manipulate_mask]

    num_manipulated = len(manipulated_indices)

    # strategic response
    X_strat, manipulated_indices = get_strategic_response_batch(X_test, w, b, alpha_np, beta)

    # # generate y_tilde
    probs = eta_model.predict_proba(X_strat)[:,1]
    y_tilde = np.random.binomial(1, probs)

    # improved users (0 -> 1)
    improved_mask = (y_test == 0) & (y_tilde == 1)

    num_improved = np.sum(improved_mask[manipulated_indices])
    
    
    
    improvement_rate = (num_improved / num_manipulated if num_manipulated > 0 else 0)

    return num_manipulated, num_improved, improvement_rate


def log_experiment(model_name, w, b, alpha_np, beta,
                   num_manip, num_imp, rate, filename):

    row = {
        "model": model_name,
        "beta": beta,
        "bias": b,
        "alpha": alpha_np.tolist(),
        "weights": w.tolist(),
        "num_manipulated": num_manip,
        "num_improved": num_imp,
        "improvement_rate": rate
    }

    df = pd.DataFrame([row])

    df.to_csv(
        filename,
        mode="a",
        header=not os.path.exists(filename),
        index=False
    )
    
    print(f"Logged experiment for {model_name} to {filename}")
    

def log_experiment_from_weights(model_name,w,b,X_test,y_test,eta_model,alpha_np,beta,filename):

    num_manip, num_imp, rate = compute_stats_from_weights(w,b,X_test,y_test,eta_model,alpha_np,beta)

    row = {
        "model": model_name,
        "beta": beta,
        "bias": b,
        "alpha": alpha_np.tolist(),
        "weights": list(w),
        "num_manipulated": num_manip,
        "num_improved": num_imp,
        "improvement_rate": rate
    }

    df = pd.DataFrame([row])

    df.to_csv(filename,mode="a",header=not os.path.exists(filename),index=False)