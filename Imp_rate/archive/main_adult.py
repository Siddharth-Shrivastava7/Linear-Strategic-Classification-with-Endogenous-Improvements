import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_preprocessing import *
from classifiers import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
set_seed(42)

X, y = load_and_process_adult('Imp_rate/adult.csv')

X_train, X_test, y_train, y_test = data_split(X, y, 0.3)

d = X_train.shape[1]  
eta_model = estimate_eta(X_train, y_train)

alpha = np.array([5,2,2,1,100])
beta = 1.0

weights_table = [
    ("Optimal Linear (f*)", [0.73,2.07,0.28,0.51,0.46], 0.24),
    ("Strategic (f*_s)", [0.73,2.08,0.21,0.51,0.47], 1.35),
    ("Strat-Imp-Aware (f*_imp)", [0.79,2.10,0.23,0.50,0.56], 1.19)
]

for name,w,b in weights_table:
    
    
    err = compute_err_imp_from_weights(
    w,
    b,
    X_test,
    eta_model,
    alpha,
    beta
    )
    
    print(name, "err_imp:", err)
    
    log_experiment_from_weights(
        model_name=name,
        w=w,
        b=b,
        X_test=X_test,
        y_test=y_test,
        eta_model=eta_model,
        alpha_np=alpha,
        beta=beta,
        filename="Imp_rate/adult_Imp_rate.csv"
    )