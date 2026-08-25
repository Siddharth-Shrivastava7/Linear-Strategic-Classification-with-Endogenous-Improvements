import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_preprocessing import *
from classifiers import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
set_seed(42)

X, y = load_and_process_adult('adult.csv')

X_train, X_test, y_train, y_test = data_split(X, y, 0.3)
train_dl = build_dataloaders(X_train, y_train)

d = X_train.shape[1]  
eta_model = estimate_eta(X_train, y_train)

# # betas_configs = [1, 0.2, 1.0, 20.0]

# # alpha_configs = {
# #     "Paper (domain)":       torch.tensor([5.0, 2.0, 2.0, 1.0, 100.0], dtype=torch.float32, device=device),
# #     "Uniform (1.0)":        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=device),
# #     "Uniform (2.0)":        torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32, device=device),
# #     "Uniform (5.0)":        torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float32, device=device),
# #     "Uniform (20.0)":        torch.tensor([20.0, 20.0, 20.0, 20.0, 20.0], dtype=torch.float32, device=device),
# #     "High education":       torch.tensor([50.0, 2.0, 2.0, 1.0, 100.0], dtype=torch.float32, device=device),
# # }


beta = 1.0
# alpha = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float32, device=device)
alpha = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float32, device=device)


# Train Optimal linear classifier (Independent of alpha and beta)

model_linear = LinearClassifier(d, 1).to(device)
# opt_linear = torch.optim.Adam(model_linear.parameters(), lr=0.01)
opt_linear = torch.optim.SGD(model_linear.parameters(), lr=0.1, momentum=0.9)

trainer_linear = optimal_linear_classifier(model_linear, train_dl, opt_linear, device)
trainer_linear.train(n_epochs=50) 

w_star_linear = model_linear.out.weight.squeeze().detach()
b_star_linear = model_linear.out.bias.item()

# Train Strategic classifier (f*_s)

model_strat = LinearClassifier(d, 1).to(device)
model_strat.load_state_dict(model_linear.state_dict()) # Warm start
# opt_s = torch.optim.Adam(model_strat.parameters(), lr=0.01)
opt_s = torch.optim.SGD(model_strat.parameters(), lr=0.1, momentum=0.9)

trainer_s = StrategicTrainer(model_strat, train_dl, opt_s, alpha, beta, device)
trainer_s.train(n_epochs=50)

w_star_strat = model_strat.out.weight.squeeze().detach()
b_star_strat = model_strat.out.bias.item()

# Train Improvement-Aware classiifier (f*_imp)

model_imp_aware = LinearClassifier(d, 1).to(device)
model_imp_aware.load_state_dict(model_linear.state_dict()) # Warm start
# opt_imp = torch.optim.Adam(model_imp_aware.parameters(), lr=0.01)
opt_imp = torch.optim.SGD(model_imp_aware.parameters(), lr=0.1, momentum=0.9)


trainer_imp = ImprovementAwareTrainer(model_imp_aware, train_dl, opt_imp, eta_model, alpha, beta, device)
trainer_imp.train(n_epochs=80) 

w_star_imp_aware = model_imp_aware.out.weight.squeeze().detach()
b_star_imp_aware = model_imp_aware.out.bias.item()


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

    log_experiment(name ,w,b,alpha_np,beta,n_manip,n_imp,rate, "adult_Imp_rate.csv")