import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_preprocessing import *
from classifiers import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
set_seed(42)

X, y = load_and_process_lawschool('Imp_rate/law_school_clean.csv')

X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y, 0.2, 0.1)
train_dl = build_dataloaders(X_train, y_train)

d = X_train.shape[1]  
eta_model = estimate_eta(X_train, y_train)


beta = 1.0
alpha = torch.tensor([1, 0.5, 1.0, 1.0, 10.0, 10.0], dtype=torch.float32, device=device)
# alpha = torch.tensor([1, 1, 1.5, 2.0, 10, 10, 0.8, 0.8], dtype=torch.float32, device=device)


# def get_best_model(trainer, model, X_val, eta_model, alpha_np, beta, n_epochs):
#     best_state = None
#     best_err = float('inf')

#     for _ in tqdm(range(n_epochs)):
#         for xb, yb in trainer.train_dl:
#             trainer.opt.zero_grad()
            
#             if isinstance(trainer, ImprovementAwareTrainer):
#                 loss = trainer.train_step(xb, yb)
#             elif isinstance(trainer, StrategicTrainer):
#                 xb_t = xb.to(trainer.device)
#                 yb_t = yb.to(trainer.device).view(-1).long()
#                 y_tilde = (2 * yb_t - 1).float()
#                 loss = trainer.strat_loss(xb_t, y_tilde)
#             else:
#                 xb_t = xb.to(trainer.device)
#                 yb_t = yb.to(trainer.device).view(-1).long()
#                 y_tilde = (2 * yb_t - 1).float()
#                 loss = trainer.loss(xb_t, y_tilde)

#             loss.backward()
#             trainer.opt.step()

#         # evaluate after each epoch
#         err = compute_err_imp(model, X_val, eta_model, alpha_np, beta)

#         if err < best_err:
#             best_err = err
#             best_state = model.state_dict()

#     # load best weights
#     model.load_state_dict(best_state)
#     return model


def get_best_model(trainer, model, X_val, eta_model, alpha_np, beta, n_epochs):
    best_state = None
    best_loss = float('inf')

    for _ in tqdm(range(n_epochs)):
        for xb, yb in trainer.train_dl:
            trainer.opt.zero_grad()
            
            if isinstance(trainer, ImprovementAwareTrainer):
                loss = trainer.train_step(xb, yb)
            elif isinstance(trainer, StrategicTrainer):
                xb_t = xb.to(trainer.device)
                yb_t = yb.to(trainer.device).view(-1).long()
                y_tilde = (2 * yb_t - 1).float()
                loss = trainer.strat_loss(xb_t, y_tilde)
            else:
                xb_t = xb.to(trainer.device)
                yb_t = yb.to(trainer.device).view(-1).long()
                y_tilde = (2 * yb_t - 1).float()
                loss = trainer.loss(xb_t, y_tilde)

            loss.backward()
            trainer.opt.step()

        # =========================
        # VALIDATION LOSS (changed)
        # =========================
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(trainer.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(trainer.device).view(-1).long()
            y_tilde_val = (2 * y_val_t - 1).float()

            if isinstance(trainer, ImprovementAwareTrainer):
                val_loss = trainer.train_step(X_val_t, y_val_t)
            elif isinstance(trainer, StrategicTrainer):
                val_loss = trainer.strat_loss(X_val_t, y_tilde_val)
            else:
                val_loss = trainer.loss(X_val_t, y_tilde_val)

            val_loss = val_loss.item()

        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model

# Train Optimal linear classifier (Independent of alpha and beta)

model_linear = LinearClassifier(d, 1).to(device)
# opt_linear = torch.optim.Adam(model_linear.parameters(), lr=0.01)
opt_linear = torch.optim.SGD(model_linear.parameters(), lr=0.1, momentum=0.9)

trainer_linear = optimal_linear_classifier(model_linear, train_dl, opt_linear, device)
model_linear = get_best_model(trainer_linear, model_linear, X_val, eta_model, alpha.cpu().numpy(), beta, n_epochs=100)

w_star_linear = model_linear.out.weight.squeeze().detach()
b_star_linear = model_linear.out.bias.item()


# Train Strategic classifier (f*_s)

model_strat = LinearClassifier(d, 1).to(device)
# model_strat.load_state_dict(model_linear.state_dict()) # Warm start
# opt_s = torch.optim.Adam(model_strat.parameters(), lr=0.01)
opt_s = torch.optim.SGD(model_strat.parameters(), lr=0.1, momentum=0.9)

trainer_s = StrategicTrainer(model_strat, train_dl, opt_s, alpha, beta, device)
model_strat = get_best_model(trainer_s, model_strat, X_val, eta_model, alpha.cpu().numpy(), beta, n_epochs=100)

w_star_strat = model_strat.out.weight.squeeze().detach()
b_star_strat = model_strat.out.bias.item()

# Train Improvement-Aware classiifier (f*_imp)

model_imp_aware = LinearClassifier(d, 1).to(device)
# model_imp_aware.load_state_dict(model_linear.state_dict()) # Warm start
# opt_imp = torch.optim.Adam(model_imp_aware.parameters(), lr=0.01)
opt_imp = torch.optim.SGD(model_imp_aware.parameters(), lr=0.1, momentum=0.9)


trainer_imp = ImprovementAwareTrainer(model_imp_aware, train_dl, opt_imp, eta_model, alpha, beta, device)
model_imp_aware = get_best_model(trainer_imp, model_imp_aware, X_val, eta_model, alpha.cpu().numpy(), beta, n_epochs=100)

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

    log_experiment(name ,w,b,alpha_np,beta,n_manip,n_imp,rate, "Imp_rate/law_school_Imp_rate.csv")