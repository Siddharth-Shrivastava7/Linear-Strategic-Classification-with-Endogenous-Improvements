import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_dataloaders(X_scaled, Y, batch_size=64):
    X = X_scaled.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)

    X_train_np, X_val_np, Y_train_np, Y_val_np = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    X_train_t = torch.from_numpy(X_train_np)
    Y_train_t = torch.from_numpy(Y_train_np)
    X_val_t = torch.from_numpy(X_val_np)
    Y_val_t = torch.from_numpy(Y_val_np)

    n_pos = torch.sum(Y_train_t == 1).item()
    n_neg = torch.sum(Y_train_t == 0).item()
    
    if n_pos == 0 or n_neg == 0: 
        weights = torch.ones_like(Y_train_t).view(-1)
    else:
        w_pos = 1.0 / n_pos
        w_neg = 1.0 / n_neg
        weights = torch.where(Y_train_t.view(-1) == 1, w_pos, w_neg)

    sampler = WeightedRandomSampler(weights, num_samples=len(Y_train_t), replacement=True)

    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, sampler=sampler)
    val_dl = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=batch_size, shuffle=False)
    
    return train_dl, val_dl, X_train_t


def load_and_process_adult(datapath):
    
    df = pd.read_csv(datapath)
    df = df.replace('?', np.nan).dropna().drop_duplicates(keep="first")
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    feature_names = ['educational-num', "capital-gain", "capital-loss", "hours-per-week", "age"]
    X = df[feature_names]
    y = df['income']
    return X, y, feature_names



def load_and_process_heloc(datapath):
    # Load dataset
    df = pd.read_csv(datapath)
    
    # Target mapping
    df['Target'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
    
    # Defined improvable features
    
    features_names = [
        "ExternalRiskEstimate",
        "MSinceOldestTradeOpen",
        "AverageMInFile",
        "NumSatisfactoryTrades",
        "PercentTradesNeverDelq",
        "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M",
        "MSinceMostRecentInqexcl7days",
    ]
    
    # Handle HELOC special values (-7, -8, -9) 
    X_raw = df[features_names].replace([-7, -8, -9], np.nan)
    y = df['Target'].values
    
    # Impute missing values with median to preserve clean matrices for PyTorch
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)
    X = pd.DataFrame(X, columns=features_names)
    
    return X, y, features_names


def load_and_process_lawschool(datapath):
    df = pd.read_csv(datapath)
    target = 'pass_bar'
    non_manipulable = ['fulltime', 'male', 'race']
    feature_names = [c for c in df.columns if c not in [target] + non_manipulable]
    X = df[feature_names]
    y = df[target]
    return X, y, feature_names