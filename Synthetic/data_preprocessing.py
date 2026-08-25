import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, SequentialSampler, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


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


# def build_dataloaders(X, Y, batch_size=64):
#     X = X.astype(np.float32)
#     if Y.ndim == 1:
#         Y = Y.reshape(-1, 1)
#     Y = Y.astype(np.float32)

#     X_train_t = torch.from_numpy(X)
#     Y_train_t = torch.from_numpy(Y)
    
#     n_pos = torch.sum(Y_train_t == 1).item()
#     n_neg = torch.sum(Y_train_t == 0).item()
    
#     if n_pos == 0 or n_neg == 0: 
#         weights = torch.ones_like(Y_train_t).view(-1)
#     else:
#         w_pos = 1.0 / n_pos
#         w_neg = 1.0 / n_neg
#         weights = torch.where(Y_train_t.view(-1) == 1, w_pos, w_neg)

#     sampler = WeightedRandomSampler(weights, num_samples=len(Y_train_t), replacement=True)
#     train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, sampler=sampler)
    
#     return train_dl

def build_dataloaders(X, Y, batch_size=64):
    
    X = X.astype(np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    Y = Y.astype(np.float32)
    
    X_train_t = torch.from_numpy(X)
    Y_train_t = torch.from_numpy(Y)
    
    train_dl = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=False)
    
    return train_dl

def load_and_process_synthetic_data(datapath):
    df = pd.read_csv(datapath)

    # Features = everything except label
    X = df.drop(columns=["label", "eta"]).values
    y = df["label"].values

    return X, y

def load_and_process_adult(datapath):
    """
    Loads Adult data locally, or fetches from OpenML.
    Restricts to 5 continuous features to match the 5D alpha vectors.
    """
    df = pd.read_csv(datapath)
    df = df.replace('?', np.nan).dropna().drop_duplicates(keep="first")
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    # Use the 5 continuous features to align with the 5D alpha configurations
    features = ['educational-num', "capital-gain", "capital-loss", "hours-per-week", "age"]
    
    X = df[features].values
    y = df['income'].values
    
    return X, y

def load_and_process_acs_income(datapath="retiring_adult_data.csv"):
    df = pd.read_csv(datapath)
    # folktables uses -1 as the NaN sentinel after postprocess; drop if any
    df = df.replace(-1, np.nan).dropna().drop_duplicates(keep="first")

    # Only the ordinal/continuous features — monotonicity is well-defined for these.
    # AGEP : age (continuous)
    # SCHL : educational attainment (ordinal, 24 levels)
    # WKHP : usual hours worked per week (continuous)
    
    feature_names = ['AGEP', 'SCHL', 'WKHP']
    X = df[feature_names].values
    y = df['label'].values
    return X, y

def load_and_process_heloc(datapath='heloc.csv'):
    # Load dataset
    df = pd.read_csv(datapath)
    
    # Target mapping
    df['Target'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
    
    # Defined improvable features
    
    features = [
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
    X_raw = df[features].replace([-7, -8, -9], np.nan)
    y = df['Target'].values
    
    # Impute missing values with median to preserve clean matrices for PyTorch
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X_raw)
    
    return X, y

def load_and_process_lawschool(datapath):
    
    df = pd.read_csv(datapath)
    target = 'pass_bar'
    
    # non_manipulable = ['fulltime', 'male', 'race']
    non_manipulable = ['fulltime', 'fam_inc', 'tier', 'male', 'race']
    
    feature_cols = [c for c in df.columns if c not in [target] + non_manipulable]
    
    X = df[feature_cols].values.astype(float)
    y = df[target].values.astype(int)
    
    return X, y