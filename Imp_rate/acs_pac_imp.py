import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from pac_baseline_updated import * 
from tqdm import tqdm

# Assuming the functions train_fstar, train_h, and improve_agent 
# from your previous codebase are already defined here.

# ═══════════════════════════════════════════════════════════
# 1. ADULT DATASET LOADER & FEATURE MASKING
# ═══════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_preprocess_acsincome(csv_path="retiring_adult_data.csv", random_state=42):
    """
    Loads adult.csv, preprocesses it, and identifies the indices of the 
    improvable features specified.
    """
    df = pd.read_csv(csv_path)
    # folktables uses -1 as the NaN sentinel after postprocess; drop if any
    df = df.replace(-1, np.nan).dropna().drop_duplicates(keep="first")
    
    # 2. Extract Target
    X_df = df.drop(columns=['label'])
    y = df['label'].values

    # Your specific improvable features
    improvable_cols = ['AGEP', 'SCHL', 'WKHP']
    
    # 3. Train-Test Split (Do this BEFORE scaling to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=random_state
    )

    # 4. Separate categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    # 5. Build and apply ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)
    ])

    # Fit ONLY on train, then transform both train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. Extract the indices of the improvable features
    out_features = preprocessor.get_feature_names_out()
    
    improvable_indices = []
    for i, feat in enumerate(out_features):
        # Extract the original feature name from the ColumnTransformer output format
        if 'cat__' in feat:
            orig_feat_name = feat.split('__')[1].rsplit('_', 1)[0]
        else:
            orig_feat_name = feat.split('__')[1]
            
        if orig_feat_name in improvable_cols:
            improvable_indices.append(i)
    
    return X_train_processed, X_test_processed, y_train, y_test, improvable_indices

# ═══════════════════════════════════════════════════════════
# 2. IMPROVEMENT RATE CALCULATOR (PGD / PAPER LOGIC)
# ═══════════════════════════════════════════════════════════

def compute_improvement_rate_pgd(
    X_test, 
    y_test, 
    h_model, 
    fstar, 
    budget_r, 
    improvable_features, 
    threshold=0.5
):
    """
    Calculates the improvement rate using the paper's PGD mechanism.
    Only agents originally classified as negative (h_orig == 0) try to improve.
    """
    h_model.eval()
    num_manipulated = 0
    num_improved = 0

    for i in tqdm(range(len(X_test))):
        x_orig = X_test[i]
        x_t = torch.FloatTensor(x_orig)
        
        # Original prediction by the decision-maker model
        h_orig = int(h_model(x_t.unsqueeze(0)).item() >= threshold)

        # If the agent is classified as negative, they react (attempt to improve)
        if h_orig == 0 and budget_r > 0:
            x_imp = improve_agent(
                x_orig, 
                h_model, 
                budget_r, 
                fstar=fstar, 
                improvable_features=improvable_features, 
                threshold=threshold,
                adversarial_tiebreak=False # Matches empirical setup
            )
            
            # Check if manipulation was successful in fooling/satisfying h
            h_imp = int(h_model(torch.FloatTensor(x_imp).unsqueeze(0)).item() >= threshold)
            
            if h_imp == 1:
                num_manipulated += 1
                
                # Check if it was a true improvement according to the ground truth
                f_imp = int(fstar.predict(x_imp.reshape(1, -1))[0])
                if f_imp == 1:
                    num_improved += 1

    improvement_rate = (num_improved / num_manipulated) if num_manipulated > 0 else 0.0

    return num_manipulated, num_improved, improvement_rate

# ═══════════════════════════════════════════════════════════
# 3. EXECUTION SCRIPT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Load data
    print("Loading ACS Income dataset...")
    X_train, X_test, y_train, y_test, improvable_features = load_and_preprocess_acsincome("retiring_adult_data.csv")
    
    # 2. Train ground truth f* on the FULL dataset (as corrected previously)
    print("Training f* oracle...")
    X_full = np.vstack((X_train, X_test))
    y_full = np.concatenate((y_train, y_test))
    fstar = train_fstar(X_full, y_full)
    
    y_fstar_train = fstar.predict(X_train)

    # 3. Train h model (Using wBCE from your configs)
    print("Training h model (wBCE wFP=1.0, wFN=1.0)...")
    h_model = train_h(X_train, y_fstar_train, w_fp=1.0, w_fn=1.0)

    # 4. Calculate Improvement Rate
    budget = 1.0 # Set your desired budget 'r'
    print(f"\nCalculating Improvement Rate for budget r={budget}...")
    
    n_manipulated, n_improved, rate = compute_improvement_rate_pgd(
        X_test, 
        y_test, 
        h_model, 
        fstar, 
        budget_r=budget, 
        improvable_features=improvable_features
    )

    print("-" * 40)
    print(f"Total Agents who successfully changed h to positive: {n_manipulated}")
    print(f"Total Agents who truly improved (f* = 1): {n_improved}")
    print(f"Improvement Rate: {rate:.2%}")
    print("-" * 40)