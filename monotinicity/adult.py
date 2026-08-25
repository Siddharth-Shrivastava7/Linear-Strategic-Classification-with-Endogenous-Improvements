import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import math

from data_preprocessing import *

def load_and_process_adult(datapath):
    
    df = pd.read_csv(datapath)
    df = df.replace('?', np.nan).dropna().drop_duplicates(keep="first")
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    feature_names = ['educational-num', "capital-gain", "capital-loss", "hours-per-week", "age"]
    X = df[feature_names]
    y = df['income']
    return X, y, feature_names

# 1. Load data
X, y, feature_names = load_and_process_adult("Ass_valid/dataset_plots/adult.csv")

model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'))
])

model_pipeline.fit(X, y)


# 2. Plotting Function for ALL Features
def plot_monotonicity_all(model, X_data, feature_names, save_path):
    n = len(feature_names)
    cols = 5
    rows = math.ceil(n / cols)
    
    # Create a grid large enough for all features
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten()

    # Use a clean subset of 50 samples for ICE plots
    clean_data = X_data.dropna()
    X_sample = clean_data.sample(n=min(50, len(clean_data)), random_state=42).copy()

    for i, feature in enumerate(feature_names):
        ax = axes[i]
        
        valid_data = X_data[feature].dropna()
        if len(valid_data) == 0:
            ax.set_title(f"No valid data", fontsize=10)
            continue
            
        min_v, max_v = valid_data.min(), valid_data.max()
        
        # Determine grid (prevent too many points for discrete variables)
        if valid_data.nunique() < 20:
            grid = np.sort(valid_data.unique())
        else:
            grid = np.linspace(min_v, max_v, 20)

        # Generate Predictions (ICE)
        ice_preds = []
        for val in grid:
            X_temp = X_sample.copy()
            X_temp[feature] = val
            ice_preds.append(model.predict_proba(X_temp)[:, 1])

        ice_preds = np.array(ice_preds).T
        avg_pred = np.mean(ice_preds, axis=0)

        # Plot Individual Conditional Expectation (blue lines)
        ax.plot(grid, ice_preds.T, color='blue', alpha=0.05)
        # Plot Partial Dependence (red line)
        ax.plot(grid, avg_pred, color='red', linewidth=3) 

        ax.set_title(f"{feature}", fontsize=11, fontweight='bold')
        ax.set_xlabel(feature, fontsize=9)
        if i % cols == 0:
            ax.set_ylabel("P(Good)", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide extra empty subplots (since 23 features don't perfectly fill a 25-slot grid)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Plot saved to: {save_path}")
    plt.show()

# 3. Run Validation
plot_monotonicity_all(model_pipeline, X, feature_names, "Ass_valid/dataset_plots/adult_features_monotonicity.pdf")