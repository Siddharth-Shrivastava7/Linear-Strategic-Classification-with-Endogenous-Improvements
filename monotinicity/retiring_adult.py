import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import math


def load_and_process_acs_income(datapath):
    df = pd.read_csv(datapath)
    # folktables uses -1 as the NaN sentinel after postprocess; drop if any
    df = df.replace(-1, np.nan).dropna().drop_duplicates(keep="first")

    # Only the ordinal/continuous features — monotonicity is well-defined for these.
    # AGEP : age (continuous)
    # SCHL : educational attainment (ordinal, 24 levels)
    # WKHP : usual hours worked per week (continuous)
    #
    # The remaining 7 features (COW, MAR, OCCP, POBP, RELP, SEX, RAC1P) are
    # unordered categoricals encoded as integer codes; ICE/PDP across them
    # would just reflect the arbitrary code ordering, not a true monotone trend.
    feature_names = ['AGEP', 'SCHL', 'WKHP']
    X = df[feature_names]
    y = df['label']
    return X, y, feature_names


# 1. Load data
X, y, feature_names = load_and_process_acs_income(
    "retiring_adult_data.csv"
)

model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'))
])

model_pipeline.fit(X, y)


# 2. Plotting Function for ALL Features
def plot_monotonicity_all(model, X_data, feature_names, save_path):
    n = len(feature_names)
    cols = min(5, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    # Make `axes` always iterable as a flat list, regardless of n
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    # Use a clean subset of 50 samples for ICE plots
    clean_data = X_data.dropna()
    X_sample = clean_data.sample(n=min(50, len(clean_data)), random_state=42).copy()

    for i, feature in enumerate(feature_names):
        ax = axes[i]

        valid_data = X_data[feature].dropna()
        if len(valid_data) == 0:
            ax.set_title("No valid data", fontsize=10)
            continue

        min_v, max_v = valid_data.min(), valid_data.max()

        # Discrete: use unique values as grid; continuous: 20-point linspace
        if valid_data.nunique() < 30:
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

        # Plot ICE (blue) and PDP (red)
        ax.plot(grid, ice_preds.T, color='blue', alpha=0.05)
        ax.plot(grid, avg_pred, color='red', linewidth=3)

        ax.set_title(f"{feature}", fontsize=11, fontweight='bold')
        ax.set_xlabel(feature, fontsize=9)
        if i % cols == 0:
            ax.set_ylabel("P(income > 50k)", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for j in range(len(feature_names), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches='tight', pad_inches=0.05)
    print(f"Plot saved to: {save_path}")
    plt.show()


# 3. Run Validation
plot_monotonicity_all(
    model_pipeline,
    X,
    feature_names,
    "retiring_adult_features_monotonicity.pdf",
)
