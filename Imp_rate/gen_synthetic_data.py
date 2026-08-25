import numpy as np
import pandas as pd
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def generate_dataset(n,d,seed,save_path,save_dir="Imp_rate"):
    
    """
    Generates dataset with:
        X ~ N(0, I_d)
        eta(x) = sigmoid(w^T x)
        y ~ Bernoulli(eta(x))

    Args:
        n: number of samples
        d: number of features
        seed: random seed
        save_path: CSV output path
        save_dir: directory to save the true weights
    """

    np.random.seed(seed)
    
    # 1. feature Vector
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, size=n)
    
    # 2. weight vector
    w = np.random.randn(d)
    
    # Save w
    np.save(os.path.join(save_dir, "true_w.npy"), w)

    # 3. Compute probabilities
    scores = X @ w
    eta = sigmoid(scores)

    # 4. label
    y = np.random.binomial(1, eta)

    # 5. Build DataFrame
    
    feature_cols = [f"X{i+1}" for i in range(d)]
    df = pd.DataFrame(X, columns=feature_cols)

    df["label"] = y
    
    df["eta"] = eta

    # 6. Save to CSV
    df.to_csv(save_path, index=False)

    print(f"Saved dataset to {save_path}")



if __name__ == "__main__":
    generate_dataset(n=20000,d=8,seed=42,save_path="Imp_rate/synthetic_dataset.csv")