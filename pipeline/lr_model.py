"""Logistic regression: preprocessing by feature type, training, save/load."""

import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_lr_pipeline(features, feature_types, cfg, sample):
    """Continuous -> standardize; ordinal -> ordered integers, standardized;
    nominal -> one-hot. `sample` is used only to check dtypes."""
    orders = cfg["feature_types"].get("ordinal_orders") or {}
    continuous = [f for f in features if feature_types[f] == "continuous"]
    ordinal = [f for f in features if feature_types[f] == "ordinal"]
    nominal = [f for f in features if feature_types[f] == "nominal"]

    numeric_ordinal = [f for f in ordinal if f not in orders]
    for f in numeric_ordinal:
        if not pd.api.types.is_numeric_dtype(sample[f]):
            raise ValueError(f"ordinal feature {f!r} is non-numeric: add its order under feature_types.ordinal_orders")
    coded_ordinal = [f for f in ordinal if f in orders]

    transformers = []
    if continuous or numeric_ordinal:
        transformers.append(("numeric", StandardScaler(), continuous + numeric_ordinal))
    if coded_ordinal:
        encoder = OrdinalEncoder(categories=[orders[f] for f in coded_ordinal])
        transformers.append(("ordinal_coded", Pipeline([("encode", encoder), ("scale", StandardScaler())]), coded_ordinal))
    if nominal:
        transformers.append(("nominal", OneHotEncoder(handle_unknown="ignore"), nominal))

    pre = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    clf = LogisticRegression(**cfg["logistic_regression"]["params"])
    return Pipeline([("preprocess", pre), ("clf", clf)])


def positive_proba(model, X):
    """P(y = 1 | x) from a fitted pipeline."""
    pos = list(model.classes_).index(1)
    return model.predict_proba(X)[:, pos]


def evaluate(model, X, y):
    p = positive_proba(model, X)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "positive_rate_true": float(np.mean(y)),
        "positive_rate_pred": float(np.mean(p >= 0.5)),
        "mean_predicted_proba": float(np.mean(p)),
    }


def raw_space_coefficients(model):
    """Coefficients on the original (unscaled) feature scale, available when
    every feature went through the StandardScaler:
        logit = intercept_raw + sum_j coef_raw[j] * x_j
    Handy for re-implementing the LR (e.g. in PyTorch) in later stages."""
    pre = model.named_steps["preprocess"]
    if [name for name, *_ in pre.transformers_ if name != "remainder"] != ["numeric"]:
        return None
    scaler = pre.named_transformers_["numeric"]
    clf = model.named_steps["clf"]
    w = clf.coef_.ravel() / scaler.scale_
    b = clf.intercept_[0] - np.sum(w * scaler.mean_)
    names = list(pre.get_feature_names_out())
    return {"intercept": float(b), "coef": dict(zip(names, map(float, w)))}


def save_lr(model, meta, paths):
    joblib.dump(model, paths["lr_model"])
    with open(paths["lr_meta"], "w") as f:
        json.dump(meta, f, indent=2)


def build_meta(model, features, feature_types, metrics, cfg):
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    return {
        "features": features,
        "feature_types": {f: feature_types[f] for f in features},
        "train_split": cfg["logistic_regression"]["train_split"],
        "params": cfg["logistic_regression"]["params"],
        "sklearn_version": sklearn.__version__,
        "standardized_space": {
            "intercept": float(clf.intercept_[0]),
            "coef": dict(zip(pre.get_feature_names_out(), map(float, clf.coef_.ravel()))),
        },
        "raw_space": raw_space_coefficients(model),
        "metrics": metrics,
    }


def load_lr(paths):
    return joblib.load(paths["lr_model"])
