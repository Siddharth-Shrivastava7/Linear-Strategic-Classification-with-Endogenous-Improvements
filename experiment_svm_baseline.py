"""Standard linear SVM vs. our saved stage 4 model in our environment.

    python experiment_svm_baseline.py --sweep configs/sweeps/svm_baseline.yaml

Both classifiers face the same agents: the cheapest feasible single-feature
move with cost alpha_j * delta <= beta (Algorithm 2's rule, see
pipeline/best_response.py); movers with y = 0 get y' ~ Bernoulli(pi_Imp).
Reports manipulating / improved counts (one draw with sample_seed, plus the
exact expectation), improvement rate and improvement error on the test split.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.svm import LinearSVC

from experiment_pli_comparison import context, ours_margin
from pipeline.best_response import cheapest_feasible_response
from pipeline.config import PROJECT_ROOT
from pipeline.data import file_sha256
from pipeline.evaluation import sample_post_labels
from stage4_train_strat_imp import DTYPE


def evaluate(ctx, margin, split, seed):
    u, y = ctx["data"][split]
    e = ctx["env"]
    y = np.asarray(y).astype(int)
    u_f, moved, feature = cheapest_feasible_response(margin, u, e["alpha"], e["beta"], e["upper"])
    with torch.no_grad():
        eta = ctx["eta"]
        pi = torch.clamp((eta(u_f) - eta(u)) / torch.clamp(1 - eta(u), min=e["delta_g"]), 0, 1).numpy()
        pred_before = (margin(u) >= 0).numpy()
    moved = moved.numpy()
    cand = moved & (y == 0)
    y_post = sample_post_labels(y, pi, cand, np.random.default_rng(seed))
    improved = cand & (y_post == 1)
    y_hat = moved | pred_before
    feats = feature.numpy()[moved]
    return {
        "agents": len(y),
        "predicted_positive_before": int(pred_before.sum()),
        "manipulating": int(moved.sum()),
        "manipulating_y0": int(cand.sum()),
        "manipulating_y1": int((moved & (y == 1)).sum()),
        "improved": int(improved.sum()),
        "improvement_rate": improved.sum() / cand.sum() if cand.any() else float("nan"),
        "expected_improved": float(pi[cand].sum()),
        "expected_improvement_rate": float(pi[cand].mean()) if cand.any() else float("nan"),
        "improvement_error": float((y_hat != y_post).mean()),
        "expected_improvement_error": float(((y_hat[~cand] != y[~cand]).sum() + (1 - pi[cand]).sum()) / len(y)),
        "error_no_response": float((pred_before != y).mean()),
        "moves_along": {ctx["feats"][int(k)]: int(v) for k, v in zip(*np.unique(feats, return_counts=True))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    args = ap.parse_args()
    with open(args.sweep) as f:
        sw = yaml.safe_load(f)
    ctx = context(str(PROJECT_ROOT / sw["base_config"]))
    paths, feats = ctx["paths"], ctx["feats"]
    out_dir = paths["root"] / "sweeps" / Path(args.sweep).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    u_tr, y_tr = ctx["data"][sw["train_split"]]
    svm = LinearSVC(**sw["svm"]).fit(u_tr.numpy(), y_tr)
    if svm.n_iter_ >= sw["svm"]["max_iter"]:
        print("warning: LinearSVC hit max_iter")
    joblib.dump(svm, out_dir / "linear_svm.joblib")
    w_svm = torch.tensor(svm.coef_.ravel(), dtype=DTYPE)
    c_svm = float(svm.intercept_[0])
    svm_margin = lambda u: u @ w_svm + c_svm  # positive iff w.u + c >= 0

    ck = torch.load(paths["strat_model"])
    methods = {"Linear SVM (hinge, C=1)": svm_margin, "Ours (stage 4 model)": ours_margin(ck["w"], ck["b"])}
    res = {name: evaluate(ctx, m, sw["test_split"], sw["sample_seed"]) for name, m in methods.items()}
    tbl = pd.DataFrame(res).T
    tbl.drop(columns=["moves_along"]).to_csv(out_dir / "env_ours_test.csv")

    meta = {
        "model_file": "linear_svm.joblib",
        "model_sha256": file_sha256(out_dir / "linear_svm.joblib"),
        "svm": sw["svm"], "train_split": sw["train_split"], "train_rows": int(len(y_tr)),
        "features": feats, "input_space": "standardized with the LR scaler (fit on lr_train)",
        "decision_rule": "positive iff w . u + c >= 0",
        "w_standardized": dict(zip(feats, svm.coef_.ravel().tolist())), "c": c_svm,
        "train_accuracy": float(svm.score(u_tr.numpy(), y_tr)),
        "test_split": sw["test_split"], "split_indices_sha256": file_sha256(paths["split_indices"]),
        "environment_ours": {"beta": ctx["env"]["beta"], "alpha": dict(zip(feats, ctx["env"]["alpha"].tolist())),
                             "delta_g": ctx["env"]["delta_g"], "clip": "feasible (max of lr_train)"},
        "sample_seed": sw["sample_seed"],
        "results": {k: {c: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                        for c, v in r.items()} for k, r in res.items()},
        "versions": {"sklearn": sklearn.__version__, "torch": torch.__version__},
    }
    with open(out_dir / "linear_svm_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    u_te, y_te = ctx["data"][sw["test_split"]]
    print(f"Linear SVM trained on {sw['train_split']} ({len(y_tr):,} rows), {svm.n_iter_} iterations; "
          f"w = {dict(zip(feats, np.round(svm.coef_.ravel(), 4)))}, c = {c_svm:.4f}")
    print(f"accuracy without agent response: train {meta['train_accuracy']:.4f}, "
          f"{sw['test_split']} {svm.score(u_te.numpy(), y_te):.4f}\n")
    for name, r in res.items():
        print(f"== {name}  on {sw['test_split']} ({r['agents']:,} agents)")
        print(f"   predicted positive before response {r['predicted_positive_before']:>6}")
        print(f"   manipulating agents                {r['manipulating']:>6}   (moves along {r['moves_along']})")
        print(f"     original y = 0 (denominator)     {r['manipulating_y0']:>6}")
        print(f"     original y = 1                   {r['manipulating_y1']:>6}")
        print(f"   improved agents (seed {sw['sample_seed']})         {r['improved']:>6}")
        print(f"   improvement rate                   {r['improvement_rate']:.4f}   (expected {r['expected_improvement_rate']:.4f}, "
              f"E[improved] {r['expected_improved']:.1f})")
        print(f"   improvement error                  {r['improvement_error']:.4f}   (expected {r['expected_improvement_error']:.4f}; "
              f"without response {r['error_no_response']:.4f})\n")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
