"""PLI's paper-default model vs. our saved stage 4 model, on the unseen test split.

    python experiment_pli_paper.py --sweep configs/sweeps/pli_paper.yaml [--reuse-model]

--reuse-model re-evaluates the saved PLI model instead of retraining it.

Both are evaluated in environment "ours" (main) and environment "PLI"
(robustness), exactly as in experiment_pli_comparison.py.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml

from experiment_pli_comparison import context, env_ours, ours_margin, pli_margin
from pipeline.baselines_pli import MODELS, make_loss, train_pli
from pipeline.best_response import evaluate_env_pli
from pipeline.config import PROJECT_ROOT
from pipeline.data import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--reuse-model", action="store_true", help="evaluate the saved PLI model, do not retrain")
    args = ap.parse_args()
    with open(args.sweep) as f:
        sw = yaml.safe_load(f)
    ctx = context(str(PROJECT_ROOT / sw["base_config"]))
    torch.set_num_threads(4)
    paths, feats = ctx["paths"], ctx["feats"]
    out_dir = paths["root"] / "sweeps" / Path(args.sweep).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    p, test = sw["pli"], sw["test_split"]

    u_tr, y_tr = ctx["data"][sw["train_split"]]
    model_path = out_dir / "pli_nn_paper_default.pt"
    if args.reuse_model:
        model = MODELS[p["model_type"]](len(feats))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded saved PLI model {model_path}")
    else:
        print(f"Training PLI {p['model_type']} ({p['loss']}, threshold {p['threshold']}, batch {p['batch_size']}, "
              f"{p['epochs']} epochs, seed {p['seed']}) on {sw['train_split']} ({len(y_tr):,} rows)...", flush=True)
        t0 = time.time()
        model = train_pli(p["model_type"], p["loss"], u_tr.float(), torch.tensor(y_tr), p["seed"],
                          p["epochs"], p["batch_size"], p["lr"])
        print(f"  done in {time.time() - t0:.0f}s")
        torch.save(model.state_dict(), model_path)

    ck = torch.load(paths["strat_model"])
    with open(paths["strat_meta"]) as f:
        best_epoch = json.load(f)["best_epoch"]
    w, b = ck["w"], ck["b"]
    tau = p["threshold"]
    methods = {
        "Ours (stage 4 model, seed 42)": {"margin": ours_margin(w, b), "h": lambda x: torch.sigmoid(x @ w - b)[:, None],
                                          "tau": 0.5, "loss": make_loss("bce")},
        "PLI paper default (NN, FP weight 2, threshold 0.9)": {
            "margin": pli_margin(model, tau), "h": lambda x: model(x.float()).double(), "tau": tau,
            "loss": make_loss(p["loss"])},
    }

    rows = []
    for name, m in methods.items():
        r = env_ours(ctx, m["margin"], test)
        r.pop("move_features")
        rows.append({"method": name, **r})
    tbl = pd.DataFrame(rows).set_index("method")
    tbl.to_csv(out_dir / "env_ours_test.csv")

    ep = sw["env_pli"]
    ids = [feats.index(f) for f in ep["improvable_features"]]
    u, _ = ctx["data"][test]
    pli_rows = [{"method": name, "r": float(rad),
                 **evaluate_env_pli(m["h"], m["tau"], m["loss"], u, ctx["fstar"], ids, float(rad), ep["step"], ep["iters"])}
                for name, m in methods.items() for rad in ep["radii"]]
    env_pli = pd.DataFrame(pli_rows)
    env_pli.to_csv(out_dir / "env_pli_test.csv", index=False)

    # Everything needed to reproduce / reload this PLI model and its numbers.
    e = ctx["env"]
    meta = {
        "model_file": model_path.name,
        "model_sha256": file_sha256(model_path),
        "method": "PLI paper default (Attias et al., ICML 2025; github.com/ripl/PLI), reimplemented in "
                  "pipeline/baselines_pli.py",
        "architecture": "NNClassifier: Linear(3,64) -> ReLU -> Linear(64,1) -> Sigmoid (float32)",
        "decision_rule": f"positive iff h(u) > {p['threshold']}",
        "training": {**p, "loss_detail": "WBCELossx(false_positive_weight=2.0, false_negative_weight=1.33)",
                     "optimizer": "Adam", "shuffle": False, "train_split": sw["train_split"],
                     "train_rows": int(len(y_tr))},
        "features": feats,
        "input_space": "standardized with the LR scaler (fit on lr_train): u = (x - shift) / scale",
        "standardization": {"shift": dict(zip(feats, ctx["eta"].shift.tolist())),
                            "scale": dict(zip(feats, ctx["eta"].scale.tolist()))},
        "data": {"dataset": ctx["cfg"]["dataset"]["path"],
                 "split_indices": str(paths["split_indices"]),
                 "split_indices_sha256": file_sha256(paths["split_indices"]),
                 "test_split": test, "test_rows": int(len(u))},
        "environment_ours": {"beta": e["beta"], "alpha": dict(zip(feats, e["alpha"].tolist())),
                             "delta_g": e["delta_g"], "clip": "feasible (max of lr_train)",
                             "eta_hat": str(paths["lr_model"])},
        "compared_with": {"model": str(paths["strat_model"]), "best_epoch": best_epoch},
        "results_env_ours_test": {k: {c: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                      for c, v in row.items()} for k, row in tbl.to_dict("index").items()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    with open(out_dir / "pli_nn_paper_default_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    pd.set_option("display.width", 250)
    print(f"\n== Environment ours, {test} ({len(u):,} unseen rows); our model = best epoch {best_epoch}")
    print(tbl[["improvement_rate", "improvement_error", "error_no_response", "neg_movers", "movers",
               "expected_improved", "predicted_positive_before"]].round(4).to_string())
    for col in ("improvement_error", "improvement_rate"):
        print(f"\n== Environment PLI, {test}: {col} by radius r")
        print(env_pli.pivot(index="method", columns="r", values=col).round(4).to_string())
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
