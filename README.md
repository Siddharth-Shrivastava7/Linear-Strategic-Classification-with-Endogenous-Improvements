# Research_coding

Dataset-agnostic pipeline. Every dataset-specific choice lives in one YAML file under `configs/`.

```
conda activate strategic-ai
python stage1_split_and_profile.py --config configs/retiring_adult.yaml   # fixed 45/45/5/5 split + feature types
python stage2_train_lr.py          --config configs/retiring_adult.yaml   # LR on lr_train, saved
python stage3_plot_lr_profiles.py  --config configs/retiring_adult.yaml   # P(y=1|x) profile plots
python stage4_train_strat_imp.py   --config configs/retiring_adult.yaml   # Algorithm 1 STRAT-IMP-AWARE
python stage5_improvement_rate.py  --config configs/retiring_adult.yaml   # improvement rate on algo_test (--split to change)
```

New dataset: copy `configs/retiring_adult.yaml`, set `dataset.*`, run stage 1, check the
printed feature-type report, fix `feature_types.overrides`, choose `logistic_regression.features`.

| Split | Share | Used for |
|---|---|---|
| `lr_train` | 45% | Stage 2: fitting the logistic regression |
| `algo_train` | 45% | Stage 4: training our algorithm |
| `algo_val` | 5% | Stage 4: validation loss / checkpoint selection |
| `algo_test` | 5% | Stage 5: unseen test set for evaluation |

Outputs (`outputs/<dataset>/`):
- `splits/`: `split_indices.json` (raw-CSV row ids per split), one CSV per split, `split_summary.json`
  (seed, sizes, label rates, dataset SHA-256). Stage 1 never overwrites an existing split without `--force`.
- `feature_types.csv/json`: detected vs. final type per feature.
- `models/logistic_regression.joblib`: sklearn pipeline (preprocessing + LR); `logistic_regression_meta.json`
  holds coefficients on both the standardized and the original feature scale, plus metrics.
- `plots/lr_profile_<feature>.png/.pdf/.csv` and `lr_profile_all.png/.pdf`.

Stage 3 profile: rows of `lr_train` are grouped by their observed value of the feature (quantile bins if
there are more than `plots.max_points` unique values). For each group: mean of the LR's P(y=1) with
a spread band, the empirical rate of y = 1, and the row count. No other feature is held fixed.

Stage 4 (`pipeline/strat_imp.py`): Algorithms 1-3 in PyTorch, float64. Features are the LR features,
standardized with the LR's own scaler (fit on `lr_train`); alpha is the cost per 1 SD. eta_hat is the
saved LR rebuilt as `FrozenLR` (buffers only, so it is never updated, but gradients flow through its
input x^f). Gradient reaches w, b through score, S(w) and q (via x^f); the improvement condition,
j* and clamp saturation are hard decisions. Updates: `w <- relu(w - lr*grad_w)`, `b <- b - lr*grad_b`.
Outputs in `models/`: `strat_imp_aware.pt` / `.json` (best w, b in standardized and raw units, decision
rule positive iff w.z - b >= 0), `strat_imp_aware_history.csv`, `strat_imp_aware_loss.png/.pdf`.

Stage 5a (`stage5_improvement_rate.py`): on `evaluation.split`, agents best-respond to the saved (w, b)
(Algorithm 2). Movers with y = 0 get a sampled post-response label y' ~ Bernoulli(pi_Imp); movers with
y = 1 keep y' = 1. Improved = moved and y = 0 and y' = 1; improvement rate = improved / movers with y = 0.
Reports one seeded draw, mean ± std over `n_repeats` draws, and the exact expectation (sum of pi_Imp).
Outputs in `evaluation/`: `improvement_rate_<split>.json`, `improvement_agents_<split>.csv` (per agent).

Sweep (`sweep_improvement_rate.py --sweep configs/sweeps/improvement_rate.yaml`): retrains Algorithm 1 per
variant (overrides of `strat_imp_aware`), never touching the stage 4 model, and records the exact expected
improvement rate on `algo_train` (selection) and `algo_val` (report), plus the share of movers pushed past the
largest observed value of the moved feature. `alpha: EQUIV` keeps agent costs equal to standard scaling.
Results: `outputs/<dataset>/sweeps/improvement_rate/results.csv`.

Bounded best responses (`pipeline/strat_imp_clipped.py`, `strat_imp_aware.clip_mode`): `none` = Algorithm 2 as
written; `clip` = x^f capped at the largest value observed in `lr_train`; `feasible` = an agent moves only if it
reaches the boundary inside that range. Used in both training and evaluation. Grid experiment:
`experiment_clipped_improvement.py --sweep configs/sweeps/clipped_improvement.yaml` ->
`outputs/<dataset>/sweeps/clipped_improvement/` (results.csv, best_settings.csv, improvement_rate_vs_beta.png/.pdf).

Split history: the original 45/45/10 split was refined to 45/45/5/5 by halving the old `algo_val`; with the same seed
`lr_train` and `algo_train` are row-for-row identical (checked). Previous outputs: `outputs/<dataset>/archive/split_45_45_10/`.

Improvement error (`experiment_improvement_error.py --sweep configs/sweeps/improvement_error.yaml`): Algorithm 1
trained on nested random subsets of `algo_train` (10 repeats per size), checkpoint by `algo_val` loss, evaluated on
`algo_test`. Improvement error = share of agents with y_hat(after best response) != y' (exact expectation + one
sampled draw). Outputs: `outputs/<dataset>/sweeps/improvement_error/` (runs.csv, summary.csv, improvement_error_vs_n.png/.pdf).

PLI comparison (`experiment_pli_comparison.py --sweep configs/sweeps/pli_comparison.yaml`): PLI (Attias et al.,
ICML 2025, github.com/ripl/PLI) is reimplemented in `pipeline/baselines_pli.py` from their notebook (MLP / linear
learners, BCE or false-positive-weighted BCE, thresholds; PGD L-inf response; decision-tree f*). Both methods train on
`algo_train`; PLI's (loss, threshold) is selected on `algo_val` by improvement error; all numbers are on `algo_test`
over 5 seeds. Two environments (`pipeline/best_response.py`): "ours" = cheapest feasible single-feature move with cost
alpha_j*delta <= beta, labels y' ~ Bernoulli(pi_Imp) (equals Algorithm 2 for a linear classifier); "PLI" = their PGD
response over radius r with decision-tree labels.
