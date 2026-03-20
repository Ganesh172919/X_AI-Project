# Architecture

This document explains how the project is structured, how the modules
relate to each other, and how data flows through the pipeline.

---

## Directory layout

```
instashap-replication/
├── config/
│   └── config.yaml            All hyperparameters in one place
├── src/                        Library code (importable package)
│   ├── __init__.py             Public API re-exports
│   ├── data_loader.py          DatasetLoader class
│   ├── black_box_model.py      BlackBoxModel class
│   ├── shap_computation.py     SHAPComputer + compute_exact_shap()
│   ├── gam_surrogate.py        SHAPSurrogate class (core InstaSHAP)
│   ├── evaluation.py           SHAPEvaluator class
│   └── utils.py                Config, logging, serialization helpers
├── scripts/                    Runnable entry points
│   ├── main.py                 Single-experiment pipeline
│   └── reproduce_results.py    Batch experiments + summary figures
├── notebooks/
│   └── replication_notebook.ipynb  Interactive walkthrough
├── tests/                      Pytest unit tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_black_box_model.py
│   ├── test_gam_surrogate.py
│   ├── test_evaluation.py
│   └── test_utils.py
├── data/                       Data directory (for custom datasets)
│   └── .gitkeep
├── results/                    Output directory (created at runtime)
│   ├── tables/                 CSV result files
│   │   ├── {dataset}_{model}_results.csv
│   │   ├── {dataset}_{model}_per_feature.csv
│   │   └── {dataset}_{model}_rankings.csv
│   ├── figures/                PNG plots
│   │   └── {dataset}_{model}/
│   │       ├── scatter_true_vs_pred_*.png
│   │       ├── per_feature_r2_*.png
│   │       ├── error_distribution_*.png
│   │       └── feature_importance_*.png
│   └── models/                 Pickled models + cached SHAP values
│       ├── {dataset}_{model}_model.pkl
│       ├── {dataset}_{model}_gam_surrogate.pkl
│       ├── {dataset}_{model}_shap_train.pkl
│       └── {dataset}_{model}_shap_test.pkl
├── docs/                       Learning documentation (this folder)
│   ├── CONCEPTS.md             Theory: SHAP, GAMs, Shapley values
│   ├── ARCHITECTURE.md         This file
│   ├── API.md                  Public API reference
│   └── AGENTS.md               Contributor guide
├── README.md                   Project overview
├── QUICKSTART.md               5-minute setup guide
├── REPLICATION_REPORT.md       Detailed replication results
├── requirements.txt            Python dependencies
├── setup.py                    pip-installable package
├── .gitignore                  Git ignore patterns
└── LICENSE                     MIT license
```

---

## Module dependency graph

```
scripts/main.py
  ├── src/data_loader.py        (DatasetLoader)
  ├── src/black_box_model.py    (BlackBoxModel)
  ├── src/shap_computation.py   (SHAPComputer, compute_exact_shap)
  ├── src/gam_surrogate.py      (SHAPSurrogate)
  ├── src/evaluation.py         (SHAPEvaluator)
  └── src/utils.py              (load_config, set_random_seed, ...)

scripts/reproduce_results.py
  └── (same imports as main.py)

src/__init__.py
  └── re-exports all public classes/functions
```

No module in `src/` imports from `scripts/`. The dependency flow is
strictly top-down: scripts -> src -> third-party libraries.

---

## Data flow through the pipeline

The following diagram traces a single experiment from start to finish.
Each arrow shows the data that is passed between steps.

```
config/config.yaml
        |
        v
+---------------------+
|  1. DatasetLoader    |
|     .load_data()     |
+--------+------------+
         |  X_train, X_test, y_train, y_test, feature_names, task_type
         v
+---------------------+
|  2. BlackBoxModel    |
|     .train()         |
|     .evaluate()      |
|     .save_model()    |
+--------+------------+
         |  trained sklearn/xgb/lgb estimator
         v
+---------------------+
|  3. compute_exact_   |
|     shap()           |
|                      |
|  Uses TreeExplainer  |
|  (or KernelExplainer)|
|  + caching support   |
+--------+------------+
         |  shap_train (n_train, n_features)
         |  shap_test  (n_test,  n_features)
         |  computation times
         v
+---------------------+
|  4. SHAPSurrogate    |
|     .train()         |
|                      |
|  Trains n GAMs:      |
|  GAM_i: X -> SHAP_i  |
|  using EBMs from     |
|  InterpretML         |
+--------+------------+
         |  fitted GAM models (one per feature)
         v
+---------------------+
|  5. SHAPSurrogate    |
|     .predict_shap()  |
|                      |
|  Evaluates all GAMs  |
|  on X_test (instant) |
+--------+------------+
         |  pred_shap (n_test, n_features)
         |  prediction time
         v
+---------------------+
|  6. SHAPEvaluator    |
|     .compute_        |
|        accuracy_     |
|        metrics()     |
|     .compute_        |
|        speed_        |
|        metrics()     |
|     .compare_        |
|        feature_      |
|        rankings()    |
|     .generate_       |
|        comparison_   |
|        plots()       |
+--------+------------+
         |  metrics dict, DataFrames, PNG files
         v
   results/tables/*.csv
   results/figures/*.png
   results/models/*.pkl
```

---

## Configuration system

All hyperparameters live in `config/config.yaml`. The structure mirrors
the pipeline steps:

```yaml
random_seed: 42           # Reproducibility

datasets:                 # Step 1
  california_housing:
    name: "California Housing"
    task: "regression"
    test_size: 0.2
  breast_cancer:
    name: "Breast Cancer"
    task: "classification"
    test_size: 0.2
  adult:
    name: "Adult Income"
    task: "classification"
    test_size: 0.2

black_box_models:         # Step 2
  random_forest:
    classification: { n_estimators: 100, max_depth: 10, ... }
    regression: { n_estimators: 100, max_depth: 10, ... }
  xgboost:
    classification: { n_estimators: 100, max_depth: 6, learning_rate: 0.1, ... }
    regression: { n_estimators: 100, max_depth: 6, learning_rate: 0.1, ... }
  lightgbm:
    classification: { n_estimators: 100, max_depth: 10, learning_rate: 0.1, ... }
    regression: { n_estimators: 100, max_depth: 10, learning_rate: 0.1, ... }

shap_config:              # Step 3
  train_sample_size: 1000
  test_sample_size: 500
  background_size: 100
  check_additivity: false

gam_config:               # Step 4
  max_iter: 5000
  max_bins: 256
  interactions: 0
  learning_rate: 0.01
  min_samples_leaf: 2

evaluation:               # Step 6
  metrics: [mse, mae, r2, pearson_correlation, spearman_correlation]
  timing_runs: 5
  top_k_features: 10

visualization:            # Plot settings
  figure_format: "png"
  dpi: 300
  style: "seaborn-v0_8-darkgrid"
  figure_size: [10, 6]

computation:              # Cross-cutting
  cache_shap: true
  use_gpu: false
  n_jobs: -1
  batch_size: 100
```

The config is loaded once at the start of a run and passed to each step.

---

## Key design decisions

### One GAM per feature

The core InstaSHAP design trains an independent GAM for each feature's
SHAP values. This means:

- Training is embarrassingly parallel (each GAM is independent).
- Total model count = number of features.
- Each GAM is small and fast to evaluate.

### Explainable Boosting Machines (EBMs)

We use `interpret.glassbox.ExplainableBoostingRegressor` as the GAM
implementation because:

- It is a mature, well-tested library from Microsoft Research.
- EBMs learn smooth shape functions via gradient boosting.
- They support optional interaction terms (set to 0 for pure additive).
- Feature importance is built-in via `term_importances()`.

### Caching

SHAP computation is the most expensive step. The pipeline caches results
to disk (controlled by `computation.cache_shap` in config). On subsequent
runs with the same parameters, cached values are loaded instead of
recomputed.

### Standardized evaluation

All experiments produce the same set of metrics (MSE, MAE, R²,
correlation, speedup, ranking overlap). This makes it easy to compare
across datasets and models and to reproduce the paper's tables and
figures.

---

## How to extend the project

### Add a new dataset

1. Add an entry to `config/config.yaml` under `datasets`:
   ```yaml
   datasets:
     my_dataset:
       name: "My Dataset"
       task: "classification"  # or "regression"
       test_size: 0.2
   ```
2. Add a `_load_my_dataset()` method to `DatasetLoader` in
   `src/data_loader.py`:
   ```python
   def _load_my_dataset(self) -> None:
       # Load data
       X, y = ...
       self.feature_names = list(X.columns)
       self.task_type = "classification"
       # Split and scale
       X_train, X_test, y_train, y_test = train_test_split(...)
       self.X_train = self.scaler.fit_transform(X_train)
       self.X_test = self.scaler.transform(X_test)
       self.y_train, self.y_test = y_train, y_test
   ```
3. Add tests in `tests/test_data_loader.py`.
4. The pipeline will pick it up automatically via `--dataset my_dataset`.

### Add a new black-box model

1. Add a config block under `black_box_models` in `config.yaml`:
   ```yaml
   black_box_models:
     my_model:
       classification: { param1: value1, ... }
       regression: { param1: value1, ... }
   ```
2. Add initialization logic in `BlackBoxModel._initialize_model()`.
3. SHAP explainer selection happens automatically (tree models get
   TreeExplainer, others get KernelExplainer).
4. Add tests in `tests/test_black_box_model.py`.

### Add a new evaluation metric

1. Implement it in `SHAPEvaluator.compute_accuracy_metrics()` in
   `src/evaluation.py`.
2. Include the metric key in `config.yaml` under `evaluation.metrics`.

### Add a new experiment script

1. Create a new file in `scripts/`.
2. Import from `src` (the package re-exports everything via
   `__init__.py`):
   ```python
   from src import (
       DatasetLoader,
       BlackBoxModel,
       SHAPSurrogate,
       SHAPEvaluator,
       compute_exact_shap,
       load_config,
       set_random_seed,
   )
   ```
3. Follow the pattern in `scripts/main.py`.

---

## Output files

After running an experiment, the following files are generated:

### Tables (`results/tables/`)

| File | Contents |
|------|----------|
| `{dataset}_{model}_results.csv` | Global metrics (MSE, MAE, R^2, correlation, speedup) |
| `{dataset}_{model}_per_feature.csv` | Per-feature metrics (MSE, MAE, R^2, correlation) |
| `{dataset}_{model}_rankings.csv` | Feature importance rankings comparison |

### Figures (`results/figures/{dataset}_{model}/`)

| File | Description |
|------|-------------|
| `scatter_true_vs_pred_*.png` | True vs predicted SHAP scatter plot |
| `per_feature_r2_*.png` | Per-feature R^2 bar chart |
| `error_distribution_*.png` | Prediction error histogram |
| `feature_importance_*.png` | Feature importance comparison (top 20) |

### Models (`results/models/`)

| File | Contents |
|------|----------|
| `{dataset}_{model}_model.pkl` | Trained black-box model |
| `{dataset}_{model}_gam_surrogate.pkl` | Trained GAM surrogates |
| `{dataset}_{model}_shap_train.pkl` | Cached training SHAP values |
| `{dataset}_{model}_shap_test.pkl` | Cached test SHAP values |
