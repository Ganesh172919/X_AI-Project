# Complete Project Analysis

This document provides a full technical analysis of the current
implementation so learners can understand both the "what" and the "why"
of the project.

## 1. Project Purpose

The repository replicates the core idea from the InstaSHAP paper:

- Train a black-box model.
- Compute exact SHAP values for a subset of data.
- Train one GAM surrogate per feature to predict that feature's SHAP value.
- Use the surrogate for near-instant SHAP prediction at inference time.

Practical goal: trade a small amount of attribution fidelity for a large
speedup in explanation generation.

## 2. Current Scope and Boundaries

### Implemented

- Datasets: `adult`, `california_housing`, `breast_cancer`
- Black-box models: `random_forest`, `xgboost`, `lightgbm`
- SHAP explainers supported in code: `tree`, `kernel`, `linear`
- Surrogate: Explainable Boosting Regressor (InterpretML)
- Metrics: global, per-feature, speed, and ranking metrics
- Output generation: CSV tables, PNG visualizations, model artifacts

### Not implemented in pipeline scripts

- Neural-network black-box models
- Advanced interaction modeling for surrogate (`interactions > 0` not used by default)
- Large-scale distributed execution
- Production API service layer

## 3. Codebase Layout (Learning Perspective)

### Core library (`src/`)

- `data_loader.py`
  - Dataset-specific loading and preprocessing
  - Handles categorical encoding for Adult dataset
  - Handles train/test split and scaling

- `black_box_model.py`
  - Unified wrapper over sklearn/xgboost/lightgbm estimators
  - Keeps training/prediction/evaluation interface consistent

- `shap_computation.py`
  - Encapsulates SHAP explainer initialization
  - Supports caching SHAP values to disk
  - Converts SHAP output formats into consistent arrays

- `gam_surrogate.py`
  - Core InstaSHAP approximation model
  - Trains one EBM per SHAP output column (feature-wise target)
  - Predicts complete SHAP matrix by stacking each GAM output

- `evaluation.py`
  - Computes fidelity metrics
  - Compares feature ranking consistency
  - Produces model-vs-surrogate plots

- `utils.py`
  - Config loading, logging setup, random seeds, object serialization

### Entry points (`scripts/`)

- `main.py`
  - Runs one dataset-model experiment end-to-end
  - Saves model, SHAP cache, metrics, and per-experiment plots

- `reproduce_results.py`
  - Runs a fixed experiment matrix
  - Produces summary tables/figures across runs

## 4. End-to-End Data Flow

1. Load config from `config/config.yaml`.
2. Load and preprocess dataset with `DatasetLoader`.
3. Train black-box model via `BlackBoxModel`.
4. Compute exact SHAP (train subset and test subset).
5. Train surrogate (`SHAPSurrogate`) on `(X_train_subset, shap_train)`.
6. Predict SHAP on test subset with surrogate.
7. Evaluate fidelity and speed with `SHAPEvaluator`.
8. Save artifacts to `results/`.

## 5. Design Choices and Tradeoffs

### Choice: One surrogate per feature

- Pros:
  - Simple decomposition
  - Easy per-feature debugging
  - Natural parallelism
- Cons:
  - More models to store
  - Ignores cross-output structure between SHAP dimensions

### Choice: EBM-based GAM surrogate

- Pros:
  - Interpretable shape functions
  - Strong tabular performance
  - Stable training behavior
- Cons:
  - Can underfit strong interactions if interactions are disabled

### Choice: Train on SHAP subset

- Pros:
  - Reduces expensive SHAP computation cost
  - Practical for iterative experimentation
- Cons:
  - Approximation quality depends on subset representativeness

## 6. Configuration-Driven Behavior

Key knobs in `config/config.yaml` with strong downstream impact:

- `shap_config.train_sample_size`
  - Controls how much ground-truth signal surrogate sees during training.
- `shap_config.test_sample_size`
  - Affects evaluation stability and run time.
- `gam_config.max_iter`, `learning_rate`, `interactions`
  - Main fidelity-vs-time controls for surrogate training.
- `computation.cache_shap`
  - Major speed lever for repeated runs.

## 7. Testing Analysis

The test suite validates each core module independently:

- `test_data_loader.py`
  - Dataset loading, basic shape checks, metadata checks

- `test_black_box_model.py`
  - Classification and regression behavior
  - `predict_proba` constraints
  - save/load consistency
  - edge-case errors

- `test_gam_surrogate.py`
  - training checks, shape mismatch checks, evaluation outputs
  - save/load prediction consistency

- `test_evaluation.py`
  - metrics and plotting outputs

- `test_utils.py`
  - config, reproducibility, logging, serialization, helpers

Observation: tests provide strong functional coverage of module contracts,
but there is limited integration testing around the full script pipelines.

## 8. Outputs and Artifact Semantics

### `results/models/`

- Black-box model checkpoint (`*_model.pkl`)
- Surrogate checkpoint (`*_gam_surrogate.pkl`)
- Cached SHAP arrays (`*_shap_train.pkl`, `*_shap_test.pkl`)

### `results/tables/`

- Per-run global metrics (`*_results.csv`)
- Per-feature metrics (`*_per_feature.csv`)
- Ranking comparisons (`*_rankings.csv`)
- Cross-run summaries (`complete_results.csv`, `table1_accuracy.csv`)

### `results/figures/`

- Per-run plots under dataset-model folder
- Summary figures from reproduction script

## 9. Known Implementation Notes

- `scripts/main.py` and `scripts/reproduce_results.py` each define a local
  config loader rather than importing `src.utils.load_config`.
- `src.__init__` currently re-exports a subset of utility functions; docs
  may mention more functions than currently exported.
- Both scripts modify `sys.path` to import `src` from repository root.

These are valid for a research-style repository, but could be refactored
for cleaner package-level usage.

## 10. Learning-Oriented Improvement Opportunities

1. Add integration tests that run a tiny end-to-end pipeline.
2. Add a single "experiment manifest" output JSON per run.
3. Harmonize script-level config loading with `src.utils.load_config`.
4. Expand docs with "failure mode" examples (e.g., poor surrogate fit).
5. Add notebook section comparing sample size vs R^2 vs speedup.

## 11. Mental Model to Keep

Think of this project as a two-stage explainer system:

- Stage A (expensive, offline): generate high-quality SHAP labels.
- Stage B (cheap, online): learn a fast student model to emulate those labels.

If Stage A labels are good and representative, Stage B can be very fast
while staying faithful enough for practical interpretability use.
