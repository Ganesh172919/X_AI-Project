# API Reference

Complete reference for every public class and function in the `src` package.

All items are re-exported from `src/__init__.py`:

```python
from src import (
    DatasetLoader,
    BlackBoxModel,
    SHAPComputer,
    SHAPSurrogate,
    SHAPEvaluator,
    compute_exact_shap,
    load_config,
    set_random_seed,
    setup_logging,
    save_object,
    load_object,
    ensure_dir,
    format_time,
)
```

---

## Quick Start Example

```python
from src import (
    DatasetLoader,
    BlackBoxModel,
    SHAPSurrogate,
    SHAPEvaluator,
    compute_exact_shap,
    set_random_seed,
)

# 1. Setup
set_random_seed(42)

# 2. Load data
loader = DatasetLoader("california_housing", test_size=0.2)
X_train, X_test, y_train, y_test = loader.load_data()

# 3. Train black-box model
model = BlackBoxModel("random_forest", task="regression", n_estimators=100)
model.train(X_train, y_train)

# 4. Compute exact SHAP values
shap_train, _ = compute_exact_shap(model.get_model(), X_train, sample_size=1000)
shap_test, exact_time = compute_exact_shap(model.get_model(), X_test, sample_size=500)

# 5. Train GAM surrogate
surrogate = SHAPSurrogate(max_iter=5000)
surrogate.train(X_train[:1000], shap_train, feature_names=loader.get_feature_names())

# 6. Predict SHAP instantly
pred_shap, pred_time = surrogate.predict_shap(X_test[:500], return_time=True)
print(f"Speedup: {exact_time / pred_time:.1f}x")

# 7. Evaluate
evaluator = SHAPEvaluator(loader.get_feature_names())
metrics = evaluator.compute_accuracy_metrics(shap_test, pred_shap)
print(f"R^2: {metrics['r2']:.4f}")
```

---

## src.data_loader

### DatasetLoader

```python
DatasetLoader(dataset_name: str, test_size: float = 0.2, random_state: int = 42)
```

Unified interface for loading and preprocessing tabular datasets.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dataset_name` | str | (required) | One of `"adult"`, `"california_housing"`, `"breast_cancer"` |
| `test_size` | float | 0.2 | Fraction of data reserved for testing |
| `random_state` | int | 42 | Random seed for reproducibility |

**Attributes (set after `load_data()`)**

| Name | Type | Description |
|------|------|-------------|
| `X_train`, `X_test` | np.ndarray | Scaled feature matrices |
| `y_train`, `y_test` | np.ndarray | Target arrays |
| `feature_names` | list[str] | Feature column names |
| `task_type` | str | `"regression"` or `"classification"` |
| `scaler` | StandardScaler | Fitted scaler instance |

**Methods**

#### `load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

Load, split, and scale the dataset. Returns `(X_train, X_test, y_train, y_test)`.

```python
loader = DatasetLoader("california_housing")
X_train, X_test, y_train, y_test = loader.load_data()
```

#### `get_feature_names() -> List[str]`

Return the list of feature names.

#### `get_train_test_split() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

Return the cached train/test split. Raises `ValueError` if `load_data()`
has not been called.

#### `describe_data() -> Dict[str, Any]`

Return a dictionary with dataset statistics (sample counts, feature count,
class distribution or target statistics).

---

## src.black_box_model

### BlackBoxModel

```python
BlackBoxModel(model_type: str = "random_forest", task: str = "classification", **model_params)
```

Wrapper for training and evaluating black-box ML models.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_type` | str | `"random_forest"` | One of `"random_forest"`, `"xgboost"`, `"lightgbm"` |
| `task` | str | `"classification"` | One of `"classification"`, `"regression"` |
| `**model_params` | dict | `{}` | Forwarded to the underlying estimator |

**Methods**

#### `train(X_train, y_train, X_val=None, y_val=None) -> BlackBoxModel`

Fit the model. For XGBoost/LightGBM, passes `eval_set` if validation data
is provided. Returns `self` for chaining.

```python
model = BlackBoxModel("xgboost", task="regression", n_estimators=200)
model.train(X_train, y_train, X_val, y_val)
```

#### `predict(X: np.ndarray) -> np.ndarray`

Return class labels or regression predictions.

#### `predict_proba(X: np.ndarray) -> np.ndarray`

Return class probabilities. Only valid for `task="classification"`.
Raises `ValueError` for regression tasks.

#### `evaluate(X_test, y_test, verbose=True) -> Dict[str, float]`

Compute and optionally log evaluation metrics.

- Classification: `accuracy`, `f1_score`, `auc_roc` (binary only)
- Regression: `mse`, `rmse`, `r2`

#### `save_model(filepath: str) -> None`

Serialize the fitted model to disk via joblib.

#### `load_model(filepath: str) -> None`

Load a previously saved model from disk.

#### `get_model() -> estimator`

Return the underlying scikit-learn compatible estimator.

---

## src.shap_computation

### SHAPComputer

```python
SHAPComputer(model, model_type: str = "tree", background_size: int = 100, check_additivity: bool = False)
```

Computes and caches exact SHAP values.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model` | estimator | (required) | Trained model object |
| `model_type` | str | `"tree"` | One of `"tree"`, `"kernel"`, `"linear"` |
| `background_size` | int | 100 | Background dataset size for KernelExplainer |
| `check_additivity` | bool | False | Whether to verify SHAP additivity |

**Methods**

#### `compute_shap_values(X, X_background=None, sample_size=None) -> Tuple[np.ndarray, float]`

Compute SHAP values. Returns `(shap_values, computation_time_seconds)`.

If `sample_size` is set and `X` has more rows, a random subset is used.

```python
computer = SHAPComputer(model, model_type="tree")
shap_values, elapsed = computer.compute_shap_values(X_test, sample_size=500)
```

#### `save_shap_values(shap_values, filepath, metadata=None) -> None`

Persist SHAP values and optional metadata to disk.

#### `load_shap_values(filepath) -> Tuple[np.ndarray, Dict]`

Load previously saved SHAP values. Returns `(shap_values, metadata)`.

#### `visualize_shap(shap_values, X, feature_names, plot_type="summary", save_path=None) -> None`

Create SHAP plots. `plot_type` can be `"summary"`, `"bar"`, or `"waterfall"`.

### compute_exact_shap (convenience function)

```python
compute_exact_shap(model, X_data, model_type="tree", sample_size=1000,
                   background_size=100, cache_path=None) -> Tuple[np.ndarray, float]
```

High-level function that handles explainer creation, caching, and SHAP
computation in one call. Returns `(shap_values, computation_time)`.

```python
from src.shap_computation import compute_exact_shap

shap_train, t = compute_exact_shap(model, X_train, cache_path="cache.pkl")
```

---

## src.gam_surrogate

### SHAPSurrogate

```python
SHAPSurrogate(max_iter=5000, max_bins=256, interactions=0,
              learning_rate=0.01, min_samples_leaf=2, random_state=42)
```

Core InstaSHAP class. Trains one GAM per feature to predict SHAP values.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `max_iter` | int | 5000 | Maximum boosting rounds per GAM |
| `max_bins` | int | 256 | Feature discretization bins |
| `interactions` | int | 0 | Pairwise interaction terms (0 = pure additive) |
| `learning_rate` | float | 0.01 | Boosting step size |
| `min_samples_leaf` | int | 2 | Minimum samples per leaf |
| `random_state` | int | 42 | Random seed |

**Attributes (set after `train()`)**

| Name | Type | Description |
|------|------|-------------|
| `gam_models` | dict[int, EBM] | Trained GAMs keyed by feature index |
| `n_features` | int | Number of features |
| `feature_names` | list[str] | Feature names |
| `is_fitted` | bool | Training flag |

**Methods**

#### `train(X_train, shap_values_train, feature_names=None, verbose=True) -> SHAPSurrogate`

Train one GAM per feature. `X_train` and `shap_values_train` must have
the same shape `(n_samples, n_features)`. Returns `self`.

```python
surrogate = SHAPSurrogate(max_iter=5000)
surrogate.train(X_train, shap_values_train, feature_names=feature_names)
```

#### `predict_shap(X_test, return_time=False) -> np.ndarray | Tuple[np.ndarray, float]`

Predict SHAP values for new data. If `return_time=True`, returns
`(predicted_shap, prediction_time_seconds)`.

```python
pred_shap, t = surrogate.predict_shap(X_test, return_time=True)
```

#### `evaluate(X_test, true_shap_values, verbose=True) -> Dict[str, float]`

Compare predictions against ground-truth SHAP values. Returns metrics
including `mse`, `mae`, `rmse`, `r2`, `pearson_correlation`,
`spearman_correlation`, `prediction_time`, `per_feature_mse`,
`per_feature_r2`, and `mean_per_feature_r2`.

#### `get_feature_importance(feature_idx: int) -> np.ndarray`

Return term importances from the GAM for the specified feature.

#### `save_model(filepath: str) -> None`

Serialize all trained GAMs and configuration to disk.

#### `load_model(filepath: str) -> None`

Load a previously saved surrogate from disk.

---

## src.evaluation

### SHAPEvaluator

```python
SHAPEvaluator(feature_names: List[str])
```

Comprehensive evaluation and visualization of SHAP prediction quality.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `feature_names` | list[str] | Names corresponding to SHAP columns |

**Methods**

#### `compute_accuracy_metrics(true_shap, pred_shap) -> Dict[str, float]`

Global metrics: `mse`, `mae`, `rmse`, `r2`, `mape`,
`pearson_correlation`, `pearson_pvalue`, `spearman_correlation`,
`spearman_pvalue`.

#### `compute_per_feature_metrics(true_shap, pred_shap) -> pd.DataFrame`

Per-feature metrics as a DataFrame with columns:
`feature`, `mse`, `mae`, `r2`, `correlation`.

#### `compute_speed_metrics(exact_time, surrogate_time, n_samples) -> Dict[str, float]`

Speed comparison: `exact_time_seconds`, `surrogate_time_seconds`,
`speedup_factor`, `exact_time_per_sample_ms`, `surrogate_time_per_sample_ms`.

#### `compare_feature_rankings(true_shap, pred_shap, top_k=10) -> Dict[str, Any]`

Feature ranking comparison. Returns:
- `top_k`, `top_k_overlap`, `top_k_overlap_ratio`
- `ranking_correlation` (Spearman)
- `ranking_df` (DataFrame with importance and rank per feature)

#### `generate_comparison_plots(true_shap, pred_shap, save_dir, dataset_name="dataset") -> None`

Generate four plots and save them to `save_dir`:
1. Scatter: true vs. predicted SHAP
2. Bar chart: per-feature R²
3. Histogram: error distribution
4. Grouped bar: feature importance comparison

---

## src.utils

### load_config

```python
load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]
```

Read a YAML file and return its contents as a dictionary.

### set_random_seed

```python
set_random_seed(seed: int = 42) -> None
```

Set random seeds for Python `random`, NumPy, and (if available) PyTorch.

### setup_logging

```python
setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger
```

Configure Python logging with a console handler and optional file handler.
Returns the root logger.

### save_object / load_object

```python
save_object(obj: Any, filepath: str) -> None
load_object(filepath: str) -> Any
```

Serialize / deserialize any Python object via joblib. `save_object`
creates parent directories automatically.

### ensure_dir

```python
ensure_dir(directory: str) -> None
```

Create a directory and all parents if they do not exist.

### format_time

```python
format_time(seconds: float) -> str
```

Convert seconds to a human-readable string (`"45.30 ms"`, `"2.10 s"`,
`"3.50 min"`, `"1.25 hr"`).

### print_dict

```python
print_dict(d: Dict[str, Any], indent: int = 0) -> None
```

Pretty-print a nested dictionary to stdout.

---

## Error Handling

All classes raise informative errors:

| Error | When raised |
|-------|-------------|
| `ValueError("Unknown dataset: ...")` | Invalid dataset name in `DatasetLoader` |
| `ValueError("Unknown model type: ...")` | Invalid model type in `BlackBoxModel` or `SHAPComputer` |
| `ValueError("Model not fitted...")` | Calling predict before train |
| `ValueError("Data not loaded...")` | Calling `get_train_test_split()` before `load_data()` |
| `ValueError("predict_proba only...")` | Calling `predict_proba()` on regression model |

---

## Type Hints

All public functions use type hints. Key types:

```python
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

# Common return types
Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # train/test split
Dict[str, float]                                        # metrics
pd.DataFrame                                            # per-feature metrics
```
