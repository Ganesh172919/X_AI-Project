# API Reference

## Table of Contents
- [Module Overview](#module-overview)
- [src.data_loader](#srcdataloader)
- [src.black_box_model](#srcblackboxmodel)
- [src.shap_computation](#srcshapcomputation)
- [src.gam_surrogate](#srcgamsurrogate)
- [src.evaluation](#srcevaluation)
- [src.utils](#srcutils)

---

## Module Overview

The InstaSHAP project is organized into 6 core modules:

| Module | Purpose | Key Classes | Lines of Code |
|--------|---------|-------------|---------------|
| **data_loader.py** | Data loading and preprocessing | `DataLoader` | 263 |
| **black_box_model.py** | Black-box model training | `BlackBoxModel` | 252 |
| **shap_computation.py** | SHAP value computation | `SHAPComputer` | 271 |
| **gam_surrogate.py** | GAM surrogate training | `GAMSurrogate` | 369 |
| **evaluation.py** | Evaluation and visualization | `Evaluator` | 442 |
| **utils.py** | Utility functions | N/A | 174 |

---

## src.data_loader

**File:** `src/data_loader.py`

### Class: DataLoader

**Purpose:** Unified interface for loading and preprocessing datasets

**Constructor:**
```python
DataLoader(dataset_name: str, test_size: float = 0.2, random_state: int = 42)
```

**Parameters:**
- `dataset_name` (str): Name of dataset ('california_housing', 'breast_cancer', 'adult')
- `test_size` (float): Proportion for test set (default: 0.2)
- `random_state` (int): Random seed for splitting (default: 42)

**Attributes:**
- `dataset_name` (str): Name of loaded dataset
- `X_train` (np.ndarray): Training features
- `X_test` (np.ndarray): Test features
- `y_train` (np.ndarray): Training labels
- `y_test` (np.ndarray): Test labels
- `feature_names` (list): List of feature names
- `task` (str): Task type ('classification' or 'regression')

**Methods:**

#### get_data()
```python
get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
```

**Returns:** (X_train, X_test, y_train, y_test, feature_names)

**Description:** Returns preprocessed data ready for training

**Example:**
```python
from src.data_loader import DataLoader

loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, feature_names = loader.get_data()
print(f"Training samples: {len(X_train)}")
print(f"Features: {feature_names}")
```

#### load_california_housing()
```python
load_california_housing() -> None
```

**Description:** Loads and preprocesses California Housing dataset

**Side Effects:**
- Sets `self.X_train`, `self.X_test`, `self.y_train`, `self.y_test`
- Sets `self.feature_names`
- Sets `self.task = 'regression'`

#### load_breast_cancer()
```python
load_breast_cancer() -> None
```

**Description:** Loads and preprocesses Breast Cancer dataset

#### load_adult()
```python
load_adult() -> None
```

**Description:** Loads and preprocesses Adult Income dataset from OpenML

---

## src.black_box_model

**File:** `src/black_box_model.py`

### Class: BlackBoxModel

**Purpose:** Wrapper for training and evaluating black-box ML models

**Constructor:**
```python
BlackBoxModel(model_type: str, task: str, **hyperparameters)
```

**Parameters:**
- `model_type` (str): Type of model ('random_forest', 'xgboost', 'lightgbm')
- `task` (str): Task type ('classification', 'regression')
- `**hyperparameters`: Model-specific hyperparameters

**Attributes:**
- `model_type` (str): Model type
- `task` (str): Task type
- `model`: Trained sklearn/xgboost/lightgbm model object

**Methods:**

#### train()
```python
train(X_train: np.ndarray, y_train: np.ndarray) -> None
```

**Description:** Trains the black-box model

**Parameters:**
- `X_train`: Training features (n_samples, n_features)
- `y_train`: Training labels (n_samples,)

**Side Effects:** Sets `self.model` to trained model

**Example:**
```python
from src.black_box_model import BlackBoxModel

model = BlackBoxModel('xgboost', 'regression')
model.train(X_train, y_train)
```

#### predict()
```python
predict(X: np.ndarray) -> np.ndarray
```

**Description:** Makes predictions on new data

**Parameters:**
- `X`: Features (n_samples, n_features)

**Returns:** Predictions (n_samples,)

**Example:**
```python
predictions = model.predict(X_test)
```

#### predict_proba()
```python
predict_proba(X: np.ndarray) -> np.ndarray
```

**Description:** Returns class probabilities (classification only)

**Parameters:**
- `X`: Features (n_samples, n_features)

**Returns:** Probabilities (n_samples, n_classes)

**Raises:** ValueError if task is not classification

####evaluate()
```python
evaluate(X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]
```

**Description:** Evaluates model on test data

**Parameters:**
- `X_test`: Test features
- `y_test`: Test labels

**Returns:** Dictionary of metrics

**Classification Metrics:**
- `accuracy`: Accuracy score
- `f1_score`: F1 score
- `roc_auc`: AUC-ROC score
- `precision`: Precision
- `recall`: Recall

**Regression Metrics:**
- `mse`: Mean squared error
- `rmse`: Root mean squared error
- `mae`: Mean absolute error
- `r2`: R² score

**Example:**
```python
metrics = model.evaluate(X_test, y_test)
print(f"Test R²: {metrics['r2']:.4f}")
```

#### save()
```python
save(filepath: str) -> None
```

**Description:** Saves model to disk using joblib

**Parameters:**
- `filepath`: Path to save file (e.g., 'models/my_model.pkl')

#### load() (static method)
```python
@staticmethod
load(filepath: str) -> BlackBoxModel
```

**Description:** Loads model from disk

**Parameters:**
- `filepath`: Path to saved model

**Returns:** Loaded BlackBoxModel instance

**Example:**
```python
model = BlackBoxModel.load('models/my_model.pkl')
predictions = model.predict(X_new)
```

---

## src.shap_computation

**File:** `src/shap_computation.py`

### Class: SHAPComputer

**Purpose:** Computes SHAP values using appropriate explainer

**Constructor:**
```python
SHAPComputer(model: BlackBoxModel, data: np.ndarray, task: str)
```

**Parameters:**
- `model`: Trained BlackBoxModel instance
- `data`: Background data for SHAP explainer (n_background, n_features)
- `task`: Task type ('classification', 'regression')

**Attributes:**
- `model`: Black-box model
- `explainer`: SHAP explainer object (TreeExplainer or KernelExplainer)
- `task`: Task type

**Methods:**

#### compute_shap_values()
```python
compute_shap_values(X: np.ndarray, method: str = 'auto') -> np.ndarray
```

**Description:** Computes SHAP values for given data

**Parameters:**
- `X`: Data to explain (n_samples, n_features)
- `method`: Explainer method ('auto', 'tree', 'kernel', 'linear')

**Returns:** SHAP values (n_samples, n_features)

**Example:**
```python
from src.shap_computation import SHAPComputer

shap_computer = SHAPComputer(model, X_train, 'regression')
shap_values = shap_computer.compute_shap_values(X_test[:100])
print(f"SHAP values shape: {shap_values.shape}")  # (100, n_features)
```

#### _get_explainer()
```python
_get_explainer(method: str) -> shap.Explainer
```

**Description:** Initializes appropriate SHAP explainer

**Parameters:**
- `method`: Explainer type

**Returns:** SHAP explainer object

**Explainer Selection Logic:**
- 'tree' → TreeExplainer (for tree-based models)
- 'kernel' → KernelExplainer (model-agnostic)
- 'linear' → LinearExplainer (for linear models)
- 'auto' → Automatically selects based on model type

---

## src.gam_surrogate

**File:** `src/gam_surrogate.py`

### Class: GAMSurrogate

**Purpose:** Trains GAM surrogates to predict SHAP values

**Constructor:**
```python
GAMSurrogate(feature_names: List[str], max_iter: int = 5000, 
             max_bins: int = 256, learning_rate: float = 0.01,
             interactions: int = 0, random_state: int = 42)
```

**Parameters:**
- `feature_names`: List of feature names
- `max_iter`: Total boosting rounds (default: 5000)
- `max_bins`: Discretization bins (default: 256)
- `learning_rate`: Step size (default: 0.01)
- `interactions`: Number of pairwise interactions (default: 0)
- `random_state`: Random seed (default: 42)

**Attributes:**
- `feature_names` (list): Feature names
- `gam_models` (dict): Dictionary mapping feature name → trained EBM
- `config` (dict): Configuration parameters

**Methods:**

#### train()
```python
train(X_train: np.ndarray, shap_values_train: np.ndarray) -> None
```

**Description:** Trains GAM surrogates (one per feature)

**Parameters:**
- `X_train`: Training features (n_samples, n_features)
- `shap_values_train`: Target SHAP values (n_samples, n_features)

**Side Effects:** Populates `self.gam_models` dictionary

**Example:**
```python
from src.gam_surrogate import GAMSurrogate

gam = GAMSurrogate(feature_names=feature_names)
gam.train(X_train, shap_values_train)
print(f"Trained {len(gam.gam_models)} GAM surrogates")
```

**Training Time:** 60-300 seconds depending on dataset size

#### predict()
```python
predict(X: np.ndarray) -> np.ndarray
```

**Description:** Predicts SHAP values using trained GAMs

**Parameters:**
- `X`: Features (n_samples, n_features)

**Returns:** Predicted SHAP values (n_samples, n_features)

**Example:**
```python
shap_predicted = gam.predict(X_test)
print(f"Predicted SHAP shape: {shap_predicted.shape}")
```

**Inference Time:** <1ms per sample (very fast!)

#### evaluate()
```python
evaluate(X_test: np.ndarray, shap_values_test: np.ndarray) -> Dict[str, float]
```

**Description:** Evaluates GAM surrogate accuracy

**Parameters:**
- `X_test`: Test features
- `shap_values_test`: True SHAP values

**Returns:** Dictionary of metrics (mse, mae, r2, correlation)

**Example:**
```python
metrics = gam.evaluate(X_test, shap_values_test)
print(f"GAM R²: {metrics['r2']:.4f}")
```

#### save()
```python
save(filepath: str) -> None
```

**Description:** Saves GAM surrogates to disk

**Parameters:**
- `filepath`: Path to save file

#### load() (static method)
```python
@staticmethod
load(filepath: str) -> GAMSurrogate
```

**Description:** Loads GAM surrogates from disk

**Returns:** Loaded GAMSurrogate instance

---

## src.evaluation

**File:** `src/evaluation.py`

### Class: Evaluator

**Purpose:** Comprehensive evaluation and visualization of SHAP predictions

**Constructor:**
```python
Evaluator(shap_true: np.ndarray, shap_pred: np.ndarray, 
          feature_names: List[str])
```

**Parameters:**
- `shap_true`: True SHAP values (n_samples, n_features)
- `shap_pred`: Predicted SHAP values (n_samples, n_features)
- `feature_names`: List of feature names

**Attributes:**
- `shap_true`: Ground truth SHAP values
- `shap_pred`: Predicted SHAP values
- `feature_names`: Feature names
- `metrics`: Computed metrics (after calling evaluate())

**Methods:**

#### evaluate()
```python
evaluate() -> Dict[str, Union[float, np.ndarray]]
```

**Description:** Computes comprehensive evaluation metrics

**Returns:** Dictionary containing:
- Global metrics: mse, mae, rmse, r2, mape
- Correlation metrics: pearson_correlation, spearman_correlation
- Per-feature metrics: per_feature_mse, per_feature_r2

**Example:**
```python
from src.evaluation import Evaluator

evaluator = Evaluator(shap_true, shap_pred, feature_names)
metrics = evaluator.evaluate()

print(f"Global R²: {metrics['r2']:.4f}")
print(f"Pearson correlation: {metrics['pearson_correlation']:.4f}")
print(f"Per-feature R²: {metrics['per_feature_r2']}")
```

#### plot_true_vs_pred()
```python
plot_true_vs_pred(save_path: str = None) -> None
```

**Description:** Plots scatter plot of true vs predicted SHAP values

**Parameters:**
- `save_path`: Optional path to save figure

**Example:**
```python
evaluator.plot_true_vs_pred(save_path='results/scatter_plot.png')
```

#### plot_per_feature_r2()
```python
plot_per_feature_r2(save_path: str = None) -> None
```

**Description:** Plots bar chart of per-feature R² scores

#### plot_error_distribution()
```python
plot_error_distribution(save_path: str = None) -> None
```

**Description:** Plots histogram of prediction errors

#### plot_feature_importance_comparison()
```python
plot_feature_importance_comparison(save_path: str = None) -> None
```

**Description:** Compares feature importance from true vs predicted SHAP

#### compute_ranking_metrics()
```python
compute_ranking_metrics(top_k: int = 10) -> Dict[str, float]
```

**Description:** Computes ranking preservation metrics

**Parameters:**
- `top_k`: Number of top features to consider

**Returns:**
- `top_k_overlap`: Proportion of top-k features that match
- `spearman_correlation`: Rank correlation
- `kendall_tau`: Alternative rank correlation

**Example:**
```python
ranking_metrics = evaluator.compute_ranking_metrics(top_k=5)
print(f"Top-5 overlap: {ranking_metrics['top_k_overlap']:.2%}")
```

---

## src.utils

**File:** `src/utils.py`

### Functions

#### load_config()
```python
load_config(config_path: str = 'config/config.yaml') -> Dict
```

**Description:** Loads configuration from YAML file

**Parameters:**
- `config_path`: Path to config file

**Returns:** Dictionary of configuration

**Example:**
```python
from src.utils import load_config

config = load_config('config/config.yaml')
print(config['random_seed'])  # 42
```

#### set_random_seed()
```python
set_random_seed(seed: int = 42) -> None
```

**Description:** Sets random seeds for reproducibility

**Parameters:**
- `seed`: Random seed value

**Side Effects:** Sets seeds for numpy, random, (and torch if available)

**Example:**
```python
from src.utils import set_random_seed

set_random_seed(42)
```

#### setup_logging()
```python
setup_logging(log_level: str = 'INFO', log_file: str = None) -> None
```

**Description:** Configures Python logging

**Parameters:**
- `log_level`: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
- `log_file`: Optional path to log file

**Example:**
```python
from src.utils import setup_logging

setup_logging(log_level='DEBUG', log_file='training.log')
```

#### save_object()
```python
save_object(obj: Any, filepath: str) -> None
```

**Description:** Saves Python object using joblib

**Parameters:**
- `obj`: Object to save
- `filepath`: Path to save file

#### load_object()
```python
load_object(filepath: str) -> Any
```

**Description:** Loads Python object from disk

**Parameters:**
- `filepath`: Path to saved file

**Returns:** Loaded object

#### ensure_dir()
```python
ensure_dir(directory: str) -> None
```

**Description:** Creates directory if it doesn't exist

**Parameters:**
- `directory`: Directory path

**Example:**
```python
from src.utils import ensure_dir

ensure_dir('results/my_experiment/')
```

#### format_time()
```python
format_time(seconds: float) -> str
```

**Description:** Formats time in human-readable format

**Parameters:**
- `seconds`: Time in seconds

**Returns:** Formatted string (e.g., "2m 34s", "1h 15m 23s")

**Example:**
```python
from src.utils import format_time

print(format_time(154.3))  # "2m 34s"
```

#### print_dict()
```python
print_dict(d: Dict, indent: int = 0) -> None
```

**Description:** Pretty-prints nested dictionaries

**Parameters:**
- `d`: Dictionary to print
- `indent`: Indentation level

**Example:**
```python
from src.utils import print_dict

metrics = {'r2': 0.972, 'mse': 0.0008}
print_dict(metrics)
```

---

## Type Hints Summary

**Common Types Used:**
```python
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional

# Feature matrix
X: np.ndarray  # Shape: (n_samples, n_features)

# Labels
y: np.ndarray  # Shape: (n_samples,)

# SHAP values
shap_values: np.ndarray  # Shape: (n_samples, n_features)

# Feature names
feature_names: List[str]

# Metrics
metrics: Dict[str, float]

# Config
config: Dict[str, Any]
```

---

## Error Handling

**Common Exceptions:**

```python
# ValueError: Invalid input
if dataset_name not in ['california_housing', 'breast_cancer', 'adult']:
    raise ValueError(f"Unknown dataset: {dataset_name}")

# FileNotFoundError: Model not found
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Model file not found: {filepath}")

# RuntimeError: Training failed
if not hasattr(self, 'model') or self.model is None:
    raise RuntimeError("Model must be trained before prediction")

# AssertionError: Shape mismatch
assert X.shape[1] == len(feature_names), "Feature count mismatch"
```

---

## Code Examples by Use Case

### Use Case 1: Simple Prediction Pipeline
```python
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel

# Load and predict
loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, names = loader.get_data()

model = BlackBoxModel('xgboost', 'regression')
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

### Use Case 2: Full InstaSHAP Pipeline
```python
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel
from src.shap_computation import SHAPComputer
from src.gam_surrogate import GAMSurrogate
from src.evaluation import Evaluator

# 1. Load data
loader = DataLoader('breast_cancer')
X_train, X_test, y_train, y_test, features = loader.get_data()

# 2. Train black-box
model = BlackBoxModel('random_forest', 'classification')
model.train(X_train, y_train)

# 3. Compute SHAP
shap_comp = SHAPComputer(model, X_train, 'classification')
shap_train = shap_comp.compute_shap_values(X_train[:1000])
shap_test = shap_comp.compute_shap_values(X_test[:500])

# 4. Train GAM
gam = GAMSurrogate(feature_names=features)
gam.train(X_train[:1000], shap_train)

# 5. Evaluate
shap_pred = gam.predict(X_test[:500])
evaluator = Evaluator(shap_test, shap_pred, features)
metrics = evaluator.evaluate()
evaluator.plot_true_vs_pred()
```

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Full Code:** [GitHub Repository](https://github.com/your-repo/instashap-replication)
