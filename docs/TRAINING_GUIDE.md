# Training Guide

## Table of Contents
- [Overview](#overview)
- [Training Workflow](#training-workflow)
- [Black-Box Model Training](#black-box-model-training)
- [GAM Surrogate Training](#gam-surrogate-training)
- [Hyperparameter Configuration](#hyperparameter-configuration)
- [Hardware Requirements](#hardware-requirements)
- [Training Best Practices](#training-best-practices)
- [Monitoring and Debugging](#monitoring-and-debugging)

---

## Overview

The InstaSHAP project involves training two types of models:

1. **Black-Box Model:** The primary predictive model (Random Forest, XGBoost, or LightGBM)
2. **GAM Surrogates:** Meta-models that learn to predict SHAP values

This guide provides comprehensive instructions for training both model types, tuning hyperparameters, and optimizing performance.

---

## Training Workflow

### Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Phase 1: Data Preparation (1-5 seconds)                │
├─────────────────────────────────────────────────────────┤
│  • Load dataset from sklearn/OpenML                     │
│  • Apply preprocessing (encoding, scaling, splitting)   │
│  • Validate data quality                                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 2: Black-Box Model Training (10-120 seconds)     │
├─────────────────────────────────────────────────────────┤
│  • Initialize model with hyperparameters                │
│  • Fit model on training data                           │
│  • Evaluate on test set                                 │
│  • Save trained model                                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 3: SHAP Computation (30-300 seconds)             │
├─────────────────────────────────────────────────────────┤
│  • Sample 1000 training instances                       │
│  • Initialize TreeExplainer                             │
│  • Compute exact SHAP values                            │
│  • Cache SHAP values to disk                            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 4: GAM Surrogate Training (60-300 seconds)       │
├─────────────────────────────────────────────────────────┤
│  • For each feature (n times):                          │
│    - Train EBM on (X → SHAP_i)                          │
│    - Save GAM_i model                                   │
│  • Validate surrogate accuracy                          │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  Phase 5: Evaluation (5-30 seconds)                     │
├─────────────────────────────────────────────────────────┤
│  • Predict SHAP values on test set                      │
│  • Compute evaluation metrics                           │
│  • Generate visualizations                              │
│  • Save results                                         │
└─────────────────────────────────────────────────────────┘

Total Training Time: 2-10 minutes (depending on dataset size)
```

### Quick Start: Single Experiment

```bash
# Train on California Housing with XGBoost
python scripts/main.py --dataset california_housing --model-type xgboost

# Train on Breast Cancer with Random Forest
python scripts/main.py --dataset breast_cancer --model-type random_forest

# Train on Adult Income with LightGBM
python scripts/main.py --dataset adult --model-type lightgbm
```

---

## Black-Box Model Training

### Training Process

#### Step 1: Model Initialization

**Code Location:** `src/black_box_model.py:BlackBoxModel.__init__()`

```python
from src.black_box_model import BlackBoxModel

# Initialize model
model = BlackBoxModel(
    model_type='xgboost',    # 'random_forest', 'xgboost', 'lightgbm'
    task='regression',        # 'classification' or 'regression'
    **hyperparameters         # Model-specific hyperparameters
)
```

**What Happens:**
- Loads hyperparameters from config file or kwargs
- Initializes sklearn/xgboost/lightgbm model object
- Sets random seed for reproducibility
- Configures verbose logging (if enabled)

#### Step 2: Model Training

**Code Location:** `src/black_box_model.py:BlackBoxModel.train()`

```python
# Train model
model.train(X_train, y_train)
```

**Training Details:**

**Random Forest:**
```python
# sklearn.ensemble.RandomForestClassifier/Regressor
model.fit(X_train, y_train)

# Training process:
# 1. Generate n_estimators bootstrap samples
# 2. For each bootstrap:
#    - Train decision tree with random feature subset at each split
#    - Grow tree to max_depth or min_samples_leaf
# 3. Store all trees in ensemble
```

**XGBoost:**
```python
# xgboost.XGBClassifier/Regressor
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Optional: for early stopping
    verbose=False
)

# Training process:
# 1. Initialize with base prediction (mean or log-odds)
# 2. For iteration t = 1 to n_estimators:
#    - Compute gradients and Hessians
#    - Train shallow tree on gradients
#    - Update ensemble with learning_rate shrinkage
#    - Check early stopping condition (if enabled)
```

**LightGBM:**
```python
# lightgbm.LGBMClassifier/Regressor
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=0
)

# Training process:
# 1. Discretize features into histograms (max_bins)
# 2. For iteration t = 1 to n_estimators:
#    - Sample data using GOSS (gradient-based one-side sampling)
#    - Build tree leaf-wise (split leaf with max gain)
#    - Limit tree size with num_leaves
#    - Update ensemble
```

**Training Time:**
- **Random Forest:** 10-60 seconds (depends on n_trees × max_depth)
- **XGBoost:** 15-90 seconds (depends on n_estimators × data size)
- **LightGBM:** 5-30 seconds (fastest due to histogram optimization)

#### Step 3: Model Evaluation

**Code Location:** `src/black_box_model.py:BlackBoxModel.evaluate()`

```python
# Evaluate model
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

**Metrics Computed:**

**Classification:**
```python
{
    'accuracy': 0.95,           # (TP + TN) / Total
    'f1_score': 0.94,           # 2 × (precision × recall) / (precision + recall)
    'roc_auc': 0.98,            # Area under ROC curve
    'precision': 0.93,          # TP / (TP + FP)
    'recall': 0.95              # TP / (TP + FN)
}
```

**Regression:**
```python
{
    'mse': 0.35,                # Mean Squared Error
    'rmse': 0.59,               # Root Mean Squared Error
    'mae': 0.42,                # Mean Absolute Error
    'r2': 0.82                  # R² (coefficient of determination)
}
```

#### Step 4: Model Serialization

```python
# Save model
model.save('models/california_housing_xgboost.pkl')

# Load model later
model = BlackBoxModel.load('models/california_housing_xgboost.pkl')
```

**Storage Format:** joblib (compressed pickle)

**File Sizes:**
- Random Forest: 10-50 MB (stores all trees)
- XGBoost: 5-20 MB (optimized storage)
- LightGBM: 3-15 MB (most compact)

---

## GAM Surrogate Training

### Training Process

#### Step 1: SHAP Value Generation

**Code Location:** `src/shap_computation.py:SHAPComputer.compute_shap_values()`

```python
from src.shap_computation import SHAPComputer

# Initialize SHAP computer
shap_computer = SHAPComputer(
    model=black_box_model,
    data=X_train,
    task='regression'
)

# Compute SHAP values
shap_values_train = shap_computer.compute_shap_values(
    X_train[:1000],           # Sample 1000 instances
    method='tree'             # Use TreeExplainer
)
```

**What Happens:**
1. **Initialization:**
   - Select SHAP explainer (TreeExplainer for tree models)
   - Sample background data (100 instances)
   - Set up caching mechanism

2. **Computation:**
   - For each instance in X_train[:1000]:
     - Compute SHAP values using TreeExplainer
     - Store in matrix (1000 × n_features)
   - Save to cache (`data/{dataset}/shap_values/train_shap.npy`)

**Computation Time:**
- **TreeExplainer:** 30-180 seconds (1000 samples)
- **KernelSHAP:** 5-30 minutes (1000 samples) - much slower
- **Caching Benefit:** Subsequent runs use cached values (instant)

#### Step 2: GAM Initialization

**Code Location:** `src/gam_surrogate.py:GAMSurrogate.__init__()`

```python
from src.gam_surrogate import GAMSurrogate

# Initialize GAM surrogate
gam_surrogate = GAMSurrogate(
    feature_names=feature_names,
    max_iter=5000,           # Total boosting rounds
    max_bins=256,            # Discretization bins
    learning_rate=0.01,      # Shrinkage factor
    interactions=0           # Pure additive (no interactions)
)
```

**Configuration:**
- **max_iter:** Controls model complexity (more iterations = better fit, longer training)
- **max_bins:** Feature discretization resolution (more bins = finer granularity)
- **learning_rate:** Step size for boosting (smaller = more conservative)
- **interactions:** Number of feature pairs to consider (0 = pure additive GAM)

#### Step 3: Training Individual GAMs

**Code Location:** `src/gam_surrogate.py:GAMSurrogate.train()`

```python
# Train GAM surrogates
gam_surrogate.train(
    X_train=X_train[:1000],           # Training features
    shap_values_train=shap_values_train  # Target SHAP values
)
```

**Training Loop:**
```python
for i, feature_name in enumerate(feature_names):
    print(f"Training GAM for feature {i+1}/{len(feature_names)}: {feature_name}")
    
    # Extract SHAP values for this feature
    y_target = shap_values_train[:, i]
    
    # Initialize EBM
    ebm = ExplainableBoostingRegressor(
        max_rounds=5000,
        max_bins=256,
        learning_rate=0.01,
        interactions=0,
        random_state=42
    )
    
    # Train EBM
    ebm.fit(X_train, y_target)
    
    # Store in dictionary
    self.gam_models[feature_name] = ebm
    
    # Log training R²
    train_pred = ebm.predict(X_train)
    train_r2 = r2_score(y_target, train_pred)
    print(f"  Training R²: {train_r2:.4f}")
```

**Training Details:**

**EBM Cyclic Boosting:**
```
Initialize: f₁, f₂, ..., fₙ = 0 (all shape functions start at zero)

For round t = 1 to max_iter:
    For feature j = 1 to n_features:
        # Compute residuals
        residuals = y_target - current_prediction
        
        # Fit small tree to feature j vs. residuals
        tree_j = DecisionTree(max_leaf_nodes=3)
        tree_j.fit(X[:, j], residuals)
        
        # Update shape function for feature j
        f_j = f_j + learning_rate × tree_j
        
        # Update predictions
        current_prediction += learning_rate × tree_j.predict(X[:, j])

Final model: ŷ = β₀ + Σ f_j(x_j)
```

**Per-Feature Training Time:**
- **Small datasets (n<5k):** 3-10 seconds per feature
- **Medium datasets (n=5k-20k):** 10-30 seconds per feature
- **Large datasets (n>20k):** 30-120 seconds per feature

**Total GAM Training Time:**
- **California Housing (8 features):** 60-120 seconds
- **Breast Cancer (30 features):** 180-300 seconds
- **Adult Income (14 features):** 120-240 seconds

#### Step 4: GAM Validation

```python
# Evaluate GAM on training data
train_metrics = gam_surrogate.evaluate(X_train, shap_values_train)

print(f"Training R²: {train_metrics['r2']:.4f}")
print(f"Training MSE: {train_metrics['mse']:.6f}")
```

**Quality Indicators:**
- **R² > 0.95:** Excellent fit (GAM captures SHAP patterns well)
- **R² = 0.90-0.95:** Good fit (acceptable for most applications)
- **R² < 0.90:** Poor fit (consider increasing max_iter or adjusting hyperparameters)

#### Step 5: GAM Serialization

```python
# Save GAM surrogates
gam_surrogate.save('models/gam_surrogate_california_housing.pkl')

# Load later
gam_surrogate = GAMSurrogate.load('models/gam_surrogate_california_housing.pkl')
```

**Storage Format:** joblib dictionary of EBM models

**File Sizes:**
- **Small models (8 features):** 5-15 MB
- **Medium models (14 features):** 10-30 MB
- **Large models (30 features):** 20-60 MB

---

## Hyperparameter Configuration

### Configuration File Structure

**Location:** `config/config.yaml`

```yaml
# Random seed for reproducibility
random_seed: 42

# Black-box model hyperparameters
black_box_models:
  random_forest:
    classification:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 5
      min_samples_leaf: 2
      max_features: 'sqrt'
      random_state: 42
    regression:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 5
      min_samples_leaf: 2
      max_features: 'sqrt'
      random_state: 42
  
  xgboost:
    classification:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
    regression:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
  
  lightgbm:
    classification:
      n_estimators: 100
      max_depth: 10
      learning_rate: 0.1
      num_leaves: 31
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
    regression:
      n_estimators: 100
      max_depth: 10
      learning_rate: 0.1
      num_leaves: 31
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42

# GAM surrogate hyperparameters
gam_config:
  max_iter: 5000            # Total boosting rounds (divided among features)
  max_bins: 256             # Feature discretization bins
  interactions: 0           # 0 = pure additive, >0 = include interactions
  learning_rate: 0.01       # Step size for boosting
  min_samples_leaf: 2       # Minimum samples per bin

# SHAP computation settings
shap_config:
  train_sample_size: 1000   # SHAP samples for training GAM
  test_sample_size: 500     # SHAP samples for evaluation
  background_size: 100      # Background data for SHAP explainer
  check_additivity: false   # Validate SHAP additivity (slow)
```

### Hyperparameter Tuning Guidelines

#### Black-Box Model Hyperparameters

**Random Forest:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_estimators` | 100 | 50-500 | More trees → better performance, slower training |
| `max_depth` | 10 | 5-20 | Deeper → more complex, risk overfitting |
| `min_samples_split` | 5 | 2-20 | Higher → smoother trees, less overfitting |
| `min_samples_leaf` | 2 | 1-10 | Higher → smoother predictions |
| `max_features` | sqrt | sqrt, log2, 0.5 | Lower → more diversity, less overfitting |

**Tuning Strategy:**
1. Start with defaults
2. If underfitting: Increase `n_estimators`, `max_depth`
3. If overfitting: Increase `min_samples_split`, `min_samples_leaf`

**XGBoost:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_estimators` | 100 | 50-1000 | More rounds → better fit (use early stopping) |
| `max_depth` | 6 | 3-10 | Deeper → more interactions, risk overfitting |
| `learning_rate` | 0.1 | 0.01-0.3 | Lower → slower convergence, better generalization |
| `subsample` | 0.8 | 0.5-1.0 | Lower → less overfitting, more stochasticity |
| `colsample_bytree` | 0.8 | 0.5-1.0 | Lower → feature diversity |
| `reg_lambda` | 1.0 | 0-10 | Higher → stronger L2 regularization |

**Tuning Strategy:**
1. Start with `learning_rate=0.1`, `n_estimators=100`
2. If underfitting: Increase `n_estimators`, `max_depth`
3. If overfitting: Decrease `learning_rate`, increase `reg_lambda`
4. Enable early stopping: `early_stopping_rounds=10`

**LightGBM:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_estimators` | 100 | 50-1000 | More rounds → better fit |
| `num_leaves` | 31 | 15-63 | More leaves → more complex (use with max_depth) |
| `max_depth` | 10 | 5-15 | Limits leaf-wise growth |
| `learning_rate` | 0.1 | 0.01-0.3 | Lower → slower, better generalization |
| `min_child_samples` | 20 | 10-100 | Higher → smoother, less overfitting |

**Tuning Strategy:**
1. Set `num_leaves = 2^(max_depth) - 1` for balanced trees
2. If overfitting: Reduce `num_leaves`, increase `min_child_samples`
3. If underfitting: Increase `num_leaves`, `n_estimators`

#### GAM Surrogate Hyperparameters

| Parameter | Default | Range | Effect | Priority |
|-----------|---------|-------|--------|----------|
| `max_iter` | 5000 | 1000-10000 | More iterations → better SHAP fit | **High** |
| `max_bins` | 256 | 64-512 | More bins → finer resolution | Medium |
| `learning_rate` | 0.01 | 0.001-0.1 | Lower → slower, smoother | Medium |
| `interactions` | 0 | 0-10 | >0 includes feature interactions | Low |
| `min_samples_leaf` | 2 | 1-10 | Higher → smoother shape functions | Low |

**Tuning Strategy:**

1. **If GAM R² < 0.90:**
   - Increase `max_iter` to 10000
   - Increase `max_bins` to 512
   - Check for data quality issues

2. **If GAM training is slow:**
   - Decrease `max_iter` to 3000
   - Decrease `max_bins` to 128
   - Sample fewer SHAP values (train_sample_size: 500)

3. **If GAM overfits (training R² >> test R²):**
   - Decrease `max_iter` to 3000
   - Increase `min_samples_leaf` to 5
   - Reduce `learning_rate` to 0.005

4. **To capture feature interactions:**
   - Set `interactions: 10` (adds top 10 pairwise interactions)
   - Note: Increases training time and complexity

### Optimal Configurations by Dataset Size

**Small Datasets (n < 1000):**
```yaml
black_box: {n_estimators: 50, max_depth: 5}
gam: {max_iter: 2000, max_bins: 128}
shap: {train_sample_size: 500}
```

**Medium Datasets (1000 < n < 10000):**
```yaml
black_box: {n_estimators: 100, max_depth: 10}
gam: {max_iter: 5000, max_bins: 256}  # Default
shap: {train_sample_size: 1000}
```

**Large Datasets (n > 10000):**
```yaml
black_box: {n_estimators: 200, max_depth: 15}
gam: {max_iter: 10000, max_bins: 512}
shap: {train_sample_size: 2000}
```

---

## Hardware Requirements

### Minimum Requirements

- **CPU:** 2 cores, 2.0 GHz
- **RAM:** 4 GB
- **Storage:** 2 GB (for datasets, models, results)
- **OS:** Windows 10, macOS 10.13+, Linux (Ubuntu 18.04+)

**Performance:** Training takes 5-15 minutes

### Recommended Requirements

- **CPU:** 4+ cores, 3.0+ GHz
- **RAM:** 8+ GB
- **Storage:** 5+ GB SSD
- **OS:** Modern 64-bit OS

**Performance:** Training takes 2-5 minutes

### Optimal Requirements

- **CPU:** 8+ cores, 3.5+ GHz (Intel i7/i9, AMD Ryzen 7/9)
- **RAM:** 16+ GB
- **Storage:** 10+ GB NVMe SSD
- **GPU:** Not required (tree models are CPU-optimized)

**Performance:** Training takes 1-3 minutes

### GPU Support

**Current Status:** Not utilized (tree-based models are CPU-optimized)

**Future Extensions:**
- Neural network black-box models (require GPU)
- Deep SHAP computation
- Large-scale GAM training (GPU-accelerated EBMs)

### Parallelization

**Current:**
- Tree models use all CPU cores (`n_jobs=-1`)
- GAM training is sequential (one feature at a time)

**Potential:**
- Parallelize GAM training across features
- Distributed SHAP computation for large datasets

**Code Example (Future):**
```python
from joblib import Parallel, delayed

# Train GAMs in parallel
gam_models = Parallel(n_jobs=-1)(
    delayed(train_gam)(X_train, shap_values[:, i]) 
    for i in range(n_features)
)
```

---

## Training Best Practices

### 1. Reproducibility

**Always Set Random Seeds:**
```python
import numpy as np
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # For torch (if using neural networks)
    # torch.manual_seed(seed)
```

**Configuration:**
```yaml
random_seed: 42  # Set in config.yaml
```

**Verify Reproducibility:**
- Run same experiment twice
- Check if metrics match exactly
- If not, identify non-deterministic components

### 2. Data Quality Checks

**Before Training:**
```python
# Check for missing values
assert not X_train.isnull().any().any(), "Missing values detected"

# Check for infinite values
assert not np.isinf(X_train).any().any(), "Infinite values detected"

# Check target distribution
print(f"Target mean: {y_train.mean():.2f}")
print(f"Target std: {y_train.std():.2f}")

# Check class balance (classification)
print(f"Class distribution: {np.bincount(y_train)}")
```

### 3. Train/Test Isolation

**Never Use Test Data During Training:**
```python
# ❌ WRONG: Fit scaler on all data
scaler.fit(np.vstack([X_train, X_test]))

# ✅ CORRECT: Fit scaler on training data only
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

### 4. Model Validation

**Cross-Validation (Optional):**
```python
from sklearn.model_selection import cross_val_score

# 5-fold CV on training data
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

**Overfitting Check:**
```python
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

if train_score - test_score > 0.1:
    print("⚠️ Warning: Model may be overfitting")
    print(f"Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
```

### 5. Incremental Training

**For Large Datasets:**
```python
# Option 1: Sample for faster iteration
X_train_sample = X_train[:5000]
y_train_sample = y_train[:5000]

# Quick training for hyperparameter search
model.fit(X_train_sample, y_train_sample)

# Option 2: Early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=10,
    verbose=False
)
```

### 6. Logging and Monitoring

**Enable Detailed Logging:**
```bash
python scripts/main.py --dataset california_housing --model-type xgboost --log-level DEBUG
```

**Log Key Metrics:**
```python
import logging

logger = logging.getLogger(__name__)
logger.info(f"Training on {len(X_train)} samples")
logger.info(f"Black-box model: {model_type}")
logger.info(f"Test R²: {test_r2:.4f}")
logger.info(f"GAM training time: {gam_time:.2f} seconds")
```

### 7. Checkpoint Saving

**Save Intermediate Results:**
```python
# After each major step
model.save('checkpoints/black_box_model.pkl')
np.save('checkpoints/shap_values_train.npy', shap_values)
gam_surrogate.save('checkpoints/gam_surrogate.pkl')
```

**Resume from Checkpoint:**
```python
if os.path.exists('checkpoints/black_box_model.pkl'):
    model = BlackBoxModel.load('checkpoints/black_box_model.pkl')
else:
    model.train(X_train, y_train)
    model.save('checkpoints/black_box_model.pkl')
```

---

## Monitoring and Debugging

### Training Progress Monitoring

**Black-Box Model:**
```python
# Enable verbose mode
model = RandomForestRegressor(verbose=2)
model.fit(X_train, y_train)

# XGBoost training output
[0]    train-rmse:1.2345
[10]   train-rmse:0.8765
[20]   train-rmse:0.5432
...
```

**GAM Surrogate:**
```python
# EBM progress (automatically logged)
Training GAM for feature 1/8: MedInc
  Training R²: 0.9845
Training GAM for feature 2/8: HouseAge
  Training R²: 0.9612
...
```

### Common Issues and Solutions

#### Issue 1: Black-Box Model Poor Performance

**Symptoms:**
- Low test accuracy/R²
- High training time

**Diagnostics:**
```python
# Check data quality
print(X_train.describe())
print(y_train.describe())

# Check for class imbalance
print(np.bincount(y_train))  # Classification only
```

**Solutions:**
- Increase model complexity: `max_depth`, `n_estimators`
- Check feature scaling: All features should have similar ranges
- Try different model type: XGBoost often outperforms Random Forest

#### Issue 2: GAM Surrogate Low R²

**Symptoms:**
- GAM R² < 0.90
- High MSE between true and predicted SHAP

**Diagnostics:**
```python
# Check per-feature R²
for i, feature in enumerate(feature_names):
    y_true = shap_values_test[:, i]
    y_pred = gam_surrogate.predict(X_test)[:, i]
    r2 = r2_score(y_true, y_pred)
    print(f"{feature}: R² = {r2:.4f}")
```

**Solutions:**
- Increase `max_iter` to 10000
- Increase `max_bins` to 512
- Use more SHAP training samples: `train_sample_size: 2000`
- Check SHAP computation quality (TreeExplainer should be exact)

#### Issue 3: Training Too Slow

**Symptoms:**
- Training takes > 15 minutes
- System becomes unresponsive

**Diagnostics:**
```python
import time

start = time.time()
model.fit(X_train, y_train)
print(f"Training time: {time.time() - start:.2f} seconds")
```

**Solutions:**
- Reduce dataset size: Sample training data
- Reduce model complexity: Lower `n_estimators`, `max_depth`
- Reduce SHAP sample size: `train_sample_size: 500`
- Reduce GAM iterations: `max_iter: 3000`
- Use LightGBM instead of XGBoost (faster)

#### Issue 4: Out of Memory

**Symptoms:**
- MemoryError during training
- System crashes

**Diagnostics:**
```python
import psutil

process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
```

**Solutions:**
- Sample training data: Use fewer samples
- Reduce SHAP sample size
- Use smaller `max_bins` (reduces GAM memory)
- Process features sequentially (already implemented for GAM)
- Close other applications

### Debugging Tools

**Python Debugger:**
```python
import pdb

# Set breakpoint
pdb.set_trace()

# Or use IPython
from IPython import embed
embed()
```

**Profiling:**
```bash
# Time profiling
python -m cProfile -o profile.stats scripts/main.py

# Memory profiling
python -m memory_profiler scripts/main.py
```

**Logging:**
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

---

## Example Training Sessions

### Example 1: Default Training

```bash
$ python scripts/main.py --dataset california_housing --model-type xgboost

2026-03-25 10:30:15 - INFO - Loading dataset: california_housing
2026-03-25 10:30:16 - INFO - Training samples: 16512, Test samples: 4128
2026-03-25 10:30:16 - INFO - Training XGBoost model...
2026-03-25 10:30:28 - INFO - Training complete. Time: 12.3s
2026-03-25 10:30:28 - INFO - Test RMSE: 0.523, Test R²: 0.826
2026-03-25 10:30:28 - INFO - Computing SHAP values...
2026-03-25 10:31:42 - INFO - SHAP computation complete. Time: 74.2s
2026-03-25 10:31:42 - INFO - Training GAM surrogates...
2026-03-25 10:32:58 - INFO - GAM training complete. Time: 76.5s
2026-03-25 10:32:58 - INFO - Evaluating GAM surrogates...
2026-03-25 10:33:05 - INFO - GAM R²: 0.972, Speedup: 51.2x
2026-03-25 10:33:05 - INFO - Results saved to results/california_housing_xgboost/
```

### Example 2: Quick Training (Small Sample)

```bash
$ python scripts/main.py --dataset breast_cancer --model-type random_forest

2026-03-25 10:35:10 - INFO - Loading dataset: breast_cancer
2026-03-25 10:35:10 - INFO - Training samples: 455, Test samples: 114
2026-03-25 10:35:10 - INFO - Training Random Forest model...
2026-03-25 10:35:15 - INFO - Training complete. Time: 4.8s
2026-03-25 10:35:15 - INFO - Test Accuracy: 0.965, Test F1: 0.958
2026-03-25 10:35:15 - INFO - Computing SHAP values...
2026-03-25 10:35:38 - INFO - SHAP computation complete. Time: 23.1s
2026-03-25 10:35:38 - INFO - Training GAM surrogates...
2026-03-25 10:38:42 - INFO - GAM training complete. Time: 184.3s
2026-03-25 10:38:42 - INFO - Evaluating GAM surrogates...
2026-03-25 10:38:45 - INFO - GAM R²: 0.981, Speedup: 42.0x
```

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Related Docs:** [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md), [CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)
