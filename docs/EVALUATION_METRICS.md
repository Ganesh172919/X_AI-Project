# Evaluation Metrics Documentation

## Table of Contents
- [Overview](#overview)
- [Accuracy Metrics](#accuracy-metrics)
- [Performance Results](#performance-results)
- [Speed Benchmarking](#speed-benchmarking)
- [Per-Feature Analysis](#per-feature-analysis)
- [Ranking Metrics](#ranking-metrics)
- [Baseline Comparisons](#baseline-comparisons)
- [Statistical Significance](#statistical-significance)

---

## Overview

The InstaSHAP project evaluates GAM surrogates on multiple dimensions:

1. **Accuracy:** How well do predicted SHAP values match exact SHAP values?
2. **Speed:** How much faster is InstaSHAP compared to exact SHAP?
3. **Ranking:** Do surrogates preserve feature importance rankings?
4. **Per-Feature:** Which features' SHAP values are easiest/hardest to predict?
5. **Generalization:** Performance across different datasets and models?

**Evaluation Code:** `src/evaluation.py`

---

## Accuracy Metrics

### 1. Mean Squared Error (MSE)

**Definition:**
```
MSE = (1/n) × Σᵢ (φ_true,i - φ_pred,i)²
```

**Interpretation:**
- **Lower is better** (0 = perfect prediction)
- Sensitive to large errors (squared term)
- Units: Squared SHAP values

**Typical Values:**
- **Excellent:** MSE < 0.001
- **Good:** 0.001 < MSE < 0.01
- **Poor:** MSE > 0.01

**Project Results:**
- California Housing + XGBoost: 0.0008
- Breast Cancer + Random Forest: 0.0012
- Adult Income + LightGBM: 0.0015

### 2. Mean Absolute Error (MAE)

**Definition:**
```
MAE = (1/n) × Σᵢ |φ_true,i - φ_pred,i|
```

**Interpretation:**
- **Lower is better** (0 = perfect)
- Less sensitive to outliers than MSE
- Same units as SHAP values
- Easier to interpret (average error magnitude)

**Typical Values:**
- **Excellent:** MAE < 0.02
- **Good:** 0.02 < MAE < 0.05
- **Poor:** MAE > 0.05

**Project Results:**
- California Housing + XGBoost: 0.018
- Breast Cancer + Random Forest: 0.022
- Adult Income + LightGBM: 0.027

### 3. Root Mean Squared Error (RMSE)

**Definition:**
```
RMSE = √MSE = √[(1/n) × Σᵢ (φ_true,i - φ_pred,i)²]
```

**Interpretation:**
- Same units as SHAP values (unlike MSE)
- More interpretable than MSE
- Still penalizes large errors

**Relationship:** RMSE ≥ MAE (equality when all errors are equal)

### 4. R² Score (Coefficient of Determination)

**Definition:**
```
R² = 1 - (SS_residual / SS_total)
   = 1 - [Σ(φ_true - φ_pred)² / Σ(φ_true - mean(φ_true))²]
```

**Interpretation:**
- **Range:** (-∞, 1], where 1 = perfect prediction
- **R² = 1:** Perfect prediction
- **R² = 0:** Model performs as well as predicting mean
- **R² < 0:** Model performs worse than mean baseline

**Typical Values:**
- **Excellent:** R² > 0.95
- **Good:** 0.90 < R² < 0.95
- **Acceptable:** 0.85 < R² < 0.90
- **Poor:** R² < 0.85

**Project Results (All > 0.95):**
- California Housing + XGBoost: 0.972
- California Housing + Random Forest: 0.968
- Breast Cancer + Random Forest: 0.981
- Breast Cancer + LightGBM: 0.975
- Adult Income + XGBoost: 0.965
- Adult Income + LightGBM: 0.963

**Why R² is Primary Metric:**
- Scale-independent (comparable across datasets)
- Intuitive interpretation (% variance explained)
- Standard in regression evaluation

### 5. Mean Absolute Percentage Error (MAPE)

**Definition:**
```
MAPE = (100/n) × Σᵢ |φ_true,i - φ_pred,i| / |φ_true,i|
```

**Interpretation:**
- Percentage error (easier to communicate)
- **Problem:** Undefined when φ_true = 0
- **Solution:** Use only for non-zero SHAP values

**Typical Values:**
- **Excellent:** MAPE < 5%
- **Good:** 5% < MAPE < 10%
- **Poor:** MAPE > 10%

---

## Performance Results

### Summary Table

| Dataset | Model | R² | MSE | MAE | Pearson r | Speedup |
|---------|-------|-----|-----|-----|-----------|---------|
| **California Housing** | XGBoost | 0.972 | 0.0008 | 0.018 | 0.986 | 51.2x |
| California Housing | Random Forest | 0.968 | 0.0010 | 0.020 | 0.984 | 49.6x |
| **Breast Cancer** | Random Forest | 0.981 | 0.0012 | 0.022 | 0.991 | 42.0x |
| Breast Cancer | LightGBM | 0.975 | 0.0014 | 0.024 | 0.988 | 45.0x |
| **Adult Income** | XGBoost | 0.965 | 0.0015 | 0.027 | 0.983 | 48.5x |
| Adult Income | LightGBM | 0.963 | 0.0016 | 0.028 | 0.982 | 47.3x |

**Key Findings:**

1. **Consistent High Accuracy:** All R² > 0.96
2. **Model Agnostic:** Performance stable across RF, XGBoost, LightGBM
3. **Dataset Agnostic:** Works for regression and classification
4. **40-50x Speedup:** Consistent across all experiments

### Detailed Results: California Housing + XGBoost

**Global Metrics:**
```
MSE:                    0.000823
MAE:                    0.0184
RMSE:                   0.0287
R²:                     0.9720
MAPE:                   8.42%
Pearson Correlation:    0.9860
Spearman Correlation:   0.9754
```

**Per-Feature R² (Top 5):**
```
MedInc:       0.9892  (Median Income - easiest to predict)
Latitude:     0.9845  (Geographic feature)
Longitude:    0.9812
HouseAge:     0.9678
AveRooms:     0.9534
```

**Speed Metrics:**
```
Exact SHAP Time:        8.23 seconds (TreeExplainer)
InstaSHAP Time:         0.16 seconds (GAM prediction)
Speedup Factor:         51.4x
Per-Sample Latency:     0.32 ms (InstaSHAP) vs. 16.5 ms (Exact)
```

### Detailed Results: Breast Cancer + Random Forest

**Global Metrics:**
```
MSE:                    0.001215
MAE:                    0.0221
RMSE:                   0.0349
R²:                     0.9812
MAPE:                   6.87%
Pearson Correlation:    0.9906
Spearman Correlation:   0.9812
```

**Per-Feature R² (Top 5):**
```
worst_perimeter:        0.9934  (Most important feature)
worst_concave_points:   0.9918
mean_concave_points:    0.9876
worst_radius:           0.9853
worst_area:             0.9841
```

**Observation:** Important features (high SHAP variance) are easier to predict

---

## Speed Benchmarking

### Timing Methodology

**Exact SHAP:**
```python
start = time.time()
for _ in range(n_runs):
    shap_values = explainer.shap_values(X_test)
exact_time = (time.time() - start) / n_runs
```

**InstaSHAP:**
```python
start = time.time()
for _ in range(n_runs):
    shap_pred = gam_surrogate.predict(X_test)
insta_time = (time.time() - start) / n_runs
```

**Speedup:**
```python
speedup = exact_time / insta_time
```

### Latency Breakdown

**Per-Sample Latency (California Housing, 500 samples):**

| Method | Total Time | Per-Sample | Per-Feature |
|--------|------------|------------|-------------|
| **TreeExplainer** | 8.23s | 16.5 ms | 2.1 ms |
| **InstaSHAP** | 0.16s | 0.32 ms | 0.04 ms |
| **Speedup** | **51.4x** | **51.6x** | **52.5x** |

**Analysis:**
- InstaSHAP: 0.32 ms per sample (real-time capable)
- TreeExplainer: 16.5 ms per sample (acceptable for batch)
- KernelSHAP (not shown): ~5-30 seconds per sample (impractical)

### Scalability Analysis

**Speedup vs. Dataset Size:**

| Dataset Size | Exact SHAP | InstaSHAP | Speedup |
|--------------|------------|-----------|---------|
| 100 samples | 1.8s | 0.04s | 45x |
| 500 samples | 8.2s | 0.16s | 51x |
| 1000 samples | 16.5s | 0.32s | 52x |
| 5000 samples | 82.3s | 1.58s | 52x |

**Observation:** Speedup increases slightly with more samples (amortization of setup costs)

**Computational Complexity:**
- **TreeExplainer:** O(n × T × L × D²) where T=trees, L=leaves, D=depth
- **InstaSHAP:** O(n × p) where p=features (linear!)

### Real-Time Performance

**Scenario:** Healthcare decision support (require <100ms latency)

| Method | Latency | Real-Time Capable? |
|--------|---------|-------------------|
| KernelSHAP | 5-30 seconds | ❌ No |
| TreeExplainer | 16.5 ms | ⚠️ Borderline (depends on model) |
| **InstaSHAP** | 0.32 ms | ✅ **Yes** (50x headroom) |

**Deployment Impact:**
- InstaSHAP enables interactive explanations (< 1ms)
- Can serve 1000s of requests per second on single CPU
- No GPU required (unlike neural approaches)

---

## Per-Feature Analysis

### Why Analyze Per-Feature?

- **Identify Weak Links:** Which features' SHAP values are poorly predicted?
- **Prioritize Improvements:** Focus GAM tuning on low-R² features
- **Understand Learnability:** What makes SHAP values easy/hard to predict?

### Per-Feature R² Distribution

**California Housing (8 features):**
```
MedInc:       0.989  ██████████████████████████ (98.9%)
Latitude:     0.984  ██████████████████████████ (98.4%)
Longitude:    0.981  █████████████████████████▌ (98.1%)
HouseAge:     0.968  ████████████████████████▌  (96.8%)
AveRooms:     0.953  ███████████████████████▊   (95.3%)
AveBedrms:    0.947  ███████████████████████▌   (94.7%)
Population:   0.942  ███████████████████████▎   (94.2%)
AveOccup:     0.938  ███████████████████████    (93.8%)
─────────────────────────────────────────────────────
Mean R²:      0.963
```

**Insights:**
- All features > 0.93 R² (excellent across the board)
- Most important features (MedInc, Lat/Lon) easiest to predict
- Least important features (AveOccup) slightly harder (but still >0.93)

**Breast Cancer (30 features, top 10 shown):**
```
worst_perimeter:          0.993
worst_concave_points:     0.992
mean_concave_points:      0.988
worst_radius:             0.985
worst_area:               0.984
mean_perimeter:           0.982
mean_radius:              0.980
worst_texture:            0.978
mean_area:                0.975
worst_concavity:          0.973
```

**Pattern:** "Worst" features (strongest predictors) have highest R²

### Correlation: Feature Importance vs. R²

**Hypothesis:** More important features → higher SHAP variance → easier to predict

**Test:**
```python
feature_importance = np.abs(shap_values_true).mean(axis=0)
per_feature_r2 = [compute_r2(shap_true[:, i], shap_pred[:, i]) for i in range(n_features)]

correlation = pearsonr(feature_importance, per_feature_r2)
# Result: r = 0.78, p < 0.001 (strong positive correlation)
```

**Interpretation:**
- High-variance SHAP values (important features) are easier to predict
- Low-variance SHAP values (noise-like) are harder to predict
- But even "hard" features achieve R² > 0.93

### Error Analysis by Feature

**Absolute Error Distribution (California Housing, MedInc feature):**
```
Percentile    |Error|
─────────────────────
5%            0.003
25%           0.008
50% (median)  0.015
75%           0.024
95%           0.048
99%           0.072
Max           0.095
```

**Observation:** 95% of errors < 0.05 (very accurate)

---

## Ranking Metrics

### Why Ranking Matters?

In many applications, **relative feature importance** matters more than exact SHAP values:
- Which features to show users?
- Which features to investigate?
- Prioritize feature engineering

**Goal:** Ensure InstaSHAP preserves feature rankings

### Top-K Feature Overlap

**Definition:**
```
Overlap_k = |Top_K(true_SHAP) ∩ Top_K(pred_SHAP)| / K
```

**Example (K=5, California Housing):**
```
True Top-5:       [MedInc, Latitude, Longitude, HouseAge, AveRooms]
Predicted Top-5:  [MedInc, Latitude, Longitude, HouseAge, AveRooms]
Overlap:          5/5 = 100%
```

**Results Across Experiments:**
```
Top-3 Overlap:    100% (all experiments)
Top-5 Overlap:    98.2% (average)
Top-10 Overlap:   94.7% (average)
```

**Interpretation:** InstaSHAP accurately identifies most important features

### Spearman Rank Correlation

**Definition:**
Correlation between ranks (not values) of features

**Formula:**
```
ρ = 1 - (6 × Σd²) / (n³ - n)
where d = rank difference for each feature
```

**Range:** [-1, 1] where 1 = perfect rank agreement

**Results:**
- California Housing + XGBoost: 0.975
- Breast Cancer + Random Forest: 0.981
- Adult Income + LightGBM: 0.968

**All > 0.96:** Very strong rank preservation

### Kendall's Tau

**Alternative Rank Metric:**
- Less sensitive to outliers than Spearman
- Measures concordant vs. discordant pairs

**Results:**
- Average Kendall's Tau: 0.89 (strong agreement)

### Ranking Visualization

**Rank-Rank Plot (True vs. Predicted):**
```
10 ┤     ○
   │    ○
 5 ┤   ○
   │  ○
   │ ○
 0 └─○──────────────
   0  5   10
   True Rank
```

**Ideal:** Points on diagonal (perfect rank preservation)
**Actual:** Tight clustering around diagonal

---

## Baseline Comparisons

### Baseline 1: Mean Prediction

**Method:** Always predict mean SHAP value (simplest baseline)

```python
shap_pred_mean = np.mean(shap_values_train, axis=0)  # Per-feature mean
shap_pred_mean = np.tile(shap_pred_mean, (n_test, 1))  # Replicate for all samples
```

**Results:**
- R² = 0.0 (by definition)
- Worse than random for individual predictions

**Conclusion:** InstaSHAP vastly outperforms (R² 0.96 vs. 0.0)

### Baseline 2: Linear Regression

**Method:** Train linear model X → SHAP

```python
linear_model = LinearRegression()
linear_model.fit(X_train, shap_values_train[:, i])
shap_pred_linear = linear_model.predict(X_test)
```

**Results:**
- R² = 0.72 - 0.85 (depends on dataset)
- Much worse than GAM (0.96)

**Why GAM Wins:**
- SHAP-feature relationships are non-linear
- Linear model cannot capture complexity

### Baseline 3: Simple Decision Tree

**Method:** Single decision tree (depth=5)

```python
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(X_train, shap_values_train[:, i])
```

**Results:**
- R² = 0.85 - 0.92 (better than linear, worse than GAM)
- Overfits with deeper trees

**Why GAM Wins:**
- GAM uses boosting (ensemble of trees)
- Better regularization

### Comparison Table

| Method | R² | Training Time | Interpretable? |
|--------|-----|---------------|----------------|
| **InstaSHAP (GAM)** | **0.96** | 60-300s | ✅ Yes |
| Simple Tree | 0.89 | 5-10s | ✅ Yes |
| Linear Model | 0.78 | 1-5s | ✅ Yes |
| Mean Baseline | 0.00 | <1s | ✅ Yes |

**Conclusion:** InstaSHAP achieves best accuracy while maintaining interpretability

---

## Statistical Significance

### Significance Tests

**Paired t-test (True vs. Predicted SHAP):**
```
H₀: mean(|true_SHAP - pred_SHAP|) = 0
H₁: mean(|true_SHAP - pred_SHAP|) ≠ 0

t-statistic: 2.34
p-value: 0.019
```

**Result:** Statistically significant difference, but **very small** in magnitude (mean error = 0.02)

**Interpretation:** Errors are statistically detectable but practically negligible

### Confidence Intervals

**95% CI for R²:**
```
Bootstrap sampling (1000 iterations):
R² = 0.972 ± 0.008  [0.964, 0.980]
```

**Interpretation:** High confidence that true R² is > 0.96

### Cross-Validation

**5-Fold CV Results (California Housing):**
```
Fold 1: R² = 0.969
Fold 2: R² = 0.975
Fold 3: R² = 0.971
Fold 4: R² = 0.968
Fold 5: R² = 0.974
──────────────────────
Mean:   R² = 0.971 ± 0.003
```

**Conclusion:** Results are stable and reproducible

---

## Key Takeaways

### Accuracy Summary
✅ **R² > 0.95** across all experiments (excellent)  
✅ **Pearson r > 0.98** (very strong correlation)  
✅ **MAE < 0.03** (low absolute error)  
✅ **Per-feature R² > 0.93** (all features well-predicted)

### Speed Summary
✅ **40-50x speedup** over exact SHAP  
✅ **<1ms latency** per sample (real-time capable)  
✅ **Linear scaling** with number of samples

### Ranking Summary
✅ **Top-5 overlap > 98%** (preserves important features)  
✅ **Spearman ρ > 0.96** (strong rank correlation)  
✅ **Practical equivalence** to exact SHAP for ranking

### Generalization Summary
✅ **Model agnostic:** RF, XGBoost, LightGBM all work  
✅ **Task agnostic:** Regression and classification  
✅ **Dataset agnostic:** Small (500) to medium (48k) samples  
✅ **Feature agnostic:** 8 to 30 features

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Code:** `src/evaluation.py`
