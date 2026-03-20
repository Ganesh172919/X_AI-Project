# Core Concepts

This document explains the key ideas behind InstaSHAP so you can understand
the project even if you are new to explainable AI (XAI).

---

## 1. Black-Box Models

A **black-box model** is any machine learning model whose internal logic is
hard to inspect directly. Examples used in this project:

| Model | Library | Why it is "black-box" |
|-------|---------|-----------------------|
| Random Forest | scikit-learn | Hundreds of decision trees vote together |
| XGBoost | xgboost | Thousands of boosted trees in sequence |
| LightGBM | lightgbm | Same idea, different leaf-wise growth |

These models give accurate predictions but do not tell you *why* a
particular prediction was made. That is the problem SHAP solves.

---

## 2. Feature Attributions

A **feature attribution** answers: "How much did each input feature
contribute to this specific prediction?"

```
Prediction = base_value + contribution_1 + contribution_2 + ... + contribution_n
```

Each `contribution_i` is the attribution for feature *i*.

---

## 3. Shapley Values (the theory)

Shapley values come from cooperative game theory (Lloyd Shapley, 1953).

**Analogy:** A team of players wins a prize. How do you fairly split the
prize among the players based on their individual contributions?

**Formal definition:** For feature *i*, the Shapley value is the average
marginal contribution of feature *i* across all possible subsets of
features:

```
phi_i = sum over all subsets S not containing i of:
        |S|! * (n - |S| - 1)! / n! * [f(S union {i}) - f(S)]
```

Where:
- `phi_i` = Shapley value for feature *i*
- `S` = a subset of features (not containing *i*)
- `f(S)` = model prediction using only features in *S*
- `n` = total number of features

**Key properties** guaranteed by Shapley values:

| Property | Meaning |
|----------|---------|
| Efficiency | Attributions sum to the prediction difference |
| Symmetry | Two features with equal marginal contributions get equal attributions |
| Dummy | A feature that never changes the prediction gets zero attribution |
| Additivity | Attributions for a sum of models equal the sum of attributions |

---

## 4. SHAP (the library)

Computing exact Shapley values requires evaluating the model on every
possible subset of features -- exponential in the number of features. The
**SHAP library** (Lundberg & Lee, 2017) provides fast algorithms:

| Explainer | When to use | Speed |
|-----------|-------------|-------|
| `TreeExplainer` | Tree-based models (RF, XGBoost, LightGBM) | Fast -- polynomial time |
| `KernelExplainer` | Any model (model-agnostic) | Slow -- samples coalitions |
| `LinearExplainer` | Linear / logistic regression | Fast |

In this project we primarily use `TreeExplainer` because our black-box
models are tree-based.

**The bottleneck:** Even TreeExplainer takes noticeable time on large
datasets (seconds to minutes per batch). This is where InstaSHAP comes in.

---

## 5. Generalized Additive Models (GAMs)

A GAM predicts a target by summing independent functions of each feature:

```
y = f_1(x_1) + f_2(x_2) + ... + f_n(x_n) + bias
```

Each `f_i` is a learned shape function (typically a smooth curve or
piecewise-constant step function). Because each feature is handled
separately, GAMs are **inherently interpretable** -- you can plot each
`f_i` and see exactly how that feature influences the prediction.

**Explainable Boosting Machines (EBMs)** are a specific GAM implementation
from Microsoft's InterpretML library. They use gradient boosting to learn
the shape functions, giving GAM accuracy with tree-model performance.

---

## 6. InstaSHAP -- the key idea

The core insight of InstaSHAP is:

> SHAP values themselves can be predicted by an interpretable model.

Instead of computing expensive Shapley values at runtime, **train a GAM to
approximate them once**, then use the GAM at prediction time.

### How it works

```
Step 1: Train black-box model M on data (X, y)
Step 2: Compute exact SHAP values for a training subset
           phi = TreeExplainer(M).shap_values(X_train)
Step 3: For each feature i, train a GAM:
           GAM_i : X  -->  phi_i
Step 4: At runtime, predict SHAP instantly:
           predicted_phi = [GAM_1(X), GAM_2(X), ..., GAM_n(X)]
```

### Why it works

- SHAP values are smooth functions of the input features (for tree models).
- GAMs are good at learning smooth, additive relationships.
- The mapping from features to SHAP values is often close to additive,
  so a GAM with no interaction terms captures most of the signal.

### Speed advantage

| Method | Time complexity | Typical speed |
|--------|----------------|---------------|
| Exact SHAP (TreeExplainer) | O(TL) per sample | Seconds |
| InstaSHAP (GAM prediction) | O(n * B) per sample | Milliseconds |

Where T = number of trees, L = max depth, n = features, B = bins per GAM.

The result is **40-100x speedup** with **R² > 0.95** accuracy.

---

## 7. Evaluation metrics

To verify the GAM surrogates are faithful, we measure:

### Accuracy metrics (predicted SHAP vs. true SHAP)

| Metric | What it measures | Ideal value |
|--------|-----------------|-------------|
| MSE | Average squared error | 0 |
| MAE | Average absolute error | 0 |
| RMSE | Root of MSE | 0 |
| R^2 | Variance explained | 1.0 |
| MAPE | Mean absolute percentage error | 0% |
| Pearson r | Linear correlation | 1.0 |
| Spearman rho | Rank correlation | 1.0 |

### Speed metrics

| Metric | What it measures |
|--------|-----------------|
| Exact time | Seconds for TreeExplainer |
| Surrogate time | Seconds for GAM prediction |
| Speedup factor | Exact / Surrogate |
| Per-sample latency | Milliseconds per sample |

### Ranking metrics

| Metric | What it measures |
|--------|-----------------|
| Top-k overlap | How many of the top-k most important features match |
| Top-k overlap ratio | Overlap count / k |
| Ranking correlation | Spearman correlation of feature importance vectors |

---

## 8. Putting it all together

```
+--------------+     +------------------+     +------------------+
|  Dataset      |---->|  Black-Box Model  |---->|  TreeSHAP        |
|  (X, y)       |     |  (RF/XGB/LGBM)   |     |  (exact values)  |
+--------------+     +------------------+     +--------+---------+
                                                       |
                                                       v
                      +------------------+     +------------------+
                      |  GAM Surrogates   |<----|  SHAP values     |
                      |  (one per feature)|     |  (ground truth)  |
                      +--------+---------+     +------------------+
                               |
                               v
                      +------------------+
                      |  Instant SHAP     |
                      |  Prediction       |
                      +------------------+
```

This is the full pipeline implemented in `scripts/main.py`.

---

## 9. Datasets used in this project

| Dataset | Task | Features | Samples | Description |
|---------|------|----------|---------|-------------|
| California Housing | Regression | 8 | 20,640 | Predict median house values |
| Breast Cancer | Classification | 30 | 569 | Classify tumors as malignant/benign |
| Adult Income | Classification | 14 | ~48,000 | Predict income >$50K |

---

## 10. Key hyperparameters

### GAM Surrogate (SHAPSurrogate)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 5000 | Maximum boosting rounds per GAM |
| `max_bins` | 256 | Feature discretization bins |
| `interactions` | 0 | Pairwise interactions (0 = pure additive) |
| `learning_rate` | 0.01 | Boosting step size |
| `min_samples_leaf` | 2 | Minimum samples per leaf |

### SHAP Computation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_sample_size` | 1000 | Samples for GAM training |
| `test_sample_size` | 500 | Samples for evaluation |
| `background_size` | 100 | Background dataset for KernelExplainer |

---

## References

- Lundberg & Lee. "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017.
- Caruana et al. "Intelligible Models for Classification and Regression." KDD 2015.
- Nori et al. "InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly." ICLR 2025.
- SHAP documentation: https://shap.readthedocs.io/
- InterpretML documentation: https://interpret.ml/
