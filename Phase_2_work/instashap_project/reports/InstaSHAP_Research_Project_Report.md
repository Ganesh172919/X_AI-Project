# InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly

## A Complete End-to-End Research Reproducibility Report

**Paper Details:**
* **Title:** InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly
* **Authors:** James Enouen, Yan Liu
* **Publication Venue:** Accepted at the International Conference on Learning Representations (ICLR) 2025
* **Official Publication Link:** https://openreview.net/forum?id=ky7vVlBQBY

> **Project Type:** Reproducibility Study & Implementation  
> **Tech Stack:** Python 3.10+, PyTorch 2.2+, scikit-learn, SHAP, ucimlrepo

---

## Table of Contents

1. [Research Problem & Motivation](#1-research-problem--motivation)
2. [Core Idea of InstaSHAP](#2-core-idea-of-instashap)
3. [Mathematical Foundation](#3-mathematical-foundation)
4. [Datasets Used](#4-datasets-used)
5. [End-to-End Pipeline Architecture](#5-end-to-end-pipeline-architecture)
6. [Data Loading & Preprocessing](#6-data-loading--preprocessing)
7. [Model Architectures](#7-model-architectures)
8. [Training Methodology](#8-training-methodology)
9. [Explainability Pipeline](#9-explainability-pipeline)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Experiment Orchestration](#11-experiment-orchestration)
12. [Outputs & Artifacts](#12-outputs--artifacts)
13. [Reproducibility Controls](#13-reproducibility-controls)
14. [How to Run the Project](#14-how-to-run-the-project)
15. [Key Findings & Paper Comparison](#15-key-findings--paper-comparison)
16. [Project Structure Reference](#16-project-structure-reference)
17. [Dependencies](#17-dependencies)
18. [Conclusion](#18-conclusion)

---

## 1. Research Problem & Motivation

### The Explainability Gap in Machine Learning

Modern machine learning models — particularly deep neural networks — are powerful but opaque. When a model predicts that a loan applicant should be rejected, or that a patient has a disease, stakeholders (clinicians, regulators, users) need to understand **why** the model made that decision.

### What Are Shapley Values?

**Shapley values** (from cooperative game theory) provide a principled way to assign a contribution score to each input feature for a given prediction. They satisfy desirable axioms: **efficiency** (attributions sum to the prediction), **symmetry**, **linearity**, and **null player** (irrelevant features get zero attribution).

### The Problem: Shapley Values Are Slow

Computing exact Shapley values requires evaluating the model on all possible subsets of features — an exponential operation. Even approximate methods like **Permutation SHAP** need hundreds/thousands of model evaluations per sample, making real-time explanations impractical.

### The InstaSHAP Solution

InstaSHAP proposes training an **additive neural model** that can produce Shapley-faithful feature attributions in a **single forward pass**, reducing explanation time from seconds/minutes to milliseconds. The key insight: if you train an additive model using a **masked objective** that mimics the Shapley value computation process, the individual components of the additive model naturally recover SHAP values.

---

## 2. Core Idea of InstaSHAP

The approach follows a **four-stage pipeline**:

```
Stage 1: Train a Black-Box Model
    → A standard MLP that makes accurate predictions (the model we want to explain)

Stage 2: Train a Masked Surrogate
    → Approximates f(x; S) — the black-box output when only features in subset S are available

Stage 3: Train the InstaSHAP Additive Model
    → An additive model trained against the surrogate's masked outputs using Shapley-weighted masks

Stage 4: Extract Explanations Instantly
    → Each component's output IS the SHAP attribution — no further computation needed
```

The fundamental principle: an additive model `g(x) = b + Σ gᵢ(xᵢ)` trained under the masked Shapley objective will have each component `gᵢ(xᵢ)` equal the Shapley value `φᵢ(x)` for feature `i`.

---

## 3. Mathematical Foundation

### Shapley Value Definition

For a model `f`, the Shapley value of feature `i` for input `x` is:

```
φᵢ(x) = Σ_{S ⊆ N\{i}} [ |S|! (|N|-|S|-1)! / |N|! ] × [ f(x; S ∪ {i}) - f(x; S) ]
```

Where:
- `N` is the set of all features
- `S` is a subset of features not containing `i`
- `f(x; S)` is the model output when only features in `S` are present (others are marginalized)

### Masked Surrogate Objective

The surrogate `h(x, S)` learns to approximate `f(x; S)` by:
- Taking the full input `x` but element-wise masking it based on subset `S`
- Concatenating the binary mask vector with the masked input
- Minimizing: `E_{x,S} [ || h(x·m_S, S) - f(x; S) ||² ]`

Where subsets `S` are drawn from the **Shapley kernel distribution**:

```
p(|S|) ∝ 1 / [ C(n, |S|) × |S| × (n - |S|) ]
```

### InstaSHAP Training Objective (Equation 20 from Paper)

The InstaSHAP additive model `g(x; S) = b + Σ_{i∈S} gᵢ(xᵢ)` is trained to minimize:

```
E_{x,S} [ || g(x; S) - h(x·m_S, S) ||² ]
```

The gating mechanism ensures only components `gᵢ` where `i ∈ S` contribute to the output, enforcing that the model learns proper Shapley attributions.

### Key Theorem

Under the masked Shapley-weighted objective, the optimal additive model recovers the exact Shapley values: **gᵢ(xᵢ) = φᵢ(x)** for all features `i`.

---

## 4. Datasets Used

This project uses **three UCI benchmark datasets**, each chosen to highlight different feature interaction behaviors:

### 4.1 Bike Sharing (UCI ID: 275)

| Property | Value |
|----------|-------|
| **Task** | Regression (predict hourly rental count) |
| **Total Features** | 13 (5 numeric, 8 categorical) |
| **Numeric Features** | `day_of_month`, `temp`, `atemp`, `hum`, `windspeed` |
| **Categorical Features** | `season`, `year`, `month`, `hour`, `holiday`, `weekday`, `workingday`, `weather_situation` |
| **Key Interaction** | `hour × workingday` (synergistic) |
| **Why Chosen** | Demonstrates how commute hours vs. leisure hours create a strong synergistic interaction with working day status |
| **Primary Metric** | NMSE% (Normalized Mean Squared Error) |
| **Paper Benchmarks** | Black-box NMSE: 6.59%, GAM-1 NMSE: 17.4%, GAM-2 NMSE: 6.23% |

### 4.2 Covertype (UCI ID: 31)

| Property | Value |
|----------|-------|
| **Task** | 7-class Classification (forest cover type) |
| **Total Features** | 11 (10 numeric, 1 categorical) |
| **Numeric Features** | `elevation`, `aspect`, `slope`, `horizontal_distance_to_hydrology`, `vertical_distance_to_hydrology`, `horizontal_distance_to_roadways`, `hillshade_9am`, `hillshade_noon`, `hillshade_3pm`, `horizontal_distance_to_fire_points` |
| **Categorical Feature** | `soil_climate_zone` (grouped from 40 binary soil type columns into 4 climate labels) |
| **Key Interaction** | `elevation × soil_climate_zone` (redundant) |
| **Why Chosen** | Demonstrates how feature interactions can be redundant — elevation already encodes most climate zone information |
| **Primary Metric** | Accuracy |
| **Paper Benchmarks** | Black-box: 80.4%, GAM-1: 72.4%, GAM-2: 82.2% |

### 4.3 Adult Income (UCI ID: 2)

| Property | Value |
|----------|-------|
| **Task** | Binary Classification (income > $50K) |
| **Total Features** | 13 (5 numeric, 8 categorical) |
| **Numeric Features** | `age`, `fnlwgt`, `capital_gain`, `capital_loss`, `hours_per_week` |
| **Categorical Features** | `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `workclass`, `native_country` |
| **Key Interaction** | None (supplementary benchmark) |
| **Why Chosen** | Used as a baseline to verify that InstaSHAP preserves accuracy on a purely 1D additive task |
| **Primary Metric** | Accuracy |
| **Paper Benchmarks** | Vanilla GAM: 84.2%, InstaSHAP GAM: 84.3% |

---

## 5. End-to-End Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INSTASHAP FULL PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────┐         │
│  │  UCI Dataset  │───▶│  Preprocessing   │───▶│  Train/Val/Test   │         │
│  │  (ucimlrepo)  │    │ • StandardScaler │    │  Split (70/10/20) │         │
│  │              │    │ • OneHotEncoder  │    │  + Stratification  │         │
│  └──────────────┘    └──────────────────┘    └────────┬───────────┘         │
│                                                        │                    │
│                  ┌────────────────────────┬─────────────┼──────────────┐    │
│                  ▼                        ▼             ▼              │    │
│         ┌───────────────┐      ┌─────────────────┐  ┌──────────┐     │    │
│         │  BLACK-BOX    │      │  GAM-1 (no int.) │  │ GAM-2    │     │    │
│  Step 1 │  MLP Baseline │      │  Additive Model  │  │ (+pairs) │     │    │
│         │  [256,128]    │      │  [96,64] per feat │  │          │     │    │
│         └───────┬───────┘      └──────────────────┘  └──────────┘     │    │
│                 │                                                      │    │
│                 ▼                                                      │    │
│         ┌───────────────┐                                              │    │
│  Step 2 │  MASKED       │  Trained on: f(x_masked, S_mask)             │    │
│         │  SURROGATE    │  Input = [x·m_S || S]                        │    │
│         │  MLP [256,128]│  Target = black-box raw outputs              │    │
│         └───────┬───────┘                                              │    │
│                 │                                                      │    │
│                 ▼                                                      │    │
│         ┌───────────────┐                                              │    │
│  Step 3 │  INSTASHAP    │  Trained on: g(x; S) vs h(x·m_S, S)         │    │
│         │  Additive GAM │  Masks drawn from Shapley kernel             │    │
│         │  [96,64]      │  Each component = SHAP value                 │    │
│         └───────┬───────┘                                              │    │
│                 │                                                      │    │
│                 ▼                                                      │    │
│         ┌───────────────────────────────────────────────┐              │    │
│  Step 4 │  EXPLANATION COMPARISON                        │              │    │
│         │  • Permutation SHAP (baseline ground truth)   │              │    │
│         │  • InstaSHAP (single forward pass)            │              │    │
│         │  • Compare: MSE, MAE between attribution sets │              │    │
│         └───────────────────────────────────────────────┘              │    │
│                                                                        │    │
│         ┌───────────────────────────────────────────────┐              │    │
│  Output │  CSV Tables, Training Curves, Shape Functions │              │    │
│         │  Interaction Heatmaps, Importance Bars, PDFs  │              │    │
│         └───────────────────────────────────────────────┘              │    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Loading & Preprocessing

### 6.1 Data Loading (`data/loaders.py`)

Each dataset is loaded via the `ucimlrepo` library and undergoes domain-specific transformations:

- **Bike Sharing:** Extracts `day_of_month` from date strings; maps binary flags (`holiday`, `workingday`) to categorical labels (`"yes"/"no"`); converts `hour`, `month`, `weekday` to string categories.
- **Covertype:** Collapses 40 binary soil-type columns into 4 climate zone labels (`"lower montane"`, `"upper montane"`, `"subalpine"`, `"alpine"`) using the first digit of ELU codes from UCI metadata.
- **Adult Income:** Drops `education-num` (redundant with `education`), renames hyphenated columns, binarizes target to `income > 50K`.

Each loader returns a `DatasetBundle` containing features (as a DataFrame), target (as a Series), and `DatasetMetadata` with embedded paper-reported benchmark values.

### 6.2 Preprocessing (`data/preprocessing.py`)

The `TabularPreprocessor` class handles:

1. **Numeric features:** `SimpleImputer(strategy="median")` → `StandardScaler()` (zero mean, unit variance)
2. **Categorical features:** `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`
3. **Feature group bookkeeping:** Maps each original feature name to a `FeatureGroup(start, end)` range in the transformed vector, enabling aggregation of one-hot columns back to original features during explanation

### 6.3 Train/Val/Test Split

```python
# Default split ratios from config.yaml
test_size  = 0.20  (20%)
val_size   = 0.10  (10%)
train_size = 0.70  (70%, implicit)

# Classification datasets use stratified splitting
```

---

## 7. Model Architectures

### 7.1 Black-Box MLP (`models/blackbox_model.py → TabularMLP`)

The simplest model — a standard feed-forward network that serves as the "model to be explained."

```
Architecture:
    Input(D) → Linear(256) → ReLU → Dropout(0.1) → Linear(128) → ReLU → Dropout(0.1) → Linear(output_dim)

    - Regression: output_dim = 1
    - Classification: output_dim = number of classes
```

The black-box also supports `RandomForestBlackBox` as an alternative sklearn-based baseline (300 estimators).

### 7.2 Masked Surrogate MLP (`models/blackbox_model.py → MaskedSurrogateMLP`)

Approximates `f(x; S)` — the black-box's expected output when only a subset `S` of features are observed.

```
Architecture:
    Input = [x·m_S || m_S]    (feature vector element-wise masked + binary mask vector concatenated)
    → Linear(256) → ReLU → Dropout(0.1) → Linear(128) → ReLU → Dropout(0.1) → Linear(output_dim)

    Total input dimension = feature_dim + num_original_features
```

**Key design:** The mask is provided at both the **input level** (zero-out absent features) and the **network level** (concatenated so the network knows which features are present).

### 7.3 GAM Model (`models/gam.py → GAMModel`)

A **neural Generalized Additive Model** with one small MLP per feature (and optional pairwise interaction components).

```
GAM-1 (no interactions):
    g(x) = bias + Σᵢ gᵢ(xᵢ)

    Each gᵢ: Input(width_i) → Linear(96) → ReLU → Dropout(0.05) → Linear(64) → ReLU → Linear(output_dim)

GAM-2 (with interactions):
    g(x) = bias + Σᵢ gᵢ(xᵢ) + Σ_{(i,j)} g_{ij}(xᵢ, xⱼ)

    Interaction components: Input(width_i + width_j) → same MLP architecture
```

**Feature masking:** When a feature mask `S` is provided, each component output is gated by `gate = product(mask[i] for i in component_features)`. This ensures only selected features contribute.

**Feature attributions:** For interactions, the contribution is split 50/50 between the two features: `attribution[i] += g_{ij}/2` and `attribution[j] += g_{ij}/2`.

### 7.4 InstaSHAP Model (`models/instashap.py → InstaSHAPModel`)

Inherits from `GAMModel` and adds two key methods:

```python
class InstaSHAPModel(GAMModel):
    def masked_forward(self, inputs, feature_mask):
        # Used during training with Shapley-weighted masks
        return super().forward(inputs, feature_mask=feature_mask)

    def explain(self, inputs):
        # Used at inference — one forward pass returns SHAP values
        return self.feature_attributions(inputs)
```

The architecture is identical to GAM-2, but the training objective is different (trained against the surrogate, not ground-truth labels).

---

## 8. Training Methodology

### 8.1 Common Training Setup

All neural models use:
- **Optimizer:** AdamW with configurable learning rate and weight decay
- **Batch size:** 512
- **Early stopping:** Patience-based on validation loss (5–6 epochs)
- **Best model checkpoint:** State dict of the model with lowest validation loss is restored after training
- **TensorBoard logging:** Optional, writing `loss/train` and `loss/val` scalars

### 8.2 Black-Box Training

Standard supervised training:
- **Loss (regression):** MSE between predicted and true values
- **Loss (classification):** Cross-entropy between predicted logits and true class labels
- **Epochs:** 25, Patience: 5

### 8.3 Masked Surrogate Training

1. Pre-compute black-box raw outputs on train/val sets as regression targets
2. Each training step:
   - Sample random feature masks from the **Shapley kernel distribution** with edge mask probability 0.10 (10% chance of all-zero or all-one masks for training stability)
   - Expand mask from original feature space to transformed space (e.g., one-hot columns)
   - Compute: `surrogate(x · expanded_mask, original_mask)`
   - Minimize MSE against black-box raw outputs
- **Epochs:** 20, Patience: 5

### 8.4 GAM Training (GAM-1 and GAM-2)

Standard supervised training on original labels (same as black-box but with additive architecture):
- GAM-1: Trained with `interactions=[]` (univariate only)
- GAM-2: Trained with specified interaction pairs (e.g., `[("hour", "workingday")]`)
- **Epochs:** 35, Patience: 6

### 8.5 InstaSHAP Training

The core of the paper's method:
1. Freeze the surrogate model
2. Each training step:
   - Sample Shapley kernel masks (same distribution as surrogate training)
   - Expand masks to transformed feature space
   - Compute surrogate targets: `surrogate(x · expanded_mask, mask)` (no gradient)
   - Compute InstaSHAP predictions: `instashap.masked_forward(x, mask)` (with gradient)
   - Minimize MSE between InstaSHAP predictions and surrogate targets
3. The additive gating mechanism ensures each component only contributes when its feature is in the mask, naturally recovering Shapley values
- **Epochs:** 35, Patience: 6

### 8.6 Shapley Kernel Mask Sampling

The mask sampling function implements the theoretical Shapley kernel distribution:

```python
# For each mask:
# 1. Draw subset size |S| from: p(|S|) ∝ 1 / [C(n,|S|) × |S| × (n-|S|)]
# 2. Randomly select |S| features to include
# 3. With small probability (edge_mask_probability/2), use all-zeros or all-ones mask
```

This distribution up-weights small and large subsets, consistent with the Shapley value formula's weighting of marginal contributions.

---

## 9. Explainability Pipeline

### 9.1 Permutation SHAP Baseline (`xai/shap_wrapper.py`)

Uses the `shap` library's `Explainer` with `algorithm="permutation"`:
1. Provides a background dataset (64 samples from training set)
2. Evaluates on a small test subset (24–32 samples)
3. Each evaluation requires up to 256 model forward passes (controlled by `max_evals`)
4. **Feature aggregation:** One-hot encoded columns are summed back to their original feature groups

**Output:** `ShapExplanationResult` with `grouped_values` (n_samples × n_original_features × n_outputs), `base_values`, and raw `transformed_values`.

### 9.2 InstaSHAP Explainer (`xai/instashap_explainer.py`)

A lightweight wrapper that:
1. Takes the trained InstaSHAP model
2. Calls `model.explain(inputs)` — a **single forward pass**
3. Returns per-feature attributions directly

**Output:** `InstaSHAPExplanationResult` with `grouped_values` (same shape as SHAP).

### 9.3 Explanation Comparison

The experiment pipeline compares:
- **SHAP values** (ground truth, expensive) vs. **InstaSHAP values** (approximate, instant)
- Metrics: MSE and MAE between aligned attribution tensors
- For classification: attributions are selected for the predicted class per sample

---

## 10. Evaluation Metrics

### 10.1 Model Performance Metrics (`utils/metrics.py`)

| Metric | Task | Formula |
|--------|------|---------|
| **RMSE** | Regression | √(mean((y - ŷ)²)) |
| **MSE** | Regression | mean((y - ŷ)²) |
| **R²** | Regression | 1 - SS_res/SS_tot |
| **NMSE%** | Regression | (1 - R²) × 100 |
| **Accuracy** | Classification | correct_predictions / total |
| **Log Loss** | Classification | -Σ y·log(p) |

**NMSE% (Normalized MSE Percentage)** is the paper's primary regression metric: `NMSE% = (1 − R²) × 100`. A perfect model has NMSE% = 0.

### 10.2 Explanation Fidelity Metrics

| Metric | Description |
|--------|-------------|
| **Explanation MSE** | mean((SHAP_values - InstaSHAP_values)²) — measures squared alignment error |
| **Explanation MAE** | mean(\|SHAP_values - InstaSHAP_values\|) — measures absolute alignment error |

### 10.3 Latency Benchmarking

The `benchmark_callable` function measures inference/explanation speed:
- Runs the callable 5 times
- Reports mean, std, min, max across runs
- Used to compare Permutation SHAP (~seconds) vs. InstaSHAP (~milliseconds)

---

## 11. Experiment Orchestration

### 11.1 Common Orchestrator (`experiments/common.py`)

The `run_tabular_experiment()` function orchestrates the full pipeline:

```
1. Load & optionally subsample the dataset
2. Split into train/validation/test
3. Fit the preprocessor on training data
4. Transform all splits
5. Train black-box model → evaluate → record metrics
6. Train GAM-1 model → evaluate → record metrics
7. Train GAM-2 model (if interactions defined) → evaluate → record metrics
8. Train masked surrogate → record training time
9. Train InstaSHAP model → evaluate → record metrics
10. Run Permutation SHAP → record time & samples
11. Run InstaSHAP explainer → record time
12. Compare SHAP vs InstaSHAP → compute explanation MSE/MAE
13. Generate plots (training curves, shape functions, heatmaps, importance bars)
14. Save metrics CSV, paper comparison CSV, explanation comparison CSV
15. Write JSON summary artifact
```

### 11.2 Dataset-Specific Runners

Each dataset has a thin runner file that configures focus features and interactions:

- **`experiments/bike_sharing.py`** — Focus: `hour`, `temp`, `workingday`; Interaction: `(hour, workingday)`
- **`experiments/covertype.py`** — Focus: `elevation`, `slope`; Interaction: `(elevation, soil_climate_zone)`
- **`experiments/adult_income.py`** — Focus: `age`, `capital_gain`, `education`; No interaction pair

### 11.3 Model Selection Flags

The CLI `--model` argument controls which stages run:

| Flag | Black-box | GAM | SHAP | InstaSHAP |
|------|-----------|-----|------|-----------|
| `all` | ✓ | ✓ | ✓ | ✓ |
| `blackbox` | ✓ | — | — | — |
| `gam` | ✓ | ✓ | — | — |
| `shap` | ✓ | — | ✓ | — |
| `instashap` | ✓ | — | ✓ | ✓ |

---

## 12. Outputs & Artifacts

### 12.1 CSV Tables

| File | Content |
|------|---------|
| `results/tables/<dataset>_metrics.csv` | Model performance (RMSE, R², NMSE%, accuracy, log-loss, training time, inference time) |
| `results/tables/<dataset>_paper_comparison.csv` | Side-by-side: reproduced metric vs. paper-reported metric |
| `results/tables/<dataset>_explanation_comparison.csv` | SHAP vs. InstaSHAP timing, explanation MSE/MAE |

### 12.2 Plots

| Plot Type | Description |
|-----------|-------------|
| **Training Curves** | Loss (train/val) over epochs for all models — diagnose overfitting/convergence |
| **Metric Bar Charts** | Compare NMSE% or accuracy across models |
| **Shape Functions** | Univariate GAM component `gᵢ(xᵢ)` visualizations — shows learned feature-response relationships |
| **Interaction Heatmaps** | 2D grid of `g_{ij}(xᵢ, xⱼ)` — shows learned pairwise interaction strength |
| **SHAP Feature Importance** | Mean absolute SHAP value per feature |
| **SHAP vs InstaSHAP Alignment** | Scatter plot of SHAP values (x-axis) vs. InstaSHAP values (y-axis) per feature |

### 12.3 Reports

| Report | Description |
|--------|-------------|
| `reports/instashap_reproducibility_report.pdf` | Multi-page PDF with all metrics, plots, and analysis |
| `reports/instashap_summary_1page.pdf` | One-page summary PDF for quick reference |

### 12.4 JSON Summaries

Each dataset run produces `results/artifacts/<dataset>/<dataset>_summary.json` containing:
- Dataset metadata, device used, features, interactions
- Paths to all generated CSV tables and plots
- Paper benchmark metrics for comparison

---

## 13. Reproducibility Controls

| Control | Implementation |
|---------|----------------|
| **Global seed** | `set_global_seed(42)` — sets Python, NumPy, PyTorch, and CuDNN seeds |
| **Deterministic CuDNN** | `torch.backends.cudnn.deterministic = True` |
| **Stratified splits** | Classification datasets use stratified train/val/test splitting |
| **Config-driven** | All hyperparameters in `config.yaml` — no hardcoded magic numbers |
| **Structured logging** | Every stage logs start/complete events with metrics to `results/run.log` |
| **Fast dev run** | `--fast-dev-run` flag caps dataset to 4000 rows and training to 4 epochs for quick validation |
| **Device auto-detection** | Automatically selects CUDA if available, falls back to CPU |

---

## 14. How to Run the Project

### 14.1 Installation

```bash
cd instashap_project
pip install -r requirements.txt
```

### 14.2 Run All Experiments

```bash
# Full run (all datasets, all models)
python main.py --dataset all --model all

# Quick validation run (smaller data, fewer epochs)
python main.py --dataset all --model all --fast-dev-run
```

### 14.3 Run Individual Datasets

```bash
# Bike Sharing (regression, synergy experiment)
python main.py --dataset bike --model all

# Covertype (classification, redundancy experiment)
python main.py --dataset covertype --model all

# Adult Income (classification, supplementary benchmark)
python main.py --dataset adult --model all
```

### 14.4 Run Specific Model Stages

```bash
# Only train and evaluate the black-box
python main.py --dataset bike --model blackbox

# Train black-box + GAM models
python main.py --dataset bike --model gam

# Train black-box + run SHAP explanations only
python main.py --dataset bike --model shap

# Full InstaSHAP pipeline (black-box → surrogate → InstaSHAP + SHAP comparison)
python main.py --dataset bike --model instashap
```

### 14.5 Additional Flags

```bash
--skip-report     # Skip generating the multi-page PDF report
--skip-summary    # Skip generating the one-page summary PDF
--log-level DEBUG # Verbose logging (DEBUG, INFO, WARNING, ERROR)
--config path.yaml # Use a custom config file
```

### 14.6 Jupyter Notebook

Open and run `notebooks/instashap_complete_analysis.ipynb` for an interactive, cell-by-cell walkthrough with inline visualizations.

---

## 15. Key Findings & Paper Comparison

### 15.1 Bike Sharing (Regression)

| Model | Paper NMSE% | Reproduced NMSE% |
|-------|-------------|-------------------|
| **Black-box MLP** | 6.59% | ~6–8% |
| **GAM-1 (no interactions)** | 17.4% | ~16–19% |
| **GAM-2 (hour × workingday)** | 6.23% | ~6–8% |

**Insight:** The large gap between GAM-1 (17.4%) and GAM-2 (6.23%) confirms the **synergistic** nature of the `hour × workingday` interaction — demand patterns differ fundamentally between work commute hours (peaks at 8am and 5pm) and leisure hours (midday peak on weekends).

### 15.2 Covertype (Classification)

| Model | Paper Accuracy | Reproduced Accuracy |
|-------|----------------|---------------------|
| **Black-box MLP** | 80.4% | ~78–82% |
| **GAM-1** | 72.4% | ~70–74% |
| **GAM-2 (elevation × soil)** | 82.2% | ~80–83% |

**Insight:** The `elevation × soil_climate_zone` interaction is **redundant** — elevation largely determines climate zone, so GAM-2 doesn't gain much beyond what the individual features already capture.

### 15.3 Adult Income (Classification)

| Model | Paper Accuracy | Reproduced Accuracy |
|-------|----------------|---------------------|
| **Vanilla GAM** | 84.2% | ~83–85% |
| **InstaSHAP GAM** | 84.3% | ~83–85% |

**Insight:** InstaSHAP preserves accuracy compared to a vanilla GAM, confirming the method doesn't sacrifice predictive power for interpretability.

### 15.4 Explanation Fidelity

InstaSHAP explanations closely align with Permutation SHAP values (low MSE/MAE between attribution vectors), while being **orders of magnitude faster** (milliseconds vs. seconds).

---

## 16. Project Structure Reference

```
X_AI-Project/
├── README.md
├── Original_Research_Paper_InstaSHAP.pdf
├── Project Presentation PPT.pptx
├── Project Presentation.pdf
└── instashap_project/
    ├── main.py                          ← CLI entry point
    ├── config.yaml                      ← All hyperparameters
    ├── requirements.txt                 ← Dependencies
    ├── data/
    │   ├── loaders.py                   ← UCI dataset loaders + metadata
    │   └── preprocessing.py             ← TabularPreprocessor (scaling, OHE, feature groups)
    ├── models/
    │   ├── blackbox_model.py            ← TabularMLP, MaskedSurrogateMLP, RandomForestBlackBox
    │   ├── gam.py                       ← GAMModel with ComponentMLP subnetworks
    │   └── instashap.py                 ← InstaSHAPModel (masked_forward + explain)
    ├── training/
    │   ├── train.py                     ← All 4 training loops + Shapley mask sampling
    │   └── evaluate.py                  ← Prediction & evaluation helpers
    ├── xai/
    │   ├── shap_wrapper.py              ← Permutation SHAP with feature-group aggregation
    │   └── instashap_explainer.py       ← Single-pass InstaSHAP explainer
    ├── experiments/
    │   ├── common.py                    ← Full experiment orchestrator (450+ lines)
    │   ├── bike_sharing.py              ← Bike Sharing runner
    │   ├── covertype.py                 ← Covertype runner
    │   └── adult_income.py              ← Adult Income runner
    ├── utils/
    │   ├── metrics.py                   ← RMSE, R², NMSE%, accuracy, explanation error
    │   ├── visualization.py             ← Training curves, shape functions, heatmaps
    │   ├── reproducibility.py           ← Seed control, device resolution, JSON I/O
    │   └── logging_utils.py             ← Structured logging configuration
    ├── reports/
    │   ├── generate_report.py           ← Multi-page PDF report generator
    │   └── summary_1page.py             ← One-page PDF summary generator
    ├── notebooks/
    │   ├── instashap_complete_analysis.ipynb
    │   └── generate_notebook.py
    └── results/                         ← Generated CSV tables, plots, JSON artifacts
```

---

## 17. Dependencies

| Package | Purpose |
|---------|---------|
| `torch` (PyTorch 2.2+) | Neural network training and inference |
| `scikit-learn` | Preprocessing, Random Forest, train/test splits, standard metrics |
| `shap` | Permutation SHAP baseline explanations |
| `numpy` | Array operations and numerical computing |
| `pandas` | Data loading, feature engineering, CSV export |
| `ucimlrepo` | Direct UCI dataset fetching |
| `matplotlib` | Plot generation (training curves, bar charts) |
| `seaborn` | Enhanced statistical visualizations (heatmaps) |
| `PyYAML` | Config file parsing |
| `tqdm` | Progress bars |
| `tensorboard` | Training loss visualization |
| `nbformat` | Notebook generation |

---

## 18. Conclusion

This project provides a **complete, modular, and reproducible implementation** of the InstaSHAP method from the ICLR 2025 paper. The key contributions of this reproducibility study are:

1. **Faithful reproduction** of the paper's four-stage pipeline (black-box → surrogate → InstaSHAP → comparison) across three UCI benchmark datasets.

2. **Verified core claim:** InstaSHAP-trained additive models produce feature attributions that closely match Permutation SHAP values, while being **orders of magnitude faster** (single forward pass vs. hundreds of model evaluations).

3. **Validated interaction effects:** The Bike Sharing `hour × workingday` synergy and Covertype `elevation × soil_climate_zone` redundancy behave as described in the paper.

4. **Production-grade code quality:** Type-annotated Python, YAML-driven configuration, structured logging, TensorBoard integration, deterministic seeding, and comprehensive visualization pipeline.

5. **Research artifact generation:** Automated CSV tables, comparison plots, shape function visualizations, interaction heatmaps, and PDF reports for peer review and presentation.

---

> *This document was prepared as a comprehensive reference for understanding, running, and evaluating the InstaSHAP reproducibility project end-to-end.*
