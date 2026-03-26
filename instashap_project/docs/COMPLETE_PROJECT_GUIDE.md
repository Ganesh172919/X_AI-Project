# InstaSHAP — Complete Project Guide

> **One document to understand the entire project: flow, internals, ML models, results, and how to present it.**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Research Paper Background](#3-research-paper-background)
4. [Project Architecture & Flow](#4-project-architecture--flow)
5. [Data Pipeline](#5-data-pipeline)
6. [ML Models Used](#6-ml-models-used)
7. [InstaSHAP Methodology (Core Innovation)](#7-instashap-methodology-core-innovation)
8. [Training Pipeline Deep Dive](#8-training-pipeline-deep-dive)
9. [Explainability Pipeline](#9-explainability-pipeline)
10. [Evaluation & Metrics](#10-evaluation--metrics)
11. [Experiment Results & Interpretation](#11-experiment-results--interpretation)
12. [Key Things to Remember](#12-key-things-to-remember)
13. [How to Explain This Project](#13-how-to-explain-this-project)
14. [Presentation Guide](#14-presentation-guide)
15. [FAQ — Common Questions & Answers](#15-faq--common-questions--answers)
16. [File-by-File Reference](#16-file-by-file-reference)
17. [Glossary](#17-glossary)

---

## 1. Executive Summary

This project is a **research-grade reproduction** of the ICLR 2025 paper **"InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly"**. 

**In one sentence:** Instead of computing expensive Shapley values through thousands of model evaluations, InstaSHAP trains an additive neural network that produces SHAP-equivalent feature attributions in a **single forward pass**.

### What the project does:
- Loads 3 UCI benchmark datasets (Bike Sharing, Covertype, Adult Income)
- Trains a **black-box MLP** as the prediction model
- Trains a **masked surrogate** to approximate how the black-box behaves under feature masking
- Trains an **InstaSHAP additive model** against the surrogate using Shapley-kernel masks
- Trains **GAM-1 and GAM-2** (Generalized Additive Models) as interpretable baselines
- Computes **permutation SHAP** as ground-truth explanations
- Compares InstaSHAP explanations vs SHAP explanations for fidelity
- Generates tables, plots, training curves, shape function plots, and PDF reports

### The core result:
InstaSHAP produces feature attributions that closely match traditional SHAP values but **orders of magnitude faster** — in one forward pass instead of hundreds/thousands of permutations.

---

## 2. Problem Statement & Motivation

### The Problem
In machine learning, we often need to explain **why** a model made a specific prediction. SHAP (SHapley Additive exPlanations) is the gold standard for this, assigning each feature an importance score based on cooperative game theory.

**The catch:** Computing SHAP values is computationally expensive.
- **Permutation SHAP** requires evaluating the model on many subsets of features
- For `d` features, exact Shapley values need `2^d` model evaluations
- Even approximate methods require `O(d²)` evaluations per sample
- This makes SHAP impractical for real-time or high-throughput applications

### The Solution (InstaSHAP)
Train a neural network with an **additive structure** so that:
- Each feature has its own small sub-network (component)
- The model is trained under a **masked objective** that mimics the Shapley value computation
- After training, each component's output directly gives the SHAP attribution for that feature
- **One forward pass = all SHAP values** (no exponential computation needed)

### Why This Matters
| Scenario | Traditional SHAP | InstaSHAP |
|----------|-----------------|-----------|
| Single prediction explanation | Seconds to minutes | Milliseconds |
| Batch of 10,000 explanations | Hours | Seconds |
| Real-time dashboard | Infeasible | Practical |
| Regulatory compliance auditing | Expensive | Cheap |

---

## 3. Research Paper Background

**Paper:** "InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly"  
**Venue:** ICLR 2025  
**Core Contribution:** A training procedure for additive models that makes their individual component outputs equal to the Shapley values of a black-box model.

### Key Paper Concepts

1. **Shapley Values**: From cooperative game theory — the unique fair allocation of a "payout" (prediction) among "players" (features).

2. **Additive Models (GAMs)**: Models of the form:  
   `f(x) = bias + g₁(x₁) + g₂(x₂) + ... + gₖ(xₖ)`  
   where each `gᵢ` is a learned function of a single feature.

3. **Masked Objective (Equation 20)**: The key equation from the paper. The InstaSHAP model is trained to predict what a surrogate model outputs when only a subset of features is visible:  
   `minimize E_S[ || InstaSHAP(x; S) - Surrogate(x; S) ||² ]`  
   where `S` is sampled from the Shapley kernel distribution.

4. **Feature Interactions**: The paper demonstrates two types:
   - **Synergistic** (hour × workingday in Bike Sharing): Features together have more impact than individually
   - **Redundant** (elevation × soil_climate_zone in Covertype): Features share overlapping information

---

## 4. Project Architecture & Flow

### Complete Pipeline Flow

```
                    ┌──────────────────────────────────────────┐
                    │           STAGE 1: DATA LOADING          │
                    │                                          │
                    │  UCI Repository → pandas DataFrame       │
                    │  Feature engineering & type assignment    │
                    │  Paper-aligned column selection           │
                    └───────────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │         STAGE 2: PREPROCESSING           │
                    │                                          │
                    │  TabularPreprocessor                     │
                    │  ├─ Numeric: Median impute → StandardScaler
                    │  ├─ Categorical: Mode impute → OneHotEncoder
                    │  └─ Feature group bookkeeping            │
                    │                                          │
                    │  Train/Val/Test split (80/10/10)         │
                    │  Stratified for classification           │
                    └───────────────────┬──────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
┌──────────────────┐       ┌──────────────────┐          ┌──────────────────┐
│  STAGE 3a:       │       │  STAGE 3b:       │          │  STAGE 3c:       │
│  BLACK-BOX MLP   │       │  GAM-1 / GAM-2   │          │  INSTASHAP       │
│                  │       │                  │          │  PIPELINE        │
│  TabularMLP      │       │  Additive model  │          │                  │
│  [256,128] + ReLU│       │  with per-feature │          │  ┌────────────┐ │
│  + Dropout       │       │  sub-networks    │          │  │ Surrogate  │ │
│                  │       │  [96,64] + ReLU  │          │  │ Training   │ │
│  Loss: MSE or CE │       │                  │          │  └─────┬──────┘ │
│  Optimizer: AdamW│       │  GAM-1: univariate│          │        │        │
│  Early stopping  │       │  GAM-2: +pairwise│          │  ┌─────▼──────┐ │
└────────┬─────────┘       └────────┬─────────┘          │  │ InstaSHAP  │ │
         │                          │                    │  │ Training   │ │
         │                          │                    │  └─────┬──────┘ │
         │                          │                    └────────┼────────┘
         │                          │                             │
         ▼                          ▼                             ▼
┌──────────────────┐       ┌──────────────────┐          ┌──────────────────┐
│  STAGE 4a:       │       │  STAGE 4b:       │          │  STAGE 4c:       │
│  PERMUTATION     │       │  SHAPE FUNCTION  │          │  INSTASHAP       │
│  SHAP VALUES     │       │  VISUALIZATION   │          │  EXPLANATIONS    │
│                  │       │                  │          │                  │
│  shap.Explainer  │       │  Per-feature     │          │  model.explain() │
│  (permutation)   │       │  response curves │          │  Single forward  │
│                  │       │  Interaction     │          │  pass             │
│  Slow but exact  │       │  heatmaps        │          │  Fast             │
└────────┬─────────┘       └────────┬─────────┘          └────────┬─────────┘
         │                          │                             │
         └──────────────┬───────────┘                             │
                        │                                         │
                        ▼                                         │
               ┌────────────────┐                                 │
               │  STAGE 5:      │◄────────────────────────────────┘
               │  COMPARISON    │
               │                │
               │  SHAP vs       │
               │  InstaSHAP     │
               │  (MSE, MAE)    │
               └────────┬───────┘
                        │
                        ▼
               ┌────────────────┐
               │  STAGE 6:      │
               │  OUTPUTS       │
               │                │
               │  CSV tables    │
               │  PNG plots     │
               │  JSON summaries│
               │  PDF reports   │
               └────────────────┘
```

### Module Dependency Map

```
main.py
 ├── config.yaml (loaded with PyYAML)
 ├── utils/reproducibility.py (seed control, device resolution)
 ├── utils/logging_utils.py (structured logging)
 └── experiments/{bike_sharing, covertype, adult_income}.py
      └── experiments/common.py (orchestrator)
           ├── data/loaders.py (UCI dataset fetching)
           ├── data/preprocessing.py (TabularPreprocessor)
           ├── training/train.py (4 trainers)
           ├── training/evaluate.py (prediction & metrics)
           ├── xai/shap_wrapper.py (permutation SHAP)
           ├── xai/instashap_explainer.py (single-pass explainer)
           ├── utils/metrics.py (RMSE, NMSE%, accuracy, log-loss)
           ├── utils/visualization.py (matplotlib plots)
           └── reports/{generate_report, summary_1page}.py
```

---

## 5. Data Pipeline

### Three Benchmark Datasets

| Property | Bike Sharing | Covertype | Adult Income |
|----------|-------------|-----------|-------------|
| **UCI ID** | 275 | 31 | 2 |
| **Task** | Regression | 7-class Classification | Binary Classification |
| **Target** | `count` (hourly bike rentals) | `cover_type` (forest type) | `income_above_50k` |
| **Numeric features** | 5 (day_of_month, temp, atemp, hum, windspeed) | 10 (elevation, aspect, slope, distances, hillshades) | 5 (age, fnlwgt, capital_gain, capital_loss, hours_per_week) |
| **Categorical features** | 8 (season, year, month, hour, holiday, weekday, workingday, weather_situation) | 1 (soil_climate_zone — grouped from 40 soil types) | 8 (workclass, education, marital_status, occupation, relationship, race, sex, native_country) |
| **Total original features** | 13 | 11 | 13 |
| **Interaction studied** | hour × workingday (synergy) | elevation × soil_climate_zone (redundancy) | None |
| **Rows** | ~17,379 | 581,012 (sampled to 60K) | ~48,842 |
| **Paper-reported metric** | NMSE%: BB=6.59, GAM1=17.4, GAM2=6.23 | Accuracy: BB=0.804, GAM1=0.724, GAM2=0.822 | Accuracy: GAM=0.842, InstaSHAP=0.843 |

### Data Preprocessing Steps

1. **Loading**: Fetched via `ucimlrepo.fetch_ucirepo(id=...)` — direct download from UCI ML Repository
2. **Feature engineering**: 
   - Bike: Parses date → `day_of_month`, maps binary codes to "yes"/"no"
   - Covertype: 40 binary soil columns → 1 categorical `soil_climate_zone` (grouped by ELU climate digit)
   - Adult: Drops redundant `education-num`, renames hyphenated columns
3. **Train/Val/Test split**: 70/10/20 by default, stratified for classification tasks
4. **Numeric**: `SimpleImputer(median)` → `StandardScaler` (zero mean, unit variance)
5. **Categorical**: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
6. **Feature group mapping**: The `TabularPreprocessor` tracks which transformed columns belong to which original feature — crucial for aggregating SHAP values back to original features

### Why Feature Group Tracking Matters

After one-hot encoding, a categorical feature like `hour` (24 categories) becomes 24 columns. When computing SHAP values on transformed data, we get 24 separate attributions. The preprocessor's `FeatureGroup` system maps them back so we can aggregate: `SHAP(hour) = sum(SHAP(hour=0), SHAP(hour=1), ..., SHAP(hour=23))`.

---

## 6. ML Models Used

### Model 1: TabularMLP (Black-Box Baseline)

**Purpose:** The model we want to explain. It's a "black box" because we treat its internal workings as opaque.

**Architecture:**
```
Input (transformed features)
   ↓
Linear(input_dim → 256) → ReLU → Dropout(0.10)
   ↓
Linear(256 → 128) → ReLU → Dropout(0.10)
   ↓
Linear(128 → output_dim)
   ↓
Output (regression value or class logits)
```

**Key details:**
- Standard feed-forward network (Multi-Layer Perceptron)
- Uses ReLU activation, dropout for regularization
- Loss: MSE for regression, CrossEntropy for classification
- Optimizer: AdamW (Adam with decoupled weight decay)
- Early stopping with patience=5 on validation loss

### Model 2: MaskedSurrogateMLP

**Purpose:** Approximates how the black-box behaves when only a **subset** of features is visible. This is the intermediate step that bridges the black-box and InstaSHAP.

**Architecture:**
```
Input: [masked_features (feature_dim), feature_mask (num_original_features)]
   ↓   ← concatenation of masked input + the binary mask itself
Linear(feature_dim + num_features → 256) → ReLU → Dropout(0.10)
   ↓
Linear(256 → 128) → ReLU → Dropout(0.10)
   ↓
Linear(128 → output_dim)
   ↓
Output (approximation of black-box with masked input)
```

**Key details:**
- Takes the **concatenation** of `x ⊙ mask_expanded` and `mask` as input
- `mask_expanded` repeats each feature's mask bit across all its one-hot columns
- Trained on random Shapley kernel masks (not uniform random!)
- The surrogate learns: "given that features in set S are visible, what would the black-box predict?"

### Model 3: GAMModel (Generalized Additive Model)

**Purpose:** A transparent, interpretable model where each feature's effect can be visualized independently. Comes in two variants:

**GAM-1 (univariate only):**
```
f(x) = bias + g₁(x₁) + g₂(x₂) + ... + gₖ(xₖ)
```
Each `gᵢ` is a small MLP:
```
ComponentMLP: Linear(feature_width → 96) → ReLU → Linear(96 → 64) → ReLU → Linear(64 → output_dim)
```

**GAM-2 (with pairwise interactions):**
```
f(x) = bias + g₁(x₁) + ... + gₖ(xₖ) + h₁₂(x₁,x₂)
```
Adds interaction components, e.g., `h(hour, workingday)` — a small MLP that takes both features' columns as input.

**Key details:**
- `nn.ModuleDict` holds one ComponentMLP per feature (and per interaction pair)
- Feature attributions: each `gᵢ(xᵢ)` IS the attribution for feature `i`
- Interaction attributions: split equally between the two features (each gets `h₁₂/2`)
- Supports optional feature masking via a gate mechanism (multiply component output by mask bit)

### Model 4: InstaSHAPModel (Core Innovation)

**Purpose:** An additive model (extending GAMModel) trained so that its component outputs **equal** the Shapley values of the black-box model.

**Architecture:** Same additive structure as GAMModel:
```
f(x; S) = bias + Σᵢ [ gᵢ(xᵢ) · mask(i) ]
```

**What makes it different from GAM:**
- **Not trained on labels** — trained against the surrogate's masked outputs
- Uses `masked_forward(x, mask)` during training to match `surrogate(x ⊙ mask, mask)`
- After training, `explain(x)` returns per-feature attributions in one forward pass
- The masked training forces each component to learn the marginal contribution of its feature — which is exactly the Shapley value

**The "Instant" part:** Once trained, getting SHAP values = one forward pass through the additive model. No permutations, no subsets, no exponential blowup.

---

## 7. InstaSHAP Methodology (Core Innovation)

### The Mathematical Foundation

#### Step 1: Shapley Values (What We Want)
For a model `f` and input `x`, the Shapley value of feature `i` is:

```
φᵢ(x) = Σ_{S ⊆ N\{i}} [|S|!(d-|S|-1)!/d!] · [f(x_S∪{i}) - f(x_S)]
```

This measures the average marginal contribution of feature `i` across all possible subsets.

#### Step 2: The Surrogate (What We Approximate)
Instead of evaluating `f(x_S)` directly (which requires marginalizing over missing features), we train a surrogate `g(x, S)` that takes the mask `S` as an explicit input:

```
g(x ⊙ expand(S), S) ≈ E_x̄[f(x_S, x̄_{N\S})]
```

The surrogate learns to predict the black-box's output under any masking pattern.

#### Step 3: The InstaSHAP Objective (Equation 20)
Train an additive model `h` such that:

```
h(x; S) = bias + Σ_{i ∈ S} hᵢ(xᵢ)
```

Minimizes:
```
L = E_{x, S~Shapley} [ || h(x; S) - g(x ⊙ expand(S), S) ||² ]
```

**Why this works:** When an additive model is forced to match masked outputs under Shapley-kernel-distributed masks, the unique optimal solution for each `hᵢ` is the Shapley value `φᵢ`.

#### Step 4: Shapley Kernel Mask Sampling
Masks are NOT sampled uniformly. The Shapley kernel assigns weights:

```
w(|S|) = 1 / [C(d, |S|) · |S| · (d - |S|)]
```

This gives **more weight** to small subsets and large subsets (where marginal contributions are most distinctive) and less weight to medium-sized subsets.

**Edge masks** (all-zeros and all-ones) are included with probability `edge_mask_probability=0.10` for training stability.

### The 4-Stage Training Pipeline

```
Stage 1: Train Black-Box MLP on (X, y)
    ↓
Stage 2: Train Masked Surrogate on (X ⊙ mask, mask) → BlackBox(X)
    ↓
Stage 3: Train InstaSHAP on masked_forward(X, mask) → Surrogate(X ⊙ mask, mask)
    ↓
Stage 4: Compare InstaSHAP.explain(X) vs PermutationSHAP(BlackBox, X)
```

---

## 8. Training Pipeline Deep Dive

### Common Training Configuration

| Parameter | Black-Box | Surrogate | GAM | InstaSHAP |
|-----------|-----------|-----------|-----|-----------|
| **Hidden dims** | [256, 128] | [256, 128] | [96, 64] | [96, 64] |
| **Dropout** | 0.10 | 0.10 | 0.05 | 0.05 |
| **Learning rate** | 0.001 | 0.001 | 0.001 | 0.001 |
| **Weight decay** | 1e-4 | 1e-5 | 1e-5 | 1e-5 |
| **Batch size** | 512 | 512 | 512 | 512 |
| **Epochs** | 25 | 20 | 35 | 35 |
| **Patience** | 5 | 5 | 6 | 6 |
| **Masks/sample** | — | 2 | — | 2 |
| **Edge mask prob** | — | 0.10 | — | 0.10 |

### All Trainers Use:
- **AdamW optimizer** (Adam with decoupled weight decay regularization)
- **Early stopping** on validation loss with configurable patience
- **Best model checkpointing** (restores best validation weights after training)
- **TensorBoard logging** for training/validation loss curves
- **Deterministic seeding** (seed=42 by default for reproducibility)

### Surrogate Training Details
- Generates random Shapley kernel masks each batch (not uniform!)
- Expands feature-level masks to column-level (one-hot columns share their parent's mask)
- Target: black-box model's raw outputs (logits for classification, regression values for regression)
- Loss: MSE between surrogate output and black-box output

### InstaSHAP Training Details  
- Same mask sampling as surrogate
- Uses `masked_forward()` which gates each component output by the mask
- Target: surrogate's output under the same mask (NOT the black-box directly)
- This two-stage approach (black-box → surrogate → InstaSHAP) improves stability

---

## 9. Explainability Pipeline

### Permutation SHAP Baseline

**Implementation:** Uses the `shap` library's `shap.Explainer` with `algorithm="permutation"`.

**How it works:**
1. Select a background dataset (64 samples from training set)
2. For each test sample, systematically permute feature presence/absence
3. Measure each feature's marginal contribution to prediction change
4. Aggregate one-hot column attributions back to original features using `preprocessor.group()`

**Output:** `ShapExplanationResult` with:
- `grouped_values`: SHAP values aggregated to original features `[n_samples, n_features, n_outputs]`
- `base_values`: Expected model output (baseline)
- `transformed_values`: Raw per-column SHAP values

**Speed:** Slow — requires `max_evals` model evaluations per sample (default: 256)

### InstaSHAP Single-Pass Explainer

**Implementation:** Wraps the trained InstaSHAP model's `explain()` method.

**How it works:**
1. Pass input through the additive model with no mask (`feature_mask=None`)
2. Each component MLP produces its output
3. Univariate component outputs = feature attributions
4. Interaction components are split 50/50 between the two features

**Output:** `InstaSHAPExplanationResult` with:
- `grouped_values`: Feature attributions `[n_samples, n_features, n_outputs]`

**Speed:** Fast — single forward pass, batched

### Comparison
The experiment orchestrator compares SHAP vs InstaSHAP using:
- **MSE** (Mean Squared Error) between attribution vectors
- **MAE** (Mean Absolute Error) between attribution vectors

Lower values = InstaSHAP more faithfully reproduces SHAP values.

---

## 10. Evaluation & Metrics

### Prediction Metrics

| Metric | Task | Formula | Meaning |
|--------|------|---------|---------|
| **RMSE** | Regression | `√(mean((y-ŷ)²))` | Prediction error in original units |
| **MSE** | Regression | `mean((y-ŷ)²)` | Average squared error |
| **R²** | Regression | `1 - MSE/Var(y)` | Fraction of variance explained |
| **NMSE%** | Regression | `100 × MSE/Var(y)` | Normalized MSE as percentage (paper uses this) |
| **Accuracy** | Classification | `correct/total` | Fraction of correct predictions |
| **Log-loss** | Classification | `-mean(Σ y·log(p))` | Cross-entropy between true and predicted probabilities |

### Explanation Fidelity Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Explanation MSE** | `mean((SHAP - InstaSHAP)²)` | How far InstaSHAP attributions are from SHAP |
| **Explanation MAE** | `mean(|SHAP - InstaSHAP|)` | Average absolute attribution difference |

### Timing Metrics
- **Training time**: Wall-clock seconds for each model
- **Inference time**: Mean latency over 5 repeated predictions on 512 samples

---

## 11. Experiment Results & Interpretation

### What to Expect from Each Dataset

#### Bike Sharing (Synergy Experiment)
- **Goal:** Show that GAM-2 (with hour × workingday interaction) dramatically outperforms GAM-1
- **Expected:** GAM-1 NMSE% ≈ 17%, GAM-2 NMSE% ≈ 6% (close to black-box)
- **Why:** Rush hour patterns differ on workdays vs weekends — the interaction captures this
- **Key visualization:** Shape function for `hour` shows different curves on working vs non-working days
- **InstaSHAP result:** InstaSHAP attributions should closely match SHAP attributions (low MSE/MAE)

#### Covertype (Redundancy Experiment)
- **Goal:** Show that elevation × soil_climate_zone interaction improves predictions
- **Expected:** GAM-2 accuracy > GAM-1 accuracy (improves by ~10 percentage points)
- **Why:** Elevation and soil climate zone carry overlapping but complementary information
- **Key visualization:** Interaction heatmap shows how prediction changes across elevation × soil zone grid

#### Adult Income (Supplementary Benchmark)
- **Goal:** Verify InstaSHAP matches vanilla GAM performance (no interactions needed)
- **Expected:** GAM ≈ InstaSHAP ≈ 84% accuracy
- **Key visualization:** Feature importance bar chart showing age, capital_gain, education as top features

### Paper vs Reproduced Values
The project generates `*_paper_comparison.csv` tables that directly compare reproduced metrics against the paper's reported values. Small differences (±2-3%) are expected due to:
- Different random seeds
- Slightly different preprocessing
- Library version differences

---

## 12. Key Things to Remember

### 🔑 Critical Concepts

1. **InstaSHAP ≠ SHAP**: InstaSHAP is a model that **approximates** SHAP values, trained to match them. It's not a different way of computing SHAP — it's a learned approximation that's extremely fast.

2. **The 4 models are NOT alternatives** — they form a pipeline:
   ```
   Black-Box → Surrogate → InstaSHAP → Explanations
   ```
   GAM-1/GAM-2 are separate baselines for comparison.

3. **Shapley kernel masks ≠ uniform random masks**: The weighting prioritizes edge-case subsets (small S and large S). This is mathematically crucial for convergence.

4. **One-hot aggregation is necessary**: SHAP operates on transformed (one-hot encoded) features, but humans think in original features. The `FeatureGroup` system handles this mapping.

5. **Surrogate is needed because**: You can't directly train InstaSHAP against the black-box with masking (the black-box doesn't accept mask inputs). The surrogate serves as a differentiable, mask-aware target.

6. **Additive structure is the key constraint**: Because `f(x;S) = bias + Σᵢ∈S gᵢ(xᵢ)`, the optimal `gᵢ` under Shapley-weighted masking is provably the Shapley value.

### 🎯 Numbers to Remember
- **Seed:** 42 (for full reproducibility)
- **Split:** 70% train / 10% val / 20% test
- **SHAP background:** 64 samples
- **SHAP evaluation:** 24-32 samples
- **Black-box architecture:** [256, 128] hidden dims
- **InstaSHAP/GAM architecture:** [96, 64] hidden dims
- **Edge mask probability:** 10%

### ⚠️ Common Pitfalls
- Running without `--fast-dev-run` on Covertype takes long (581K rows → sampled to 60K)
- SHAP computation is the slowest stage — can be skipped for quick iteration
- Feature mask expansion is the most complex preprocessing step — understand it well

---

## 13. How to Explain This Project

### 30-Second Elevator Pitch
> "We reproduced a cutting-edge ICLR 2025 paper that makes AI explanations instant. Normally, explaining one prediction takes hundreds of model evaluations. Our implementation trains a specialized neural network that produces the same explanations in a single forward pass — making real-time explainable AI practical."

### 2-Minute Technical Summary
> "SHAP values are the gold standard for explaining ML predictions, but they're computationally expensive because they require evaluating all possible feature subsets. The InstaSHAP paper showed that if you train an additive neural network under a Shapley-kernel-weighted masking objective, the individual component outputs converge to exact SHAP values.
>
> Our project implements this from scratch:
> 1. We train a black-box MLP on three UCI benchmark datasets
> 2. We train a surrogate that learns the black-box's behavior under feature masking
> 3. We train an additive InstaSHAP model against the surrogate with Shapley-kernel masks
> 4. We show that InstaSHAP's instant explanations closely match expensive permutation SHAP
>
> The key result: equivalent explanations, orders of magnitude faster."

### For Non-Technical Audiences
> "Imagine a doctor uses an AI system to diagnose patients. The AI says 'you have condition X' but can't explain why. With SHAP, we can say 'the AI focused mainly on your blood pressure and age.' But computing that explanation takes minutes. Our project implements a method that gives the same explanation instantly — crucial for real-time healthcare decisions."

---

## 14. Presentation Guide

### Recommended Slide Structure

**Slide 1: Title**
- "InstaSHAP: Making AI Explanations Instant"
- Subtitle: Reproduction of ICLR 2025 Paper

**Slide 2: The Problem**
- ML models are powerful but opaque
- SHAP explanations are faithful but slow (show complexity diagram)
- Real-time explainability is needed in healthcare, finance, regulatory compliance

**Slide 3: What is SHAP?**
- Game theory roots: "fair" allocation of prediction credit to features
- Visual: stacked bar chart showing SHAP values for one prediction
- Problem: exponential computation

**Slide 4: The InstaSHAP Solution**
- Train an additive model to MATCH SHAP values
- Key insight: masked training under Shapley kernel = optimal Shapley recovery
- Show the pipeline: Black-Box → Surrogate → InstaSHAP

**Slide 5: Our Implementation**
- 3 datasets, 5 model types, full pipeline
- Technology: Python, PyTorch, shap library, scikit-learn
- Show the project structure diagram

**Slide 6: Datasets**
- Table of 3 datasets with task types, feature counts, interaction highlights

**Slide 7: Model Architecture**
- Black-box: MLP [256,128]
- Additive models: per-feature ComponentMLP [96,64]
- Visual: diagram showing additive structure

**Slide 8: Key Results — Bike Sharing**
- Show NMSE% comparison: Black-box vs GAM-1 vs GAM-2
- Shape function plot for `hour`
- Interaction heatmap for hour × workingday

**Slide 9: Key Results — Explanation Fidelity**
- SHAP vs InstaSHAP alignment scatter plots
- MSE/MAE numbers
- Speed comparison: seconds vs milliseconds

**Slide 10: Conclusion & Takeaways**
- InstaSHAP successfully recovers SHAP values in one forward pass
- Additive structure + Shapley kernel training = theoretical guarantee
- Practical implications for real-time explainable AI

### Tips for Q&A
- **"Why not just use LIME?"** — LIME uses local linear approximations and has no theoretical guarantees of matching Shapley values. InstaSHAP provably converges to exact Shapley values.
- **"Why the surrogate step?"** — The black-box doesn't accept mask inputs. The surrogate is a differentiable approximation that does.
- **"What if the black-box changes?"** — You'd need to retrain the surrogate and InstaSHAP. This is amortized — one retraining enables unlimited instant explanations.
- **"How do you handle categorical features?"** — One-hot encoding + aggregation. The FeatureGroup system tracks the mapping.

---

## 15. FAQ — Common Questions & Answers

### Q: What ML model is used for training?
**A:** Multiple models are trained in a pipeline:
1. **TabularMLP** (2-layer [256,128] MLP with ReLU and dropout) as the black-box prediction model
2. **MaskedSurrogateMLP** (2-layer [256,128] MLP) as the mask-aware approximation
3. **GAMModel** (per-feature ComponentMLPs [96,64]) as interpretable baselines
4. **InstaSHAPModel** (extends GAMModel with masked training) as the final explanation model

All use **PyTorch** with **AdamW optimizer** and **early stopping**.

### Q: What is the loss function?
**A:** 
- Black-box & GAM: MSE (regression) or CrossEntropy (classification)
- Surrogate: MSE between surrogate output and black-box output
- InstaSHAP: MSE between InstaSHAP masked output and surrogate masked output

### Q: What libraries/frameworks are used?
**A:** PyTorch ≥2.2 (neural networks), scikit-learn ≥1.4 (preprocessing, metrics), shap ≥0.45 (permutation SHAP baseline), ucimlrepo (dataset loading), matplotlib+seaborn (plotting), pandas+numpy (data manipulation), PyYAML (configuration), tensorboard (training monitoring).

### Q: How is reproducibility ensured?
**A:** Global seed (42) controls Python's `random`, NumPy's `RandomState`, and PyTorch's random generators. CuDNN deterministic mode is enabled. Classification splits use stratification. All hyperparameters are in `config.yaml`.

### Q: What datasets are used and why these specific ones?
**A:** Three UCI datasets chosen because the paper uses them to demonstrate different interaction types:
- **Bike Sharing** — demonstrates synergistic interaction (hour × workingday)
- **Covertype** — demonstrates redundant interaction (elevation × soil zone)
- **Adult Income** — supplementary benchmark with no highlighted interactions

### Q: What is the difference between GAM-1 and GAM-2?
**A:** GAM-1 has only univariate components (one sub-network per feature). GAM-2 adds pairwise interaction components (e.g., a sub-network for `hour × workingday`). GAM-2 can capture feature interactions that GAM-1 misses, often closing the performance gap with the black-box.

### Q: What are Shapley kernel masks?
**A:** Binary masks that determine which features are "visible" during training. They're sampled from the Shapley kernel distribution, which weights subset sizes by `1/[C(d,|S|) · |S| · (d-|S|)]`. This is NOT uniform random — it favors small and large subsets where marginal contributions are most informative.

### Q: Why is a surrogate needed? Can't InstaSHAP train directly against the black-box?
**A:** The black-box MLP doesn't accept mask inputs — it expects full feature vectors. The surrogate bridges this gap by learning `f(x;S)` (what the black-box would predict if only features S were present). It takes both the masked input and the mask itself as input, making it differentiable with respect to the mask.

### Q: How fast is InstaSHAP compared to traditional SHAP?
**A:** InstaSHAP produces attributions in a single forward pass (milliseconds for a batch). Traditional permutation SHAP requires `O(d · max_evals)` forward passes per sample (seconds to minutes). The speedup is typically 100-1000× depending on the number of features and evaluation budget.

### Q: What outputs does the project generate?
**A:** 
- **CSV tables:** Model metrics, paper comparison values, explanation comparison
- **PNG plots:** Training curves, feature importance bars, shape functions, interaction heatmaps, SHAP vs InstaSHAP alignment
- **JSON summaries:** Full experiment metadata for each dataset
- **PDF reports:** Multi-page reproducibility report and one-page summary

### Q: How can I run the project?
**A:**
```bash
cd instashap_project
pip install -r requirements.txt
python main.py --dataset all --model all --fast-dev-run  # Quick validation
python main.py --dataset all --model all                  # Full run
```

### Q: What does `--fast-dev-run` do?
**A:** Limits datasets to 4,000 rows max and training to 4 epochs. Useful for verifying the pipeline works without waiting for full training.

---

## 16. File-by-File Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `main.py` | CLI entry point | `main()`, `parse_args()`, `load_config()` |
| `config.yaml` | All hyperparameters | Seed, device, training configs, dataset configs |
| `data/loaders.py` | Load UCI datasets | `load_bike_sharing()`, `load_covertype()`, `load_adult_income()`, `DatasetBundle` |
| `data/preprocessing.py` | Feature transformation | `TabularPreprocessor`, `FeatureGroup`, `make_splits()` |
| `models/blackbox_model.py` | Black-box & surrogate | `TabularMLP`, `MaskedSurrogateMLP`, `RandomForestBlackBox` |
| `models/gam.py` | Additive models | `GAMModel`, `ComponentMLP`, `ComponentSpec` |
| `models/instashap.py` | InstaSHAP model | `InstaSHAPModel.masked_forward()`, `.explain()` |
| `training/train.py` | All 4 trainers | `train_blackbox_model()`, `train_masked_surrogate()`, `train_gam_model()`, `train_instashap_model()` |
| `training/evaluate.py` | Prediction helpers | `predict_raw_outputs()`, `evaluate_supervised_model()` |
| `xai/shap_wrapper.py` | SHAP baseline | `ShapBaselineExplainer.explain()` with aggregation |
| `xai/instashap_explainer.py` | Single-pass explainer | `InstaSHAPExplainer.explain()` |
| `experiments/common.py` | Full experiment orchestrator | `run_tabular_experiment()` |
| `experiments/bike_sharing.py` | Bike experiment runner | `run()` with synergy focus |
| `experiments/covertype.py` | Covertype experiment runner | `run()` with redundancy focus |
| `experiments/adult_income.py` | Adult experiment runner | `run()` supplementary |
| `utils/metrics.py` | Metric computation | `regression_metrics()`, `classification_metrics()`, `explanation_error()` |
| `utils/visualization.py` | Plot generation | `plot_training_curves()`, `plot_shape_function()`, `plot_interaction_heatmap()` |
| `utils/reproducibility.py` | Seed control | `set_global_seed()`, `resolve_device()` |
| `utils/logging_utils.py` | Structured logging | `configure_logging()`, `format_log_event()` |
| `reports/generate_report.py` | Multi-page PDF | `generate_full_report()` |
| `reports/summary_1page.py` | One-page PDF | `generate_one_page_summary()` |

---

## 17. Glossary

| Term | Definition |
|------|-----------|
| **SHAP** | SHapley Additive exPlanations — a game-theoretic method for explaining ML predictions |
| **Shapley Value** | The unique fair allocation of a model's prediction among its input features |
| **InstaSHAP** | A trained additive model that produces Shapley values in one forward pass |
| **GAM** | Generalized Additive Model — `f(x) = Σ gᵢ(xᵢ)`, each component depends on one feature |
| **Black-box** | A model whose internals are treated as opaque; we only observe inputs and outputs |
| **Surrogate** | A mask-aware model that approximates the black-box's behavior under feature masking |
| **Feature Mask** | A binary vector indicating which features are "visible" (1) or "hidden" (0) |
| **Shapley Kernel** | A probability distribution over mask sizes that weights subsets for optimal Shapley estimation |
| **Edge Masks** | All-zeros or all-ones masks included for training stability |
| **One-Hot Encoding** | Representing categorical values as binary vectors (one column per category) |
| **Feature Group** | Mapping from one original feature to its transformed columns after preprocessing |
| **Synergy** | Features that interact to produce more combined impact than their individual effects |
| **Redundancy** | Features that share overlapping information |
| **NMSE%** | Normalized Mean Squared Error as a percentage: `100 × MSE/Var(y)` |
| **Early Stopping** | Halting training when validation loss stops improving (patience epochs) |
| **AdamW** | Adam optimizer with decoupled weight decay (improved L2 regularization) |
| **Shape Function** | The learned response curve `gᵢ(xᵢ)` of one GAM component |
| **ComponentMLP** | A small MLP dedicated to learning one feature's (or feature pair's) contribution |
| **UCI Repository** | University of California Irvine ML Repository — standard benchmark dataset source |
| **Permutation SHAP** | SHAP variant that uses feature permutations to estimate marginal contributions |
| **Amortized Explanation** | Pay upfront (training) to get cheap future explanations (single forward pass) |

---

*Generated for the InstaSHAP Reproducibility Project — March 2026*
