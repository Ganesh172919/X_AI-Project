# Phase 3: InstaSHAP with Three Research Innovations

## Improving InstaSHAP via Empirical-Background Masking, Curriculum-Weighted Shapley Training, and Multi-Surrogate Ensembling

> **Base Paper:** InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly  
> **Authors:** James Enouen, Yan Liu  
> **Venue:** International Conference on Learning Representations (ICLR) 2025  
> **Paper Link:** https://openreview.net/forum?id=ky7vVlBQBY

> **Extension:** Three targeted research improvements addressing fundamental limitations in the masking strategy, training schedule, and explanation stability of the original InstaSHAP pipeline.

> **Dataset:** Forest Covertype (UCI ML Repository, ID: 31) — 7-class classification with 10 numeric + 1 categorical feature  
> **Tech Stack:** Python 3.10+, PyTorch 2.2+, scikit-learn, SHAP, ucimlrepo

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Original InstaSHAP Background](#2-original-instashap-background)
3. [Complete Gap Analysis](#3-complete-gap-analysis)
4. [Innovation 1: Empirical-Background Masking](#4-innovation-1-empirical-background-masking)
5. [Innovation 2: Curriculum-Weighted Shapley Training](#5-innovation-2-curriculum-weighted-shapley-training)
6. [Innovation 3: Multi-Surrogate Ensemble](#6-innovation-3-multi-surrogate-ensemble)
7. [Experiment Design](#7-experiment-design)
8. [Pipeline Architecture](#8-pipeline-architecture)
9. [Project Structure & Code Organization](#9-project-structure--code-organization)
10. [Installation & Setup](#10-installation--setup)
11. [How to Run](#11-how-to-run)
12. [Configuration Reference](#12-configuration-reference)
13. [Metrics & Evaluation](#13-metrics--evaluation)
14. [Generated Deliverables](#14-generated-deliverables)
15. [Reproducibility](#15-reproducibility)
16. [References](#16-references)

---

## 1. Project Overview

### What This Project Does

This project takes the **InstaSHAP** method from ICLR 2025 — which produces Shapley-faithful feature attributions in a single neural network forward pass — and identifies **three critical weaknesses** in the original implementation. We then design, implement, and rigorously evaluate **three targeted innovations** that address these weaknesses:

| Innovation | Gap Addressed | Core Idea |
|------------|--------------|-----------|
| **Empirical-Background Masking** | Off-distribution masked inputs | Replace absent features with real training data instead of zeros |
| **Curriculum-Weighted Training** | Uniform coalition difficulty | Progressive easy→hard mask scheduling during surrogate training |
| **Multi-Surrogate Ensemble** | Single-surrogate fragility | Average predictions from 3 independent surrogates + variance as confidence |

### Why Covertype?

We focus on the **Forest Covertype** dataset because:
- It contains a **4-category one-hot feature** (`soil_climate_zone`) where zero-masking is maximally harmful (zeroing all one-hot columns creates an impossible "no category" state)
- The `elevation × soil_climate_zone` interaction is **well-documented** in the paper as a redundant interaction, making it ideal for studying how masking strategy affects interaction learning
- Phase 2 results showed the weakest InstaSHAP performance on Covertype (**48.75% accuracy** vs paper's GAM-2 target of 82.2%), confirming substantial room for improvement
- The 7-class classification task is complex enough to expose surrogate approximation errors

### What Makes This a Research Contribution

This is not simply a re-implementation. The project:
1. **Identifies gaps** in a published ICLR 2025 paper through systematic analysis
2. **Proposes defensible improvements** grounded in 2025–2026 research literature
3. **Implements layered innovations** that enable clean ablation analysis
4. **Provides rigorous experimental evidence** via 3-seed comparison with statistical reporting
5. **Produces reusable artifacts** (code, tables, plots, PDFs) suitable for a research supplement

---

## 2. Original InstaSHAP Background

### The Core Problem InstaSHAP Solves

Computing Shapley values — the gold standard for feature attribution — requires evaluating a model on **all 2^n possible feature subsets**. For n=11 features (Covertype), that's 2,048 subsets × hundreds of evaluation samples. InstaSHAP collapses this into a **single forward pass** by training an additive model whose components naturally recover Shapley values.

### The Four-Stage Pipeline

```
Stage 1: Black-Box MLP         → Train accurate classifier (the model to explain)
Stage 2: Masked Surrogate      → Learn to approximate f(x; S) for any subset S
Stage 3: InstaSHAP GAM         → Additive model trained against surrogate under Shapley masks
Stage 4: Explain                → Each component output IS the SHAP attribution
```

### Key Mathematical Foundation

**Shapley value** of feature i for input x:
```
φᵢ(x) = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)!/|N|!] × [f(x; S∪{i}) - f(x; S)]
```

**Masked surrogate objective:**
```
h(x, S) minimizes E_{x,S}[||h(x·m_S, S) - f(x; S)||²]
```

**InstaSHAP training objective (Eq. 20 from paper):**
```
g(x; S) = b + Σ_{i∈S} gᵢ(xᵢ) minimizes E_{x,S}[||g(x; S) - h(x·m_S, S)||²]
```

**Key theorem:** Under masks drawn from the Shapley kernel distribution, the optimal additive model recovers exact Shapley values: **gᵢ(xᵢ) = φᵢ(x)**.

---

## 3. Complete Gap Analysis

We identified **5 weaknesses** in the original InstaSHAP paper and implementation. The **3 most impactful** are addressed as innovations:

### Gap 1: Zero-Masking Creates Off-Distribution Inputs 🔴 CRITICAL

**The Problem:** The original code simulates absent features via `x * mask`, which replaces masked features with **zero in standardized space**:

```python
# Original Phase 2 code (train.py, line 260):
predictions = model(inputs * expanded_mask, feature_mask)
```

This creates two critical issues:
- **Invalid categorical states:** For one-hot groups like `soil_climate_zone` (4 categories), zeroing all columns creates a state [0,0,0,0] meaning "no category" — this **never occurs in real data**. The surrogate learns to predict from physically impossible inputs.
- **Broken correlation structure:** For correlated features like `elevation` × `soil_climate_zone`, zero-masking destroys the natural covariance. The surrogate cannot learn the true conditional expectation E[f(x) | x_S] when the masked inputs violate the data manifold.

**Evidence from Phase 2:**
| Metric | Phase 2 Value | Paper Target | Gap |
|--------|--------------|--------------|-----|
| InstaSHAP Accuracy | 48.75% | ~82% (GAM-2) | -33 points |
| Explanation MSE vs SHAP | 0.110 | Near-zero | Very poor |
| Explanation MAE vs SHAP | 0.292 | Near-zero | Very poor |

**Status:** ✅ Addressed by Innovation 1

### Gap 2: Uniform-Difficulty Mask Training 🟠 HIGH

**The Problem:** The Shapley kernel distribution samples masks from:
```
p(|S|) ∝ 1 / [C(n, |S|) × |S| × (n − |S|)]
```

This up-weights extreme coalition sizes (|S|=1 and |S|=n-1) **uniformly throughout training**. But:
- Sparse coalitions (|S|=1,2) are the **hardest** to learn — predicting from almost no features when the surrogate hasn't yet learned basic feature-prediction mappings
- Near-full coalitions (|S|=n-1) are the **easiest** — almost all features present
- Training all difficulties equally from epoch 1 wastes capacity on tasks the model isn't ready for

**Status:** ✅ Addressed by Innovation 2

### Gap 3: Single-Surrogate Fragility 🟠 HIGH

**The Problem:** InstaSHAP trains against a **single surrogate model**. This creates:
- **Error cascade:** If the surrogate approximates f(x;S) poorly for certain feature combinations, every InstaSHAP explanation inherits that error
- **No uncertainty signal:** Point-estimate attributions with no way to know which are reliable (2025–2026 research on "explanation multiplicity" confirms this is a real-world problem)
- **Seed sensitivity:** Different random seeds produce different surrogates with different error profiles; a single surrogate amplifies this variance

**Status:** ✅ Addressed by Innovation 3

### Gap 4: No Higher-Order Interaction Detection 🟡 MEDIUM

**The Problem:** The InstaSHAP GAM architecture supports only manually specified pairwise interactions. It cannot automatically detect or score which interactions matter most.

**Status:** ❌ Out of scope (would require architecture changes beyond surrogate training)

### Gap 5: No Explanation Confidence Scores 🟡 MEDIUM

**The Problem:** InstaSHAP outputs deterministic point-estimate attributions with no uncertainty quantification. Users cannot distinguish high-confidence explanations from noisy ones.

**Status:** ⚠️ Partially addressed by Innovation 3 (ensemble variance provides a confidence proxy)

---

## 4. Innovation 1: Empirical-Background Masking

### Concept

Instead of zeroing absent features, replace them with values from **real training rows**:

```
Zero-masking (original):     x_masked = x * mask          → x_absent = 0
Background masking (ours):   x_masked = x * mask + z * (1-mask)   → x_absent = z (from training data)
```

where `z` is a randomly sampled training row (the "background" sample).

### Why It Works

1. **Preserves categorical validity:** When `soil_climate_zone` is masked, the replacement comes from a real training row — so the one-hot encoding is a valid [0,0,1,0] or [1,0,0,0], never [0,0,0,0]
2. **Preserves marginal distribution:** Background values are drawn from the empirical distribution of the training data, so masked inputs stay on the data manifold
3. **Approximates true conditional expectation:** The SHAP framework defines f(x;S) = E[f(x) | x_S], which requires marginalizing over absent features using their distribution. Sampling from training data is a Monte Carlo approximation to this integral

### Implementation Details

- A **background bank** of 256 random training rows is pre-sampled and stored
- During training: K=1 background sample per coalition (fast, since we sample many masks)
- During evaluation: K=4 background samples per coalition (precise, averaged)
- For each masked input, feature groups are replaced **as whole groups** (all one-hot columns together)
- Background-averaged blackbox outputs serve as the surrogate training target

### Key File: `masking/background_mask.py`

Core function `apply_background_mask()` constructs masked inputs by iterating over feature groups and swapping absent groups with background row values.

### Literature Support

| Reference | Contribution |
|-----------|-------------|
| Lundberg & Lee (NeurIPS 2017) | SHAP defines f(x;S) as conditional expectation; zero-masking violates this |
| Aas et al. (2019) | Shows feature-independence assumption creates errors for correlated features |
| Frye et al. (2020) | Argues explanations should stay on the data manifold; zero-masking pushes off-manifold |
| ViaSHAP (ICML 2025) | Demonstrates baseline selection is critical; recommends context-aware baselines |
| 2026 XAI Surveys | Recent work uses clustered training data for representative baselines |

### New Metric: Coalition Fidelity

```
Coalition Fidelity MSE = E[||surrogate(x_masked, m) - f_true(x_masked)||²]
```
where f_true is computed by averaging the blackbox over multiple background replacements. This directly measures how well each masking strategy helps the surrogate learn the correct value function.

---

## 5. Innovation 2: Curriculum-Weighted Shapley Training

### Concept

Instead of sampling masks from a **static** Shapley kernel distribution, use a **3-phase temperature-controlled schedule** that progressively increases difficulty:

| Phase | Epoch Range | Temperature | Coalition Bias | Rationale |
|-------|------------|-------------|----------------|-----------|
| **Warm-up** | 0% – 25% | τ = 3.0 | Favor large |S| | Let surrogate learn with most features visible |
| **Standard** | 25% – 65% | τ = 1.0 | Exact Shapley kernel | Transition to theoretically correct distribution |
| **Hard** | 65% – 100% | τ = 0.3 | Emphasize small |S| | Focus on hardest, most informative coalitions |

### Temperature Mechanism

The tempered Shapley kernel modifies weights via:
```
p_τ(|S|) ∝ [1 / (C(n,|S|) × |S| × (n-|S|))]^(1/τ)
```

- **τ > 1 (warm-up):** Flattens the distribution → more uniform coalition sizes → more large coalitions
- **τ = 1 (standard):** Exact Shapley kernel → theoretically correct
- **τ < 1 (hard):** Sharpens the distribution → concentrates on extreme sizes → forces learning from sparse information

### Why It Works

1. **Stable early learning:** The surrogate first learns basic feature-prediction relationships with most features visible (easy coalitions), establishing a solid foundation
2. **Efficient capacity use:** Hard coalitions (|S|=1,2) are introduced only after the surrogate has "graduated" from basic patterns
3. **Better final quality:** The surrogate spends its final training epochs on the coalitions that matter most for Shapley value accuracy — the sparse ones that carry highest Shapley kernel weight

### Key File: `masking/curriculum.py`

Core function `curriculum_shapley_masks()` takes `epoch` and `total_epochs` parameters and returns masks sampled from the tempered distribution.

### Literature Support

| Reference | Contribution |
|-----------|-------------|
| Bengio et al. (ICML 2009) | Foundational curriculum learning: easy→hard improves generalization |
| Pruning as Cooperative Game (ICLR 2026) | Stratified Monte Carlo coalition sampling for cooperative games |
| Turaco (2025) | Complexity-guided sampling: surrogates should over-sample complex regions |
| Progressive Alignment (NeurIPS 2025) | Multi-phase training improves convergence in complex models |

### New Metric: Convergence Speed

```
Convergence Epoch = first epoch where val_loss ≤ best_val_loss / 0.95
```
Measures how many epochs until the surrogate reaches 95% of its final best quality. Faster convergence = more training-efficient.

---

## 6. Innovation 3: Multi-Surrogate Ensemble

### Concept

Train **M=3 independent surrogates** with different random seeds, then:
1. **Average** their predictions to create a smoother, more robust training signal for InstaSHAP
2. **Compute variance** across surrogates as an **explanation confidence score**

```
Architecture:
    Blackbox f(x)
        ├── Surrogate₁ (seed=42+777)     ──┐
        ├── Surrogate₂ (seed=42+1777)    ──┼── mean → InstaSHAP training target
        └── Surrogate₃ (seed=42+2777)    ──┘
                                            └── variance → confidence score
```

### Why It Works

1. **Error cancellation:** Individual surrogate errors are partially uncorrelated (different random initializations, different mask sequences). Averaging reduces the total error by up to √M
2. **Smoother training signal:** InstaSHAP receives a less noisy target, leading to better Shapley value recovery
3. **Uncertainty quantification:** High variance across surrogates for a given (x, S) indicates the approximation is unreliable — this information was previously invisible to the user
4. **Marginal cost:** Training 3 surrogates costs 3× surrogate training time, but surrogate training is typically <30% of total pipeline time, so the overall increase is modest (~2× total time)

### Implementation Details

- `SurrogateEnsemble` class wraps `nn.ModuleList` of M surrogates
- `forward()` averages all surrogate outputs for training
- `forward_all()` returns individual outputs for variance computation
- Each surrogate uses a different seed for weight initialization and mask sampling
- The same architecture and hyperparameters are used for all ensemble members

### Key File: `models/blackbox_model.py` (SurrogateEnsemble class) + `training/train.py` (train_surrogate_ensemble function)

### Literature Support

| Reference | Contribution |
|-----------|-------------|
| Explanation Multiplicity (2026) | Shows SHAP explanations vary significantly across runs; ensembling reduces variance |
| Multiplicative Smoothing MuS (2025) | Formal stability guarantees for attributions via smoothing |
| Conformal Prediction for XAI (2025) | Integrates uncertainty quantification with SHAP |
| LeverageSHAP (2025) | Sample-efficient SHAP with better convergence guarantees via variance reduction |
| Jethani et al. (2021) — FastSHAP | Identifies surrogate quality as the key bottleneck; ensembling directly addresses this |

### New Metric: Explanation Stability Score

```
Stability(x) = 1 - mean_i(std_i / (|mean_i| + ε))
```
where std_i and mean_i are computed across the M ensemble surrogates for feature i. A score of 1.0 = perfectly consistent explanations; lower = less reliable. This can be reported per-sample or as a dataset-wide average.

---

## 7. Experiment Design

### 4-Variant Ablation Study

The three innovations are **layered** — each builds on the previous — enabling a clean ablation analysis:

| Variant ID | Masking Strategy | Curriculum | Ensemble | What It Tests |
|------------|-----------------|------------|----------|---------------|
| `instashap_zero` | Zero (x × mask) | ✗ | ✗ | Baseline reproduction of Phase 2 |
| `instashap_bg` | Background | ✗ | ✗ | Isolated impact of Innovation 1 |
| `instashap_curriculum` | Background | ✓ | ✗ | Marginal impact of Innovation 2 |
| `instashap_full` | Background | ✓ | ✓ | Full pipeline with all 3 innovations |

### Shared Components

All 4 variants share:
- The **same blackbox** MLP (trained once per seed)
- The **same GAM-1 and GAM-2** models (trained once per seed)
- The **same train/val/test split** (determined by seed)
- The **same Permutation SHAP ground truth** (computed once per seed)
- The **same preprocessor** (fitted once on training data)

This controlled design ensures that **any performance difference between variants is attributable solely to the masking/training/ensemble innovations**.

### Multi-Seed Protocol

- **3 seeds:** 42, 123, 7
- All metrics reported as **mean ± std** across seeds
- 3 seeds is standard for academic XAI studies (balances compute cost with statistical confidence)
- `--fast-dev-run` uses 1 seed for rapid validation

### Models Trained Per Seed

| Model | Architecture | Training Objective | Shared? |
|-------|-------------|-------------------|---------|
| `blackbox` | MLP [256, 128] | Cross-entropy vs true labels | ✓ All variants |
| `gam1` | GAM [96, 64] per feature | Cross-entropy vs true labels | ✓ All variants |
| `gam2` | GAM [96, 64] with (elevation, soil) | Cross-entropy vs true labels | ✓ All variants |
| `surrogate_zero` | Masked MLP [256, 128] | MSE vs blackbox outputs (zero-mask) | `instashap_zero` only |
| `surrogate_bg` | Masked MLP [256, 128] | MSE vs blackbox outputs (bg-mask) | `instashap_bg` only |
| `surrogate_curriculum` | Masked MLP [256, 128] | MSE vs blackbox outputs (bg + curriculum) | `instashap_curriculum` only |
| `surrogate_ensemble` | 3 × Masked MLP [256, 128] averaged | MSE vs blackbox outputs (bg + curriculum) | `instashap_full` only |
| `instashap_*` | GAM [96, 64] with interactions | MSE vs respective surrogate | One per variant |

---

## 8. Pipeline Architecture

### End-to-End Flow (Per Seed)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING & PREPROCESSING                                                 │
│     load_covertype() → TabularPreprocessor (StandardScaler + OneHotEncoder)       │
│     → 70/10/20 stratified split                                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│  2. SHARED MODELS (trained once)                                                 │
│     Black-Box MLP [256→128→7]   │   GAM-1 [96→64 × 11]   │   GAM-2 [+interaction]│
├──────────────────────────────────────────────────────────────────────────────────┤
│  3. BACKGROUND BANK (pre-sampled from training data, 256 rows)                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│  4. VARIANT BRANCHES (run in sequence)                                           │
│                                                                                  │
│     ┌─ instashap_zero ──────────┐   ┌─ instashap_bg ────────────┐              │
│     │  Surrogate: x*mask (zero) │   │  Surrogate: x*mask + bg   │              │
│     │  InstaSHAP: vs surrogate  │   │  InstaSHAP: vs surrogate  │              │
│     └───────────────────────────┘   └────────────────────────────┘              │
│                                                                                  │
│     ┌─ instashap_curriculum ────┐   ┌─ instashap_full ──────────┐              │
│     │  Surrogate: bg + curriculum│   │  3× Surrogate (ensemble)  │              │
│     │  InstaSHAP: vs surrogate  │   │  InstaSHAP: vs avg(3)     │              │
│     └───────────────────────────┘   └────────────────────────────┘              │
├──────────────────────────────────────────────────────────────────────────────────┤
│  5. PERMUTATION SHAP (ground truth, 24 test samples, 256 max evals)             │
├──────────────────────────────────────────────────────────────────────────────────┤
│  6. COMPARISON: each variant's InstaSHAP attributions vs SHAP ground truth       │
│     → MSE, MAE, Spearman ρ                                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│  7. RESULTS: CSV tables, JSON artifacts, plots, PDF reports                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Project Structure & Code Organization

```
Phase_3_work/
│
├── main.py                              ← CLI entry point (argparse)
├── config.yaml                          ← All hyperparameters (YAML)
├── requirements.txt                     ← Python dependencies
├── README.md                            ← This file
├── AI_USAGE.md                          ← AI usage declaration
│
├── data/                                ← Data loading & preprocessing
│   ├── __init__.py
│   ├── loaders.py                       ← Covertype UCI loader + DatasetBundle
│   └── preprocessing.py                 ← TabularPreprocessor + FeatureGroup + background bank
│
├── masking/                             ← Masking strategies (core innovation)
│   ├── __init__.py
│   ├── config.py                        ← MaskingConfig dataclass with factory methods
│   ├── zero_mask.py                     ← Baseline: x * expanded_mask (Phase 2 reproduction)
│   ├── background_mask.py              ← Innovation 1: empirical-background replacement
│   └── curriculum.py                    ← Innovation 2: temperature-scheduled mask sampling
│
├── models/                              ← Neural network architectures
│   ├── __init__.py
│   ├── blackbox_model.py                ← TabularMLP, MaskedSurrogateMLP, SurrogateEnsemble
│   ├── gam.py                           ← GAMModel with ComponentMLP + feature gating
│   └── instashap.py                     ← InstaSHAPModel (masked_forward + explain)
│
├── training/                            ← Training loops
│   ├── __init__.py
│   ├── train.py                         ← 4 training functions + ensemble builder
│   └── evaluate.py                      ← predict_raw_outputs, predict_classes, etc.
│
├── experiments/                         ← Experiment orchestration
│   ├── __init__.py
│   └── covertype_comparison.py          ← Full 4-variant × 3-seed pipeline
│
├── xai/                                 ← Explainability methods
│   ├── __init__.py
│   ├── instashap_explainer.py           ← Single-pass InstaSHAP attribution
│   └── shap_wrapper.py                  ← Permutation SHAP with group aggregation
│
├── utils/                               ← Shared utilities
│   ├── __init__.py
│   ├── metrics.py                       ← Standard + innovation-specific metrics
│   ├── visualization.py                 ← 10+ plot types including comparison charts
│   ├── reproducibility.py              ← Seed control, device, JSON I/O
│   └── logging_utils.py                ← Structured logging
│
├── reports/                             ← Report generators + output PDFs
│   ├── generate_experiment_report.py    ← Multi-page PDF (matplotlib PdfPages)
│   ├── generate_research_gap.py         ← 1-page compact gap PDF
│   ├── phase3_experiment_report.md      ← Generated markdown companion
│   └── phase3_research_gap_1page.md     ← Generated gap markdown
│
└── results/                             ← Generated at runtime
    ├── tables/
    │   ├── covertype_model_metrics.csv
    │   └── covertype_explanation_comparison.csv
    ├── plots/covertype/
    │   ├── innovation_accuracy_bars.png
    │   ├── innovation_mse_bars.png
    │   ├── innovation_rho_bars.png
    │   ├── all_models_accuracy.png
    │   ├── innovation_radar.png
    │   └── seed_*/                      ← Per-seed plots
    │       ├── convergence_comparison.png
    │       ├── explanation_scatter.png
    │       └── gam2_shape_functions.png
    ├── artifacts/covertype/
    │   ├── covertype_summary.json       ← Aggregated metrics
    │   └── per_seed_results.json        ← Raw per-seed data
    └── run.log                          ← Structured log file
```

### Key Module Responsibilities

| Module | Lines (approx) | Responsibility |
|--------|----------------|---------------|
| `data/loaders.py` | ~120 | UCI dataset fetching, soil-type climate grouping, DatasetBundle |
| `data/preprocessing.py` | ~180 | StandardScaler, OneHotEncoder, FeatureGroup mapping, background bank |
| `masking/config.py` | ~60 | MaskingConfig dataclass with factory methods for each variant |
| `masking/zero_mask.py` | ~30 | Baseline zero-masking (x × expanded_mask) |
| `masking/background_mask.py` | ~70 | Background replacement + averaged blackbox targets |
| `masking/curriculum.py` | ~80 | Temperature-controlled Shapley kernel + 3-phase schedule |
| `models/blackbox_model.py` | ~100 | TabularMLP, MaskedSurrogateMLP, SurrogateEnsemble |
| `models/gam.py` | ~110 | GAMModel with component gating and attributions |
| `models/instashap.py` | ~20 | InstaSHAPModel (thin wrapper over GAMModel) |
| `training/train.py` | ~300 | All 4 training stages + ensemble builder |
| `training/evaluate.py` | ~50 | Inference helpers (raw outputs, classes, probabilities) |
| `experiments/covertype_comparison.py` | ~350 | Full orchestrator with plotting and aggregation |
| `utils/metrics.py` | ~100 | Standard + 5 innovation-specific metrics |
| `utils/visualization.py` | ~280 | 10+ plot types for comparison and analysis |
| `reports/generate_experiment_report.py` | ~200 | Multi-page PDF with tables and embedded plots |
| `reports/generate_research_gap.py` | ~130 | 1-page compact PDF |
| `main.py` | ~60 | argparse CLI with 4 subcommands |

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- ~2 GB free disk space for dataset download and results
- GPU optional (CUDA auto-detected; falls back to CPU)

### Install Dependencies

```bash
cd Phase_3_work
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.2 | Neural network training and inference |
| `scikit-learn` | ≥ 1.4 | Preprocessing, splits, metrics |
| `shap` | ≥ 0.45 | Permutation SHAP baseline |
| `numpy` | ≥ 1.26 | Array operations |
| `pandas` | ≥ 2.2 | Data manipulation, CSV export |
| `ucimlrepo` | ≥ 0.0.7 | UCI dataset fetching |
| `matplotlib` | ≥ 3.8 | Plotting + PDF generation |
| `seaborn` | ≥ 0.13 | Statistical visualizations |
| `scipy` | ≥ 1.13 | Spearman rank correlation |
| `PyYAML` | ≥ 6.0 | Config parsing |
| `tqdm` | ≥ 4.66 | Progress bars |
| `tensorboard` | ≥ 2.16 | Optional training visualization |

---

## 11. How to Run

### Full Experiment (Recommended — 3 Seeds)

```bash
python main.py --variant compare
```
- Runs all 4 variants × 3 seeds
- Generates tables, plots, JSON, and PDF reports
- **Estimated time:** 30–45 minutes on CPU, 15–20 minutes on GPU

### Quick Smoke Test

```bash
python main.py --variant compare --fast-dev-run
```
- 1 seed, 4,000 rows, 4 epochs per stage
- Validates entire pipeline in ~5 minutes
- Use this to verify setup before full run

### Individual Variants

```bash
python main.py --variant baseline      # Zero-mask pipeline only
python main.py --variant improved      # All 3 innovations only
```

### Regenerate Reports Only

```bash
python main.py --report-only           # Regenerate PDFs from existing results/
```

### Additional Flags

| Flag | Description |
|------|-------------|
| `--config path.yaml` | Use a custom config file |
| `--log-level DEBUG` | Verbose logging |
| `--skip-report` | Skip PDF generation (just metrics + plots) |

---

## 12. Configuration Reference

All hyperparameters are in `config.yaml`. Key sections:

### Global Settings
```yaml
global:
  seeds: [42, 123, 7]           # Multi-seed experiment
  device: auto                   # auto | cpu | cuda
  shap_background_size: 64      # Background samples for Permutation SHAP
  shap_eval_samples: 24         # Test samples for SHAP comparison
  shap_max_evals: 256           # Max evaluations per SHAP sample
```

### Masking Configuration (Innovations)
```yaml
masking:
  background_bank_size: 256      # Rows in background bank (Innovation 1)
  background_samples_train: 1    # K per coalition during training
  background_samples_eval: 4     # K per coalition during evaluation
  curriculum_warmup_frac: 0.25   # Fraction of epochs for warm-up (Innovation 2)
  curriculum_standard_frac: 0.40 # Fraction for standard Shapley kernel
  ensemble_size: 3               # Number of surrogates (Innovation 3)
```

### Training Hyperparameters
```yaml
training:
  blackbox:     { hidden_dims: [256,128], dropout: 0.10, lr: 0.001, epochs: 25, patience: 5 }
  gam:          { hidden_dims: [96,64],   dropout: 0.05, lr: 0.001, epochs: 35, patience: 6 }
  surrogate:    { hidden_dims: [256,128], dropout: 0.10, lr: 0.001, epochs: 20, patience: 5 }
  instashap:    { hidden_dims: [96,64],   dropout: 0.05, lr: 0.001, epochs: 35, patience: 6 }
```

---

## 13. Metrics & Evaluation

### Predictive Performance Metrics

| Metric | Task | Formula | Used For |
|--------|------|---------|----------|
| **Accuracy** | Classification | correct/total | All models |
| **Log-Loss** | Classification | -Σ y·log(p) | All models |

### Explanation Fidelity Metrics (vs Permutation SHAP)

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Explanation MSE** | mean((SHAP - InstaSHAP)²) | Squared alignment error (lower = better) |
| **Explanation MAE** | mean(\|SHAP - InstaSHAP\|) | Absolute alignment error (lower = better) |
| **Spearman ρ** | Rank correlation per sample, averaged | Feature importance ranking agreement (higher = better) |

### Innovation-Specific Metrics

| Metric | Innovation | What It Measures |
|--------|-----------|-----------------|
| **Coalition Fidelity MSE** | 1 | How well surrogate approximates true masked value function |
| **Convergence Epoch** | 2 | Epochs until surrogate reaches 95% of best val-loss |
| **Explanation Stability Score** | 3 | 1 - mean CV across ensemble (higher = more stable) |
| **Per-Feature Confidence** | 3 | Per-feature reliability from ensemble variance |

### Runtime Metrics

| Metric | What It Measures |
|--------|-----------------|
| Surrogate training time | Wall-clock seconds for surrogate training |
| InstaSHAP training time | Wall-clock seconds for InstaSHAP training |
| SHAP computation time | Permutation SHAP total time (baseline) |

---

## 14. Generated Deliverables

### CSV Tables

| File | Content |
|------|---------|
| `results/tables/covertype_model_metrics.csv` | Accuracy and log-loss for blackbox, GAM-1, GAM-2, and all InstaSHAP variants |
| `results/tables/covertype_explanation_comparison.csv` | MSE, MAE, Spearman ρ, convergence epoch for each variant |

### JSON Artifacts

| File | Content |
|------|---------|
| `results/artifacts/covertype/covertype_summary.json` | Aggregated mean±std metrics across all seeds |
| `results/artifacts/covertype/per_seed_results.json` | Raw metrics for each individual seed |

### Plots

| Plot | Type | Description |
|------|------|-------------|
| `innovation_accuracy_bars.png` | Grouped bar | 4-variant accuracy comparison with error bars |
| `innovation_mse_bars.png` | Grouped bar | 4-variant explanation MSE comparison |
| `innovation_rho_bars.png` | Grouped bar | 4-variant Spearman ρ comparison |
| `all_models_accuracy.png` | Bar chart | All 7 models with paper benchmark lines |
| `innovation_radar.png` | Radar chart | Multi-metric radar across variants |
| `convergence_comparison.png` | Line plot | Surrogate val-loss curves with 95% threshold |
| `explanation_scatter.png` | Multi-panel | SHAP vs InstaSHAP per feature per variant |
| `gam2_shape_functions.png` | Line/bar | GAM-2 learned shape functions |
| `*_curves.png` | Line plot | Per-model training/validation loss curves |

### PDF Reports

| File | Content |
|------|---------|
| `reports/phase3_experiment_report.pdf` | Multi-page: title, gaps, innovations, results tables, all plots, conclusions |
| `reports/phase3_research_gap_1page.pdf` | Compact 1-page: gaps, innovations, results table, references |

### Markdown Companions

| File | Content |
|------|---------|
| `reports/phase3_experiment_report.md` | Markdown version with embedded plot links |
| `reports/phase3_research_gap_1page.md` | Markdown version of research gap summary |

---

## 15. Reproducibility

### Deterministic Execution

| Control | Implementation |
|---------|---------------|
| **Global seeds** | `set_global_seed()` → Python, NumPy, PyTorch, CuDNN |
| **Deterministic CuDNN** | `torch.backends.cudnn.deterministic = True` |
| **Stratified splits** | Classification uses `sklearn.model_selection.train_test_split(stratify=y)` |
| **Config-driven** | All hyperparameters in `config.yaml` — no hardcoded magic numbers |
| **Structured logging** | Every stage logs to `results/run.log` |
| **Device auto-detection** | CUDA if available, CPU fallback |

### Expected Timeline

| Stage | CPU Time | GPU Time |
|-------|----------|----------|
| Data loading + preprocessing | ~10s | ~10s |
| Black-box training | ~30s | ~10s |
| GAM-1 + GAM-2 training | ~60s | ~20s |
| 4 surrogate variants | ~4 min | ~2 min |
| 4 InstaSHAP variants | ~4 min | ~2 min |
| Permutation SHAP | ~30s | ~15s |
| Plotting + reports | ~10s | ~10s |
| **Total per seed** | **~10 min** | **~5 min** |
| **Total 3 seeds** | **~30 min** | **~15 min** |

---

## 16. References

1. **Enouen, J. & Liu, Y.** (ICLR 2025). *InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly.* https://openreview.net/forum?id=ky7vVlBQBY
2. **Lundberg, S. & Lee, S.** (NeurIPS 2017). *A Unified Approach to Interpreting Model Predictions.*
3. **Jethani, N. et al.** (2021). *FastSHAP: Real-Time Shapley Value Estimation.*
4. **Aas, K. et al.** (2019). *Explaining Individual Predictions When Features Are Dependent.*
5. **Frye, C. et al.** (2020). *Shapley Explainability on the Data Manifold.*
6. **Tsai, C. et al.** (JMLR 2024). *Faith-Shap: The Faithful Shapley Interaction Index.*
7. **Muschalik, M. et al.** (NeurIPS 2024). *shapiq: Shapley Interactions for Machine Learning.*
8. **ViaSHAP** (ICML 2025). *Integrated Shapley Training with Context-Aware Baselines.*
9. **Pruning as a Cooperative Game** (ICLR 2026). *Stratified Monte Carlo Coalition Sampling for Neural Network Pruning.*
10. **Explanation Multiplicity** (2026). *Stability of Feature Attributions Across Multiple Runs.*
11. **Bengio, Y. et al.** (ICML 2009). *Curriculum Learning.*
12. **LeverageSHAP** (2025). *Sample-Efficient SHAP with Variance Reduction via Leverage Score Sampling.*

---

> *This project was developed as a Phase 3 research extension demonstrating the ability to identify gaps in published research, propose defensible improvements, and validate them through rigorous experimentation.*
