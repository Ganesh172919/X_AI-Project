# Phase 3: Research Gap and Extension — Architecture Documentation

## Table of Contents

- [Overview](#overview)
- [Research Gap](#research-gap)
- [Module Structure](#module-structure)
- [Extension Modules (`extension/`)](#extension-modules-extension)
- [Gap Analysis (`gap_analysis/`)](#gap-analysis-gap_analysis)
- [References (`references/`)](#references-references)
- [Experiment Layer (`experiments/`)](#experiment-layer-experiments)
- [Phase 3 ↔ Phase 2 Dependencies](#phase-3--phase-2-dependencies)

---

## Overview

Phase 3 addresses a concrete limitation of the original InstaSHAP method: **purely additive surrogates cannot represent pairwise or higher-order feature interactions**. When the black-box model's behavior depends on feature interactions, the additive surrogate's fidelity drops, and the derived InstaSHAP attributions become less accurate.

The extension implements an **Interaction-Aware InstaSHAP** pipeline that replaces the additive-only EBM surrogate with a GA²M-style surrogate that includes pairwise interaction terms. The interaction contributions are fairly allocated back to individual features using an equal-split rule.

---

## Research Gap

### The Limitation

The original InstaSHAP (Phase 2) fits an EBM surrogate with `interactions=0`:

```
f_surrogate(x) = β₀ + f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ)
```

This additive structure cannot capture terms like `f₁₂(x₁, x₂)`. If the black-box model uses such interactions, the surrogate is misspecified:

1. **Surrogate fidelity drops** — The additive surrogate's R² when predicting black-box outputs decreases
2. **Attribution accuracy drops** — The InstaSHAP values diverge from Exact SHAP (lower Pearson correlation)
3. **Feature-removal faithfulness degrades** — Removing top features by InstaSHAP attribution causes less prediction change than removing by Exact SHAP

### Why It Matters

- **Practically**: Real tabular datasets (e.g., `friedman1` with its `x₁·x₂` term) contain interaction effects
- **Theoretically**: Shapley values are defined on the true model behavior. A misspecified surrogate leads to inaccurate Shapley approximations
- **Evidence**: The `friedman1` benchmark has a known `x₁·x₂` interaction; additive InstaSHAP performs measurably worse here than on purely additive datasets like `diabetes`

### Proposed Extension

Replace the additive surrogate with a **GA²M-style surrogate**:

```
f_surrogate(x) = β₀ + f₁(x₁) + ... + fₚ(xₚ) + f₁₂(x₁,x₂) + ...
```

The pairwise interaction terms `fᵢⱼ(xᵢ, xⱼ)` are allocated to features `i` and `j` using a 50/50 split rule, consistent with the general term allocator from Phase 2's `compute_instashap_values()`.

---

## Module Structure

```
phase3/
├── extension/
│   ├── interaction_aware_surrogate.py   # GA²M-style surrogate fitting
│   ├── enhanced_instashap.py            # Interaction-aware attribution
│   └── adaptive_surrogate.py            # Adaptive upgrade strategy
├── gap_analysis/
│   └── research_gap.md                  # Written gap analysis
├── references/
│   └── supporting_references.md         # Bibliography
├── experiments/
│   ├── experiment_gap_demonstration.py  # Show where additive breaks
│   ├── experiment_extension_accuracy.py # Original vs extension accuracy
│   ├── experiment_extension_runtime.py  # Runtime benchmarks
│   └── experiment_comparison.py         # Cross-dataset comparison
└── notebooks/
    └── extension_walkthrough.ipynb      # Interactive walkthrough
```

---

## Extension Modules (`extension/`)

### `interaction_aware_surrogate.py`

**File**: `phase3/extension/interaction_aware_surrogate.py`
**Role**: Fits a GA²M-style EBM surrogate with pairwise interaction terms. This is the Phase 3 counterpart to `phase2/models/gam_surrogate.py`.

#### `InteractionAwareSurrogateBundle`

```python
@dataclass
class InteractionAwareSurrogateBundle(SurrogateBundle):
    interaction_terms: list[str] | None = None
```

Extends the Phase 2 `SurrogateBundle` with an additional field:

| Field | Type | Description |
|-------|------|-------------|
| `surrogate` | `Any` | Fitted `ExplainableBoostingRegressor` (inherited) |
| `feature_names` | `list[str]` | Feature names (inherited) |
| `interactions` | `int \| list[tuple[int, ...]]` | Interaction config (inherited) |
| `artifact_path` | `Path \| None` | Saved model path (inherited) |
| `interaction_terms` | `list[str] \| None` | Names of the pairwise interaction terms discovered by the EBM |

#### `_normalize_interaction_pairs(feature_names, interaction_pairs) -> list[tuple[int, int]] | None`

Converts user-friendly feature name pairs into EBM term indices.

**Examples**:
- `[("x_1", "x_2")]` → `[(0, 1)]`
- `[(0, 2), (1, 3)]` → `[(0, 2), (1, 3)]`
- `None` → `None` (triggers auto-discovery)

**Raises**: `KeyError` if a feature name is not in `feature_names`.

#### `train_interaction_aware_surrogate(X_train, black_box_predictions, feature_names, interaction_pairs=None, interaction_count=5, random_state=42, save_dir=None) -> InteractionAwareSurrogateBundle`

Fits a GA²M-style surrogate with selected or automatically discovered interactions.

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | `pd.DataFrame` | (required) | Training features |
| `black_box_predictions` | `np.ndarray` | (required) | Black-box outputs |
| `feature_names` | `list[str]` | (required) | Feature names |
| `interaction_pairs` | `list[tuple] \| None` | `None` | Explicit pairs (names or indices). `None` = auto-discover |
| `interaction_count` | `int` | `5` | Number of interaction pairs if auto-discovering |
| `random_state` | `int` | `42` | Random seed |
| `save_dir` | `Path \| str \| None` | `None` | If provided, saves model |

**Logic**:
1. If `interaction_pairs` is provided, normalize to index pairs
2. If `interaction_pairs` is `None`, use `min(interaction_count, max(1, len(feature_names) // 2))` as the number of interactions
3. Call `train_gam_surrogate()` with `interactions=normalized_pairs` (delegates to Phase 2)
4. Extract `interaction_terms` from the fitted EBM's `term_names_` and `term_features_` — any term with `len(feature_group) > 1` is an interaction term
5. Return `InteractionAwareSurrogateBundle`

#### `interaction_surrogate_fidelity(surrogate, X_eval, black_box_predictions) -> dict[str, float]`

Alias for `evaluate_surrogate_fidelity()`. Exists to keep the Phase 3 API readable at call sites.

---

### `enhanced_instashap.py`

**File**: `phase3/extension/enhanced_instashap.py`
**Role**: Computes interaction-aware InstaSHAP attributions from a GA²M-style surrogate.

#### `EnhancedInstaShapOutput`

```python
@dataclass
class EnhancedInstaShapOutput:
    values: pd.DataFrame
    base_value: float
    centered_term_values: pd.DataFrame
    reference_term_means: pd.Series
    interaction_breakdown: pd.DataFrame
```

Extends `InstaShapOutput` from Phase 2 with an additional field:

| Field | Shape | Description |
|-------|-------|-------------|
| `values` | `(n_samples, n_features)` | Per-feature attributions (inherited) |
| `base_value` | scalar | Expected prediction (inherited) |
| `centered_term_values` | `(n_samples, n_terms)` | All term contributions (inherited) |
| `reference_term_means` | `(n_terms,)` | Reference means (inherited) |
| `interaction_breakdown` | `(n_samples, n_interaction_terms)` | Centered values for interaction terms only |

The `interaction_breakdown` field allows analysis of how much each specific pairwise interaction contributes to the final attributions.

#### `compute_interaction_aware_instashap(surrogate, X, reference_data, feature_names=None) -> EnhancedInstaShapOutput`

Computes InstaSHAP values while fairly allocating interaction terms.

**Algorithm**:
1. Call `compute_instashap_values()` from Phase 2 — this already handles multi-feature terms by dividing the centered contribution equally across participating features
2. Extract interaction term names (terms where `len(feature_group) > 1`)
3. Filter `centered_term_values` to only interaction columns
4. Return `EnhancedInstaShapOutput`

**Key insight**: The equal-split logic in `compute_instashap_values()` already allocates pairwise interaction terms as 50/50 to the two features. Phase 3 does not need a separate allocator — it only needs to ensure the surrogate includes interaction terms in the first place.

---

### `adaptive_surrogate.py`

**File**: `phase3/extension/adaptive_surrogate.py`
**Role**: Implements an adaptive strategy that automatically chooses between additive and interaction-aware surrogates based on fidelity.

#### `AdaptiveSurrogateResult`

```python
@dataclass
class AdaptiveSurrogateResult:
    surrogate: object
    chosen_mode: str
    additive_metrics: dict[str, float]
    final_metrics: dict[str, float]
```

Decision record for the adaptive strategy.

| Field | Type | Description |
|-------|------|-------------|
| `surrogate` | `object` | The chosen fitted surrogate model |
| `chosen_mode` | `str` | `"additive"` or `"interaction_aware"` |
| `additive_metrics` | `dict[str, float]` | Fidelity metrics for the additive surrogate (`mae`, `rmse`, `r2`) |
| `final_metrics` | `dict[str, float]` | Fidelity metrics for the finally chosen surrogate |

#### `fit_adaptive_surrogate(X_train, train_predictions, X_validation, validation_predictions, feature_names, fidelity_threshold=0.95, interaction_pairs=None, interaction_count=5) -> AdaptiveSurrogateResult`

**Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X_train` | `pd.DataFrame` | (required) | Training features |
| `train_predictions` | `np.ndarray` | (required) | Black-box outputs on training data |
| `X_validation` | `pd.DataFrame` | (required) | Validation features |
| `validation_predictions` | `np.ndarray` | (required) | Black-box outputs on validation data |
| `feature_names` | `list[str]` | (required) | Feature names |
| `fidelity_threshold` | `float` | `0.95` | R² threshold for additive surrogate acceptance |
| `interaction_pairs` | `list[tuple] \| None` | `None` | Explicit interaction pairs |
| `interaction_count` | `int` | `5` | Auto-discovery interaction count |

**Decision Logic**:

```
1. Fit additive surrogate (interactions=0)
2. Evaluate R² on validation data
3. IF additive R² >= fidelity_threshold (0.95):
     → Return additive surrogate
4. ELSE:
     Fit interaction-aware surrogate
     Evaluate R² on validation data
     IF interaction R² <= additive R²:
       → Return additive surrogate (interactions didn't help)
     ELSE:
       → Return interaction-aware surrogate
```

This conservative upgrade strategy ensures that interaction terms are only used when they demonstrably improve fidelity, avoiding unnecessary model complexity.

---

## Gap Analysis (`gap_analysis/`)

### `research_gap.md`

**File**: `phase3/gap_analysis/research_gap.md`

Written analysis documenting:
- The specific limitation: additive surrogates miss interaction structure
- Why it matters (practically and theoretically)
- How it is operationalized on the `friedman1` benchmark
- The proposed GA²M extension with fair allocation
- Supporting literature references

---

## References (`references/`)

### `supporting_references.md`

**File**: `phase3/references/supporting_references.md`

Formatted bibliography covering:
1. Lundberg & Lee — SHAP (NeurIPS 2017)
2. Lou, Caruana, Gehrke & Hooker — GA²M (KDD 2013)
3. Bordt & von Luxburg — Shapley values to GAMs and back (JMLR 2023)
4. Tsai, Yeh & Ravikumar — Faith-Shap (JMLR 2023)
5. Jethani et al. — InstaSHAP (ICLR 2025)

---

## Experiment Layer (`experiments/`)

### `experiment_gap_demonstration.py`

**File**: `phase3/experiments/experiment_gap_demonstration.py`
**Purpose**: Demonstrates where additive-only InstaSHAP breaks on the interaction-heavy `friedman1` benchmark.

**CLI Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | `str` | `friedman1` | Only `friedman1` supported |
| `--model-name` | `str` | `xgboost` | Black-box model |
| `--explain-samples` | `int` | `128` | Number of test samples to explain |
| `--background-size` | `int` | `100` | Background dataset size |
| `--output-dir` | `Path` | `phase3/results/gap_demonstration` | Output directory |

**Workflow**:
1. Load `friedman1`, train black-box model
2. Fit additive-only InstaSHAP explainer
3. Compute Exact SHAP (ground truth) and additive InstaSHAP
4. Measure additive surrogate fidelity (R²) and alignment metrics
5. Generate scatter plots showing the gap

**Outputs**:
- `gap_summary.csv` — Fidelity and alignment metrics
- `exact_shap_values.csv`, `original_instashap_values.csv` — Attribution matrices
- `gap_surrogate_fidelity_scatter.png` — Black-box vs surrogate predictions
- `gap_shap_alignment_scatter.png` — Exact SHAP vs InstaSHAP

---

### `experiment_extension_accuracy.py`

**File**: `phase3/experiments/experiment_extension_accuracy.py`
**Purpose**: Compares original InstaSHAP (additive) vs Interaction-Aware InstaSHAP against Exact SHAP on `friedman1`.

**CLI Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | `str` | `friedman1` | Dataset |
| `--model-name` | `str` | `xgboost` | Black-box model |
| `--explain-samples` | `int` | `128` | Test samples to explain |
| `--background-size` | `int` | `100` | Background size |
| `--output-dir` | `Path` | `phase3/results/extension_accuracy` | Output directory |

**Workflow**:
1. Load data, train black-box model
2. Fit additive InstaSHAP explainer, get attributions
3. Fit interaction-aware surrogate with `interaction_pairs=[("x_1", "x_2")]` and `interaction_count=3`
4. Compute interaction-aware InstaSHAP via `compute_interaction_aware_instashap()`
5. Compute Exact SHAP
6. Compare both methods on alignment metrics and surrogate fidelity
7. Generate bar charts and scatter plots

**Outputs**:
- `extension_accuracy_summary.csv` — Metrics for both methods
- `exact_shap_values.csv`, `original_instashap_values.csv`, `interaction_aware_instashap_values.csv`
- `extension_accuracy_metrics.png` — Grouped bar chart
- `extension_accuracy_scatter.png` — Overlay scatter plot

---

### `experiment_extension_runtime.py`

**File**: `phase3/experiments/experiment_extension_runtime.py`
**Purpose**: Benchmarks all four explanation methods including the Interaction-Aware extension.

**CLI Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | `str` | `friedman1` | Dataset |
| `--model-name` | `str` | `xgboost` | Black-box model |
| `--sample-sizes` | `list[int]` | `50 100 250` | Sample counts for scaling |
| `--background-size` | `int` | `75` | Background size |
| `--kernel-nsamples` | `str/int` | `auto` | KernelSHAP samples |
| `--output-dir` | `Path` | `phase3/results/extension_runtime` | Output directory |

**Four methods compared**:
1. Exact SHAP
2. KernelSHAP
3. Original InstaSHAP (additive surrogate)
4. Interaction-Aware InstaSHAP (GA²M surrogate)

**Outputs**:
- `extension_runtime.csv` — Runtime table
- `extension_runtime.png` — Line plot

---

### `experiment_comparison.py`

**File**: `phase3/experiments/experiment_comparison.py`
**Purpose**: Comprehensive cross-dataset comparison across all four explanation methods.

**CLI Arguments**:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--datasets` | `list[str]` | `friedman1 diabetes breast_cancer` | Datasets |
| `--model-name` | `str` | `xgboost` | Black-box model |
| `--explain-samples` | `int` | `96` | Test samples |
| `--background-size` | `int` | `75` | Background size |
| `--kernel-nsamples` | `str/int` | `auto` | KernelSHAP samples |
| `--output-dir` | `Path` | `phase3/results/comparison` | Output directory |

**Per dataset**:
1. Train black-box model
2. Compute Exact SHAP, KernelSHAP, original InstaSHAP, interaction-aware InstaSHAP
3. Measure alignment (Pearson, MAE), runtime, and surrogate R² for each method
4. Note: Interaction-aware uses `interaction_pairs=[("x_1", "x_2")]` only for `friedman1`; auto-discovery for others

**Outputs**:
- `comparison_summary.csv` — All metrics across datasets and methods
- `comparison_accuracy.png` — Bar chart of Pearson correlations
- `comparison_runtime.png` — Bar chart of runtimes

---

## Phase 3 ↔ Phase 2 Dependencies

Phase 3 heavily reuses Phase 2 modules. The dependency graph:

```
phase2/utils.py
  ← phase3/extension/interaction_aware_surrogate.py  (uses SEED)
  ← phase3/extension/adaptive_surrogate.py           (indirect, via gam_surrogate)
  ← phase3/experiments/*                             (uses compute_alignment_metrics, seed_everything, etc.)

phase2/models/gam_surrogate.py
  ← phase3/extension/interaction_aware_surrogate.py  (uses SurrogateBundle, train_gam_surrogate, evaluate_surrogate_fidelity)
  ← phase3/extension/adaptive_surrogate.py           (uses train_gam_surrogate, evaluate_surrogate_fidelity)

phase2/models/instashap.py
  ← phase3/extension/enhanced_instashap.py           (uses compute_instashap_values, InstaShapOutput)

phase2/models/base_model.py
  ← phase3/experiments/*                             (uses train_black_box_model, predict_black_box)

phase2/data/data_loader.py
  ← phase3/experiments/*                             (uses load_dataset)

phase2/explainers/exact_shap.py
  ← phase3/experiments/*                             (uses compute_exact_shap)

phase2/explainers/instashap_explainer.py
  ← phase3/experiments/*                             (uses InstaSHAPExplainer)

phase2/explainers/kernel_shap.py
  ← phase3/experiments/experiment_extension_runtime.py  (uses compute_kernel_shap)
  ← phase3/experiments/experiment_comparison.py          (uses compute_kernel_shap)
```

Phase 3 does **not** duplicate any Phase 2 logic — it always imports and delegates.
