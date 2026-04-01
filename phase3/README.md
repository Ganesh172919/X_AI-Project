# Phase 3 — Interaction-Aware InstaSHAP Extension

**Course:** DS357 — Explainable AI

**Paper:** *"InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly"* (ICLR 2025)

---

## Research Gap

InstaSHAP relies on **purely additive surrogates** (GAMs/EBMs) that cannot capture **feature interactions**. On datasets where the black-box model relies on pairwise or higher-order interactions (e.g., XOR patterns, multiplicative effects), the additive surrogate is a poor fit, leading to inaccurate Shapley values.

## Proposed Extension

**Interaction-Aware InstaSHAP** replaces the additive GAM with a **GA²M** (Generalized Additive Model with Pairwise Interactions):

```
f(x) = f_0 + Σ_j f_j(x_j) + Σ_{(j,k)} f_{jk}(x_j, x_k)
```

The closed-form Shapley computation is extended: each pairwise interaction `f_{jk}` is split equally between features `j` and `k`:

```
φ_j = (f_j(x_j) - E[f_j]) + Σ_{k≠j} 0.5 × (f_{jk}(x_j, x_k) - E[f_{jk}])
```

An **adaptive strategy** automatically selects between additive and interaction-aware surrogates based on fidelity.

---

## Project Structure

```
phase3/
├── README.md
├── requirements.txt
├── gap_analysis/
│   └── research_gap.md         ← Gap identification & justification
├── extension/
│   ├── interaction_aware_surrogate.py  ← GA²M surrogate fitting
│   ├── enhanced_instashap.py           ← Extended Shapley computation
│   └── adaptive_surrogate.py           ← Auto-select additive vs GA²M
├── experiments/
│   ├── experiment_gap_demonstration.py  ← Show where additive fails
│   ├── experiment_extension_accuracy.py ← GA²M improves accuracy
│   ├── experiment_extension_runtime.py  ← GA²M still fast
│   └── experiment_comparison.py         ← Full method comparison
├── results/                    ← Generated plots & CSVs
├── notebooks/
│   └── extension_walkthrough.ipynb
└── references/
    └── supporting_references.md
```

---

## Setup

```bash
cd phase3
pip install -r requirements.txt
```

Phase 3 imports from Phase 2 as a sibling package. Ensure the project root is on your Python path.

---

## Running Experiments

```bash
# Step 1: Demonstrate the gap
python experiments/experiment_gap_demonstration.py

# Step 2: Show extension improves accuracy
python experiments/experiment_extension_accuracy.py

# Step 3: Show runtime is still fast
python experiments/experiment_extension_runtime.py

# Step 4: Comprehensive comparison
python experiments/experiment_comparison.py
```

---

## Key Expected Results

| Dataset | Additive R² | GA²M R² | Additive Pearson | GA²M Pearson |
|---------|------------|---------|-----------------|-------------|
| XOR Synthetic | ~0.70-0.85 | ~0.95+ | ~0.70-0.80 | ~0.93+ |
| California Housing | ~0.98+ | ~0.98+ | ~0.95+ | ~0.95+ |

**Key insight:** GA²M dramatically helps on interaction-heavy data while maintaining similar performance on additive-dominated data.

---

## References

See `references/supporting_references.md` for full citations.
