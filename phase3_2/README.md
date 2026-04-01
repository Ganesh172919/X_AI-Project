# Phase 3: InstaSHAP Extension - Interaction-Aware InstaSHAP

## Project Overview

This phase identifies a meaningful limitation in the InstaSHAP paper and proposes, implements, and experimentally validates an improvement.

### Identified Research Gap

**Inability to Capture Feature Interactions**

InstaSHAP relies on purely additive surrogate models (GAMs/EBMs) that decompose predictions into independent per-feature contributions. By construction, these models cannot capture pairwise or higher-order feature interactions. In datasets where feature interactions are significant, the additive surrogate will be a poor approximation of the black-box model, leading to inaccurate Shapley value estimates.

### Proposed Solution

**Interaction-Aware InstaSHAP** using GA²M (Generalized Additive Model with pairwise interactions):
- Augment the additive surrogate with pairwise interaction terms
- Extend the closed-form Shapley computation to handle interactions
- Adaptive strategy to automatically enable interactions when needed

## Project Structure

```
phase3/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── gap_analysis/
│   └── research_gap.md         # Detailed gap analysis
├── extension/
│   ├── enhanced_instashap.py       # Extended Shapley computation
│   ├── interaction_aware_surrogate.py  # GA²M surrogate
│   └── adaptive_surrogate.py       # Adaptive fidelity-based selection
├── experiments/
│   ├── experiment_gap_demonstration.py  # Show the gap exists
│   ├── experiment_extension_accuracy.py # Extension accuracy
│   ├── experiment_extension_runtime.py  # Extension runtime
│   └── experiment_comparison.py         # Comprehensive comparison
├── results/                     # Generated plots and tables
├── notebooks/
│   └── extension_walkthrough.ipynb  # End-to-end walkthrough
└── references/
    └── supporting_references.md     # Literature references
```

## Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run gap demonstration:**
   ```bash
   python experiments/experiment_gap_demonstration.py
   ```

3. **Run extension experiments:**
   ```bash
   python experiments/experiment_extension_accuracy.py
   python experiments/experiment_extension_runtime.py
   python experiments/experiment_comparison.py
   ```

4. **View notebook walkthrough:**
   ```bash
   jupyter notebook notebooks/extension_walkthrough.ipynb
   ```

## Key Contributions

1. **Gap Identification:** Formal analysis of InstaSHAP's additive limitation
2. **Interaction-Aware Surrogate:** GA²M implementation with pairwise terms
3. **Extended Shapley Formula:** Fair allocation of interaction contributions
4. **Adaptive Strategy:** Automatic surrogate selection based on fidelity
5. **Comprehensive Evaluation:** Accuracy, runtime, and fidelity comparisons

## Expected Results

- **Improved accuracy** on interaction-heavy datasets
- **Similar accuracy** on non-interaction datasets (no harm)
- **Modest runtime increase** while still much faster than Exact SHAP
- **Better surrogate fidelity** when interactions matter

## References

See `references/supporting_references.md` for the literature supporting this extension.
