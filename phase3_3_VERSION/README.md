# Phase 3: Research Gap and Extension

This phase identifies a concrete limitation in the original InstaSHAP setup and implements a research extension to address it.

## Chosen Research Gap

**Gap:** purely additive surrogates cannot represent pairwise or higher-order feature interactions.  
**Consequence:** when the black-box model relies on interaction structure, the surrogate fidelity drops and the derived InstaSHAP attributions become less accurate.

## Proposed Extension

This folder implements an **Interaction-Aware InstaSHAP** pipeline:

- `interaction_aware_surrogate.py`: trains a GA²M-style surrogate with pairwise interactions
- `enhanced_instashap.py`: analytically allocates pairwise interaction terms back to features
- `adaptive_surrogate.py`: optional upgrade rule that keeps the additive model when it is already faithful enough

## Setup

From the repository root:

```bash
pip install -r phase3/requirements.txt
```

Phase 3 reuses the Phase 2 package modules, so installing the same dependency set is sufficient.

## Running the Experiments

Gap demonstration:

```bash
python -m phase3.experiments.experiment_gap_demonstration
```

Extension accuracy:

```bash
python -m phase3.experiments.experiment_extension_accuracy
```

Extension runtime:

```bash
python -m phase3.experiments.experiment_extension_runtime
```

Comprehensive comparison:

```bash
python -m phase3.experiments.experiment_comparison
```

## Outputs

Generated artifacts are saved inside `phase3/results/`:

- `gap_demonstration/`
- `extension_accuracy/`
- `extension_runtime/`
- `comparison/`

## Notebook and References

- Walkthrough notebook: [extension_walkthrough.ipynb](./notebooks/extension_walkthrough.ipynb)
- Gap analysis: [research_gap.md](./gap_analysis/research_gap.md)
- Supporting literature: [supporting_references.md](./references/supporting_references.md)
