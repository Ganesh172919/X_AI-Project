# Phase 3 Deliverables

## What Was Added
- `run_phase3_experiment.py`: Before-vs-after experiment for additive vs interaction-aware SHAP surrogate.
- `RESEARCH_GAP_EXPLANATION.md`: One-page research gap explanation with literature references.
- `results/tables/before_after_summary.csv`: Main comparison metrics.
- `results/tables/before_after_per_feature.csv`: Per-feature metrics for both variants.
- `results/figures/before_after_metrics.png`: Visual before-vs-after summary.

## How To Re-run
From project root:

```bash
c:/Users/RAVIPRAKASH/X_AI-Project/.venv/Scripts/python.exe Phase3/run_phase3_experiment.py --dataset california_housing --model-type random_forest --train-sample-size 120 --test-sample-size 80 --improved-interactions 4
```

## Core Finding
Compared with additive surrogate (`interactions=0`), the interaction-aware surrogate (`interactions=4`) improved SHAP approximation fidelity on California Housing + Random Forest:
- Better R2 and lower MAE/MSE
- Higher Pearson/Spearman correlation
- Preserved feature ranking quality
- Trade-off: lower speedup vs exact SHAP due to added interaction complexity
