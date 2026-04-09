# Continuous Improvement With Other Datasets

This document focuses on continuing the Phase 3 improvement across more datasets.

## Immediate continuation path

- Step 1: Use Adult Income to demonstrate the masking improvement at the coalition-validity level.
- Step 2: Make the Phase 3 experiment runner dataset-generic.
- Step 3: Add a dataset-specific reporting template so the files are named dynamically instead of using Covertype-specific names.
- Step 4: Compare multiple datasets with the same masking diagnostics before pushing for full retraining on all of them.

## Best dataset order

1. Adult Income.
2. Bank Marketing or German Credit.
3. Telco Churn.
4. Larger structured tabular datasets after the workflow is stable.

## Why this order works

- Adult is already supported by the repo loaders.
- The masking weakness is easier to demonstrate on category-heavy datasets.
- Credit or churn datasets make the value of realistic hidden groups easy to explain to reviewers.
- Once the story is stable, larger datasets can test whether the improvement still scales.

## Metrics to keep across every dataset

- Predictive accuracy or regression error.
- SHAP alignment metrics such as MAE and rank correlation.
- Coalition fidelity metrics.
- Runtime metrics.
- Masking-validity metrics such as hidden categorical validity rate and nearest-train distance.

## Why this matters

If Covertype is the only dataset, reviewers may think the mixed result is dataset-specific. A continuation path across multiple datasets lets you show whether the limitation and its fix generalize.
