# Adult Dataset Extension Summary

This document summarizes the new Adult Income diagnostic assets created for the repository.

## Created assets

- `Phase_3_work/results/adult_masking_diagnostic/adult_masking_diagnostic_summary.csv`
- `Phase_3_work/results/adult_masking_diagnostic/adult_masking_diagnostic_summary.json`
- `Phase_3_work/results/adult_masking_diagnostic/adult_masking_diagnostic_comparison.png`
- `Phase_3_work/results/adult_masking_diagnostic/adult_masking_diagnostic_report.md`
- `Phase_3_work/results/adult_masking_diagnostic/adult_masking_diagnostic_report.pdf`
- `Phase_3_work/notebooks/phase3_adult_masking_diagnostic.ipynb`

## Why these assets matter

- They give you a dataset where the Phase 3 masking improvement is clearly visible at the coalition-construction level.
- They are faster to run and explain than a full new end-to-end dataset retraining workflow.
- They create a clean bridge from the current Covertype result to a broader multi-dataset extension story.

| strategy | hidden_categorical_valid_rate | hidden_categorical_invalid_rate | hidden_numeric_exact_zero_rate | nearest_train_distance_mean |
| --- | --- | --- | --- | --- |
| zero_mask | 0.0000 | 1.0000 | 1.0000 | 2.1893 |
| empirical_background | 1.0000 | 0.0000 | 0.0000 | 1.3329 |
