# Phase 3 Dataset Masking Diagnostic Report

This report compares datasets at the coalition-construction level to show where the Phase 3 masking improvement is easiest to demonstrate.

## Summary table

| dataset | categorical_features | one_hot_expansion | zero_hidden_categorical_valid_rate | bg_hidden_categorical_valid_rate | zero_hidden_numeric_exact_zero_rate | bg_hidden_numeric_exact_zero_rate | zero_nearest_train_distance_mean | bg_nearest_train_distance_mean | showcase_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adult_income | 8 | 81 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 2.1893 | 1.3329 | 0.7971 |

## Interpretation

- The best showcase dataset in this diagnostic is `adult_income`.
- Adult categorical validity improves from 0.0000 to 1.0000.
- The current saved Covertype pipeline remains mixed: accuracy 0.6843 vs 0.6774, explanation MAE 0.3591 vs 0.3795, but Spearman 0.5650 vs 0.5835 slightly favors the new branch.
- This supports using Adult as the next dataset for demonstrating the masking improvement while keeping Covertype as the honest current benchmark.
