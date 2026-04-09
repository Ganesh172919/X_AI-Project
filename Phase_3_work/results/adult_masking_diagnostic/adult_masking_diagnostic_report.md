# Adult masking diagnostic

This diagnostic does not retrain the full Phase 3 pipeline. Instead, it measures the exact weakness that the Phase 3 improvement targets: unrealistic coalition construction after tabular preprocessing.

## Why Adult Income is a good next dataset

- It contains many categorical feature groups, so zero-masking creates many impossible all-zero one-hot states.
- It is already supported by the repository data loaders.
- It is a strong follow-on dataset for Phase 3 because the masking problem is easier to demonstrate here than on Covertype.

## Summary table

| strategy | dataset | hidden_feature_fraction | hidden_categorical_groups_evaluated | hidden_categorical_valid_rate | hidden_categorical_invalid_rate | hidden_numeric_entries_evaluated | hidden_numeric_exact_zero_rate | nearest_train_distance_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero_mask | adult_income | 0.5102 | 1049 | 0.0000 | 1.0000 | 649 | 1.0000 | 2.1893 |
| empirical_background | adult_income | 0.5102 | 4196 | 1.0000 | 0.0000 | 2596 | 0.0000 | 1.3329 |

## Interpretation

- `zero_mask` leaves hidden categorical groups valid only 0.0000 of the time in this diagnostic because all-zero one-hot groups are invalid category states.
- `empirical_background` keeps hidden categorical groups valid 1.0000 of the time because it copies hidden transformed groups from real rows.
- `zero_mask` sets hidden numeric values to exact zero at rate 1.0000, while `empirical_background` does so at rate 0.0000.
- The mean nearest-train distance is 2.1893 for `zero_mask` versus 1.3329 for `empirical_background`.
- This dataset therefore showcases the Phase 3 masking improvement more clearly than Covertype at the coalition-construction level.
