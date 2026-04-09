# Why Adult Can Improve The Story And Why Covertype Did Not Improve Enough

This document explains why the masking improvement is easier to show on Adult Income than on Covertype.

## Why Adult Income is a strong next dataset

- Adult Income has many categorical groups, so the invalid all-zero one-hot problem becomes very visible.
- The new masking strategy directly fixes that issue by copying hidden groups from real transformed rows.
- The Adult masking diagnostic therefore shows the exact coalition-construction improvement more clearly.

| strategy | hidden_categorical_valid_rate | hidden_categorical_invalid_rate | hidden_numeric_exact_zero_rate | nearest_train_distance_mean |
| --- | --- | --- | --- | --- |
| zero_mask | 0.0000 | 1.0000 | 1.0000 | 2.1893 |
| empirical_background | 1.0000 | 0.0000 | 0.0000 | 1.3329 |

## Why Covertype did not improve enough

- Covertype still has a real masking problem, but the final end-to-end metrics depend on more than masking realism.
- The empirical_background target is harder for the surrogate and final additive model to learn.
- The current training budget may be too small for the harder objective.
- One interaction pair is not enough to capture all remaining structure in the data.
- Better coalition realism does not automatically guarantee better SHAP alignment or predictive accuracy.

## The reason behind the mixed result

- The improvement fixes one weakness in the pipeline but exposes another: optimization capacity.
- In other words, the new coalition target is better grounded in the data but also harder to approximate.
- That is why the Covertype result is honest and useful even though it is not a full win.

## Best interpretation

The masking idea is correct in spirit. Adult Income shows that clearly at the coalition-validity level. Covertype shows that a better masking rule alone is not always enough to improve the full InstaSHAP pipeline without additional modeling or training changes.
