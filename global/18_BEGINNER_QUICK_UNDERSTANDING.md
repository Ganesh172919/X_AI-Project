# Beginner Guide To This Project

This document is written for a beginner who wants to understand the project quickly without reading the whole repository first.

## What this project is

- This is an Explainable AI project based on InstaSHAP.
- The goal is to make SHAP-style explanations much faster by training an additive explainer ahead of time.
- The repository has three phases: proposal, replication, and improvement.

## The simple story

1. Train a black-box model.
2. Train a surrogate that learns the black-box under different feature masks.
3. Train an additive InstaSHAP model that imitates the surrogate.
4. Compare the learned explanations to a slower SHAP baseline.

## What Phase 3 changed

- The old approach used `zero_mask`, which hides features by writing zeros into transformed columns.
- The new approach uses `empirical_background`, which fills hidden feature groups from real training rows.
- This is meant to make masked coalitions more realistic.

## Why the Covertype result is mixed

- The masking idea is valid, but the harder and more realistic coalition objective also makes learning harder.
- In the current saved Covertype run, the new branch is slightly better on some secondary signals but not on the main headline metrics.
- That means the idea is promising, but the full pipeline still needs tuning.

## Why Adult Income is a better showcase for the masking idea

- Adult Income has many categorical feature groups.
- Under `zero_mask`, hidden categorical groups become all-zero one-hot vectors, which are invalid category states.
- Under `empirical_background`, those hidden groups remain valid because they come from real transformed rows.

## Adult masking diagnostic snapshot

| strategy | hidden_categorical_valid_rate | hidden_categorical_invalid_rate | hidden_numeric_exact_zero_rate | nearest_train_distance_mean |
| --- | --- | --- | --- | --- |
| zero_mask | 0.0000 | 1.0000 | 1.0000 | 2.1893 |
| empirical_background | 1.0000 | 0.0000 | 0.0000 | 1.3329 |

## Best files for a beginner

- `global/README.md` for the full documentation hub.
- `global/01_PROJECT_UNDERSTANDING_GUIDE.md` for the fast overview.
- `global/06_LATEST_PHASE3_IMPROVEMENT_ANALYSIS.md` for the direct answer to the improvement question.
- `Phase_3_work/README.md` for the current runnable branch.
- `Phase_3_work/instashap_project/masking.py` to see the actual improvement in code.

## Best short summary

The project reproduced InstaSHAP, identified that zero-masking creates unrealistic coalition states in transformed tabular data, and implemented a background-aware masking fix. The current Covertype result is mixed, but Adult Income shows the masking improvement itself very clearly.
