# SHAP Aggregation Deep Dive

## Purpose

This document explains the most subtle part of the SHAP baseline integration: aggregation from transformed columns back to original features.

## Why Aggregation Is Needed

The models are trained on transformed matrices, not raw feature frames.

This means one original feature may correspond to:

- 1 transformed column for a numeric feature
- many transformed columns for a categorical feature

## Example

Original feature:

```text
education
```

Possible transformed columns:

```text
education=HS-grad
education=Bachelors
education=Masters
...
```

SHAP returns values at the transformed-column level, so the project sums these columns to get one original-feature attribution.

## Aggregation Rule

For each original feature:

```text
feature attribution = sum of SHAP values across that feature's transformed indices
```

## Why This Is a Fair Comparison

InstaSHAP returns original-feature-group attributions directly. To compare SHAP and InstaSHAP numerically, SHAP must be mapped into the same feature space.

## Where This Logic Lives

- `xai/shap_wrapper.py`
- `TabularPreprocessor.feature_groups`

## Important Caveat

Aggregating transformed-column SHAP values is a necessary practical comparison device, but it does not remove all representation differences between SHAP and additive-model explanations.

