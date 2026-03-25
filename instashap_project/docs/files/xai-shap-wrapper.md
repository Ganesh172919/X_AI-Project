# File Reference: `xai/shap_wrapper.py`

## Purpose

This file wraps the SHAP baseline and converts SHAP values on transformed columns back into original feature groups.

## Main Objects

- `ShapExplanationResult`
- `ShapBaselineExplainer`

## Key Steps

1. Call SHAP on transformed inputs
2. Get explanation values per transformed column
3. Sum one-hot encoded columns back into original feature groups

## Why It Matters

Without this aggregation step, SHAP would be harder to compare directly to the additive component structure used by InstaSHAP.

