# File Reference: `data/preprocessing.py`

## Purpose

This file provides the shared preprocessing pipeline and tracks how original features map to transformed columns.

## Main Objects

- `FeatureGroup`
- `SplitBundle`
- `TabularPreprocessor`
- `make_splits()`

## Why It Is Important

The project needs explanations at the **original feature level**, even though training happens on transformed matrices. This file preserves that mapping.

## Main Behaviors

- Median imputation for numeric features
- Standardization for numeric features
- Most-frequent imputation for categorical features
- One-hot encoding for categorical features
- Feature-group bookkeeping for explanation aggregation
- Train/validation/test splitting

## Critical Concept

`expand_feature_mask(...)` bridges the gap between:

- masks over original features
- transformed columns used by the model

That logic is essential for masked surrogate training and InstaSHAP training.

