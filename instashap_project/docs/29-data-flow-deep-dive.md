# Data Flow Deep Dive

## Purpose

This document traces how data changes shape as it moves through the project.

## Stage 1: Raw dataset bundle

Each loader returns:

- `features`: pandas `DataFrame`
- `target`: pandas `Series`
- `metadata`: `DatasetMetadata`

At this stage:

- features are still human-readable
- categorical values are still semantic categories
- no train/test split has happened yet

## Stage 2: Split bundle

`make_splits(...)` creates:

- `X_train`
- `X_val`
- `X_test`
- `y_train`
- `y_val`
- `y_test`

For classification:

- stratification is preserved

## Stage 3: Preprocessor fit

`TabularPreprocessor.fit(...)` learns:

- numeric imputers
- numeric scaling statistics
- categorical imputers
- one-hot vocabularies
- feature-group mappings

This stage is critical because it defines the mapping from:

- original features
- transformed matrix columns

## Stage 4: Transformed matrices

The model-facing tensors are numpy arrays:

- `X_train`
- `X_val`
- `X_test`

These arrays are:

- imputed
- scaled for numeric columns
- one-hot encoded for categorical columns

## Stage 5: Original-feature masks

Masks are sampled over **original features**, not transformed columns.

Example:

```text
[1, 0, 1, 1, 0, ...]
```

This means:

- keep original feature 1
- drop original feature 2
- keep original feature 3

## Stage 6: Expanded masks

The preprocessor expands original-feature masks to transformed-column masks.

This is necessary because:

- one original categorical feature may map to many one-hot columns

Example:

```text
original feature: occupation
one-hot columns: occupation=Sales, occupation=Tech-support, ...
```

If `occupation` is masked out, all of its one-hot columns are masked out together.

## Stage 7: Masked model inputs

Masked transformed input is computed by:

```text
masked_input = transformed_input * expanded_mask
```

This is the main bridge between:

- additive explanation theory
- practical tensor computation

## Stage 8: Explanation outputs

SHAP baseline:

- values start at transformed-column level
- then are aggregated back to original features

InstaSHAP:

- values are produced directly at original-feature group level

## Key Takeaway

The whole project depends on one invariant:

> the original feature grouping must be preserved all the way from data loading to explanation output.

