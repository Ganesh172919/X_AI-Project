# File Reference: `training/train.py`

## Purpose

This is the main training engine of the repository.

## Main Contents

- `TrainingResult`
- `train_blackbox_model(...)`
- `train_masked_surrogate(...)`
- `train_gam_model(...)`
- `train_instashap_model(...)`
- `sample_shapley_feature_masks(...)`

## Critical Concepts

### Shapley-kernel sampling

Masks are sampled according to subset-size weights:

- proportional to `1 / (C(d, s) * s * (d - s))`

### Edge masks

Optional empty/full masks can be sampled for stability.

### Early stopping

Validation loss is tracked and best weights are restored.

## Why It Matters

If you want to change optimization strategy, losses, sampling, or early stopping behavior, this is the main file.

