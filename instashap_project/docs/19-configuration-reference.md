# Configuration Reference

## Purpose

This document explains every major configuration section in `config.yaml` and how changing it affects the project.

## `global`

### `seed`

Controls deterministic behavior across:

- Python
- NumPy
- PyTorch

### `device`

Allowed patterns:

- `auto`
- `cpu`
- `cuda`

### `output_root`

Base directory for generated outputs.

### `artifact_root`

Logical root for saved experiment summaries and TensorBoard logs.

### `shap_background_size`

Number of training examples used as the SHAP background distribution.

Larger values:

- may improve stability
- increase runtime

### `shap_eval_samples`

How many test samples are used for SHAP / InstaSHAP explanation comparison.

### `shap_max_evals`

Controls the number of model evaluations SHAP is allowed to use.

### `fast_dev_run`

When enabled:

- shrinks datasets
- reduces epochs
- speeds up validation runs

## `training.blackbox`

Important fields:

- `model_type`
- `hidden_dims`
- `dropout`
- `lr`
- `weight_decay`
- `batch_size`
- `epochs`
- `patience`

## `training.gam`

Controls the additive GAM-1 and GAM-2 training loops.

## `training.surrogate`

Controls masked surrogate training.

Additional important fields:

- `masks_per_sample`
- `edge_mask_probability`

## `training.instashap`

Controls the additive masked model training.

The most important tradeoffs are:

- hidden size vs interpretability
- epochs vs runtime
- regularization vs stability

## `datasets.*`

Each dataset section may contain:

- `max_rows`
- `test_size`
- `val_size`
- `interaction_pairs`
- `shap_sample_size`

## Recommended Tuning Strategy

1. Keep `fast_dev_run: true` while testing code changes
2. Disable it for real experiments
3. Increase epochs first
4. Increase dataset size second
5. Tune hidden dimensions third

