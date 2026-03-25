# File Reference: `config.yaml`

## Purpose

`config.yaml` is the central configuration file for hyperparameters, dataset sizes, training settings, and global runtime options.

## Main Sections

- `global`
- `training.blackbox`
- `training.gam`
- `training.surrogate`
- `training.instashap`
- `datasets.bike`
- `datasets.covertype`
- `datasets.adult`

## Key Settings

- random seed
- device selection
- SHAP evaluation size
- batch sizes
- epochs
- learning rates
- dataset-specific interaction pairs

## Why It Matters

This file is the safest place to tune experiments without editing code.

## Common Extensions

- Increase epochs for stronger reproduction
- Add new dataset configs
- Add more interaction pairs

