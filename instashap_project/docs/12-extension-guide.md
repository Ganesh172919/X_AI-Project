# Extension Guide

## If You Want to Add a New Dataset

1. Add a loader in `data/loaders.py`
2. Define numeric and categorical features in the metadata
3. Add dataset config in `config.yaml`
4. Add a runner in `experiments/`
5. Update docs and README

## If You Want to Add a New Interaction

1. Update `interaction_pairs` in `config.yaml`
2. Ensure the dataset metadata supports those feature names
3. Re-run the experiment
4. Inspect new interaction heatmaps

## If You Want to Add a New Model

Good integration points:

- `models/`
- `training/train.py`
- `training/evaluate.py`
- `experiments/common.py`

## If You Want Better Reproduction Quality

Recommended directions:

- stronger black-box architecture
- better interaction selection than the current manual/config-driven choices
- larger training budgets
- more careful surrogate modeling
- more faithful treatment of classification logits and calibration

## If You Want Better Documentation for New Code

Follow the current pattern:

- add a topic guide if the feature introduces a new concept
- add a file-reference page in `docs/files/`
- link it from `README.md` and `docs/index.md`

