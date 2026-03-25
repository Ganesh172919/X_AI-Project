# File Reference: `training/evaluate.py`

## Purpose

This file standardizes model evaluation across model types.

## Main Functions

- `predict_raw_outputs(...)`
- `predict_targets(...)`
- `evaluate_supervised_model(...)`

## Why It Matters

Different models in the project use different APIs:

- PyTorch neural models
- sklearn random forest wrapper

This file provides a consistent interface for evaluation and downstream explainers.

