# File Reference: `models/blackbox_model.py`

## Purpose

This file contains the flexible predictive baselines:

- MLP black-box
- masked surrogate MLP
- random forest wrapper

## Main Classes

- `TabularMLP`
- `MaskedSurrogateMLP`
- `RandomForestBlackBox`

## Roles

### `TabularMLP`

Used as the default black-box baseline for predictive performance.

### `MaskedSurrogateMLP`

Learns masked behavior of the black-box model. It takes both:

- masked transformed inputs
- original-feature mask vector

### `RandomForestBlackBox`

Optional sklearn-based baseline wrapper.

## Extension Points

- Replace the MLP with a better tabular architecture
- Add gradient-boosted trees or modern tabular transformers

