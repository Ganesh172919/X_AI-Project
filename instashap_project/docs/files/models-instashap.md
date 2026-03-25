# File Reference: `models/instashap.py`

## Purpose

This file defines the `InstaSHAPModel`, which extends the additive GAM structure with the masked-forward behavior needed for InstaSHAP training.

## Main Class

- `InstaSHAPModel`

## Key Methods

- `masked_forward(...)`
- `explain(...)`

## How It Differs from `GAMModel`

Architecturally it is close to `GAMModel`, but its intended use is different:

- trained on masked surrogate targets
- used to produce instant attributions after training

## Why It Matters

If you want to modify the explanation logic or change how pairwise contributions are assigned to features, this is one of the main files to inspect.

