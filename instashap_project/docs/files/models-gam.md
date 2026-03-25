# File Reference: `models/gam.py`

## Purpose

This file implements generalized additive neural models.

## Main Objects

- `ComponentSpec`
- `ComponentMLP`
- `GAMModel`

## Core Idea

Each additive term gets its own small neural network:

- one network per univariate feature
- optionally one network per pairwise interaction

The final prediction is:

- bias term
- plus all active component outputs

## Important Methods

- `forward(...)`
- `component_contributions(...)`
- `feature_attributions(...)`
- `single_component(...)`

## Why It Matters

This file is the backbone for:

- GAM-1
- GAM-2
- the additive structure inherited by InstaSHAP

