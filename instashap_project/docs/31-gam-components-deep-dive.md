# GAM Components Deep Dive

## Purpose

This document explains how the additive model is represented internally.

## Additive Structure

The model is:

```text
F(x) = bias + sum_i f_i(x_i) + sum_(i,j) f_ij(x_i, x_j)
```

In code:

- `bias` is a trainable vector
- each `f_i` is a small neural network
- each `f_ij` is a small neural network

## Why Small Per-Component Networks

Instead of one monolithic network, the project uses component-specific subnetworks because:

- each term stays interpretable
- each term has clear ownership over specific features
- pairwise effects can be visualized directly

## Component Specification

`ComponentSpec` defines:

- which feature tuple the component owns
- what its input width is in transformed-column space

## Component Inputs

Each component reads only the transformed columns belonging to its feature subset.

Examples:

- univariate numeric feature: one scaled scalar
- univariate categorical feature: one one-hot vector
- pairwise component: concatenation of both feature groups

## Mask Gating

During masked training:

- component contribution is multiplied by `1(T subseteq S)`

In implementation, that means:

- if any feature in the component is missing, the whole component is zeroed out

## Attribution Recovery

After training:

- univariate terms go directly to their features
- pairwise terms are split equally between both participating features

## Why This File Matters

If you want to:

- add higher-order interactions
- change contribution splitting logic
- change per-component architectures

then `models/gam.py` is the main file to study.

