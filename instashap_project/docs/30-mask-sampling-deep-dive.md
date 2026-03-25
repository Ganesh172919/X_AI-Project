# Mask Sampling Deep Dive

## Purpose

This document explains how feature masks are sampled and why mask sampling is central to InstaSHAP.

## What a Mask Represents

A mask represents a subset of original features that are considered "present" in a coalition:

```text
S subseteq [d]
```

In implementation, this becomes a binary vector:

```text
[1, 0, 1, 1, 0]
```

## Why Mask Sampling Exists

The masked objective requires the model to understand how predictions behave when only some features are available. This is the mechanism that links:

- black-box masked behavior
- additive explanatory behavior

## Shapley Kernel Sampling

The project samples subset sizes using:

```text
1 / (C(d, s) * s * (d - s))
```

where:

- `d` is the number of original features
- `s` is the subset size

This follows the least-squares Shapley framing discussed in the paper.

## Sampling Steps in Practice

1. Sample subset size `s`
2. Randomly choose `s` original features
3. Create binary mask vector
4. Optionally include empty/full masks with `edge_mask_probability`

## Why Edge Masks Help

Empty and full masks can help stabilize learning because they expose:

- fully absent feature contexts
- fully present feature contexts

These edge cases may be especially useful for learning bias terms and full coalition behavior.

## Where This Logic Lives

- `training/train.py`
- function: `sample_shapley_feature_masks(...)`

## Important Limitation

The current implementation samples masks stochastically and efficiently, but it does not exhaustively enumerate coalition structure. That is a practical tradeoff.

