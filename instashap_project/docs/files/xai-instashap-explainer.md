# File Reference: `xai/instashap_explainer.py`

## Purpose

This file provides a thin inference wrapper for the trained InstaSHAP model.

## Main Objects

- `InstaSHAPExplanationResult`
- `InstaSHAPExplainer`

## What It Does

- batches transformed inputs
- runs `model.explain(...)`
- returns grouped feature attributions

## Why It Matters

This is the simplest path from a trained InstaSHAP model to actual explanation outputs.

