# Code Reading Guide

## Purpose

This document gives a recommended order for reading the codebase if you want to understand it efficiently.

## Reading Path A: Understand the whole pipeline

1. `main.py`
2. `experiments/common.py`
3. `data/loaders.py`
4. `data/preprocessing.py`
5. `training/train.py`
6. `training/evaluate.py`
7. `models/gam.py`
8. `models/instashap.py`
9. `xai/shap_wrapper.py`
10. `utils/visualization.py`

## Reading Path B: Focus on theory-to-code mapping

1. `docs/06-instashap-methodology.md`
2. `models/gam.py`
3. `training/train.py`
4. `xai/instashap_explainer.py`

## Reading Path C: Focus on experiments

1. `experiments/bike_sharing.py`
2. `experiments/covertype.py`
3. `experiments/adult_income.py`
4. `experiments/common.py`

## Reading Path D: Focus on extension

1. `docs/12-extension-guide.md`
2. `docs/17-development-workflow.md`
3. `docs/files/...`

## Key Insight

The fastest way to understand the codebase is to treat `experiments/common.py` as the center and then branch outward to the modules it calls.

