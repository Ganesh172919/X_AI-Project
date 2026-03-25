# File Reference: `experiments/common.py`

## Purpose

This is the central experiment orchestration file.

## Main Objects

- `ExperimentResult`
- `run_tabular_experiment(...)`

## Responsibilities

- split data
- fit preprocessor
- train selected models
- evaluate metrics
- run SHAP and InstaSHAP explanations
- save tables and plots
- build JSON summaries
- log stage progress

## Why It Matters

If you want to understand the full pipeline from raw dataset to saved artifacts, this is the most important file after `main.py`.

