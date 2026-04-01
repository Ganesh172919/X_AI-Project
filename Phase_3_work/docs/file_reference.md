# File Reference

This file gives a quick map of the most important source files in the Phase 3 repository.

## Root Files

### `main.py`

Main CLI entrypoint.

Use this file to understand:

- supported commands
- config loading
- run-vs-report-only behavior

### `config.yaml`

Primary experiment configuration.

Use this file to understand:

- seed choices
- training budgets
- masking parameters
- dataset-level settings

### `README.md`

Top-level project description and run guide.

### `AI_USAGE.md`

Assignment compliance file describing AI usage and verification.

## Core Package

### `instashap_project/masking.py`

Most important Phase 3 extension file.

Use this file to understand:

- masking strategy types
- background bank creation
- zero-mask vs empirical-background masking behavior

### `instashap_project/experiments/common.py`

Main experiment orchestrator.

Use this file to understand:

- seed loop
- training order
- metric aggregation
- artifact writing

### `instashap_project/reporting.py`

Report-generation module.

Use this file to understand:

- how Markdown reports are created
- how PDFs are generated
- how narrative text is tied to saved artifacts

## Data And Preprocessing

### `instashap_project/data/loaders.py`

Contains the dataset loader logic reused from the previous work, including the Covertype grouped soil-climate representation.

### `instashap_project/data/preprocessing.py`

Contains:

- train/validation/test splitting
- numeric and categorical preprocessing
- original-feature grouping logic

This file is important because the masking logic depends on feature-group tracking.

## Models

### `instashap_project/models/blackbox_model.py`

Contains:

- black-box MLP
- surrogate model
- optional random forest wrapper

### `instashap_project/models/gam.py`

Contains:

- additive component model
- interaction component support
- feature attribution reconstruction

### `instashap_project/models/instashap.py`

Thin wrapper around the additive GAM model for masked training and one-pass explanations.

## Training And Evaluation

### `instashap_project/training/train.py`

Contains the main training loops:

- black-box training
- surrogate training
- GAM training
- InstaSHAP training

### `instashap_project/training/evaluate.py`

Prediction and evaluation helpers used throughout the experiment flow.

## Explainers

### `instashap_project/xai/shap_wrapper.py`

Permutation SHAP wrapper with grouped-feature aggregation.

### `instashap_project/xai/instashap_explainer.py`

One-pass explanation wrapper for trained InstaSHAP models.

## Utilities

### `instashap_project/utils/metrics.py`

Metric definitions for:

- predictive quality
- explanation fidelity
- runtime benchmarks

### `instashap_project/utils/visualization.py`

Plot-generation helpers for:

- metric comparison
- training curves
- alignment plots
- shape functions
- interaction heatmaps

### `instashap_project/utils/reproducibility.py`

Seed control, path helpers, and JSON writing.

### `instashap_project/utils/logging_utils.py`

Structured logging used by the CLI and experiment pipeline.

## Tests

### `tests/test_cli.py`

Checks that the CLI parses the main modes correctly.

### `tests/test_masking.py`

Checks:

- zero-mask visible-feature preservation
- empirical-background categorical validity

## Reports And Result Files

### `reports/`

Final human-readable project deliverables.

### `results/`

Machine-generated tables, plots, logs, and JSON summaries.
