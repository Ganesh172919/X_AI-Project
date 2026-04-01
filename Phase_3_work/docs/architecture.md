# Architecture

## High-Level Layout

The project is organized as a self-contained package under `instashap_project/`, plus root-level execution and documentation files.

```text
Phase_3_work/
|- main.py
|- config.yaml
|- README.md
|- AI_USAGE.md
|- docs/
|- instashap_project/
|  |- data/
|  |- experiments/
|  |- models/
|  |- training/
|  |- utils/
|  |- xai/
|  |- masking.py
|  \- reporting.py
|- reports/
|- results/
\- tests/
```

## Execution Flow

The main runtime flow is:

1. `main.py` parses CLI flags and loads the config.
2. The Covertype experiment runner is selected.
3. The experiment module:
   - loads the dataset
   - builds splits and preprocessing
   - trains baseline and improved branches
   - computes metrics
   - writes tables, plots, and JSON summaries
4. The reporting module regenerates Markdown and PDF reports from saved artifacts.

This separation is deliberate: experiment execution and report generation share the same saved data, which reduces the chance that the narrative drifts away from the actual outputs.

## Module Responsibilities

### `main.py`

Responsible for:

- CLI parsing
- config loading
- fast-dev flag propagation
- experiment execution
- report-only regeneration

### `instashap_project/data/`

Responsible for:

- dataset loading
- tabular preprocessing
- split creation
- original-feature group bookkeeping

The preprocessor is especially important because the masking logic must respect original feature groups even after transformation.

### `instashap_project/masking.py`

Responsible for:

- declaring masking configuration
- building the background bank
- constructing masked coalition batches
- switching between `zero_mask` and `empirical_background`

This is the central Phase 3 extension point.

### `instashap_project/models/`

Contains:

- black-box model definitions
- additive GAM definitions
- InstaSHAP additive wrapper

The surrogate model was adapted so it can consume:

- masked realization
- full original transformed input as context
- feature mask

This helps the background-aware branch learn the coalition objective more faithfully.

### `instashap_project/training/`

Responsible for:

- supervised training
- masked surrogate training
- additive InstaSHAP training
- coalition-mask sampling
- shared training result containers

This is where the baseline and improved branches are kept under a common training interface.

### `instashap_project/experiments/`

Responsible for:

- dataset-specific orchestration
- seed loops
- metrics aggregation
- artifact writing

The `common.py` module is the main coordinator for the Phase 3 Covertype study.

### `instashap_project/xai/`

Responsible for:

- permutation SHAP wrapper
- one-pass InstaSHAP explanation wrapper

These modules provide the explanation outputs used in the comparison tables and plots.

### `instashap_project/utils/`

Provides:

- logging
- reproducibility helpers
- metrics
- plotting helpers

These utilities keep the experiment orchestration file smaller and easier to review.

### `instashap_project/reporting.py`

Responsible for:

- reading the saved summary JSON and CSV outputs
- generating Markdown reports
- generating PDF reports with tables and figures
- producing a report manifest

This means the final write-up is reproducible and can be regenerated without retraining.

## Design Decisions

### Standalone Package

The Phase 3 work does not import the Phase 2 package at runtime. This avoids hidden dependencies and makes the deliverable easier for a grader to run independently.

### Covertype-Only Scope

The repository intentionally removes unused experiment entrypoints from the final Phase 3 workflow. That keeps the scope aligned with the actual research question and prevents confusion.

### Artifact-First Reporting

The code writes tables and plots first, then generates reports from those artifacts. This is safer than hardcoding results or rendering report text from in-memory values that may differ from saved outputs.

### Honest Reporting

The reporting module interprets results conservatively. It does not assume the improved branch always wins, and it updates the narrative based on the saved comparison table.
