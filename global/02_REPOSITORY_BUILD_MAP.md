# Repository Build Map

This document maps the whole repository as a build and execution system rather than as a loose collection of files.

## Top-Level Areas

- Phase_1_work holds the proposal artifact.
- Phase_2_work holds the replication package and saved replication outputs.
- Phase_3_work holds the current runnable extension and its saved evidence.
- instashap_presentation holds communication assets rather than training code.
- The root README and phase3-architecture note are context files, not the final runtime truth source.

## Phase 2 Build Flow

- Read config.yaml and select a dataset and model stage from main.py.
- Load data through data/loaders.py and preprocess it in data/preprocessing.py.
- Train black-box, GAM, surrogate, and InstaSHAP models through training/train.py.
- Compute SHAP baselines through xai/shap_wrapper.py.
- Save metrics, plots, reports, and notebook outputs.

## Phase 3 Build Flow

- Read Phase_3_work/main.py and config.yaml.
- Load Covertype and create deterministic multi-seed splits.
- Build the background bank from real transformed training rows.
- Train zero_mask and empirical_background branches under the same broader architecture.
- Compare prediction, explanation, coalition, and runtime metrics.
- Generate Markdown and PDF reports from the saved artifacts.

## Trust Hierarchy

- For current metrics, trust Phase_3_work/results/tables/*.csv first.
- For current implementation, trust Phase_3_work/*.py first.
- For replication context, trust Phase_2_work/instashap_project/*.py and its README.
- For presentation support, trust instashap_presentation/ after checking it against the current CSV tables.
- Treat the root README and phase3-architecture.md as context, not final evidence.

## Common Confusions

- The root README talks about an EBM plan that is not the current runnable Phase 3 implementation.
- The root phase3-architecture note describes an interaction-aware idea that is different from the current runnable masking improvement.
- Some older narrative docs mention outcomes that do not match the latest Phase 3 CSV summaries.
- The global folder exists to remove exactly these confusions.
