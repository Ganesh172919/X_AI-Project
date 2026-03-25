# Project Structure

## Top-Level Layout

```text
instashap_project/
|- README.md
|- config.yaml
|- main.py
|- requirements.txt
|- data/
|- models/
|- xai/
|- training/
|- experiments/
|- utils/
|- docs/
|- notebooks/
|- results/
\- reports/
```

## Folder-by-Folder Meaning

### `data/`

Responsible for:

- dataset loading
- dataset metadata
- preprocessing
- train/validation/test splits

### `models/`

Responsible for:

- black-box predictors
- additive GAM models
- InstaSHAP additive architecture

### `xai/`

Responsible for:

- SHAP baseline explanations
- direct InstaSHAP explanations

### `training/`

Responsible for:

- training loops
- loss computation
- mask sampling
- generic evaluation logic

### `experiments/`

Responsible for:

- end-to-end experiment orchestration
- dataset-specific experiment configuration

### `utils/`

Responsible for:

- metrics
- plotting
- logging
- reproducibility helpers

### `reports/`

Responsible for:

- full PDF report generation
- one-page summary generation

### `results/`

Generated experiment artifacts:

- tables
- plots
- summaries
- logs

## Where to Make Common Changes

- Add a new dataset: `data/` and `experiments/`
- Add a new model: `models/`, `training/`, `experiments/`
- Change plots: `utils/visualization.py`
- Change metrics: `utils/metrics.py`
- Change CLI behavior: `main.py`
