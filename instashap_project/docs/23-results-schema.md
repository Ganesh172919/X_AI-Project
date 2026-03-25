# Results Schema

## Purpose

This document explains the structure of the generated CSV, JSON, plot, and log artifacts.

## Metrics CSV Schema

Pattern:

- `results/tables/*_metrics.csv`

Possible columns:

- `model`
- `rmse`
- `mse`
- `r2`
- `nmse_pct`
- `accuracy`
- `log_loss`
- `training_seconds`
- `inference_seconds_mean`

The exact set depends on task type.

## Paper Comparison CSV Schema

Pattern:

- `results/tables/*_paper_comparison.csv`

Columns:

- `model`
- `reproduced`
- `paper`

## Explanation Comparison CSV Schema

Pattern:

- `results/tables/*_explanation_comparison.csv`

Possible rows:

- surrogate runtime row
- SHAP runtime row
- InstaSHAP runtime row
- SHAP vs InstaSHAP error row

Possible columns:

- `model`
- `training_seconds`
- `seconds_total`
- `samples`
- `mse`
- `mae`

## Dataset Summary JSON Schema

Pattern:

- `results/artifacts/<dataset>/<dataset>_summary.json`

Fields:

- `dataset`
- `task`
- `device`
- `features`
- `interaction_pairs`
- `metrics_table`
- `paper_comparison_table`
- `explanation_table`
- `plots`
- `paper_metadata`

## Run Log

Pattern:

- `results/run.log`

Format:

```text
time | level | event | key=value ...
```

## Plot Naming Convention

Typical patterns:

- `<dataset>_training_curves.png`
- `<dataset>_accuracy.png`
- `<dataset>_nmse_pct.png`
- `<dataset>_shape_<feature>.png`
- `<dataset>_interaction_<feature1>_<feature2>.png`
- `<dataset>_shap_importance.png`
- `<dataset>_shap_vs_instashap_alignment.png`

