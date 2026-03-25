# Evaluation Metrics

## Predictive Metrics

### Regression

- `rmse`
- `mse`
- `r2`
- `nmse_pct`

`nmse_pct` is especially useful because the paper reports normalized error-style quantities for Bike Sharing.

### Classification

- `accuracy`
- `log_loss`

## Explanation Metrics

The project also compares explanation outputs using:

- `mse`
- `mae`

These are computed between grouped SHAP attributions and grouped InstaSHAP attributions on the evaluation samples.

## Runtime Metrics

The project records:

- training time
- average inference time
- explanation runtime for SHAP and InstaSHAP

## Baseline Comparisons

Each dataset can produce:

- black-box vs additive model comparisons
- reproduced vs paper table comparisons
- SHAP vs InstaSHAP explanation comparisons

## Where Metrics Are Saved

- `results/tables/*_metrics.csv`
- `results/tables/*_paper_comparison.csv`
- `results/tables/*_explanation_comparison.csv`

