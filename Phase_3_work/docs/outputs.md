# Outputs And Artifacts

## Output Philosophy

The project saves machine-readable and human-readable artifacts separately:

- CSV files for tables and grading
- PNG files for plots
- JSON files for run summaries
- Markdown files for readable report text
- PDF files for final deliverables

This makes the project easier to review and easier to regenerate.

## Tables

The main tables are saved under `results/tables/`.

### `covertype_predictive_metrics.csv`

Per-seed predictive metrics for all evaluated models.

Typical columns:

- `seed`
- `model`
- `accuracy`
- `log_loss`

### `covertype_predictive_summary.csv`

Grouped mean and standard deviation of predictive metrics by model.

### `covertype_explanation_fidelity.csv`

Per-seed explanation fidelity of each InstaSHAP branch against permutation SHAP.

Typical columns:

- `seed`
- `model`
- `mse`
- `mae`
- `spearman`

### `covertype_explanation_summary.csv`

Grouped summary of explanation-fidelity metrics.

### `covertype_coalition_fidelity.csv`

Per-seed surrogate error against the black-box coalition value function.

Typical columns:

- `seed`
- `model`
- `masking_strategy`
- `mse`
- `mae`

### `covertype_coalition_summary.csv`

Grouped summary of coalition-fidelity metrics.

### `covertype_runtime_metrics.csv`

Per-seed timing information for training, prediction, and explanation stages.

### `covertype_runtime_summary.csv`

Grouped timing summary by model.

## Plots

Plots are saved under `results/plots/covertype/`.

### Aggregate Comparison Plots

- `covertype_accuracy_comparison.png`
- `covertype_explanation_mae_comparison.png`
- `covertype_explanation_runtime_comparison.png`

### Explanation Alignment Plots

- `covertype_instashap_zero_alignment.png`
- `covertype_instashap_bg_alignment.png`

These plots show feature-level alignment error relative to permutation SHAP.

### Shape And Interaction Plots

- `covertype_shape_elevation.png`
- `covertype_shape_soil_climate_zone.png`
- `covertype_shape_aspect.png`
- `covertype_interaction_elevation_soil_climate_zone.png`

These are generated from the improved additive model to support interpretation.

### Seed-Specific Plots

The project also writes seed-specific training curves and SHAP importance plots for debugging and analysis.

## JSON Artifacts

### `results/artifacts/covertype/covertype_phase3_summary.json`

This is the main machine-readable run summary. It records:

- selected dataset and variant
- seed list
- saved table paths
- saved plot paths

### `results/artifacts/covertype/seed_<seed>/seed_summary.json`

Per-seed summary of the saved artifacts.

## Reports

### Markdown

- `reports/phase3_experiment_report.md`
- `reports/phase3_research_gap_1page.md`

These are useful for quick review inside the code editor.

### PDF

- `reports/phase3_experiment_report.pdf`
- `reports/phase3_research_gap_1page.pdf`

These are the assignment-facing deliverables.

## Regeneration Rule

The reports are regenerated from the saved summary JSON and CSV files, not from hardcoded text. That means if the underlying metrics change, rerunning `python main.py --report-only` refreshes the final write-up.
