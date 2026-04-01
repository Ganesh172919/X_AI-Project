# Experiment Guide

## Supported Commands

### Compare Run

```bash
python main.py --dataset covertype --variant compare
```

Runs the complete Phase 3 comparison:

- black-box baseline
- GAM-1
- GAM-2
- zero-mask surrogate + `instashap_zero`
- background-aware surrogate + `instashap_bg`
- permutation SHAP evaluation
- table, plot, JSON, Markdown, and PDF generation

### Baseline-Only Run

```bash
python main.py --dataset covertype --variant baseline
```

Runs only the zero-mask branch and the shared reference models.

### Improved-Only Run

```bash
python main.py --dataset covertype --variant improved
```

Runs only the empirical-background branch and the shared reference models.

### Fast-Dev Smoke Test

```bash
python main.py --dataset covertype --variant compare --fast-dev-run
```

Useful for:

- checking that the project executes successfully
- validating that plots and reports are created
- iterating on implementation details quickly

### Report Regeneration

```bash
python main.py --report-only
```

This reads saved artifacts and rebuilds:

- `reports/phase3_experiment_report.md`
- `reports/phase3_experiment_report.pdf`
- `reports/phase3_research_gap_1page.md`
- `reports/phase3_research_gap_1page.pdf`

## Config Summary

The project uses `config.yaml` for:

- device selection
- seed list
- dataset size cap
- split ratios
- masking parameters
- training hyperparameters

Important Phase 3-specific settings:

- `global.seeds`
- `dataset.interaction_pairs`
- `masking.background_bank_size`
- `masking.background_samples_train`
- `masking.background_samples_eval`

## What Happens In One Full Compare Run

For each seed:

1. Set the random state.
2. Load and sample Covertype.
3. Create train, validation, and test splits.
4. Fit the tabular preprocessor.
5. Build the transformed training background bank.
6. Train the black-box model.
7. Train `gam1`.
8. Train `gam2`.
9. Compute permutation SHAP on evaluation samples.
10. Train `surrogate_zero`.
11. Train `instashap_zero`.
12. Train `surrogate_bg`.
13. Train `instashap_bg`.
14. Compute predictive, explanation, coalition, and runtime metrics.
15. Save per-seed and aggregate artifacts.

After the seed loop completes:

1. Save summary CSVs.
2. Save aggregate plots.
3. Write the summary JSON.
4. Regenerate Markdown and PDF reports.

## How To Read The Results

### Predictive Summary

Use this to answer:

- does the improved branch preserve classification quality?
- how far are the additive explainers from the black-box?

### Explanation Summary

Use this to answer:

- which one-pass explainer is closer to permutation SHAP?
- is the difference visible in MAE, MSE, or rank correlation?

### Coalition Summary

Use this to answer:

- is the surrogate learning the intended coalition function accurately?
- is the masking strategy improving the surrogate target or making optimization harder?

### Runtime Summary

Use this to answer:

- how much speed advantage remains relative to permutation SHAP?
- what is the cost of the background-aware branch?

## Expected Reviewer Workflow

A grader or reviewer can validate the project in this order:

1. install dependencies
2. run the fast-dev compare command
3. inspect `results/tables/`
4. inspect `results/plots/covertype/`
5. read the Markdown reports
6. open the PDFs
7. run `python -m unittest discover -s tests -v`

This sequence should be enough to understand the project without reading every source file first.
