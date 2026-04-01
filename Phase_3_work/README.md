# Phase 3: Covertype Background-Aware InstaSHAP Extension

This repository is a standalone Phase 3 project built inside `Phase_3_work`. It extends the InstaSHAP tabular pipeline with a focused research-gap study on the **Covertype** dataset.

## Documentation Hub

Detailed documentation is available in `docs/`.

- `docs/index.md`
- `docs/overview.md`
- `docs/methodology.md`
- `docs/architecture.md`
- `docs/experiments.md`
- `docs/outputs.md`
- `docs/file_reference.md`

Recommended reading order:

1. `docs/overview.md`
2. `docs/methodology.md`
3. `docs/architecture.md`
4. `docs/experiments.md`
5. `docs/outputs.md`
6. `docs/file_reference.md`

## What Changed From Phase 2

The Phase 2 reproduction used zero-masking in transformed feature space. That is simple, but it can create unrealistic coalition samples for tabular data after scaling and one-hot encoding.

Phase 3 adds an **empirical-background masking** path:

- `zero_mask`: baseline Phase 2 style coalition masking
- `empirical_background`: hidden original feature groups are filled using real transformed training rows chosen by similarity on the visible features

The experiment compares:

- `blackbox`
- `gam1`
- `gam2`
- `instashap_zero`
- `instashap_bg`

against permutation SHAP on the same Covertype split logic and seed set.

## Research Gap

The identified gap is that zero-masking in transformed feature space can distort the masked coalition value function for tabular data. In Covertype, this matters because feature groups such as `elevation` and `soil_climate_zone` carry meaningful dependence structure.

The proposed improvement is a data-aware masking/value-function approximation that:

- replaces hidden feature groups with values from real training rows
- keeps one-hot categorical groups valid
- averages coalition behavior over multiple empirical background samples

This is **not** a full conditional-SHAP estimator. It is a targeted, scoped improvement to the current implementation.

## References Used

Local references used to design and document the extension:

- `Phase_2_work/instashap_project/data/loaders.py`
- `Phase_2_work/instashap_project/reports/Original_Research_Paper_InstaSHAP.pdf`
- `Phase_2_work/instashap_project/reports/InstaSHAP_Research_Project_Report.md`
- `Phase_2_work/instashap_project/reports/InstaSHAP_Architecture.md`

Online references cited in the generated reports:

- Lundberg and Lee, SHAP: https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Jethani et al., FastSHAP: https://arxiv.org/abs/2107.07436
- Aas et al., dependent-feature SHAP: https://arxiv.org/abs/1903.10464
- Frye et al., Shapley explainability on the data manifold: https://arxiv.org/abs/2006.01272
- Tsai et al., Faith-Shap: https://jmlr.org/papers/v24/22-0202.html

## Project Structure

```text
Phase_3_work/
|- config.yaml
|- main.py
|- README.md
|- AI_USAGE.md
|- docs/
|- instashap_project/
|- reports/
|- results/
|- tests/
\- project_goal/
```

## Internal Package Structure

```text
instashap_project/
|- data/
|- experiments/
|- models/
|- training/
|- utils/
|- xai/
|- masking.py
\- reporting.py
```

The most important Phase 3 extension points are:

- `instashap_project/masking.py`
- `instashap_project/training/train.py`
- `instashap_project/experiments/common.py`
- `instashap_project/reporting.py`

## Setup

```bash
pip install -r requirements.txt
```

## Run Commands

Smoke test:

```bash
python main.py --dataset covertype --variant compare --fast-dev-run
```

Full baseline:

```bash
python main.py --dataset covertype --variant baseline
```

Full improved:

```bash
python main.py --dataset covertype --variant improved
```

Full comparison:

```bash
python main.py --dataset covertype --variant compare
```

Regenerate reports from saved artifacts:

```bash
python main.py --report-only
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Main Outputs

Tables:

- `results/tables/covertype_predictive_metrics.csv`
- `results/tables/covertype_predictive_summary.csv`
- `results/tables/covertype_explanation_fidelity.csv`
- `results/tables/covertype_explanation_summary.csv`
- `results/tables/covertype_coalition_fidelity.csv`
- `results/tables/covertype_coalition_summary.csv`
- `results/tables/covertype_runtime_metrics.csv`
- `results/tables/covertype_runtime_summary.csv`

JSON:

- `results/artifacts/covertype/covertype_phase3_summary.json`
- `results/artifacts/covertype/seed_<seed>/seed_summary.json`

Plots:

- `results/plots/covertype/covertype_accuracy_comparison.png`
- `results/plots/covertype/covertype_explanation_mae_comparison.png`
- `results/plots/covertype/covertype_explanation_runtime_comparison.png`
- `results/plots/covertype/covertype_instashap_zero_alignment.png`
- `results/plots/covertype/covertype_instashap_bg_alignment.png`
- `results/plots/covertype/covertype_shape_elevation.png`
- `results/plots/covertype/covertype_shape_soil_climate_zone.png`
- `results/plots/covertype/covertype_shape_aspect.png`
- `results/plots/covertype/covertype_interaction_elevation_soil_climate_zone.png`

Reports:

- `reports/phase3_experiment_report.md`
- `reports/phase3_experiment_report.pdf`
- `reports/phase3_research_gap_1page.md`
- `reports/phase3_research_gap_1page.pdf`

## Current Status

The implementation has been executed with `--fast-dev-run`, and the reports were generated from the saved artifacts. The report text is intentionally honest: it describes the empirical-background masking method, summarizes the observed outcomes, and does not overclaim a win when the metrics are mixed.

For final submission-quality numbers, run the full comparison command without `--fast-dev-run` so the PDF reflects the full training budget.

## Reviewer Checklist

If someone is reviewing this project quickly, this order works well:

1. read this `README.md`
2. read `docs/overview.md`
3. read `docs/methodology.md`
4. run the fast-dev compare command
5. inspect `results/tables/`
6. inspect `reports/phase3_experiment_report.md`
7. open the generated PDFs
8. run `python -m unittest discover -s tests -v`
