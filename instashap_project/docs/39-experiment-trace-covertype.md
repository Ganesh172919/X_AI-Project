# Experiment Trace: Covertype

## Purpose

This walkthrough explains the Covertype experiment step by step.

## Command

```bash
python main.py --dataset covertype --model all
```

## Execution Trace

1. UCI Covertype is loaded through `load_covertype()`
2. Raw soil indicators are grouped into `soil_climate_zone`
3. The compact 11-feature representation is created
4. Train/val/test splits are created with stratification
5. Black-box classifier trains
6. GAM-1 classifier trains
7. GAM-2 classifier trains with `elevation x soil_climate_zone`
8. Surrogate model learns masked black-box behavior
9. InstaSHAP additive model trains
10. SHAP and InstaSHAP explanations are compared
11. Tables, plots, and summary JSON are saved

## Main Outputs

- `covertype_metrics.csv`
- `covertype_paper_comparison.csv`
- `covertype_explanation_comparison.csv`
- `covertype_shape_elevation.png`
- `covertype_shape_soil_climate_zone.png`
- `covertype_interaction_elevation_soil_climate_zone.png`

## Main Question This Experiment Answers

Can a low-order additive interaction help reveal a redundant, correlated interaction structure that a 1D explanation would miss?

