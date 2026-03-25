# Experiment Trace: Adult Income

## Purpose

This walkthrough explains the Adult experiment from beginning to end.

## Command

```bash
python main.py --dataset adult --model all
```

## Execution Trace

1. UCI Adult is loaded through `load_adult_income()`
2. `education-num` is removed for the 13-feature representation
3. The train/val/test split is created with stratification
4. Preprocessing fits numeric scaling and categorical one-hot encoding
5. Black-box classifier trains
6. GAM-1 trains
7. Surrogate learns masked black-box behavior
8. InstaSHAP trains
9. SHAP explanations run
10. InstaSHAP explanations run
11. Metrics, explanation comparisons, and plots are written

## Main Outputs

- `adult_metrics.csv`
- `adult_paper_comparison.csv`
- `adult_explanation_comparison.csv`
- `adult_shape_age.png`
- `adult_shape_capital_gain.png`
- `adult_shape_education.png`

## Main Question This Experiment Answers

How competitive is a mostly 1D additive explanation setting on a classical tabular classification benchmark?

