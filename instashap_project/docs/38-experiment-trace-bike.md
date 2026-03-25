# Experiment Trace: Bike Sharing

## Purpose

This is a step-by-step walkthrough of what happens when you run the Bike Sharing experiment.

## Command

```bash
python main.py --dataset bike --model all
```

## Execution Trace

1. `main.py` loads config and selects the Bike runner
2. `experiments/bike_sharing.py` loads Bike Sharing through `load_bike_sharing()`
3. `experiments/common.py` creates splits
4. `TabularPreprocessor` fits transformations and feature groups
5. Black-box model trains on regression targets
6. GAM-1 trains
7. GAM-2 trains with `hour x workingday`
8. Masked surrogate trains on black-box masked outputs
9. InstaSHAP trains against masked surrogate outputs
10. SHAP runs on evaluation samples
11. InstaSHAP explanations run on the same samples
12. Metrics, plots, and summary JSON are written

## Main Outputs

- `bike_metrics.csv`
- `bike_paper_comparison.csv`
- `bike_explanation_comparison.csv`
- `bike_shape_hour.png`
- `bike_shape_workingday.png`
- `bike_interaction_hour_workingday.png`

## Main Question This Experiment Answers

Does a low-order additive interaction model capture the key Bike Sharing synergy better than a purely univariate additive explanation?

