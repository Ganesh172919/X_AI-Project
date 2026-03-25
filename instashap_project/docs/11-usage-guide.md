# Usage Guide

## Training the Project

### Run everything

```bash
python main.py --dataset all --model all
```

### Run a single dataset

```bash
python main.py --dataset bike --model all
python main.py --dataset covertype --model all
python main.py --dataset adult --model all
```

### Run only specific model families

```bash
python main.py --dataset bike --model gam
python main.py --dataset bike --model shap
python main.py --dataset bike --model instashap
```

## Example Terminal Output

```text
21:55:27 | INFO | experiment.start | dataset="bike" selected_model="all" task="regression"
21:55:30 | INFO | stage.complete | dataset="bike" primary_metric=204.8466348729589 stage="blackbox"
21:55:37 | INFO | stage.start | dataset="bike" stage="shap"
21:55:54 | INFO | experiment.complete | dataset="bike" summary_path="...bike_summary.json"
```

## Inference / Prediction Workflow

The project does not expose a separate deployment API yet, but the internal path is:

1. Load the dataset
2. Fit the preprocessor
3. Transform inputs
4. Load or train a model
5. Call `predict_targets(...)` or the specific model directly

## Explanation Workflow

### SHAP baseline

- Train black-box model
- Build `ShapBaselineExplainer`
- Explain transformed evaluation samples
- Aggregate transformed columns back to original features

### InstaSHAP

- Train black-box model
- Train masked surrogate
- Train InstaSHAP model
- Build `InstaSHAPExplainer`
- Explain transformed evaluation samples in one forward pass

## Example Outputs

Common output files:

- `results/tables/adult_metrics.csv`
- `results/tables/bike_explanation_comparison.csv`
- `results/plots/covertype/covertype_training_curves.png`
- `results/artifacts/bike/bike_summary.json`

