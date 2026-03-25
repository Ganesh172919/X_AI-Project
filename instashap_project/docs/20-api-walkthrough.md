# API Walkthrough

## Purpose

This document explains the practical programmatic interfaces of the project for people who want to use the modules from Python instead of only from the CLI.

## Dataset API

Typical usage:

```python
from instashap_project.data.loaders import load_bike_sharing

bundle = load_bike_sharing()
print(bundle.features.head())
print(bundle.target.head())
print(bundle.metadata)
```

Main return type:

- `DatasetBundle`

## Preprocessing API

Typical usage:

```python
from instashap_project.data.preprocessing import TabularPreprocessor, make_splits

splits = make_splits(bundle, test_size=0.2, val_size=0.1, seed=42)
preprocessor = TabularPreprocessor(bundle.metadata).fit(splits.X_train)
X_train = preprocessor.transform(splits.X_train)
```

## Model API

### Black-box

```python
from instashap_project.models.blackbox_model import TabularMLP
```

### Additive GAM

```python
from instashap_project.models.gam import GAMModel
```

### InstaSHAP

```python
from instashap_project.models.instashap import InstaSHAPModel
```

## Training API

Main entry points:

- `train_blackbox_model(...)`
- `train_gam_model(...)`
- `train_masked_surrogate(...)`
- `train_instashap_model(...)`

All return:

- `TrainingResult`

## Evaluation API

Main helpers:

- `predict_raw_outputs(...)`
- `predict_targets(...)`
- `evaluate_supervised_model(...)`

## Experiment API

Dataset runners:

```python
from instashap_project.experiments.bike_sharing import run

result = run(config=config, selected_model="all")
```

Main orchestrator:

- `run_tabular_experiment(...)`

## Explanation API

SHAP:

```python
from instashap_project.xai.shap_wrapper import ShapBaselineExplainer
```

InstaSHAP:

```python
from instashap_project.xai.instashap_explainer import InstaSHAPExplainer
```

