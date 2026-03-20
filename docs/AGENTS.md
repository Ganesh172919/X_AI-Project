# Contributor Guide

This guide is for anyone working on the codebase -- humans and AI agents
alike. It covers coding conventions, testing, and how to run the project.

---

## Environment setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .

# Verify installation
python -c "import shap, interpret, xgboost, lightgbm; print('Setup complete!')"
```

---

## Running experiments

```bash
# Single experiment with default settings
python scripts/main.py --dataset california_housing --model-type random_forest

# With different dataset and model
python scripts/main.py --dataset breast_cancer --model-type xgboost

# Full reproduction (all dataset + model combinations)
python scripts/reproduce_results.py

# Interactive notebook
jupyter notebook notebooks/replication_notebook.ipynb

# Available options:
#   --dataset: adult, california_housing, breast_cancer
#   --model-type: random_forest, xgboost, lightgbm
#   --config: path to config file (default: config/config.yaml)
#   --log-level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

---

## Running tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Single test file
pytest tests/test_gam_surrogate.py -v

# Single test function
pytest tests/test_gam_surrogate.py::test_surrogate_train -v

# Quick smoke test
pytest tests/ -v -x --tb=short
```

---

## Coding conventions

- **Python 3.8+** compatible.
- **Type hints** on all public function signatures.
- **Docstrings** on every module, class, and public method (Google style).
- **No comments** in code unless explicitly asked for by the user.
- Use `logging` instead of `print()` in library code.
- Prefer NumPy vectorized operations over Python loops.
- Keep `src/` modules free of side effects at import time.
- Configuration goes in `config/config.yaml`, not hardcoded in source.
- Use `pathlib.Path` for file path operations.
- Use `joblib` for serialization of models and large arrays.

---

## Project structure rules

- `src/` contains only importable library code.
- `scripts/` contains runnable entry points that import from `src/`.
- `tests/` mirrors `src/` (one test file per module, prefixed `test_`).
- `results/` is generated at runtime and should not be committed.
- `data/` is for custom datasets and should not be committed.
- No module in `src/` should import from `scripts/`.
- All public classes/functions are re-exported from `src/__init__.py`.

---

## Adding a new dataset

1. Add config entry under `datasets` in `config/config.yaml`:
   ```yaml
   datasets:
     my_dataset:
       name: "My Dataset"
       task: "classification"  # or "regression"
       test_size: 0.2
   ```

2. Add `_load_my_dataset()` method to `DatasetLoader` in `src/data_loader.py`:
   ```python
   def _load_my_dataset(self) -> None:
       # Load your data
       X, y = ...
       self.feature_names = list(X.columns)
       self.task_type = "classification"
       
       # Split and scale
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=self.test_size, random_state=self.random_state
       )
       self.X_train = self.scaler.fit_transform(X_train)
       self.X_test = self.scaler.transform(X_test)
       self.y_train = y_train
       self.y_test = y_test
   ```

3. Update the dispatch in `load_data()`:
   ```python
   elif self.dataset_name == "my_dataset":
       self._load_my_dataset()
   ```

4. Add tests in `tests/test_data_loader.py`.

---

## Adding a new model type

1. Add config block under `black_box_models` in `config.yaml`:
   ```yaml
   black_box_models:
     my_model:
       classification:
         param1: value1
       regression:
         param1: value1
   ```

2. Extend `_initialize_model()` in `BlackBoxModel`:
   ```python
   elif self.model_type == "my_model":
       if self.task == "classification":
           self.model = MyClassifier(**self.model_params)
       else:
           self.model = MyRegressor(**self.model_params)
   ```

3. Add tests in `tests/test_black_box_model.py`.

---

## Dependencies

All dependencies are pinned with minimum versions in `requirements.txt`
and `setup.py`. When adding a new dependency:

1. Add it to both `requirements.txt` and `setup.py`.
2. Verify it is compatible with Python 3.8+.
3. Import it inside the function that uses it if it is optional.

Core dependencies:
- `numpy`, `pandas`, `scikit-learn` - data processing
- `shap` - exact SHAP computation
- `interpret` - GAM (EBM) implementation
- `xgboost`, `lightgbm` - gradient boosting models
- `matplotlib`, `seaborn` - visualization
- `pyyaml` - configuration
- `joblib` - serialization

---

## Linting and type checking

Run before submitting changes:

```bash
# Syntax check
python -m py_compile src/*.py
python -m py_compile scripts/*.py

# Optional: ruff for linting
ruff check src/ scripts/

# Optional: mypy for type checking
mypy src/ --ignore-missing-imports
```

---

## Configuration reference

Key settings in `config/config.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `random_seed` | - | Global seed for reproducibility |
| `datasets.*.test_size` | - | Train/test split ratio |
| `shap_config.train_sample_size` | - | Samples for GAM training |
| `shap_config.test_sample_size` | - | Samples for evaluation |
| `gam_config.max_iter` | - | EBM boosting rounds |
| `gam_config.interactions` | - | 0 for pure additive GAM |
| `computation.cache_shap` | - | Enable SHAP value caching |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Import errors | Run from project root; add root to `PYTHONPATH` |
| Memory errors | Reduce `train_sample_size` / `test_sample_size` in config |
| Slow SHAP | Set `cache_shap: true` in config |
| OpenML timeout | Pipeline falls back to synthetic data automatically |
| GPU not used | Set `use_gpu: true` in config (requires CUDA) |
| Plot errors | Install `matplotlib` with GUI backend or use `Agg` |

---

## Common workflows

### Reproduce a specific experiment
```bash
python scripts/main.py --dataset california_housing --model-type random_forest
```

### Run all experiments
```bash
python scripts/reproduce_results.py
```

### Load cached results
```python
from src.utils import load_object

# Load trained model
model = load_object("results/models/california_housing_random_forest_model.pkl")

# Load cached SHAP values
shap_data = load_object("results/models/california_housing_random_forest_shap_train.pkl")
shap_values = shap_data["shap_values"]
```

### Evaluate a trained surrogate
```python
from src import SHAPSurrogate, SHAPEvaluator

surrogate = SHAPSurrogate()
surrogate.load_model("results/models/california_housing_random_forest_gam_surrogate.pkl")

pred_shap = surrogate.predict_shap(X_test)
evaluator = SHAPEvaluator(feature_names)
metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)
```
