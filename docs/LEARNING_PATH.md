# Learning Path

Use this path to understand the project quickly and deeply.

## Level 1 - 20 minutes (Big Picture)

Goal: understand what the project does.

1. Read `../QUICKSTART.md`.
2. Read `CONCEPTS.md` sections 1-8.
3. Run one command:

```bash
python scripts/main.py --dataset california_housing --model-type random_forest
```

4. Inspect outputs in:
   - `results/tables/`
   - `results/figures/california_housing_random_forest/`

## Level 2 - 60 minutes (How It Works)

Goal: understand module responsibilities.

Read in this order:

1. `src/data_loader.py`
2. `src/black_box_model.py`
3. `src/shap_computation.py`
4. `src/gam_surrogate.py`
5. `src/evaluation.py`
6. `scripts/main.py`

While reading, answer these checkpoints:

- Where is exact SHAP computed?
- Where is surrogate trained?
- Which metrics represent fidelity vs speed?
- Which files are saved at each stage?

## Level 3 - 2 to 3 hours (Experiment Like a Researcher)

Goal: build intuition about tradeoffs.

### Experiment A: sample size sensitivity

Change in `config/config.yaml`:

- `shap_config.train_sample_size`: 500, 1000, 2000

Track:

- `r2`
- `pearson_correlation`
- `speedup_factor`

### Experiment B: surrogate complexity sensitivity

Change in `config/config.yaml`:

- `gam_config.max_iter`: 1000, 5000, 10000
- `gam_config.learning_rate`: 0.05, 0.01

Track:

- train time
- prediction time
- per-feature `r2`

### Experiment C: model family comparison

Run:

```bash
python scripts/main.py --dataset california_housing --model-type random_forest
python scripts/main.py --dataset california_housing --model-type xgboost
python scripts/main.py --dataset california_housing --model-type lightgbm
```

Compare ranking overlap and overall fidelity.

## Level 4 - Validate Understanding with Tests

Run:

```bash
pytest tests/ -v
```

Then map tests to modules:

- Data loading logic: `tests/test_data_loader.py`
- Model wrappers: `tests/test_black_box_model.py`
- Surrogate behavior: `tests/test_gam_surrogate.py`
- Metrics/plots: `tests/test_evaluation.py`
- Utilities: `tests/test_utils.py`

## Suggested Exercises

1. Add a new dataset loader method and wire it into config.
2. Add one new evaluation metric to `SHAPEvaluator`.
3. Add one integration test that runs a tiny full pipeline.
4. Add a script that exports run metadata as JSON.

## Common Pitfalls

- Comparing SHAP arrays with mismatched sample counts.
- Forgetting to align feature order between training and evaluation.
- Over-interpreting MAPE for near-zero SHAP values.
- Running large SHAP computations without caching.

## Fast Debug Checklist

If results look wrong:

1. Confirm `task_type` from `DatasetLoader`.
2. Confirm model and dataset pair is valid.
3. Confirm SHAP output shape equals feature count.
4. Confirm surrogate sees aligned `X_train` and `shap_train` shapes.
5. Check logs for fallback behavior (e.g., Adult synthetic fallback).

## Graduation Criteria

You understand the project when you can:

- Explain why one GAM is trained per SHAP feature.
- Predict how changing `train_sample_size` affects speed and fidelity.
- Run and interpret both `scripts/main.py` and `scripts/reproduce_results.py`.
- Modify one module and add/update matching tests confidently.
