# Development Workflow

## Recommended Workflow for Extending the Project

### 1. Start with the documentation

Read in this order:

- `README.md`
- `docs/index.md`
- `docs/16-project-structure.md`
- the relevant file reference pages in `docs/files/`

### 2. Use fast-dev mode first

Before making large changes, validate the pipeline quickly:

```bash
python main.py --dataset adult --model all --fast-dev-run
```

### 3. Change configuration before changing code

If the change is experimental rather than architectural, prefer editing:

- `config.yaml`

### 4. Make isolated code changes

Common safe patterns:

- new dataset loader
- new plot type
- new metric
- new experiment variant

### 5. Re-run a targeted experiment

Avoid rerunning everything while iterating. Use the smallest relevant command:

```bash
python main.py --dataset bike --model gam --fast-dev-run
```

### 6. Inspect saved artifacts

After the run, review:

- `results/tables/`
- `results/plots/`
- `results/artifacts/`
- `results/run.log`

## Best Practices for Future Work

- keep new features modular
- add new documentation pages with each major change
- preserve original feature-group mappings for explanation comparability
- compare new methods against the existing baselines in `experiments/common.py`

