# FAQ

## Why are my results not matching the paper exactly?

Most likely because:

- you used `--fast-dev-run`
- your dataset subset is smaller
- hyperparameters are not fully tuned
- the project uses a clean modular reproduction, not the official paper code

## Why does Covertype use a grouped soil feature instead of all 40 raw soil columns?

To align the practical representation more closely with the paper's compact tabular discussion of a terrain-plus-soil interaction.

## Why are SHAP values computed on transformed columns first?

Because models are trained on transformed matrices. The project then aggregates those transformed-column attributions back to original features.

## Why is a surrogate model needed?

The surrogate approximates masked black-box behavior. InstaSHAP then learns an additive approximation of that masked function.

## Is this a deployment-ready inference API?

Not yet. The codebase is production-structured and modular, but it is primarily a reproducibility and research workflow project.

## Where should I start if I want to extend the project?

Read:

- `README.md`
- `docs/index.md`
- `docs/16-project-structure.md`
- `docs/17-development-workflow.md`
