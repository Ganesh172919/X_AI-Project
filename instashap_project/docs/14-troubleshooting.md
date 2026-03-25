# Troubleshooting

## Slow Runs

Symptoms:

- SHAP stage takes much longer than model inference

What to do:

- use `--fast-dev-run`
- reduce `global.shap_eval_samples`
- reduce `global.shap_background_size`
- run a single dataset at a time

## Weak Reproduction Numbers

Symptoms:

- metrics are far from the paper

What to check:

- Are you using `--fast-dev-run`?
- Did you reduce dataset size via `max_rows`?
- Are epochs too low in `config.yaml`?
- Is the chosen interaction pair sufficient?

## PowerShell Screen Reader Warning

This is a PowerShell warning, not a project error. It does not affect the project itself.

## TensorFlow / SHAP Startup Noise

The project suppresses most of this noise, but SHAP internals can still be environment-dependent. The actual run status is controlled by the structured logs.

## Missing GPU

The project works on CPU. GPU is optional.

