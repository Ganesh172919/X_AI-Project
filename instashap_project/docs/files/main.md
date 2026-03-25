# File Reference: `main.py`

## Purpose

`main.py` is the command-line entry point for the project. It loads configuration, initializes logging, dispatches dataset runners, and triggers report generation.

## Responsibilities

- Parse CLI arguments
- Load `config.yaml`
- Set global seeds
- Configure structured logging
- Select dataset runners
- Generate the full PDF report and one-page summary

## Important Arguments

- `--dataset`
- `--model`
- `--config`
- `--fast-dev-run`
- `--skip-report`
- `--skip-summary`
- `--log-level`

## Why It Matters

If you want to change how the whole project runs, this is the first file to inspect.

## Common Extensions

- Add new CLI flags
- Add new datasets to the runner registry
- Add export or deployment steps after training

