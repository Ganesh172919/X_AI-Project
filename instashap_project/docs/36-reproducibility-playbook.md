# Reproducibility Playbook

## Purpose

This document gives a practical checklist for running the project in a more reproducible, research-style way.

## Step 1: Fix the environment

- pin dependencies with `requirements.txt`
- use one Python version consistently
- avoid changing library versions mid-experiment

## Step 2: Fix the random seed

The project already does this through:

- `global.seed`

Still, keep the seed unchanged across comparison runs.

## Step 3: Fix the config

Before comparing methods:

- save a stable `config.yaml`
- record whether `fast_dev_run` was used

## Step 4: Separate smoke tests from benchmark runs

Use:

- smoke-test runs for code validation
- full runs for reported comparisons

Do not mix the two when interpreting results.

## Step 5: Keep generated artifacts

Important artifacts to archive:

- metrics CSVs
- paper comparison CSVs
- explanation comparison CSVs
- dataset summary JSONs
- full report PDF
- run log

## Step 6: Document deviations

If you changed:

- hidden dimensions
- training epochs
- interaction pairs
- dataset size

record those deviations before comparing against prior runs.

