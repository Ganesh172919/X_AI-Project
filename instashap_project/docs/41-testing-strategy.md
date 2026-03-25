# Testing Strategy

## Purpose

The project does not yet contain a full test suite, so this document explains the current validation philosophy and a future testing plan.

## Current Validation Strategy

The project currently relies on:

- compile checks
- smoke-test runs
- end-to-end artifact generation
- inspection of structured logs

## What Is Already Validated in Practice

- imports resolve
- datasets load
- preprocessing fits
- training loops execute
- explanation wrappers run
- reports generate

## Recommended Future Automated Tests

### Unit tests

- dataset loader output schema
- feature-group mapping correctness
- mask expansion correctness
- SHAP aggregation correctness

### Integration tests

- one fast-dev run per dataset
- report generation after synthetic mock outputs

### Regression tests

- compare output schema and column names
- verify that core artifacts still exist after refactors

## Why This Matters

As the project grows, automated tests will become more important than manual inspection alone.

