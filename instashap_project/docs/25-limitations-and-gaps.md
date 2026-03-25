# Limitations and Gaps

## Purpose

Good documentation should also be honest about what is incomplete or approximate.

## Current Limitations

### 1. No automated interaction detection

The paper discusses interaction selection ideas, but this reproduction currently uses manually configured interaction pairs.

### 2. Numerical parity with the paper is not guaranteed

Especially when:

- using `--fast-dev-run`
- subsetting Covertype
- changing training budgets

### 3. No dedicated deployment package

The project is modular, but it does not yet expose:

- saved checkpoint registry
- REST API
- batch inference CLI

### 4. No comprehensive unit test suite yet

The project has been validated through runtime smoke tests and artifact generation, but not through a formal automated test suite.

### 5. Vision experiments are out of scope here

The paper includes vision-side discussion, but this repository focuses on the tabular setting.

## Why These Limits Exist

They are tradeoffs made to prioritize:

- clarity
- modularity
- reproducibility
- educational value

## Best Next Improvements

- add tests
- add checkpoint save/load
- add automated interaction ranking
- add more systematic hyperparameter search

