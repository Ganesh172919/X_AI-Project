# Research Assumptions and Design Decisions

## Why This Document Exists

The project already explains what it does. This document explains **why certain implementation choices were made**, especially where the paper leaves room for engineering interpretation.

## Core Assumptions

### 1. The tabular experiments are the primary reproducibility target

This repository focuses on:

- Bike Sharing
- Covertype
- Adult Income

The paper also includes vision experiments, but this project intentionally prioritizes the tabular setting to keep the codebase practical and understandable.

### 2. `ucimlrepo` is the canonical dataset loader

Even though some datasets can also be loaded through `sklearn` or direct CSV URLs, this project uses:

- `fetch_ucirepo(id=275)`
- `fetch_ucirepo(id=31)`
- `fetch_ucirepo(id=2)`

This keeps dataset provenance explicit and consistent.

### 3. Explanation comparability matters more than maximizing raw predictive performance

The project favors:

- interpretable feature engineering
- original-feature group tracking
- grouped explanation outputs

over aggressive feature engineering or opaque preprocessing tricks.

### 4. Pairwise interactions are manually selected through configuration

The paper discusses interaction detection ideas such as Archipelago and sparse interaction selection. This reproduction currently uses:

- a config-driven interaction list

instead of a fully automated interaction discovery pipeline.

This is a deliberate simplification to keep the implementation easier to audit.

### 5. Classification explanations are handled on raw vector outputs

The project compares explanations on raw model outputs for classification settings, which is closer to the paper's discussion of using logits or log-probability style outputs rather than direct one-hot decisions.

## Engineering Approximations

### Masked surrogate

The masked surrogate is a practical approximation of the masked function behavior:

- it learns `f(x; S)` as a function of the transformed input and mask
- it is then used as the target for the additive InstaSHAP model

### Covertype compression

The raw UCI Covertype dataset has 54 columns. The project compresses this to a smaller representation to match the paper's 10 numeric + 1 soil-feature style discussion.

### Adult 13-feature representation

The project drops `education-num` to keep the Adult experiment aligned with the paper's supplementary 13-feature framing.

## Consequences of These Assumptions

- The project is easier to understand and extend
- The project may not numerically match the paper exactly
- The artifact structure is cleaner and better for learning and iteration

## If You Want a More Paper-Exact Reproduction

Possible future upgrades:

- add automatic interaction detection
- add alternative surrogate constructions
- tune architectures and optimization more aggressively
- run longer experiments on larger data subsets
