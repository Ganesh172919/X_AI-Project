# Project Overview

## Purpose

This Phase 3 repository is a standalone research extension built on top of the ideas reproduced in Phase 2. Its purpose is not to create a completely different explainability method, but to identify a specific weakness in the current InstaSHAP-style implementation and study one concrete improvement rigorously.

The project focuses on the **Covertype** dataset and asks the following question:

> Does a more realistic coalition-construction strategy improve the quality of tabular InstaSHAP training and explanations?

## Core Problem

The baseline pipeline masks hidden features by multiplying transformed inputs with zeros. That is attractive because it is simple and cheap, but it is not always semantically faithful for tabular data:

- standardized numeric features become artificial zeros that may not correspond to realistic hidden values
- one-hot categorical groups can become all-zero vectors, which may not represent a valid category
- the surrogate and the additive explainer may learn from coalition states that are statistically implausible

This matters most when the dataset contains dependence or structured relationships across feature groups. Covertype is a strong candidate because the project already models the interaction between `elevation` and `soil_climate_zone`.

## Main Contribution

The Phase 3 contribution is an **empirical-background masking** branch that:

1. keeps coalition masks at the original feature-group level
2. fills hidden transformed feature groups using values taken from real transformed training rows
3. selects those rows using similarity on the visible part of the coalition
4. averages across multiple background samples to approximate a more data-aware coalition value

This yields two directly comparable experiment branches:

- `instashap_zero`
- `instashap_bg`

## What The Project Delivers

- a standalone runnable package in `Phase_3_work`
- a Covertype-only comparison pipeline
- generated CSV tables, plots, JSON summaries, Markdown reports, and PDF reports
- assignment-facing documentation and AI usage declaration
- a small test suite for CLI parsing and masking behavior

## Honest Scope

This repository does **not** claim to implement full conditional SHAP. The implemented idea is a practical empirical-background approximation that addresses the zero-masking weakness in the current codebase.

## Reader Takeaway

The project should be read as a targeted research study with a narrow but meaningful systems question:

- keep the architecture family fixed
- change the coalition construction
- compare before vs after with transparent reporting
