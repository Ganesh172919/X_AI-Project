# Research Context

## Paper Context

The reference paper connects three ideas:

- **Generalized Additive Models (GAMs)**
- **Functional ANOVA**
- **SHAP / Shapley values**

The paper argues that additive models and SHAP explanations are more tightly connected than they are often presented in the literature.

## Key Research Claims

The paper motivates the project through several observations:

- If a low-order additive model can match the black-box model's predictive quality, then SHAP-style explanations become more trustworthy.
- Standard SHAP explanations can blur feature interactions, especially in correlated settings.
- A masked training objective can turn additive components into purified explanation terms.

## Why These Three Datasets

The tabular experiments cover different interaction regimes:

- **Bike Sharing**: a strong **synergistic** interaction between `hour` and `workingday`
- **Covertype**: a more **redundant / correlated** interaction between `elevation` and grouped soil information
- **Adult Income**: a supplementary benchmark for stable 1D additive behavior

## What Is Reproduced Here

This project reproduces the tabular experiment style, not the full vision experiments from the paper. The emphasis is on:

- Data loading
- Additive model training
- Masked surrogate training
- InstaSHAP training
- SHAP baseline comparison
- Artifact generation

## Important Reproduction Note

This repository is a **research reproduction**, not the official paper codebase. Some implementation decisions are faithful to the paper's objective and structure, while other details are engineering approximations chosen to keep the project modular and understandable.
