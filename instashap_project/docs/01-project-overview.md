# Project Overview

## Problem Statement

The project reproduces the paper **"InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly."** The main question is:

> Can we train an interpretable additive model that directly outputs SHAP-style explanations in one forward pass while remaining useful for real tabular prediction tasks?

This matters because standard post-hoc explanation methods are often:

- Too slow for large-scale repeated use
- Limited in how they represent interactions
- Hard to inspect at the function level

## Real-World Use Case

The project targets practical ML settings where explanations are part of the workflow:

- Demand forecasting systems, where stakeholders want to know why predictions changed
- Structured classification tasks, where correlated features can create misleading 1D explanations
- Model auditing settings, where additive shape functions are easier to review than opaque latent features

## Objective Type

This repository supports two task types:

- **Regression**
  - Bike Sharing
- **Classification**
  - Covertype
  - Adult Income

## Expected Outcomes

After running the project, you should be able to:

- Train a black-box baseline
- Train GAM-1 and GAM-2 style additive models
- Train a masked surrogate for the black-box model
- Train an InstaSHAP model using a masked objective
- Compare SHAP and InstaSHAP explanations
- Generate paper-comparison tables and visual artifacts

## Core Idea in One Sentence

Instead of computing SHAP values separately at inference time, InstaSHAP trains an additive model so that its learned components can be interpreted as SHAP-style attributions immediately.
