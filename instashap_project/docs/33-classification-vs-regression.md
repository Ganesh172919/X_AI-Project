# Classification vs Regression in This Project

## Purpose

This document explains how the project behaves differently depending on task type.

## Regression Path

Used for:

- Bike Sharing

### Predictive training

- objective: mean squared error

### Evaluation

- RMSE
- MSE
- R2
- normalized MSE percentage

### Output shape

- scalar output per sample

## Classification Path

Used for:

- Covertype
- Adult

### Predictive training

- objective: cross-entropy

### Evaluation

- accuracy
- log-loss

### Output shape

- vector output per sample

## Explanation Comparison Difference

For regression:

- explanation tensor naturally corresponds to one scalar output

For classification:

- explanations are evaluated with respect to the predicted class output index

This allows SHAP and InstaSHAP to be compared per sample for the class that the model actually selects.

## Practical Effect

Classification explanations are slightly more complex because:

- they require output-index selection
- they involve vector outputs rather than scalars

