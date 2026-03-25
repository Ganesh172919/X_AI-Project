# Exploratory Data Analysis (EDA)

## Why EDA Matters Here

The paper is about explanation quality and feature interactions. That means EDA is not just "nice to have"; it directly informs:

- Which interactions are important
- Whether 1D explanations are likely to be misleading
- How correlated inputs may distort interpretation

## What the Project Already Produces

The experiment pipeline saves useful EDA-like artifacts automatically:

- Feature importance summaries
- Learned shape functions
- Pairwise interaction heatmaps
- Training curves

These are saved under:

- `results/plots/bike/`
- `results/plots/covertype/`
- `results/plots/adult/`

## Dataset-Specific EDA Questions

### Bike Sharing

Questions to ask:

- How does demand vary by hour?
- How different are workdays and non-workdays?
- Does the `hour x workingday` interaction dominate other effects?

### Covertype

Questions to ask:

- How strongly is elevation associated with class changes?
- How much information is already encoded by soil grouping?
- Are elevation and soil partially redundant?

### Adult Income

Questions to ask:

- Which socioeconomic variables drive most separability?
- Are the major drivers primarily 1D or interaction-heavy?
- Is an additive model already sufficient?

## How to Perform More EDA

Use the notebook:

- `notebooks/instashap_analysis.ipynb`

Recommended additions:

- Class balance plots
- Numeric feature histograms
- Correlation matrices
- Crosstabs for categorical features
- Pairwise scatter plots for the focus interaction features

## Interpretation Guidance

When reading the plots, keep these distinctions in mind:

- A strong **shape function** means a strong marginal additive effect.
- A strong **interaction heatmap** means the model needed pairwise structure beyond 1D effects.
- A mismatch between SHAP importance and additive interaction structure suggests explanation oversimplification.
