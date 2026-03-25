# Model Architecture and Approach

## Modeling Stack

The repository implements four main model families:

1. **Black-box baseline**
   - MLP by default
   - Optional random forest wrapper
2. **GAM-1**
   - One learned function per original feature
3. **GAM-2**
   - GAM-1 plus selected pairwise interaction components
4. **InstaSHAP**
   - Additive model trained under the masked objective from the paper

## Why These Models

### Black-box model

Used to answer:

- How strong is an unconstrained predictive baseline?
- What does SHAP explain when the predictive model is flexible?

### GAM-1

Used to answer:

- Can simple 1D additive effects explain the task?
- Is the gap between black-box and additive performance large?

### GAM-2

Used to answer:

- Does adding low-order feature interaction structure close the performance gap?

### InstaSHAP

Used to answer:

- Can an additive model be trained so that its components are directly reusable as SHAP-like explanations?

## Pipeline View

```text
raw UCI data
-> dataset-specific cleanup
-> shared preprocessing
-> train black-box baseline
-> train optional GAM-1 / GAM-2
-> train masked surrogate of black-box outputs
-> train InstaSHAP on masked surrogate targets
-> evaluate predictions and explanations
-> save tables, plots, and reports
```

## Feature Engineering

Feature engineering is intentionally modest and interpretable:

- Bike: date decomposition and categorical casting
- Covertype: grouped climate-zone soil feature
- Adult: aligned 13-feature representation by dropping `education-num`

The project avoids opaque feature engineering because the goal is interpretability and reproducibility.

