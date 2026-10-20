# InstaSHAP Engineering Guide

## What this folder contains

This folder is the real implementation of the standalone Phase 3 pipeline.
If you want to understand how InstaSHAP works in code, start here.

Main submodules:

- `data/` loads and preprocesses datasets
- `models/` defines the black-box, surrogate, GAM, and InstaSHAP models
- `training/` trains each stage
- `xai/` runs SHAP and InstaSHAP explanation wrappers
- `masking.py` defines the coalition masking logic
- `reporting.py` turns outputs into reports

## The core idea in plain English

The project builds explanations in two learning stages:

```text
black-box model
    ->
surrogate learns masked coalition behavior
    ->
InstaSHAP learns an additive explanation model
```

This lets the final explainer return feature attributions in one pass.

## Step-by-step: how InstaSHAP works in this codebase

### Step 1: load and group the dataset

The Covertype loader reduces the raw dataset to:

- 10 numeric features
- 1 grouped categorical feature called `soil_climate_zone`

This matters because the code treats explanations at the original feature-group level, not at the one-hot-column level.

Relevant files:

- `data/loaders.py`
- `data/preprocessing.py`

### Step 2: preprocess without losing feature groups

`TabularPreprocessor` does two jobs:

1. standardizes numeric features and one-hot encodes categorical ones
2. remembers which transformed columns belong to each original feature

That group bookkeeping is essential.
Without it, SHAP values on one-hot columns would be hard to interpret.

### Step 3: train the black-box predictor

The black-box is usually an MLP in this project.
Its job is simple:

```text
predict the target as well as possible
```

This is the model we want to explain.

Relevant files:

- `models/blackbox_model.py`
- `training/train.py`

### Step 4: sample Shapley-style coalition masks

The code samples binary masks over original features.

Example:

```text
features = [elevation, aspect, slope, soil_climate_zone]
mask     = [1,         0,      1,     0]
```

Meaning:

- keep `elevation`
- hide `aspect`
- keep `slope`
- hide `soil_climate_zone`

This is the coalition game behind SHAP.

Relevant file:

- `training/train.py`

Function:

- `sample_shapley_feature_masks(...)`

### Step 5: build masked inputs

This is where the Phase 3 research gap lives.

Two masking strategies exist:

#### A. Zero-mask baseline

```text
masked_input = original_input * expanded_mask
```

Good:

- easy
- fast

Bad:

- standardized zero may not mean "missing"
- one-hot all-zero blocks can be unrealistic

#### B. Empirical-background masking

For each masked feature group:

- pick real background rows from training data
- keep visible features from the original row
- fill hidden features from background rows
- average outputs across several background samples

This is more data-aware.

Relevant file:

- `masking.py`

Key functions:

- `build_background_bank(...)`
- `build_masked_batch(...)`

## Why empirical-background masking is an improvement

Suppose a row has:

```text
elevation = very high
soil_climate_zone = alpine
```

If we hide `soil_climate_zone` with zero-mask, the one-hot block becomes all zeros.
That may represent no valid category at all.

If we hide it with empirical background:

- the hidden category is copied from a real row
- the masked sample still looks like something the model could plausibly see

That makes the coalition game more realistic.

## Step 6: train the surrogate model

The surrogate does not predict the original label directly.
It predicts:

```text
what the black-box would output
for a given masked coalition
```

That is a different task.

The surrogate receives:

- masked inputs
- the original full input
- the coalition mask itself

This helps it approximate the coalition value function.

Relevant classes:

- `MaskedSurrogateMLP`

Relevant function:

- `train_masked_surrogate(...)`

## Step 7: train the InstaSHAP model

The final InstaSHAP model is built on top of `GAMModel`.

The structure is:

```text
prediction =
    bias
  + sum of one-feature components
  + optional pairwise interaction components
```

During training, the model sees:

- the full input
- a coalition mask

Its target is:

- the surrogate's coalition output

Relevant classes:

- `GAMModel`
- `InstaSHAPModel`

Relevant function:

- `train_instashap_model(...)`

## Step 8: explain in one forward pass

After training, explanation is simple.

The model computes feature components directly.
Those components are then treated as SHAP-style attributions.

For pairwise interactions, the code splits the interaction equally between the two features.

```text
interaction(elevation, soil_climate_zone)
    ->
half goes to elevation
half goes to soil_climate_zone
```

Relevant methods:

- `GAMModel.feature_attributions(...)`
- `InstaSHAPModel.explain(...)`
- `InstaSHAPExplainer.explain(...)`

## Real example with Covertype

Imagine the black-box strongly predicts:

```text
cover type = class 3
```

for a sample with:

- high elevation
- alpine soil climate
- steep slope

The explanation process becomes:

```text
test sample
    ->
estimate how prediction changes when some feature groups are hidden
    ->
train surrogate on that masked behavior
    ->
train additive explainer against the surrogate
    ->
read direct attributions:
       elevation          +0.42
       soil_climate_zone  +0.35
       slope              +0.08
       aspect             -0.03
       ...
```

That is the main engineering value of InstaSHAP:

```text
front-load the expensive work,
then make explanation cheap
```

## What is the best approach here?

Inside this codebase, the best practical approach is:

1. keep feature groups intact
2. use realistic masking for hidden groups
3. compare against a trusted reference explainer
4. keep the final explanation model simple enough to inspect

That means:

- `TabularPreprocessor`
- empirical-background masking
- permutation SHAP as the reference
- additive or low-order interaction models

## Current limitations of InstaSHAP in this implementation

### Limitation 1: surrogate quality controls everything

If the surrogate learns the wrong coalition function, the final explanation is also wrong.

```text
bad surrogate
    ->
bad InstaSHAP targets
    ->
bad explanations
```

### Limitation 2: additive structure is still restrictive

The final model is interpretable because it is structured.
But that structure also limits expressiveness.

If the black-box uses:

- higher-order interactions
- discontinuous logic
- complex dependence patterns

the explainer may underfit.

### Limitation 3: empirical background is not true conditional SHAP

It is closer to realistic masking, but still an approximation.

### Limitation 4: evaluation reference is also approximate

The project compares against permutation SHAP.
That is useful, but it is not the same as exact SHAP on every model family.

### Limitation 5: training budget may be too small

The current results suggest the improved masking idea may need:

- more training time
- better surrogate capacity
- more careful tuning

## Why InstaSHAP fails in some use cases

### Use case: strong interaction logic

Example:

```text
approval = 1 only if income is high AND debt is low
```

An additive explainer can blur the pairwise condition into misleading single-feature importance.

### Use case: impossible masked inputs

Example:

```text
one-hot category hidden by setting all category columns to zero
```

The model may behave unpredictably because it never saw that pattern in training.

### Use case: sequence models

Example:

```text
"The movie was not good"
```

If you hide the word `not`, meaning changes completely.
Language features are not independent switches.

## Can I integrate InstaSHAP into a fine-tuned model?

### Good answer

Yes, if the fine-tuned model has:

- fixed-size inputs
- stable feature semantics
- a scalar or small-vector output target
- meaningful masking rules

Examples:

- fraud detection on tabular features
- medical risk scoring from structured patient data
- sentiment classification on frozen sentence embeddings

### Weak answer

Maybe, if you first convert the model into stable groups such as:

- image patches
- sentence spans
- learned latent feature blocks

### Bad answer

Directly on raw generative LLM token sequences, this is usually not a good idea.

## Why you usually cannot use this directly in an LLM

### The feature problem

LLMs do not have a small set of stable human-readable features.
They have sequences of tokens whose meaning depends on context.

### The masking problem

Token removal creates unnatural prompts.

Example:

```text
Original: "I do not recommend this product."
Mask "not": "I do recommend this product."
```

That is not "missing information".
That is a different sentence.

### The output problem

This project assumes outputs like:

- class probabilities
- regression scores

LLMs generate long sequences.
Explaining one generated answer requires defining:

- which token to explain
- which logit to explain
- which decoding step matters

### What happens if you still try

If you apply this directly to an LLM:

1. coalition masking corrupts prompt semantics
2. surrogate learns prompt artifacts
3. attribution becomes unstable across decoding steps
4. explanations look precise but are not trustworthy

## Best next technical improvements

If you want better InstaSHAP in this folder, the strongest next steps are:

1. Combine empirical-background masking with interaction-aware surrogates.
2. Increase surrogate capacity without losing stability.
3. Add ablations on background sample count.
4. Add experiments where zero-mask is known to create impossible states.
5. Compare against exact SHAP on a smaller tractable benchmark.

## Mental model to remember

```text
SHAP is the target idea
surrogate is the approximation bridge
InstaSHAP is the fast explanation model
masking strategy decides whether the bridge is built on realistic data
```

If the bridge is weak, the fast explanation will also be weak.
