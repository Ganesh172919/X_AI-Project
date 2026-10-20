# Results And Failure Analysis

## What this folder contains

This folder stores the evidence for the Phase 3 experiment.

Key subfolders:

- `tables/` for CSV metrics
- `plots/` for graphs
- `artifacts/` for summary JSON and seed-level outputs
- `run.log` for execution logging

## The main result in one sentence

The background-aware masking idea is more realistic than zero-mask, but the current saved run does not show better explanation fidelity.

## Current saved summary

### Predictive summary

```text
blackbox        accuracy 0.6942
gam1            accuracy 0.6977
gam2            accuracy 0.7056
instashap_zero  accuracy 0.6098
instashap_bg    accuracy 0.6135
```

### Explanation fidelity summary

```text
instashap_zero  mae 0.2805  spearman 0.4951
instashap_bg    mae 0.3109  spearman 0.4518
```

### Coalition fidelity summary

```text
surrogate_zero  mse 0.2863  mae 0.4047
surrogate_bg    mse 0.3793  mae 0.4715
```

## ASCII graph view

### Predictive accuracy

```text
blackbox        0.6942  ############################
gam1            0.6977  ############################
gam2            0.7056  #############################
instashap_zero  0.6098  ########################
instashap_bg    0.6135  ########################
```

### Explanation quality

Lower MAE is better:

```text
instashap_zero  0.2805  ############################
instashap_bg    0.3109  ###############################
```

Higher Spearman is better:

```text
instashap_zero  0.4951  #########################
instashap_bg    0.4518  #######################
```

### Coalition fidelity

Lower MSE is better:

```text
surrogate_zero  0.2863  ############################
surrogate_bg    0.3793  ######################################
```

## What improved

### Improvement 1: predictive accuracy

`instashap_bg` is slightly better than `instashap_zero` on accuracy.

This suggests the background-aware training signal may help the final model behave a little better as a predictor.

### Improvement 2: realism of masked inputs

This is not directly shown by one metric, but it is a real design improvement.

The new method:

- preserves valid one-hot patterns more often
- uses real training rows for hidden features
- reduces the artificiality of masked coalitions

### Improvement 3: research quality

The project now has a defendable limitation and a concrete fix.
That is an improvement even when the final score is mixed.

## Where the existing system fails

### Failure A: zero-mask creates unrealistic feature states

This is the original system failure that Phase 3 tries to fix.

Example:

```text
numeric feature after scaling:
0 may mean average, not missing

categorical one-hot group after zero-mask:
[0, 0, 0, 0]
may mean "no category", which is impossible
```

### Failure B: improved masking makes learning harder

The new target is more realistic, but also more variable.
The surrogate has to learn a more complex masked expectation.

That can raise:

- surrogate error
- downstream explanation error

### Failure C: final additive explainer may be too simple

The final model is still structured for interpretability.
That is good for explanation speed, but it can leave accuracy on the table when the coalition function is complex.

## Why InstaSHAP fails in particular use cases

### Use case 1: strong feature interactions

Example:

```text
prediction changes only when feature A and feature B appear together
```

A purely additive explanation can miss that joint logic.

### Use case 2: strong feature dependence

Example:

```text
elevation and soil zone move together in real data
```

If masked samples break that dependence, explanations can become unrealistic.

### Use case 3: out-of-manifold masked points

If masking creates points far from real data, the black-box can behave erratically there.
Then the explanation learns from unstable targets.

## Why the "improved" version did not win yet

The best current explanation is:

1. the masking idea is better conceptually
2. the resulting surrogate problem is harder
3. the current model/training budget was not enough
4. the final explanations therefore became weaker than the simpler baseline

This is a normal research outcome.

## What would show better work in InstaSHAP

If you want a stronger future result, focus on the following:

### 1. Stronger surrogate

- deeper surrogate
- better regularization
- more careful early stopping

### 2. Better background design

- tune background bank size
- tune train/eval background sample counts
- compare nearest-neighbor vs random background selection

### 3. Combined extension

- empirical-background masking
- interaction-aware surrogate

This is likely the highest-value next experiment.

### 4. Better failure-case benchmarks

Create examples where:

- zero-mask clearly breaks category validity
- interactions are known in advance
- exact SHAP is available for smaller models

### 5. More honest plots

Useful future plots:

- per-feature attribution error
- calibration of surrogate vs black-box on masked coalitions
- comparison of masked sample realism

## Real beginner example

Imagine a forest sample:

```text
elevation = high
soil_climate_zone = alpine
aspect = north-facing
```

Old system:

```text
hide soil_climate_zone
-> set one-hot group to all zeros
-> maybe impossible example
```

New system:

```text
hide soil_climate_zone
-> borrow a real soil zone pattern from training data
-> more realistic masked sample
```

But then:

```text
surrogate must learn an average over several realistic alternatives
```

That is harder than learning one simple zero-mask rule.

## The most important research takeaway

```text
realistic explanation targets
and
easy-to-learn explanation targets
are not always the same thing
```

That sentence explains the whole result very well.

## Bottom line

The current results do not prove that background-aware InstaSHAP is better overall.
They do prove that:

- zero-mask is a real limitation
- a principled fix was implemented
- the fix changes the learning problem in a meaningful way
- more work is needed to turn realism into better explanation fidelity
