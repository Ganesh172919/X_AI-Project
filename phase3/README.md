# Phase 3: Interaction-Aware InstaSHAP Extension

This folder is the first clear Phase 3 interaction-aware prototype in the repository.

## What problem this folder studies

Original InstaSHAP is fast because it uses an additive surrogate. The downside is that additive models cannot represent strong feature interactions.

Simple example:

```text
True model: y = x1 * x2

If x1 changes alone, the effect depends on x2.
An additive model cannot represent that exactly.
```

That mismatch causes two problems:
- the surrogate can become a weak approximation of the black-box
- the resulting feature attributions can drift away from exact SHAP

## What this folder proposes

Use an interaction-aware GA2M-style surrogate:

```text
f(x) = bias
     + sum of single-feature effects
     + sum of pairwise interaction effects
```

Then split each pairwise interaction contribution equally across the two features when producing InstaSHAP values.

## How InstaSHAP works in this branch

```text
sample coalition masks
    ->
fit an additive surrogate first
    ->
measure whether the additive surrogate is faithful enough
    ->
upgrade to a pairwise interaction-aware surrogate when needed
    ->
convert surrogate terms into fast per-feature explanations
```

The key change is that the surrogate is allowed to learn pairwise terms instead of only single-feature terms.

## Flow

```text
train black-box
    ->
fit additive surrogate
    ->
check fidelity
    ->
if fidelity is poor, fit interaction-aware surrogate
    ->
compute InstaSHAP values from the surrogate terms
```

## What to open inside this folder

- `gap_analysis/research_gap.md` -> why the additive assumption fails
- `extension/interaction_aware_surrogate.py` -> how pairwise terms are added
- `extension/enhanced_instashap.py` -> how interaction terms are allocated
- `extension/adaptive_surrogate.py` -> how the upgrade decision is made
- `experiments/experiment_gap_demonstration.py` -> where failure is shown

## Best use case for this folder

Use this folder when the failure source is:

```text
the model depends on feature interactions
```

Typical examples:
- `friedman1`
- XOR-like patterns
- price models where location and size interact
- risk models where a feature matters only when another feature is high

## What improves here

- the surrogate can represent pairwise effects instead of forcing everything into separate feature terms
- the explanation rule can assign part of an interaction back to each participating feature
- interaction-heavy samples become more explainable in principle than under a purely additive surrogate

## What still fails here

- unrealistic masking is still unsolved
- higher-order interactions beyond pairs are still not handled directly
- better surrogate fidelity does not guarantee better final attribution metrics
- raw LLM generation remains a poor fit for the current tabular-style setup

## Main limitation of this folder

This branch improves representation, but it does not solve the masking problem. If masked coalitions are unrealistic, explanation quality can still suffer even with a better surrogate.

## Can this be used with a fine-tuned model?

Yes, if the fine-tuned model still uses stable feature groups and predicts a fixed target.

No, not directly for raw LLM generation, because:
- tokens are not stable tabular features
- masking changes language meaning
- outputs are sequences, not one fixed scalar

## Best next folder after this one

If you want the most complete version of this track, open `../phase3_3_VERSION/README.md`.
