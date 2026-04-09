# Reports Interpretation Guide

## Why this folder matters

This folder is where the project turns raw experiment outputs into a human-readable story.

Main generated files:

- `phase3_experiment_report.md`
- `phase3_experiment_report.pdf`
- `phase3_research_gap_1page.md`
- `phase3_research_gap_1page.pdf`
- `phase3_report_manifest.json`

## How the reports are generated

Report generation is handled by `instashap_project/reporting.py`.

The process is:

```text
read summary JSON
    ->
load result CSV tables
    ->
build compact before-vs-after table
    ->
write markdown
    ->
render PDF pages
    ->
attach saved plots
```

This means the reports are not hand-written opinions.
They are derived from saved experiment artifacts.

## What the main report is trying to answer

The report asks:

```text
Did the background-aware masking idea improve the Phase 3 InstaSHAP pipeline?
```

The answer from the saved run is:

```text
partly in concept,
slightly in predictive accuracy,
but not yet in explanation fidelity
```

## How to read the "Before vs After" section

The compact comparison uses two models:

- `instashap_zero`
- `instashap_bg`

Meaning:

- `instashap_zero` = old zero-mask coalition construction
- `instashap_bg` = improved empirical-background coalition construction

## Current report numbers

```text
instashap_zero
  accuracy_mean             0.6098
  explanation_mae_mean      0.2805
  explanation_spearman_mean 0.4951
  coalition_mse_mean        0.2863
  explain_seconds_mean      0.0035

instashap_bg
  accuracy_mean             0.6135
  explanation_mae_mean      0.3109
  explanation_spearman_mean 0.4518
  coalition_mse_mean        0.3793
  explain_seconds_mean      0.0050
```

## Beginner interpretation of each metric

### Accuracy

Higher is better.
This tells us whether the final model predicts the right class often.

### Explanation MAE

Lower is better.
This measures how far InstaSHAP explanations are from the permutation SHAP reference.

### Explanation Spearman

Higher is better.
This tells us whether the ranking of feature importance is similar to the reference explanation.

### Coalition MSE

Lower is better.
This checks whether the surrogate matches the black-box under the same masking game.

### Explanation runtime

Lower is better.
This is the core speed advantage of InstaSHAP.

## Visual summary

```text
Metric                     zero-mask      background-mask      Better
----------------------------------------------------------------------
Predictive accuracy        0.6098         0.6135               background
Explanation MAE            0.2805         0.3109               zero-mask
Explanation Spearman       0.4951         0.4518               zero-mask
Coalition fidelity MSE     0.2863         0.3793               zero-mask
Explain runtime (sec)      0.0035         0.0050               zero-mask
```

## What improved

The improved method did make one thing better:

- slightly higher predictive accuracy

It also improved the research story because the masking construction is more realistic and easier to justify.

## What did not improve

The saved run shows weaker:

- explanation MAE
- explanation rank correlation
- coalition fidelity
- explanation speed

This means the project should not claim:

```text
our new method is universally better
```

The honest claim is:

```text
our new masking idea is more realistic,
but it is harder to optimize and did not yet outperform the baseline in explanation fidelity
```

## Why the reports are still valuable even with mixed results

Because they prove that:

1. the research gap was real
2. the implementation changed the correct part of the pipeline
3. the comparison was run fairly
4. the conclusion is based on evidence, not guessing

That is strong research behavior.

## How to explain the result to a beginner

Use this story:

```text
The old system used fake hidden-feature values.
The new system used more realistic hidden-feature values.
That sounds better, and conceptually it is better.
But the new learning problem also became harder.
So the model did not yet learn explanations that matched SHAP better.
```

## How to explain it in a presentation

### One-line version

We improved coalition realism, but not final explanation fidelity.

### Slightly longer version

We replaced zero-mask coalitions with empirical-background coalitions because zero-mask produces unrealistic tabular inputs.
The change slightly improved predictive accuracy, but the learned surrogate and final explanations still underperformed the simpler baseline in the current run.

## Report flow diagram

```text
summary JSON
    ->
load predictive, explanation, coalition, runtime tables
    ->
compress into before-vs-after summary
    ->
add literature references
    ->
write markdown and PDF
```

## Real example to understand the failure

Suppose the model sees:

- high elevation
- alpine soil zone

When one of these is hidden using realistic background rows, the masked outputs may vary more because the data context is richer.
That richer target can help realism, but it also makes surrogate learning harder.

So:

```text
better target realism
does not automatically mean
easier imitation by the surrogate
```

## Best way to improve future reports

1. Add ablation tables for background sample count.
2. Add a table showing impossible one-hot states under zero-mask.
3. Add per-feature failure analysis.
4. Add exact-SHAP comparison on a smaller tractable benchmark.
5. Add interaction-aware plus background-aware combined results.

## Final takeaway

This folder shows that Phase 3 is not just "write a better method".
It is "write a testable improvement, evaluate it honestly, and explain why the result happened."
