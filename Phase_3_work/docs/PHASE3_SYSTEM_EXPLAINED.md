# Phase 3 System Explained

## Why this file exists

This is a fresh, code-first explanation of the Phase 3 work in this repository.
It does not depend on the older docs.
It explains:

- what Phase 3 is trying to prove
- how InstaSHAP works in the project
- what changed across the Phase 3 folders
- where the current system succeeds
- where it still fails
- which approach is best for dgit ifferent XAI limitations

## The short version

Phase 3 is about finding a weakness in the original InstaSHAP-style setup and then building an improvement that can be tested fairly.

This repository actually shows two different Phase 3 directions:

1. `phase3`, `phase3_2`, and `phase3_3_VERSION`
   These folders study the interaction problem:
   additive surrogates fail when the black-box relies on feature interactions.

2. `Phase_3_work`
   This folder studies the masking/value-function problem:
   zero-masking creates unrealistic tabular samples after scaling and one-hot encoding.

Those are not the same limitation.
They are two different reasons why XAI can fail.

## The Phase 3 folders and what they mean

| Folder | Main idea | Best for understanding |
| --- | --- | --- |
| `phase3` | First interaction-aware extension | Why additive surrogates break on interaction-heavy tasks |
| `phase3_2` | Similar extension branch | Intermediate duplicate of the interaction idea |
| `phase3_3_VERSION` | Most complete saved interaction-aware branch | Frozen results on synthetic interaction data |
| `Phase_3_work` | Standalone reproducible Covertype project | Full pipeline, results, reports, and practical limitations |

## What improved across the project

### Improvement path

```text
Original additive idea
    ->
Interaction-aware surrogate idea
    ->
Standalone experiment pipeline
    ->
Background-aware masking for tabular coalitions
    ->
Saved reports, plots, tests, and reproducible outputs
```

### What each step improved

| Stage | Improvement | Why it matters |
| --- | --- | --- |
| Early `phase3` | Adds pairwise interactions to the surrogate | Helps when black-box behavior is not purely additive |
| `phase3_3_VERSION` | Keeps comparison artifacts and summary tables | Makes the interaction-gap story easier to inspect |
| `Phase_3_work` | Builds a complete end-to-end pipeline around Covertype | Easier to run, compare, and report |
| `Phase_3_work` masking update | Replaces zero-mask with empirical background rows | Produces more realistic masked samples for tabular data |

## The big Phase 3 question

The whole project is really asking:

```text
When InstaSHAP gives a weak explanation,
is the problem caused by:

1. a weak surrogate family?
2. unrealistic feature masking?
3. data dependence between features?
4. interaction effects?
5. all of the above?
```

That is the correct beginner mental model.

## End-to-end project flow

```text
raw dataset
    ->
preprocessing with grouped original features
    ->
train black-box model
    ->
sample coalition masks
    ->
construct masked inputs
       zero-mask
       or
       empirical-background mask
    ->
train surrogate on coalition outputs
    ->
train additive InstaSHAP model to imitate the surrogate
    ->
produce one-pass feature attributions
    ->
compare against permutation SHAP
    ->
save tables, plots, summary JSON, markdown report, PDF report
```

## How InstaSHAP works here, in beginner language

Think of SHAP as a slow accountant.
It asks:

```text
How much did each feature contribute,
after checking many feature combinations?
```

InstaSHAP tries to learn a model that can answer that much faster.

Instead of recomputing many coalitions at explanation time, it does most of the work earlier:

1. Train a black-box predictor.
2. Create masked versions of the data.
3. Train a surrogate to imitate the black-box on those masked inputs.
4. Train an additive model so each feature component becomes a direct explanation term.
5. At explanation time, run one forward pass and read feature contributions directly.

That is why it is called "instant" compared with classical SHAP.

## Real example from this Phase 3 work

The main dataset is Covertype.
One important pair is:

- `elevation`
- `soil_climate_zone`

Why this pair matters:

```text
high elevation + alpine soil zone
can strongly support one forest cover type

but high elevation alone is not enough
and soil zone alone is not enough
```

This is a good example of where XAI can fail if it assumes features act independently.

## The best approach for XAI limitations in this repository

There is no single best approach for every limitation.
The best approach depends on why the explanation is failing.

### If the problem is unrealistic masking

Best approach in this repo:

- empirical-background masking from `Phase_3_work`

Why:

- zeroing standardized numeric features creates fake points
- zeroing one-hot groups can create impossible categorical states
- using real background rows keeps masked samples closer to the data manifold

### If the problem is missing interactions

Best approach in this repo:

- the interaction-aware surrogate idea from `phase3` and `phase3_3_VERSION`

Why:

- additive surrogates cannot model XOR-like or multiplicative behavior well
- pairwise interaction terms make the surrogate more expressive

### If the problem is both masking and interactions

Best practical research direction:

- combine realistic masking with interaction-aware surrogates

That combination is not fully implemented as one unified pipeline yet.
It is the most natural next improvement.

## Where the current system still fails

### Failure mode 1: additive bottleneck

Even in `Phase_3_work`, the final InstaSHAP model is still additive with optional pairwise sharing.
If the true black-box depends on higher-order effects, explanations can still be distorted.

### Failure mode 2: masked value function is still approximate

Empirical-background masking is better than zero-mask, but it is still not true conditional SHAP.

It says:

```text
replace hidden features with values from real rows
and average
```

That is useful, but not mathematically identical to the full conditional expectation.

### Failure mode 3: optimization is hard

Better theory does not guarantee better trained models.
In the saved Covertype run, the improved masking idea is more realistic, but the final surrogate and explanation metrics are still weaker than the zero-mask baseline.

This is a very important research lesson.

## Honest reading of the current Covertype results

From `Phase_3_work/results/tables`:

```text
instashap_zero accuracy_mean  = 0.6098
instashap_bg   accuracy_mean  = 0.6135

instashap_zero explanation_mae_mean = 0.2805
instashap_bg   explanation_mae_mean = 0.3109

instashap_zero spearman_mean = 0.4951
instashap_bg   spearman_mean = 0.4518

surrogate_zero coalition_mse_mean = 0.2863
surrogate_bg   coalition_mse_mean = 0.3793
```

Interpretation:

- predictive accuracy improved slightly
- explanation fidelity got worse
- coalition fidelity got worse
- runtime stayed fast for both InstaSHAP variants

So the improvement is meaningful as a research direction, but not yet a final win.

## Why this matters for a beginner

Many beginners assume:

```text
better idea -> better metric -> finished
```

Real research often looks like this:

```text
better idea
    ->
better realism
    ->
harder optimization
    ->
mixed results
    ->
new hypothesis
    ->
next experiment
```

That is exactly what is happening here.

## Can InstaSHAP be integrated with a fine-tuned model?

### Yes, if the model looks like structured prediction

Good fit:

- tabular classifiers
- tabular regressors
- fixed-length embedding models
- fine-tuned encoders where each input dimension has stable meaning
- multimodal models after feature extraction into stable vectors

### Maybe, but only with care

- time-series models with fixed windows
- vision models after region/patch grouping
- text classifiers after stable token-group or span-group design

### Poor fit

- autoregressive LLM generation
- open-ended sequence generation
- systems where features are not fixed or semantically stable

## Why InstaSHAP is a weak fit for LLMs

In raw LLM generation, the core assumptions break.

```text
InstaSHAP assumption:
fixed features
fixed output target
meaningful masking
surrogate can imitate the explanation game

LLM reality:
variable-length tokens
context dependence everywhere
masking changes sentence meaning
output is a generated sequence, not one scalar prediction
```

If you try to use this approach directly on an LLM:

1. token masking may create nonsense prompts
2. nearby tokens change each other's meaning
3. one prompt can generate many different outputs
4. the coalition game becomes unstable
5. the surrogate may learn prompt artifacts instead of causal language behavior

So the answer is:

```text
not impossible in a restricted setup,
but not a good direct method for raw LLM generation
```

## What to do next if you want the strongest Phase 3 story

Best next research path:

1. Keep empirical-background masking.
2. Add an interaction-aware surrogate on top of it.
3. Increase surrogate capacity carefully.
4. Compare against stronger SHAP references.
5. Add failure-case experiments where zero-mask is obviously unrealistic.

## Recommended reading order inside `Phase_3_work`

1. `project_goal/GOAL_TO_EXECUTION_MAP.md`
2. `instashap_project/INSTASHAP_ENGINEERING_GUIDE.md`
3. `results/RESULTS_FAILURE_ANALYSIS.md`
4. `reports/REPORTS_INTERPRETATION_GUIDE.md`
5. `tests/TESTS_AND_TRUST_BOUNDARIES.md`

## One-sentence conclusion

Phase 3 shows that fast XAI is not just about making explanation runtime small; it is about making the surrogate family, the masking strategy, and the evaluation setup match the real structure of the data.
