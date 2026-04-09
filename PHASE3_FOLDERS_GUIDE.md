# Phase 3 Folders Guide

This file is the quickest way to understand all Phase 3 work in this repository.

## 1. Which Phase 3 folder should you read first?

| Folder | Main idea | Best for | Current status |
| --- | --- | --- | --- |
| `phase3/` | First interaction-aware InstaSHAP design | Understanding the original research gap | Good architecture-first prototype |
| `phase3_2/` | Cleaner write-up of the same interaction-aware direction | Assignment framing and beginner reading | Better explanation, still mostly proposal-oriented |
| `phase3_3_VERSION/` | Most complete interaction-aware branch | Seeing saved results for interaction-aware InstaSHAP | Best folder for the interaction-gap track |
| `Phase_3_work/` | Standalone Covertype project with background-aware masking and pair interaction support | End-to-end runnable project and richer documentation | Best folder for the tabular masking-gap track |

## 2. What InstaSHAP is doing in this repository

At a high level, every InstaSHAP-style branch in this repo follows this pattern:

```text
Original sample x
    |
    v
Sample coalition mask S
    |
    v
Hide some features and keep others
    |
    v
Ask the black-box model for the masked prediction f(x, S)
    |
    v
Train a fast surrogate to imitate those coalition outputs
    |
    v
Train an additive or low-order interpretable model
    |
    v
Explain a new sample in one forward pass
```

The speed comes from learning a reusable explainer instead of recomputing SHAP from scratch for every new point.

## 3. The two different Phase 3 improvement tracks

### Track A: Interaction-aware InstaSHAP

Folders: `phase3/`, `phase3_2/`, `phase3_3_VERSION/`

Problem being studied:
- Original InstaSHAP is additive.
- Additive models miss feature interactions like `x1 * x2`.

Improvement:
- Add pairwise interaction terms through a GA2M-style surrogate.
- Split each pairwise term fairly across the two features.

Real example:

```text
True rule: crop_yield = rain * fertilizer

If rain is low, fertilizer helps little.
If rain is high, fertilizer helps a lot.

An additive explainer says:
  effect(rain) + effect(fertilizer)

But the real effect is:
  effect(rain) + effect(fertilizer) + effect(rain,fertilizer)
```

Best matching repository example:
- `friedman1`, where `x_1` and `x_2` interact.

### Track B: Background-aware masking

Folder: `Phase_3_work/`

Problem being studied:
- Zero-masking in transformed tabular space creates unrealistic samples.
- Standardized numeric zero may not be realistic.
- All-zero one-hot groups may be invalid categories.

Improvement:
- Fill hidden feature groups from real training rows.
- Match those rows using the visible features.
- Average multiple completions.

Real example:

```text
Visible features:
  elevation = high
  slope = medium

Hidden feature:
  soil_climate_zone

Zero-mask approach:
  soil_climate_zone -> [0, 0, 0, 0]
  This is not a real category.

Empirical-background approach:
  Copy soil_climate_zone from a real training row
  with similar visible terrain values.
```

## 4. Which approach is best?

There is no single winner for every failure mode.

- If the main problem is **feature interaction**, the best existing folder is `phase3_3_VERSION/`.
- If the main problem is **unrealistic masking on real tabular data**, the best existing folder is `Phase_3_work/`.
- If you want the **best next research direction**, combine both ideas:
  1. realistic masking/value function
  2. interaction-aware surrogate
  3. adaptive switching when extra complexity is not needed

That combined approach is the strongest recommendation across this repository.

### Best approach by failure mode

| Failure mode | Best current folder | Why |
| --- | --- | --- |
| Pairwise interaction is missing | `phase3_3_VERSION/` | It has the clearest interaction-aware surrogate story plus saved evidence. |
| Tabular masking creates unrealistic samples | `Phase_3_work/` | It directly studies empirical-background masking on a runnable pipeline. |
| You want the easiest end-to-end project to run | `Phase_3_work/` | It has the cleanest CLI, docs, tests, reports, and saved artifacts. |
| You want the original interaction-gap explanation | `phase3/` | It is the simplest first-pass interaction-aware prototype. |
| You want a presentation-ready version of the interaction idea | `phase3_2/` | It frames the same interaction gap in a cleaner assignment style. |
| You want the strongest future research direction | combine both tracks | One fixes interaction misspecification; the other fixes unrealistic coalition construction. |

## 5. What improved, and what did not

### Interaction-aware branch (`phase3_3_VERSION/`)

Saved `friedman1` results show:
- surrogate fidelity improved: `R^2` went from about `0.9195` to `0.9473`
- runtime increased: about `0.0066s` to `0.0131s`
- final attribution metrics did **not** clearly improve in the saved run

Meaning:
- the surrogate represented the black-box better
- but that did not automatically become better SHAP alignment everywhere

### Background-aware branch (`Phase_3_work/`)

Saved Covertype results show:
- predictive accuracy improved slightly: `0.6098` -> `0.6135`
- explanation MAE got worse: `0.2805` -> `0.3109`
- Spearman got worse: `0.4951` -> `0.4518`
- coalition fidelity got worse: surrogate MSE `0.2863` -> `0.3793`
- explanation time increased slightly: `0.0035s` -> `0.0050s`

Meaning:
- the idea is scientifically useful and plausible
- but the current surrogate capacity and training budget are not yet enough to turn it into a clean win

## 6. Why InstaSHAP fails in some use cases

### Failure mode A: interactions are real, but the explainer is additive

Example:

```text
loan_risk = income_level * debt_ratio
```

If the model only learns separate pieces for `income_level` and `debt_ratio`, it misses the fact that the combination matters.

### Failure mode B: masking creates fake data

Example:

```text
hidden one-hot group = [0, 0, 0, 0]
```

That may not correspond to any real category, so the black-box and surrogate learn from unnatural coalitions.

### Failure mode C: the surrogate is too weak for the coalition function

Even if the idea is correct, the surrogate may still underfit:
- too few epochs
- too little capacity
- not enough masked samples
- noisy or unstable coalition targets

### Failure mode D: explanation target is not a simple fixed-size prediction

This is where raw LLM use becomes difficult.

### Failure mode -> cause -> consequence -> fix

| Failure mode | Root cause | Consequence | Best fix in this repo |
| --- | --- | --- | --- |
| Interaction-heavy samples are explained poorly | additive surrogate misses joint effects | surrogate fit and attributions drift from exact SHAP | add pairwise interaction terms |
| One-hot groups become invalid when hidden | zero-masking creates fake categorical states | coalition values do not reflect real data behavior | empirical-background masking |
| Correlated tabular features behave unrealistically when hidden | hidden values are replaced independently or with zeros | surrogate learns from off-manifold samples | use real background rows matched on visible features |
| Better surrogate fit still does not improve explanations enough | downstream InstaSHAP model still underfits or is mismatched | final MAE / correlation may stay weak | raise capacity, samples, or combine both Phase 3 ideas |
| Raw LLM explanation becomes unstable | tokens are not stable tabular features and outputs are sequences | masked prompts become unnatural and hard to trust | explain grouped prompt/document/tool blocks instead of raw tokens |

## 7. Can you integrate InstaSHAP into a fine-tuned model?

### Yes, if the fine-tuned model still has stable feature groups

Good candidates:
- tabular classifier
- tabular regressor
- risk model
- fraud model
- medical score model
- multimodal system after features are converted into a fixed vector

What you need:
- a fixed input schema
- a clear output target such as a class score or regression value
- a meaningful masking rule for each feature group

### Maybe, with care

Possible but harder:
- fine-tuned encoder models where you explain sentence chunks, sections, or retrieved documents
- systems where you define groups like prompt blocks, tools, or evidence sources

At that point you are adapting the idea, not using the current code unchanged.

## 8. Why this repository does not directly fit raw LLM explanation

Raw LLMs are a poor match for the current InstaSHAP setup because:

1. tokens are not independent features
2. masking tokens changes grammar and meaning
3. outputs are sequences, not one fixed scalar
4. next-token distributions depend on previous generated tokens
5. coalition space becomes huge
6. a simple additive surrogate is usually too weak

If you try anyway, you usually get:
- unstable masked prompts
- surrogate mismatch
- explanations that are hard to trust
- large training and storage cost

A better LLM adaptation would explain:
- retrieved documents
- prompt sections
- tools used
- memory blocks
- final answer score or chosen class

That would be a grouped, task-specific explainer rather than the current tabular InstaSHAP pipeline.

## 9. Goal files and what you can do next

The formal Phase 3 goal lives in:
- `Phase_3_work/project_goal/goal.txt`
- `Phase_3_work/project_goal/project.md`

What those goal files allow you to do:
- define a research gap clearly
- implement one concrete improvement
- compare before vs after
- produce graphs, tables, and a one-page explanation

Best next project step:

```text
Combine:
  interaction-aware surrogate
  + background-aware masking
  + adaptive selection

Then evaluate on:
  friedman1
  covertype
  at least one mostly additive dataset
```

## 10. Recommended reading order

1. `PHASE3_FOLDERS_GUIDE.md`
2. `phase3_3_VERSION/README.md`
3. `Phase_3_work/README.md`
4. `Phase_3_work/docs/phase3_beginner_guide.md`
5. `Phase_3_work/project_goal/README.md`
