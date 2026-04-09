# Phase 3 Beginner Guide

This guide explains the full Phase 3 story in simple language.

## 1. The short version

Phase 3 is asking one honest research question:

```text
What is the main weakness in the current InstaSHAP-style system,
and how can we improve it without losing the speed advantage?
```

This repository explores two answers:

1. interaction-aware surrogates in `phase3/`, `phase3_2/`, and `phase3_3_VERSION/`
2. background-aware masking in `Phase_3_work/`

The strongest future direction is to combine both.

## 2. What InstaSHAP means here

Instead of running expensive SHAP calculations every time, InstaSHAP trains a model that learns to output SHAP-like explanations instantly.

Think of it like this:

```text
Exact SHAP:
  slow but very direct

InstaSHAP:
  train once, explain fast later
```

## 3. How the Phase 3 pipeline works across the project

```text
Step 1: Train a black-box model
Step 2: Create masked coalitions of features
Step 3: Ask the black-box what it predicts for those coalitions
Step 4: Train a surrogate to imitate those coalition outputs
Step 5: Train an interpretable additive model against the surrogate
Step 6: Use one forward pass to return per-feature explanations
Step 7: Compare those explanations against SHAP
```

In code, the main Phase 3 standalone flow is:

```text
main.py
  -> instashap_project/experiments/common.py
     -> data loading and preprocessing
     -> black-box training
     -> surrogate training
     -> InstaSHAP training
     -> SHAP comparison
     -> plots, tables, reports
```

## 4. The beginner mental model

Imagine a doctor-risk model with three inputs:

- age
- blood pressure
- smoking status

We want to know:

```text
How much did each feature contribute to the final prediction?
```

SHAP answers this by checking many "what if this feature were missing?" cases.
That is powerful but expensive.

InstaSHAP learns a fast approximation of that process.

## 5. Real example: why additive InstaSHAP fails

Suppose the true model is:

```text
risk = smoking * blood_pressure
```

Then the model does not really care about each feature alone. It cares about the combination.

### Case A

```text
smoking = 0
blood_pressure = high
risk may still be modest
```

### Case B

```text
smoking = 1
blood_pressure = high
risk jumps a lot
```

A purely additive explainer tries:

```text
effect(smoking) + effect(blood_pressure)
```

But the real behavior needs:

```text
effect(smoking)
+ effect(blood_pressure)
+ effect(smoking, blood_pressure)
```

That is the core reason `phase3_3_VERSION/` adds pairwise interaction terms.

## 6. Real example: why zero-mask fails on tabular data

Suppose you have a one-hot encoded feature:

```text
soil_climate_zone = [1, 0, 0, 0]
```

If you hide it by zero-masking, you get:

```text
[0, 0, 0, 0]
```

That is not a valid category. So the model is being asked to evaluate a fake sample.

The `Phase_3_work/` branch tries to fix that by filling hidden values from real training rows.

## 7. What each Phase 3 folder is doing

## `phase3/`

Purpose:
- first structured version of the interaction-aware research gap

What to learn there:
- why additive surrogates miss interactions
- how a GA2M-style surrogate helps
- what experiments are needed

## `phase3_2/`

Purpose:
- cleaner wording of the interaction-aware proposal

What to learn there:
- assignment-ready framing
- clearer explanation of expected contributions

## `phase3_3_VERSION/`

Purpose:
- most complete interaction-aware folder

What to learn there:
- saved experimental outputs
- actual gap demonstration
- practical tradeoff between fidelity and speed

## `Phase_3_work/`

Purpose:
- standalone runnable Phase 3 project on Covertype

What to learn there:
- real experiment pipeline
- tabular masking problem
- background-aware masking
- report generation

## 8. What improved in the repository

### Improvement 1: explicit interaction handling

In `phase3_3_VERSION/`:
- pairwise interaction terms are added to the surrogate
- interaction contributions are split across the involved features
- saved surrogate fidelity improved on `friedman1`

The good news:
- this is a better model of interaction-heavy data

The catch:
- better surrogate fit did not become a universal win in the saved explanation metrics

### Improvement 2: more realistic masking

In `Phase_3_work/`:
- hidden feature groups are filled with real background values
- one-hot groups stay valid
- similarity on visible features is used to choose background rows

The good news:
- it is a more realistic coalition construction

The catch:
- the saved run still shows weaker SHAP alignment than the zero-mask branch

## 9. What the saved results say

### Interaction-aware branch (`phase3_3_VERSION/`)

Saved `friedman1` numbers:

| Method | Surrogate R2 | Pearson | MAE | Runtime |
| --- | --- | --- | --- | --- |
| Original InstaSHAP | 0.9195 | 0.9285 | 0.2918 | 0.0066s |
| Interaction-aware InstaSHAP | 0.9473 | 0.9225 | 0.3144 | 0.0131s |

Interpretation:
- the surrogate got better
- the final explanation metrics did not improve in the same run
- this means the idea is promising, but the current pipeline still needs tuning

### Background-aware branch (`Phase_3_work/`)

Saved Covertype numbers:

| Method | Accuracy | Explanation MAE | Spearman | Coalition MSE | Explain time |
| --- | --- | --- | --- | --- | --- |
| `instashap_zero` | 0.6098 | 0.2805 | 0.4951 | 0.2863 | 0.0035s |
| `instashap_bg` | 0.6135 | 0.3109 | 0.4518 | 0.3793 | 0.0050s |

Interpretation:
- predictive accuracy improved slightly
- explanation quality did not improve in the saved run
- the idea is valid as a research extension, but not yet a complete success

## 10. Why the current system fails in some use cases

### Use case A: strong interactions

Problem:
- additive structure is too simple

Typical failure:
- surrogate predicts okay on average
- feature attributions are wrong for interaction-heavy samples

### Use case B: correlated features

Problem:
- independent or zero-style hiding creates unrealistic inputs

Typical failure:
- surrogate learns from fake coalitions
- explanation faithfulness drops

### Use case C: transformed categorical inputs

Problem:
- all-zero one-hot groups are invalid

Typical failure:
- coalition value function no longer matches real data behavior

### Use case D: large language models

Problem:
- features are tokens, spans, or hidden states, not stable tabular groups

Typical failure:
- masking breaks language
- output is sequence-level, not simple scalar-level
- surrogate becomes too weak or too expensive

## 11. Can InstaSHAP be integrated with a fine-tuned model?

### Good answer: yes, but only for the right kind of fine-tuned model

You can use an InstaSHAP-style pipeline if the model has:
- fixed input groups
- fixed output target
- a meaningful masking rule

Examples that fit:
- fine-tuned fraud classifier on customer features
- fine-tuned healthcare classifier on lab values
- fine-tuned demand forecaster on structured business data

### Partial answer: maybe, with grouping

Possible if you explain:
- document chunks
- prompt sections
- retrieved evidence blocks
- metadata groups

That is a custom grouped explainer, not a direct reuse of the current code.

## 12. Why this code should not be used directly for raw LLM explanation

Here is the simplest explanation:

```text
InstaSHAP in this repo assumes:
  stable feature groups
  meaningful masking
  fixed-size outputs
  low-order interpretable structure

Raw LLMs violate all four.
```

### What happens if you try anyway?

1. You choose tokens as features.
2. You mask some tokens.
3. The prompt becomes unnatural.
4. The model output changes for grammar reasons, not just semantic reasons.
5. The surrogate now has to imitate a huge sequence model.
6. Explanations become hard to interpret and easy to misuse.

So the answer is:
- not impossible in theory
- not a good use of the current implementation

## 13. The best approach for XAI limitations in this repo

If we step back, the XAI limitations here come from two places:

```text
representation problem  -> missing interactions
data realism problem    -> unrealistic masking
```

So the best research direction is:

```text
background-aware masking
    +
interaction-aware surrogate
    +
adaptive selection
    +
stronger fidelity evaluation
```

That is the best combined approach visible from the current repository.

## 14. Goal files: what they mean for your project

Open:
- `../project_goal/goal.txt`
- `../project_goal/project.md`

These files define the assignment requirement:

1. identify a research gap
2. implement one improvement
3. compare before and after
4. document results clearly

That means you can already do all of the following with this repo:

- explain the chosen research gap
- show code changes for the improvement
- run compare experiments
- generate plots and tables
- produce a one-page gap write-up
- discuss why the method still fails in some cases

## 15. Suggested presentation flow

If you need to explain the project to a beginner, use this order:

```text
1. What SHAP does
2. Why SHAP is slow
3. Why InstaSHAP is faster
4. Where InstaSHAP can fail
5. How Phase 3 tries to fix that
6. What improved
7. What still did not improve
8. What the next better version should combine
```

## 16. Suggested next experiments

1. combine interaction-aware surrogate with background-aware masking
2. test on `friedman1` and `covertype`
3. add one mostly additive dataset as a safety check
4. increase surrogate capacity and training budget
5. test calibration and explanation stability, not only MAE and rank metrics

## 17. File map for the fastest understanding

- `../README.md`
- `../project_goal/README.md`
- `../instashap_project/experiments/common.py`
- `../instashap_project/masking.py`
- `../instashap_project/models/gam.py`
- `../instashap_project/training/train.py`
- `../instashap_project/xai/instashap_explainer.py`
- `../reports/phase3_experiment_report.md`
