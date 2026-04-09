# Phase 3: Covertype Background-Aware InstaSHAP Extension

This folder is the most runnable, beginner-friendly Phase 3 project in the repository. It studies a specific failure mode in tabular InstaSHAP:

```text
zero-masking can create unrealistic coalitions after scaling and one-hot encoding
```

It also keeps one pairwise interaction (`elevation`, `soil_climate_zone`) in the learned GAM/InstaSHAP models, so this branch is already closer to a practical combined Phase 3 system than the earlier folders.

## Read This First

- `docs/phase3_beginner_guide.md`
- `project_goal/README.md`
- `reports/phase3_experiment_report.md`

## What This Folder Tries To Improve

Baseline behavior:
- hide feature groups by writing zeros into transformed columns

Problem:
- numeric zero may not be realistic after standardization
- one-hot all-zero may be an invalid category
- surrogate training sees fake coalition states

Improvement:
- fill hidden feature groups from real transformed training rows
- select background rows using similarity on visible features
- average multiple completions per coalition

This is called `empirical_background` masking.

## How InstaSHAP Works In This Branch

```text
1. sample a coalition mask over original feature groups
2. keep visible feature groups from the real sample
3. hide the others using either:
      - zero_mask
      - empirical_background
4. ask the black-box for coalition outputs
5. train a surrogate to imitate those coalition outputs
6. train an additive one-pass InstaSHAP model on the surrogate targets
7. compare its explanations against permutation SHAP
```

In this branch, the main research change is step 3.

## Simple Flow

```text
raw Covertype data
    ->
preprocess into grouped transformed features
    ->
train black-box model
    ->
sample coalition masks
    ->
build masked coalitions
       |-> zero_mask branch
       |-> empirical_background branch
    ->
train surrogate on coalition outputs
    ->
train InstaSHAP model on surrogate outputs
    ->
compare against permutation SHAP
    ->
write tables, plots, JSON, Markdown, PDF
```

## Folder Guide

```text
Phase_3_work/
|- main.py                 -> CLI entrypoint
|- config.yaml             -> seeds, training budget, masking settings
|- docs/                   -> full explanation of the project
|- instashap_project/      -> implementation package
|- reports/                -> generated Markdown and PDF write-ups
|- results/                -> tables, plots, JSON summaries
|- tests/                  -> small regression tests
\- project_goal/           -> assignment goal and its interpretation
```

## The Main Question

```text
Does more realistic coalition construction help InstaSHAP on Covertype?
```

## What The Saved Results Show

| Method | Accuracy | Explanation MAE | Spearman | Coalition MSE | Explain time |
| --- | --- | --- | --- | --- | --- |
| `instashap_zero` | 0.6098 | 0.2805 | 0.4951 | 0.2863 | 0.0035s |
| `instashap_bg` | 0.6135 | 0.3109 | 0.4518 | 0.3793 | 0.0050s |

Honest interpretation:
- predictive accuracy improved slightly
- explanation alignment did not improve in the saved run
- coalition fidelity also got worse
- the idea is still valid as a research gap extension, but it needs more tuning

This is a good example of responsible XAI work: a reasonable idea does not automatically become a better explainer unless the full pipeline is strong enough.

## Why InstaSHAP Still Fails Here Sometimes

1. the coalition target is harder after realistic background filling
2. the surrogate may need more capacity or more masked samples
3. one interaction pair is included, but the real data may contain richer structure
4. SHAP alignment depends on both the masking rule and the surrogate quality

## Failure Mode Table

| Failure mode | Cause | Consequence | Best next action |
| --- | --- | --- | --- |
| invalid hidden categories | zero-masking creates all-zero one-hot groups | fake coalition samples | use empirical-background masking |
| weak surrogate coalition fit | masked objective becomes harder after background filling | worse coalition MSE / MAE | increase capacity, samples, or training budget |
| mixed explanation quality | better data realism does not guarantee a better explainer | SHAP alignment can still degrade | combine masking fix with stronger surrogate design |
| remaining feature dependence | one pairwise interaction is not the whole data structure | model still misses some behavior | combine with the interaction-aware Phase 3 track |

## Can This Be Used With A Fine-Tuned Model?

Yes, if the fine-tuned model still works on fixed feature groups and has a clear prediction target.

Good fit:
- tabular classifier
- tabular regressor
- structured risk model

Bad fit:
- raw generative LLM prompt-to-text explanation

Why not raw LLM:
- masking text breaks meaning
- outputs are sequences, not one fixed score
- token interactions are far more complex than the current additive setup

If you tried to use this exact pipeline on a raw LLM anyway, the likely outcome would be:
- unstable masked prompts
- explanations that reflect grammar breakage instead of true reasoning
- a very weak surrogate approximation
- high compute cost with low trustworthiness

## Best Next Improvement

The best next step across the whole repository is to combine:

```text
background-aware masking
    +
interaction-aware surrogate
    +
adaptive model selection
```

That would address both major Phase 3 failure sources:
- unrealistic coalitions
- missing interactions

## Quick Commands

Install:

```bash
pip install -r requirements.txt
```

Smoke test:

```bash
python main.py --dataset covertype --variant compare --fast-dev-run
```

Full comparison:

```bash
python main.py --dataset covertype --variant compare
```

Reports only:

```bash
python main.py --report-only
```

Tests:

```bash
python -m unittest discover -s tests -v
```
