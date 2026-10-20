# Goal To Execution Map

## Goal file summary

The goal in `goal.txt` is clear:

```text
Identify a limitation in the original paper
and propose one improvement with references and experiments.
```

This folder answers the practical question:

```text
What exactly can I do in Phase 3?
```

## What the Phase 3 goal expects

From the project goal file, the required work is:

1. define a real research gap
2. implement an improvement
3. compare before vs after
4. provide code, results, and a short explanation

## What this project actually chose as the research gap

In `Phase_3_work`, the chosen research gap is:

```text
zero-masking in transformed tabular space creates unrealistic coalition samples
```

That is a valid and defensible limitation because:

- scaled numeric zero is often not a natural "missing" value
- one-hot all-zero categorical groups may represent an impossible category state
- explanation training can become biased by unrealistic masked inputs

## What the implemented improvement is

The improvement is:

```text
empirical-background masking
```

Meaning:

- when a feature group is hidden, the code copies that group from real training rows
- multiple background rows are sampled
- black-box and surrogate outputs are averaged across those masked realizations

## What all you can do with the current Phase 3 project

### 1. Reproduce the current experiment

You can run:

```text
baseline only
improved only
before-vs-after comparison
report regeneration
```

The CLI in `main.py` supports:

- `--variant baseline`
- `--variant improved`
- `--variant compare`
- `--report-only`

### 2. Study the whole pipeline end to end

You can inspect:

- dataset creation
- preprocessing
- masking design
- surrogate training
- InstaSHAP training
- explanation evaluation
- reporting

### 3. Use the project as a research template

You can replace:

- the dataset
- the black-box model
- the masking strategy
- the interaction list
- the surrogate capacity

### 4. Show both success and failure honestly

This is important.
Phase 3 is not only about proving a win.
It is also about proving that you understood the limitation and tested it honestly.

## Deliverables mapped to the repository

| Goal requirement | Where it lives in this project |
| --- | --- |
| Research gap | `instashap_project/masking.py` and generated reports |
| Code update | `instashap_project/` pipeline |
| Experimental comparison | `results/tables/` and `results/plots/` |
| 1-page explanation | `reports/phase3_research_gap_1page.md` and PDF |
| Reproducibility | `main.py`, `config.yaml`, `requirements.txt` |

## Simple beginner flow

```text
read goal
    ->
find weakness in original method
    ->
write one focused fix
    ->
compare old vs new
    ->
explain why results happened
```

That is the whole Phase 3 logic.

## The best approach for this goal

The strongest academic approach is:

1. choose one limitation only
2. make the fix narrow and testable
3. do not change five things at once
4. evaluate with clear metrics
5. report both gains and failures

This repository mostly follows that pattern well.

## What improved relative to the original setup

### Conceptual improvement

The original baseline assumed hidden features could be replaced with zeros.
The new version assumes hidden features should be replaced with realistic background values from actual data.

### Experimental improvement

The project now has:

- saved tables
- saved plots
- summary JSON
- markdown and PDF reports
- tests for masking and CLI behavior

### Research-quality improvement

The project is easier to defend in a report because it now has:

- a specific limitation
- a specific code change
- clear evaluation outputs

## What did not improve enough yet

The current saved metrics show that the improved masking idea did not beat the baseline on explanation fidelity in the present run.

That does not destroy the project.
It changes the conclusion:

```text
the idea is promising and more realistic,
but current optimization/training settings are not enough
to turn that into a clear performance win
```

That is a valid Phase 3 conclusion.

## How the existing system fails

### Failure case A: impossible coalition samples

Example:

```text
soil_climate_zone is one-hot encoded
hidden with zero-mask
all category bits become 0
```

That may not represent any real forest condition.

### Failure case B: dependence between features

Example:

```text
elevation and soil_climate_zone are related
```

If hidden features are replaced badly, the masked value function can drift away from realistic data.

### Failure case C: explanation target mismatch

Even if the masking idea is better, the surrogate and final explainer still need enough capacity to learn the new objective.

## What you can do next from this goal

### Option 1: strengthen the same direction

- tune surrogate capacity
- tune background sample count
- add more seeds
- add ablations

### Option 2: combine both Phase 3 ideas

- empirical-background masking
- interaction-aware surrogate

This would be the strongest "next phase" direction.

### Option 3: create failure-focused experiments

Design datasets where zero-mask is clearly unrealistic.
This can make the research claim much easier to show.

## Suggested viva or report explanation

If someone asks, "What did your Phase 3 work do?", the best short answer is:

```text
We identified that zero-masking in transformed tabular space creates unrealistic coalitions.
We replaced it with empirical-background masking that uses real training rows for hidden features.
We then compared baseline and improved InstaSHAP on Covertype using predictive, explanation, coalition, and runtime metrics.
The idea improved realism and slightly improved accuracy, but explanation fidelity was still weaker in the current training regime.
```

That is honest, clear, and academically strong.
