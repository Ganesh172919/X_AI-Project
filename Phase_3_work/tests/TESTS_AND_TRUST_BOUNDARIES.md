# Tests And Trust Boundaries

## Why this file exists

This folder tells you how much of the Phase 3 system is actually verified.

That is important because XAI projects can look convincing even when the explanation logic is only lightly tested.

## What the current tests cover

### `test_masking.py`

This test checks two important masking facts:

1. zero-mask keeps visible features unchanged
2. empirical-background masking preserves valid one-hot categorical structure

That is useful because the whole Phase 3 research gap depends on masking behavior being correct.

### `test_cli.py`

This test checks:

- default CLI variant parsing
- `--report-only` parsing

That is a light interface-level test.

## What the current tests do not cover

The current suite does not yet verify:

- end-to-end experiment execution
- surrogate fidelity improvement under specific settings
- explanation fidelity stability across seeds
- report content correctness
- shape-function plot generation
- interaction attribution correctness
- regression against saved result tables

So the present test suite is helpful, but still small.

## Trust boundary map

```text
CLI parsing                     tested lightly
masking correctness             tested partially
data loading                    not directly tested
preprocessing group mapping     not directly tested
black-box training              not directly tested
surrogate training              not directly tested
InstaSHAP attribution logic     not directly tested end to end
report generation               not directly tested
saved metrics consistency       not directly tested
```

## Why this matters for InstaSHAP

In XAI, there are two kinds of failure:

1. the code is wrong
2. the code is correct, but the explanation assumption is wrong

Both matter.

Tests mostly protect against type 1.
Experiments and ablations protect against type 2.

This project currently has more evidence for the experiment side than for the automated validation side.

## Real example of why tests are needed

Suppose a masking bug accidentally changes visible features too.
The final explanation can still produce numbers, plots, and reports.
Everything may look professional.
But the explanation target is broken from the start.

That is why even small masking tests are valuable.

## Why the system can still fail even if tests pass

Passing tests does not mean the explanations are trustworthy in every setting.

Example:

```text
test passes:
one-hot stays valid

but method still fails:
surrogate cannot learn the harder coalition objective well
```

That is exactly why the current results are mixed.

## Best additional tests to add next

### 1. Group bookkeeping tests

Verify that:

- each original feature expands to the correct transformed columns
- grouped SHAP aggregation matches expected shapes

### 2. Masking realism tests

Verify that empirical-background masking:

- never creates invalid one-hot states
- keeps visible groups unchanged
- samples only from the background bank

### 3. Surrogate consistency tests

On a tiny synthetic problem:

- compare surrogate output to black-box coalition outputs
- assert error stays below a threshold

### 4. Explanation sanity tests

On a simple additive synthetic function:

```text
y = 2*x1 - 3*x2
```

InstaSHAP should recover feature contributions close to the truth.

### 5. Interaction tests

On a pairwise synthetic function:

```text
y = x1 * x2
```

The project should show where additive explanations fail and whether interaction-aware extensions help.

### 6. Report regression tests

After a fixed run:

- check that markdown reports are generated
- check that required headings and tables are present

## How to show better work academically

If you want stronger evidence in Phase 3, do not rely only on final result tables.
Add:

- unit tests for masking and grouping
- synthetic sanity checks
- ablation studies
- fixed-seed regression tests

This makes your explanation claims easier to trust.

## Can these tests make the method valid for LLMs?

No.
They only verify the current tabular pipeline.

LLMs fail for deeper reasons:

- feature definitions are unstable
- token masking changes semantics
- outputs are sequences, not one scalar

So even perfect tests here would not make this a reliable raw-LLM explanation method.

## What happens if you still force this onto an LLM

You could create tests that pass for code execution, but the explanation itself could still be misleading because the coalition game is ill-defined.

That is a trust-boundary issue, not just a bug issue.

## Best beginner takeaway

```text
Tests can show the pipeline is behaving as designed.
They cannot prove the design is the right explanation game for every model class.
```

That is one of the most important lessons in XAI engineering.
