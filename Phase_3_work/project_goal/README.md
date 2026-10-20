# Project Goal Notes

This folder contains the assignment-facing goal definition for Phase 3.

## Files in this folder

- `goal.txt` -> short original goal statement
- `project.md` -> formatted explanation of the same goal

## What the goal is asking you to do

The assignment is not asking for a brand-new XAI method from scratch.
It is asking for a careful research extension:

```text
1. Find one real limitation
2. Propose one justified improvement
3. Implement it
4. Compare before vs after
5. Document the evidence honestly
```

## How the current repository answers that goal

### In the interaction-aware folders

The gap is:
- additive surrogates miss interactions

The improvement is:
- add pairwise interaction terms

### In `Phase_3_work/`

The gap is:
- zero-masking creates unrealistic tabular coalitions

The improvement is:
- use empirical background filling from real rows

## What you can do with the current codebase

- run the baseline branch
- run the improved branch
- run the compare branch
- regenerate reports from saved artifacts
- explain the gap with concrete examples
- show where the method still fails
- propose a stronger next version by combining both Phase 3 ideas

## Best beginner explanation

If you need one sentence:

```text
Phase 3 is about finding where InstaSHAP breaks, fixing one part of that problem,
and proving with experiments whether the fix actually helps.
```

## Best next step after reading this folder

Open:
- `../README.md`
- `../docs/phase3_beginner_guide.md`
- `../reports/phase3_experiment_report.md`
