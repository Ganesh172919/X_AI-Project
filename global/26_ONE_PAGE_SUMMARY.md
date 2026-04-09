# Phase 3 One Page Summary

## What we have done

- Reproduced the InstaSHAP tabular pipeline in a modular repository.
- Built a focused Phase 3 extension around a real limitation: unrealistic transformed-space zero masking.
- Implemented `empirical_background` masking so hidden feature groups come from real transformed training rows.
- Evaluated the improvement on Covertype with predictive, explanation, coalition, and runtime metrics.
- Added an Adult Income masking diagnostic to show the masking improvement more clearly on a category-heavy dataset.

## Current Covertype result

- Accuracy: zero 0.6843 vs bg 0.6774
- Explanation MAE: zero 0.3591 vs bg 0.3795
- Spearman: zero 0.5650 vs bg 0.5835
- Coalition MSE: zero 0.2021 vs bg 0.2016
- Interpretation: the new masking idea is valid, but the current end-to-end Covertype result is mixed rather than a full win.

## New Adult diagnostic result

- Hidden categorical validity: zero 0.0000 vs bg 1.0000
- Hidden categorical invalid rate: zero 1.0000 vs bg 0.0000
- Hidden numeric exact-zero rate: zero 1.0000 vs bg 0.0000
- Nearest-train distance mean: zero 2.1893 vs bg 1.3329
- Interpretation: Adult Income is a better dataset for showing the masking improvement itself, even before full retraining.

## Best next step

- Generalize Phase 3 to new datasets with dataset-specific configs and report names.
- Strengthen the surrogate branch for the harder empirical_background objective.
- Keep the project honest: report diagnostic gains separately from full end-to-end gains.
