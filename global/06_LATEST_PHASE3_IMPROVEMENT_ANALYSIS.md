# Latest Phase 3 Improvement Analysis

This document gives the exact answer to the user question about the latest Phase 3 improvement and whether it improved InstaSHAP performance.

## Direct Answer

- The latest runnable Phase 3 improvement is empirical_background masking in Phase_3_work/instashap_project/masking.py.
- The root phase3-architecture.md is a conceptual interaction-aware note, not the same thing as the current runnable Phase_3_work branch.
- The root README still describes an earlier EBM-flavored plan, so it should not be treated as the final truth source for the current runnable system.
- The current saved Phase 3 artifact snapshot favors instashap_zero on accuracy (0.6843 vs 0.6774) and explanation MAE (0.3591 vs 0.3795).
- The current saved Phase 3 artifact snapshot slightly favors instashap_bg on Spearman rank alignment (0.5835 vs 0.5650) and coalition MSE (0.2016 vs 0.2021).
- The limitation is still a good research choice because it is specific, code-level, measurable, and easy to explain to reviewers.

## What Changed In Practice

- Before: hidden feature groups were zeroed in transformed space.
- After: hidden feature groups can be completed from real transformed training rows.
- Before: a coalition could represent impossible category states.
- After: a coalition is more likely to stay on realistic transformed patterns.
- Before and after still share the same broad black-box -> surrogate -> additive explainer pipeline.

## Current Metric Interpretation

- Accuracy currently favors instashap_zero (0.6843) over instashap_bg (0.6774).
- Explanation MAE currently favors instashap_zero (0.3591) over instashap_bg (0.3795).
- Spearman rank correlation currently favors instashap_bg (0.5835) over instashap_zero (0.5650).
- Coalition MSE currently very slightly favors surrogate_bg (0.2016) over surrogate_zero (0.2021).
- Explanation runtime currently favors instashap_zero (0.0100s) over instashap_bg (0.0121s).
- The net result is mixed, not a broad overall win.

## Is It A Good Limitation

- Yes, because it is specific and easy to tie to one implementation point in the codebase.
- Yes, because it is easy to explain why standardized zeros and all-zero one-hot vectors are often unrealistic.
- Yes, because it creates measurable hypotheses around coalition fidelity and explanation fidelity.
- Yes, because even a mixed outcome still teaches something real about the pipeline.

## What To Avoid Saying

- Do not claim that the latest branch proved overall performance improvement.
- Do not claim that the current branch is the same as the interaction-aware concept note.
- Do not mix stale narrative docs with the current Phase 3 CSV tables.

## Best Next Step

- Increase surrogate capacity or training budget for the empirical_background branch.
- Add direct invalid-state metrics so the limitation is measured even more explicitly.
- Test on another mixed categorical dataset such as Adult Income or a credit/churn benchmark.
- Combine background-aware masking with interaction-aware modeling rather than treating them as separate stories.
