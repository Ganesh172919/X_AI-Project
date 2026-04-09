# Phase 3 Extension Reference

This is the main technical summary of the current runnable Phase 3 system.

## Runnable Entry Layer

- Phase_3_work/main.py is the CLI entrypoint.
- Phase_3_work/config.yaml defines seeds, training budgets, and masking settings.
- The CLI supports baseline, improved, compare, and report-only flows.

## Core Improvement

- masking.py implements zero_mask and empirical_background.
- A background bank is built from real transformed training rows.
- Hidden groups can be filled from similar background rows based on visible features.
- Multiple background completions can be averaged.

## Experiment Orchestration

- experiments/common.py coordinates data preparation, training, evaluation, and artifact writing.
- The branch evaluates blackbox, GAM-1, GAM-2, instashap_zero, and instashap_bg.
- Permutation SHAP remains the explanation reference baseline.
- The outputs are persisted as CSV summaries, per-seed tables, plots, JSON summaries, Markdown, and PDF.

## Training Path

- training/train.py samples Shapley-style masks.
- The surrogate learns masked coalition outputs from the black-box.
- The InstaSHAP model learns to imitate the surrogate under the same masking strategy.
- The improved strategy changes the coalition values being learned, not the overall additive shape of the explainer.

## Verification

- tests/test_cli.py checks the command-line behavior.
- tests/test_masking.py checks that empirical_background keeps one-hot groups valid.
- This makes the Phase 3 branch safer to defend because the core improvement has explicit tests.

## Current Outcome

- The latest runnable Phase 3 improvement is empirical_background masking in Phase_3_work/instashap_project/masking.py.
- The root phase3-architecture.md is a conceptual interaction-aware note, not the same thing as the current runnable Phase_3_work branch.
- The root README still describes an earlier EBM-flavored plan, so it should not be treated as the final truth source for the current runnable system.
- The current saved Phase 3 artifact snapshot favors instashap_zero on accuracy (0.6843 vs 0.6774) and explanation MAE (0.3591 vs 0.3795).
- The current saved Phase 3 artifact snapshot slightly favors instashap_bg on Spearman rank alignment (0.5835 vs 0.5650) and coalition MSE (0.2016 vs 0.2021).
- The limitation is still a good research choice because it is specific, code-level, measurable, and easy to explain to reviewers.
