# Global Documentation Hub

This folder is the complete understanding layer for the repository. It was generated from the current workspace so the project can be explained, presented, and defended without mixing older planning notes with the latest runnable code and metrics.

## What This Folder Solves

- The latest runnable Phase 3 improvement is empirical_background masking in Phase_3_work/instashap_project/masking.py.
- The root phase3-architecture.md is a conceptual interaction-aware note, not the same thing as the current runnable Phase_3_work branch.
- The root README still describes an earlier EBM-flavored plan, so it should not be treated as the final truth source for the current runnable system.
- The current saved Phase 3 artifact snapshot favors instashap_zero on accuracy (0.6843 vs 0.6774) and explanation MAE (0.3591 vs 0.3795).
- The current saved Phase 3 artifact snapshot slightly favors instashap_bg on Spearman rank alignment (0.5835 vs 0.5650) and coalition MSE (0.2016 vs 0.2021).
- The limitation is still a good research choice because it is specific, code-level, measurable, and easy to explain to reviewers.

## Best Reading Order

- `01_PROJECT_UNDERSTANDING_GUIDE.md` for the fastest full-project orientation.
- `02_REPOSITORY_BUILD_MAP.md` for the folder and execution map.
- `03_PHASE_PROGRESS.md` for the story from Phase 1 to Phase 3.
- `04_PHASE2_REPLICATION_REFERENCE.md` for the replication baseline.
- `05_PHASE3_EXTENSION_REFERENCE.md` for the current runnable extension.
- `06_LATEST_PHASE3_IMPROVEMENT_ANALYSIS.md` for the exact Phase 3 question and answer.
- `07_DATASET_STRATEGY.md` for dataset recommendations and why Covertype is the best current choice.
- `08_PRESENTATION_PLAYBOOK.md` for speaking notes and viva support.
- `09_BUILD_RUN_REPRODUCIBILITY.md` for commands and operational guidance.
- `10_APPENDIX_FILE_CATALOG_A.md` to `12_APPENDIX_FILE_CATALOG_C.md` for deep file-by-file traceability.
- `13_APPENDIX_RESULTS_AND_METRICS.md` for the current saved metric tables.
- `14_GLOSSARY_FAQ.md` for short definitions and ready-to-use answers.

## Current Phase 3 Snapshot

| Metric | instashap_zero | instashap_bg | Current winner |
| --- | --- | --- | --- |
| Accuracy mean | 0.6843 | 0.6774 | instashap_zero |
| Log loss mean | 0.7815 | 0.8168 | instashap_zero |
| Explanation MAE mean | 0.3591 | 0.3795 | instashap_zero |
| Explanation Spearman mean | 0.5650 | 0.5835 | instashap_bg |
| Coalition MSE mean | 0.2021 | 0.2016 | instashap_bg |
| Explain time mean seconds | 0.0100 | 0.0121 | instashap_zero |

## Strongest Safe Claim

The project successfully turns a real limitation in transformed-space zero masking into runnable code and measurable evidence. The current results are mixed rather than universally positive, which makes the final Phase 3 claim narrow, honest, and academically defensible.

## New extension docs

- `18_BEGINNER_QUICK_UNDERSTANDING.md` for a beginner-friendly explanation.
- `19_PHASE3_IMPROVEMENT_ROADMAP.md` for what to improve, how to improve it, and what to expect.
- `20_COVERTYPE_FAILURE_AND_ADULT_SHOWCASE.md` for why Covertype stayed mixed and why Adult is a better masking showcase.
- `21_LLM_AND_DL_APPLICABILITY.md` for whether InstaSHAP can be used with LLMs and deep learning models.
- `22_CONTINUOUS_DATASET_IMPROVEMENT_PLAN.md` for the multi-dataset continuation path.
- `23_PHASE3_DATASET_EXTENSION_PROMPT.md` for a reusable extension prompt.
- `24_ADULT_DATASET_EXTENSION_SUMMARY.md` for the new Adult diagnostic assets.
- `25_PHASE3_IMPROVEMENT_PRESENTATION_MASTER.md` for the long-form Phase 3 presentation script.
- `26_ONE_PAGE_SUMMARY.md` and `26_ONE_PAGE_SUMMARY.pdf` for a short summary of what has been done.
