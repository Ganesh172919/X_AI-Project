# Phase 3 Improvement Quickstart

This is the fastest document for understanding the Phase 3 improvement work in this folder.

## What changed

- Phase 3 replaces transformed-space zero masking with `empirical_background` masking.
- Hidden feature groups are filled from real transformed training rows instead of synthetic zeros.
- This makes coalition construction more realistic.

## What the current repo now includes

- New dataset-level masking diagnostics for Adult Income, Bike Sharing, and Covertype.
- New plots in `results/plots/diagnostics/`.
- New tables in `results/tables/`.
- New beginner, roadmap, dataset, and LLM/DL docs in `docs/`.
- A one-page summary in `reports/`.
- A dataset comparison notebook and a reusable prompt.

## Best current dataset for showing the masking improvement

- `adult_income` is the strongest showcase dataset in the new diagnostic ranking.
- Adult hidden categorical validity improves from 0.0000 to 1.0000.
- Adult hidden numeric exact-zero rate drops from 1.0000 to 0.0000.
