# Build, Run, And Reproducibility

This document is the operational guide for running the repository and checking the evidence paths.

## Phase 3 Commands

- pip install -r Phase_3_work/requirements.txt
- python Phase_3_work/main.py --dataset covertype --variant compare
- python Phase_3_work/main.py --dataset covertype --variant baseline
- python Phase_3_work/main.py --dataset covertype --variant improved
- python Phase_3_work/main.py --report-only
- python -m unittest discover -s Phase_3_work/tests -v

## Phase 2 Commands

- pip install -r Phase_2_work/instashap_project/requirements.txt
- python Phase_2_work/instashap_project/main.py --dataset all --model all
- python Phase_2_work/instashap_project/main.py --dataset covertype --model instashap
- python Phase_2_work/instashap_project/main.py --dataset bike --model all --fast-dev-run

## Where To Look For Outputs

- Phase_3_work/results/tables for metric summaries.
- Phase_3_work/results/plots for presentation plots.
- Phase_3_work/results/artifacts for JSON summaries and seed artifacts.
- Phase_3_work/reports for Markdown and PDF reporting outputs.
- Phase_2_work/instashap_project/results for replication artifacts.
- instashap_presentation for HTML and PPTX presentation assets.

## Troubleshooting Rules

- If tables and markdown disagree, trust the current CSV tables.
- If you need the latest runnable Phase 3 branch, stay inside Phase_3_work.
- If the masking claim is questioned, open tests/test_masking.py and masking.py together.
- If the presentation numbers are stale, update the presentation narrative, not the evidence.

## Reproducibility Notes

- Phase 3 uses multiple seeds from config.yaml.
- The repo stores structured artifacts so claims can be audited.
- The global docs here were generated from the actual current workspace state.
