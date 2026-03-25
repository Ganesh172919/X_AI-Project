# Validation Checklist

## Purpose

Use this checklist whenever you modify the project and want confidence that it still works.

## Minimum Validation

- [ ] `pip install -r requirements.txt` succeeds
- [ ] `python main.py --dataset adult --model all --fast-dev-run` succeeds
- [ ] `results/run.log` is updated
- [ ] `adult_metrics.csv` is regenerated
- [ ] `instashap_reproducibility_report.pdf` is generated

## Stronger Validation

- [ ] `python main.py --dataset all --model all --fast-dev-run` succeeds
- [ ] all three dataset summary JSON files are regenerated
- [ ] plots are created for all datasets
- [ ] SHAP comparison tables exist for all datasets

## Reproduction Validation

- [ ] disable `--fast-dev-run`
- [ ] increase epochs where needed
- [ ] compare `*_paper_comparison.csv`
- [ ] inspect whether trends align with the paper even if numbers differ

## Documentation Validation

- [ ] new code is described in the docs
- [ ] new files are linked from the documentation hub if relevant
- [ ] README links remain correct

