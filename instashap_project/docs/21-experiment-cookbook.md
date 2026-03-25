# Experiment Cookbook

## Purpose

This document gives practical recipes for common experiment goals.

## Recipe 1: Validate the whole stack quickly

```bash
python main.py --dataset all --model all --fast-dev-run
```

Use this when:

- you changed code
- you want to verify artifacts still build
- you need a quick end-to-end sanity check

## Recipe 2: Focus only on Bike Sharing interactions

```bash
python main.py --dataset bike --model gam
```

Best for:

- studying additive shape functions
- inspecting the `hour x workingday` interaction

## Recipe 3: Compare SHAP and InstaSHAP on Adult

```bash
python main.py --dataset adult --model instashap --fast-dev-run
```

Best for:

- checking explanation agreement
- inspecting `adult_explanation_comparison.csv`

## Recipe 4: Re-run only Covertype after config tuning

```bash
python main.py --dataset covertype --model all
```

Best for:

- tuning interaction quality
- improving accuracy before comparing to the paper

## Recipe 5: Iterate in the notebook

Open:

- `notebooks/instashap_analysis.ipynb`

Best for:

- inspecting data interactively
- trying alternative preprocessing ideas

## Recipe 6: Generate reports from existing outputs

```bash
python reports/generate_report.py
python reports/summary_1page.py
```

Best for:

- refreshing PDFs after a completed run
- sharing artifacts without rerunning training

