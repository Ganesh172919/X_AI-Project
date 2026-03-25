# Performance Optimization Guide

## Purpose

This document explains how to make the project faster or more scalable.

## Main Runtime Costs

The most expensive parts are usually:

- SHAP baseline explanation
- Covertype preprocessing and training at larger scale
- repeated report generation if many plots are created

## Quick Wins

### Use fast-dev mode

```bash
python main.py --dataset all --model all --fast-dev-run
```

### Reduce SHAP work

Tune in `config.yaml`:

- `shap_background_size`
- `shap_eval_samples`
- `shap_max_evals`

### Run one dataset at a time

```bash
python main.py --dataset adult --model all
```

## Model-Side Optimizations

- lower hidden dimensions
- fewer epochs
- larger batch sizes if memory allows
- disable stages you do not need

## Infrastructure-Side Optimizations

- use GPU if available
- keep results from prior runs instead of rerunning reports unnecessarily
- inspect `results/run.log` to identify the slowest stage

## Most Important Tradeoff

The biggest speed-vs-fidelity tradeoff is SHAP. InstaSHAP is fast at inference specifically because training absorbs the complexity earlier.

