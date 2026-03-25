# TensorBoard and Logs

## Purpose

This document explains how to inspect runtime logs and TensorBoard event files.

## Structured Run Log

Main log file:

- `results/run.log`

Log format:

```text
time | level | event | key=value ...
```

Useful events:

- `run.start`
- `experiment.start`
- `stage.start`
- `stage.complete`
- `experiment.complete`
- `report.ready`

## TensorBoard Logs

Training loops optionally write TensorBoard event files under:

- `results/artifacts/<dataset>/blackbox_logs/`
- `results/artifacts/<dataset>/gam1_logs/`
- `results/artifacts/<dataset>/gam2_logs/`
- `results/artifacts/<dataset>/surrogate_logs/`
- `results/artifacts/<dataset>/instashap_logs/`

## How to Launch TensorBoard

From the project directory:

```bash
tensorboard --logdir results/artifacts
```

## What to Look At

- train loss curves
- validation loss curves
- relative convergence of black-box vs additive models
- whether surrogate and InstaSHAP losses stabilize

## Why It Matters

The saved plots are useful summaries, but TensorBoard gives a more granular view during debugging and tuning.

