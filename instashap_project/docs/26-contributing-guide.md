# Contributing Guide

## Purpose

This guide explains how to make changes cleanly and safely.

## Before You Change Code

1. Read the relevant concept docs
2. Read the related file-reference docs
3. Run a fast-dev baseline

## Suggested Development Loop

1. Make a small code change
2. Run:

```bash
python main.py --dataset adult --model all --fast-dev-run
```

3. Check:

- terminal logs
- `results/run.log`
- changed tables and plots

## Documentation Rule

For any major new feature:

- update `README.md`
- update `docs/index.md`
- add or revise at least one relevant `docs/` page

## Good Contributions

- new datasets
- cleaner report pages
- stronger additive architectures
- more faithful reproduction settings
- tests
- checkpoint loading/saving

