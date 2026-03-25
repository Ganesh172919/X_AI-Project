# Report Reading Guide

## Purpose

The project generates PDFs and several tables. This document explains how to read them in a useful order.

## Main Report Files

- `reports/instashap_reproducibility_report.pdf`
- `reports/instashap_summary_1page.pdf`

## Recommended Reading Order

### 1. Start with the one-page summary

Use it to understand:

- what datasets were run
- what the best quick outcomes were
- whether the run was a smoke test or a stronger experiment

### 2. Read the full report overview pages

These pages explain:

- the dataset
- the task type
- which methods were compared

### 3. Read the metrics page for each dataset

Ask:

- Which model has the best predictive metric?
- How large is the gap between black-box and additive models?

### 4. Read the paper-comparison page

Ask:

- Are reproduced values close to the reported paper values?
- If not, is the run using fast-dev settings?

### 5. Read the explanation-comparison page

Ask:

- How close are SHAP and InstaSHAP explanations numerically?
- Which datasets have more divergence?

### 6. Review the plots

Important plot types:

- feature importance
- shape functions
- interaction heatmaps
- training curves

## What to Look for per Dataset

### Bike

- whether `hour x workingday` is clearly visible
- whether GAM-2 improves over GAM-1

### Covertype

- whether the interaction structure improves accuracy
- whether elevation and soil grouping show correlated structure

### Adult

- whether the 1D additive structure is already competitive
- whether InstaSHAP remains close to GAM-1 and the black-box baseline

