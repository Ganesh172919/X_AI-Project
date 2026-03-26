# X_AI-Project — InstaSHAP Reproducibility Suite

**Paper Details:**
* **Title:** InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly
* **Authors:** James Enouen, Yan Liu
* **Publication Venue:** Accepted at the International Conference on Learning Representations (ICLR) 2025
* **Official Publication Link:** https://openreview.net/forum?id=ky7vVlBQBY

> **Reproducing and extending the ICLR 2025 paper:**  
> *"InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#prerequisites)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c.svg)](#prerequisites)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

This project is a **research-grade, modular reproduction** of the InstaSHAP method for obtaining instant Shapley value explanations from additive neural models. Instead of running expensive permutation-based SHAP for every prediction, InstaSHAP trains an additive model under a masked objective so that **SHAP-style feature attributions are recovered in a single forward pass**.

### Key Capabilities

- **Three UCI Benchmark Datasets** — Bike Sharing (regression), Covertype (classification), Adult Income (classification)
- **Full Modeling Stack** — Black-box MLP baselines, GAM-1/GAM-2 additive models, masked surrogate models, and InstaSHAP
- **Explainability Pipeline** — Permutation SHAP baseline + InstaSHAP single-pass explanations with alignment comparison
- **Research Artifacts** — Metrics tables, training curves, shape function plots, interaction heatmaps, PDF reports
- **Reproducibility** — Deterministic seeding, structured logging, configurable hyperparameters via YAML

---

## Project Structure

```
X_AI-Project/
├── README.md                          ← You are here
└── instashap_project/
    ├── main.py                        ← CLI entry point
    ├── config.yaml                    ← Hyperparameters & dataset settings
    ├── requirements.txt               ← Python dependencies
    ├── data/
    │   ├── loaders.py                 ← UCI dataset loaders (Bike, Covertype, Adult)
    │   └── preprocessing.py           ← TabularPreprocessor (scaling, one-hot, feature groups)
    ├── models/
    │   ├── blackbox_model.py          ← TabularMLP, MaskedSurrogateMLP, RandomForestBlackBox
    │   ├── gam.py                     ← GAMModel with univariate + pairwise components
    │   └── instashap.py               ← InstaSHAPModel (masked additive, Eq. 20)
    ├── training/
    │   ├── train.py                   ← Trainers: blackbox, surrogate, GAM, InstaSHAP
    │   └── evaluate.py                ← Prediction & evaluation helpers
    ├── xai/
    │   ├── shap_wrapper.py            ← Permutation SHAP with feature-group aggregation
    │   └── instashap_explainer.py     ← Single-pass InstaSHAP explainer
    ├── experiments/
    │   ├── common.py                  ← Full experiment orchestrator
    │   ├── bike_sharing.py            ← Bike Sharing runner (synergy experiment)
    │   ├── covertype.py               ← Covertype runner (redundancy experiment)
    │   └── adult_income.py            ← Adult Income runner (supplementary)
    ├── utils/
    │   ├── metrics.py                 ← Regression/classification metrics, explanation error
    │   ├── visualization.py           ← Training curves, shape functions, heatmaps, bar charts
    │   ├── reproducibility.py         ← Seed control, device resolution, JSON I/O
    │   └── logging_utils.py           ← Structured logging configuration
    ├── reports/
    │   ├── generate_report.py         ← Multi-page PDF reproducibility report
    │   └── summary_1page.py           ← One-page PDF summary
    ├── notebooks/
    │   ├── instashap_complete_analysis.ipynb  ← Complete end-to-end notebook
    │   └── generate_notebook.py       ← Script to regenerate the notebook
    ├── results/                       ← Generated outputs (tables, plots, artifacts)
    └── docs/                          ← 42 documentation files covering every aspect
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd instashap_project
pip install -r requirements.txt
```

### Run All Experiments (Fast Mode)

```bash
python main.py --dataset all --model all --fast-dev-run
```

### Run a Single Dataset

```bash
python main.py --dataset bike --model instashap
python main.py --dataset covertype --model gam
python main.py --dataset adult --model shap
```

### Use the Notebook

Open `instashap_project/notebooks/instashap_complete_analysis.ipynb` in Jupyter and run all cells. The notebook replicates the entire pipeline interactively with visualizations.

---

## How It Works

### Pipeline Overview

```
┌──────────────┐    ┌────────────────┐    ┌──────────────────┐
│  UCI Dataset │───▶│ Preprocessing  │───▶│ Train/Val/Test   │
│  (ucimlrepo) │    │ (scale + OHE)  │    │ Split            │
└──────────────┘    └────────────────┘    └──────┬───────────┘
                                                  │
                    ┌─────────────────────────────┼──────────────────────┐
                    ▼                             ▼                      ▼
           ┌───────────────┐            ┌─────────────────┐    ┌──────────────┐
           │  Black-Box    │            │  GAM-1 / GAM-2  │    │   Masked     │
           │  MLP Baseline │            │  Additive Model │    │   Surrogate  │
           └───────┬───────┘            └────────┬────────┘    └──────┬───────┘
                   │                             │                     │
                   ▼                             ▼                     ▼
           ┌───────────────┐            ┌─────────────────┐    ┌──────────────┐
           │  Permutation  │            │  Shape Function  │    │  InstaSHAP   │
           │  SHAP Values  │            │  Visualizations  │    │  Model       │
           └───────┬───────┘            └─────────────────┘    └──────┬───────┘
                   │                                                   │
                   └────────────────┬──────────────────────────────────┘
                                    ▼
                           ┌────────────────┐
                           │  Compare SHAP  │
                           │  vs InstaSHAP  │
                           └────────┬───────┘
                                    ▼
                           ┌────────────────┐
                           │  Tables, Plots │
                           │  PDF Reports   │
                           └────────────────┘
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Black-Box** | Standard MLP trained on raw labels — the model we want to explain |
| **GAM-1** | Additive model with one subnetwork per feature (no interactions) |
| **GAM-2** | GAM-1 plus pairwise interaction components (e.g., hour × workingday) |
| **Masked Surrogate** | Approximates `f(x; S)` — the black-box output under feature masking |
| **InstaSHAP** | Additive model trained against the surrogate's masked outputs (Eq. 20) |
| **SHAP Baseline** | Permutation SHAP computed on the black-box for ground-truth comparison |

---

## Datasets

| Dataset | Task | Features | Interaction Highlight | UCI ID |
|---------|------|----------|-----------------------|--------|
| **Bike Sharing** | Regression | 13 (5 numeric, 8 categorical) | hour × workingday (synergistic) | 275 |
| **Covertype** | Classification (7 classes) | 11 (10 numeric, 1 categorical) | elevation × soil_climate_zone (redundant) | 31 |
| **Adult Income** | Binary Classification | 13 (5 numeric, 8 categorical) | None (supplementary benchmark) | 2 |

---

## Configuration

All hyperparameters are controlled via `config.yaml`:

- **Global**: seed, device, SHAP settings, fast-dev-run toggle
- **Training**: per-model settings (hidden dims, dropout, learning rate, epochs, patience)
- **Datasets**: per-dataset settings (max rows, split ratios, interaction pairs, SHAP sample size)

---

## Outputs

After running experiments, find results in:

| Path | Content |
|------|---------|
| `results/tables/*_metrics.csv` | Model performance metrics |
| `results/tables/*_paper_comparison.csv` | Reproduced vs paper-reported values |
| `results/tables/*_explanation_comparison.csv` | SHAP vs InstaSHAP fidelity |
| `results/plots/<dataset>/` | Training curves, shape functions, heatmaps |
| `results/artifacts/<dataset>/` | JSON summaries, TensorBoard logs |
| `reports/instashap_reproducibility_report.pdf` | Full multi-page PDF report |
| `reports/instashap_summary_1page.pdf` | One-page summary PDF |

---

## Documentation

The `docs/` directory contains **42 detailed guides** covering every aspect of the project — from research context and methodology to experiment traces and troubleshooting. Start with [docs/index.md](instashap_project/docs/index.md).

---

## Reproducibility

- Global seed (default: 42) controls Python, NumPy, and PyTorch randomness
- Classification splits use stratification
- `--fast-dev-run` reduces dataset size and epochs for quick validation
- Structured logging saves a run log at `results/run.log`
- CuDNN is set to deterministic mode

---

## Dependencies

Core: `numpy`, `pandas`, `torch`, `scikit-learn`, `shap`, `ucimlrepo`, `matplotlib`, `seaborn`, `PyYAML`, `tqdm`, `nbformat`, `tensorboard`, `jupyter`

See [requirements.txt](instashap_project/requirements.txt) for version pins.
