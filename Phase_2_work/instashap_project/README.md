# InstaSHAP Reproducibility Project

This repository is a research-grade, modular reproduction of the ICLR 2025 paper **"InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly."** The project is designed for developers, data scientists, and researchers who want to understand the method end-to-end, reproduce the tabular experiments, inspect the generated artifacts, and extend the implementation for new datasets or new explanation methods.

The project focuses on three core goals:

1. Reproduce the paper's key tabular experiments on **Bike Sharing**, **Covertype**, and **Adult Income**.
2. Implement the full modeling stack: **black-box baselines**, **GAM-1 / GAM-2 additive models**, **masked surrogate models**, **SHAP baselines**, and **InstaSHAP**.
3. Generate reusable research artifacts: **tables**, **plots**, **JSON summaries**, **PDF reports**, and a **notebook**.

## What This Project Solves

In many real-world ML systems, we want explanations that are:

- Faithful to the predictive model
- Fast enough for repeated use
- Easy to inspect and reason about
- Structured enough to reveal interactions between features

Traditional SHAP explanations are useful, but they can be expensive and may blur together feature interactions. InstaSHAP addresses this by training an additive model under a masked objective so that SHAP-style feature attributions can be recovered in a **single forward pass**.

## Project Highlights

- Reproduces the paper's tabular framing of **synergistic** and **redundant** interactions
- Uses **`ucimlrepo`** to load the requested UCI datasets
- Supports **regression** and **classification**
- Implements **paper-aligned masked training** for InstaSHAP
- Produces clean experiment outputs in:
  - `results/tables/`
  - `results/plots/`
  - `results/artifacts/`
  - `reports/`
- Uses structured logs and deterministic seeding for reproducibility

## Quick Start

```bash
pip install -r requirements.txt
python main.py --dataset all --model all --fast-dev-run
```

For a single dataset:

```bash
python main.py --dataset bike --model instashap
python main.py --dataset covertype --model gam
python main.py --dataset adult --model shap
```

## Recommended Reading Order

If you want to understand the project deeply, follow this order:

1. [Documentation Hub](docs/index.md)
2. [Project Overview](docs/01-project-overview.md)
3. [Dataset Description](docs/03-dataset-description.md)
4. [Model Architecture](docs/05-model-architecture.md)
5. [InstaSHAP Methodology](docs/06-instashap-methodology.md)
6. [Training Process](docs/07-training-process.md)
7. [Usage Guide](docs/11-usage-guide.md)
8. [Extension Guide](docs/12-extension-guide.md)
9. [File Reference](docs/files/main.md) and the rest of `docs/files/`

## Repository Structure

```text
instashap_project/
|- README.md
|- config.yaml
|- main.py
|- requirements.txt
|- docs/
|- data/
|- models/
|- xai/
|- training/
|- experiments/
|- utils/
|- reports/
|- notebooks/
\- results/
```

## Main Outputs

After a run, the most important outputs are:

- `results/tables/*_metrics.csv`
- `results/tables/*_paper_comparison.csv`
- `results/tables/*_explanation_comparison.csv`
- `results/plots/<dataset>/...`
- `results/artifacts/<dataset>/<dataset>_summary.json`
- `reports/instashap_reproducibility_report.pdf`
- `reports/instashap_summary_1page.pdf`

## Reproducibility Notes

- Global seed is controlled in `config.yaml`
- Classification splits are stratified
- `--fast-dev-run` reduces dataset size and training epochs for quick validation
- The implementation uses structured logging and saves a run log at `results/run.log`
