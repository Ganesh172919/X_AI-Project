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

## Documentation Map

Core guides:

- [Documentation Hub](docs/index.md)
- [Project Overview](docs/01-project-overview.md)
- [Research Context](docs/02-research-context.md)
- [Dataset Description](docs/03-dataset-description.md)
- [EDA Guide](docs/04-eda-guide.md)
- [Model Architecture](docs/05-model-architecture.md)
- [InstaSHAP Methodology](docs/06-instashap-methodology.md)
- [Training Process](docs/07-training-process.md)
- [Evaluation Metrics](docs/08-evaluation-metrics.md)
- [Experiment Reproduction](docs/09-experiment-reproduction.md)
- [Installation and Setup](docs/10-installation-and-setup.md)
- [Usage Guide](docs/11-usage-guide.md)
- [Extension Guide](docs/12-extension-guide.md)
- [Results and Artifacts](docs/13-results-and-artifacts.md)
- [Troubleshooting](docs/14-troubleshooting.md)
- [Glossary](docs/15-glossary.md)
- [Project Structure](docs/16-project-structure.md)
- [Development Workflow](docs/17-development-workflow.md)
- [Research Assumptions](docs/18-research-assumptions.md)
- [Configuration Reference](docs/19-configuration-reference.md)
- [API Walkthrough](docs/20-api-walkthrough.md)
- [Experiment Cookbook](docs/21-experiment-cookbook.md)
- [Report Reading Guide](docs/22-report-reading-guide.md)
- [Results Schema](docs/23-results-schema.md)
- [FAQ](docs/24-faq.md)
- [Limitations and Gaps](docs/25-limitations-and-gaps.md)
- [Contributing Guide](docs/26-contributing-guide.md)
- [Validation Checklist](docs/27-validation-checklist.md)
- [System Flow Diagrams](docs/28-system-flow-diagrams.md)
- [Data Flow Deep Dive](docs/29-data-flow-deep-dive.md)
- [Mask Sampling Deep Dive](docs/30-mask-sampling-deep-dive.md)
- [GAM Components Deep Dive](docs/31-gam-components-deep-dive.md)
- [SHAP Aggregation Deep Dive](docs/32-shap-aggregation-deep-dive.md)
- [Classification vs Regression](docs/33-classification-vs-regression.md)
- [Performance Optimization](docs/34-performance-optimization.md)
- [TensorBoard and Logs](docs/35-tensorboard-and-logs.md)
- [Reproducibility Playbook](docs/36-reproducibility-playbook.md)
- [Code Reading Guide](docs/37-code-reading-guide.md)
- [Bike Experiment Trace](docs/38-experiment-trace-bike.md)
- [Covertype Experiment Trace](docs/39-experiment-trace-covertype.md)
- [Adult Experiment Trace](docs/40-experiment-trace-adult.md)
- [Testing Strategy](docs/41-testing-strategy.md)
- [Roadmap](docs/42-roadmap.md)

File-by-file reference:

- [Main Entry Point](docs/files/main.md)
- [Configuration](docs/files/config.md)
- [Requirements](docs/files/requirements.md)
- [Data Loaders](docs/files/data-loaders.md)
- [Preprocessing](docs/files/data-preprocessing.md)
- [Black-Box Models](docs/files/models-blackbox-model.md)
- [GAM Models](docs/files/models-gam.md)
- [InstaSHAP Model](docs/files/models-instashap.md)
- [SHAP Wrapper](docs/files/xai-shap-wrapper.md)
- [InstaSHAP Explainer](docs/files/xai-instashap-explainer.md)
- [Training Module](docs/files/training-train.md)
- [Evaluation Module](docs/files/training-evaluate.md)
- [Experiment Orchestrator](docs/files/experiments-common.md)
- [Bike Experiment](docs/files/experiments-bike-sharing.md)
- [Covertype Experiment](docs/files/experiments-covertype.md)
- [Adult Experiment](docs/files/experiments-adult-income.md)
- [Report Generator](docs/files/reports-generate-report.md)
- [1-Page Summary Generator](docs/files/reports-summary-1page.md)
- [Reproducibility Utilities](docs/files/utils-reproducibility.md)
- [Metrics Utilities](docs/files/utils-metrics.md)
- [Visualization Utilities](docs/files/utils-visualization.md)
- [Logging Utilities](docs/files/utils-logging-utils.md)
- [Notebook Guide](docs/files/notebook-instashap-analysis.md)
- [Package `__init__` Files](docs/files/package-init-files.md)
