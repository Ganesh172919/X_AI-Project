# 📁 01 — Project Setup Prompts

> **Total Prompts in this file:** 30
> **Category:** Requirements, Architecture, Tech Stack, Environment Setup, Folder Structure


### Prompt 1: Clarifying Project Requirements from a Research Paper

**Goal:**
Translate a research paper into a concrete set of functional requirements for a reproducibility project.

**Prompt:**
```
I need to reproduce the results of a research paper titled "[Paper Title]" published at [Venue].
Here is a summary of the paper:
- It proposes [method name] which [brief description of contribution].
- The experiments use [datasets] and compare against [baselines].
- Key metrics reported are [list metrics].

Help me break this down into:
1. A list of functional requirements (what the code must do).
2. Non-functional requirements (reproducibility, performance, documentation).
3. Minimum viable deliverables vs. stretch goals.
4. Any ambiguities in the paper that I should resolve before coding.

Please be specific and actionable.
```

**Why this is useful:**
Teaches you to decompose a vague research goal into structured engineering requirements — a critical skill for any project.

**Example Use Case:**
Before starting the InstaSHAP project, you'd paste the ICLR 2025 paper summary and get a clear requirements checklist covering data loading, model training, SHAP computation, and report generation.

---

### Prompt 2: Comparing Frameworks for Your ML Project

**Goal:**
Make an informed decision about which ML framework to use (e.g., PyTorch vs. TensorFlow vs. JAX).

**Prompt:**
```
I'm building a research project that requires:
- Custom neural network architectures (MLPs with additive structure)
- Masked forward passes where subsets of features are zeroed out
- Integration with the SHAP library for permutation-based explanations
- GPU acceleration for training but the project should also run on CPU
- Easy debugging and prototyping

Compare PyTorch, TensorFlow, and JAX for this use case. For each, explain:
1. Ease of implementing custom forward passes
2. Compatibility with SHAP/explainability libraries
3. Debugging experience
4. Community support and documentation quality
5. Your recommendation and why
```

**Why this is useful:**
Prevents you from picking a framework based on popularity alone — forces you to evaluate practical fit.

**Example Use Case:**
For InstaSHAP, PyTorch was chosen because its eager execution makes masked forward passes and custom additive architectures straightforward to debug.

---

### Prompt 3: Designing the Module Architecture

**Goal:**
Plan a clean, modular project architecture before writing any code.

**Prompt:**
```
I'm building a Python project with the following components:
- Data loading and preprocessing (3 UCI datasets, mixed numeric/categorical features)
- Multiple model types: black-box MLP, GAM (additive model), masked surrogate, InstaSHAP
- Training pipelines with early stopping, logging, and checkpointing
- Explainability: permutation SHAP and single-pass InstaSHAP explanations
- Visualization: training curves, shape function plots, heatmaps
- Report generation: multi-page PDF reports with tables and figures
- Configuration management via YAML
- Utility modules: metrics, reproducibility (seeding), logging

Propose a folder/module structure that:
1. Separates concerns cleanly (data, models, training, evaluation, visualization)
2. Avoids circular imports
3. Makes it easy to add new datasets or models later
4. Follows Python packaging best practices

Show the directory tree and briefly explain each module's responsibility.
```

**Why this is useful:**
Designing architecture up front prevents spaghetti code and makes the project extensible.

---

### Prompt 4: Setting Up a Virtual Environment with Exact Dependencies

**Goal:**
Create a reproducible Python environment with pinned dependencies.

**Prompt:**
```
I'm starting a Python 3.10+ ML project that needs these core libraries:
numpy, pandas, torch, scikit-learn, shap, ucimlrepo, matplotlib, seaborn, PyYAML, tqdm, nbformat, tensorboard, jupyter

Help me:
1. Write a requirements.txt with minimum version pins (e.g., numpy>=1.26)
2. Explain why pinning minimum versions (not exact) is appropriate for a research project
3. Show the exact commands to create a venv, activate it, and install dependencies on both Windows and Linux
4. Suggest how to handle the PyTorch CPU vs GPU installation difference
5. Explain when I should switch to a pyproject.toml or setup.cfg instead
```

**Why this is useful:**
Dependency management is the #1 cause of "it works on my machine" failures. This teaches reproducibility from day one.

---

### Prompt 5: Understanding YAML-Based Configuration

**Goal:**
Design a centralized YAML config that controls all hyperparameters and experiment settings.

**Prompt:**
```
I want to manage all my project settings in a single YAML file. My project needs:
- Global settings: random seed, device (auto/cpu/cuda), output directories
- Per-model training settings: hidden_dims, dropout, lr, weight_decay, batch_size, epochs, patience
- Per-dataset settings: max_rows, train/val/test split ratios, interaction pairs, SHAP sample size

Design a config.yaml with:
1. Sensible default values for an ML research project
2. Clear grouping (global, training, datasets)
3. Comments explaining each parameter
4. Nested structure for per-model and per-dataset blocks

Then show me Python code to load this config and access nested values safely (handling missing keys).
```

**Why this is useful:**
Centralizing configuration prevents hard-coded magic numbers scattered across files and makes experiments reproducible.

---

### Prompt 6: Implementing a CLI Entry Point with argparse

**Goal:**
Build a flexible command-line interface for running experiments.

**Prompt:**
```
I need a main.py CLI entry point for my ML project. It should support:
- --dataset: choose which dataset to run (bike, covertype, adult, or all)
- --model: choose which model pipeline (blackbox, gam, shap, instashap, or all)
- --config: path to YAML config file (default: config.yaml in the same directory)
- --fast-dev-run: flag to use smaller data subsets and fewer epochs for testing
- --skip-report: skip PDF report generation
- --log-level: DEBUG/INFO/WARNING/ERROR

Write the argparse setup. Then explain:
1. How choices= validates input automatically
2. When to use action="store_true" vs type=bool
3. How to make --config default to a path relative to the script, not the working directory
4. Best practices for organizing the main() function flow
```

**Why this is useful:**
A well-designed CLI makes your project usable by others and is essential for scripted experiment pipelines.

---

### Prompt 7: Setting Up Structured Logging

**Goal:**
Replace print statements with proper structured logging.

**Prompt:**
```
My ML project currently uses print() statements for status messages. I want to switch to Python's logging module. Requirements:
- Log to both console (stdout) and a file (results/run.log)
- Console should show INFO and above; file should capture DEBUG and above
- Each log entry should include timestamp, level, module name, and message
- I want a helper function to format structured events like:
  format_log_event("train.epoch", epoch=5, loss=0.42, val_loss=0.45)
  → "[train.epoch] epoch=5 | loss=0.42 | val_loss=0.45"

Show me:
1. A configure_logging() function that sets up handlers and formatters
2. The format_log_event() helper
3. How to use them in training loops without cluttering the code
4. Explain why print() is problematic for production/research code
```

**Why this is useful:**
Structured logging makes debugging easier, creates audit trails, and is a professional development practice.

---

### Prompt 8: Understanding .gitignore for ML Projects

**Goal:**
Configure version control to exclude generated artifacts properly.

**Prompt:**
```
I'm setting up a Git repository for a Python ML project. The project generates:
- __pycache__/ and .pyc files
- results/ folder with CSV tables, PNG plots, JSON artifacts, and PDF reports
- Jupyter notebook checkpoints (.ipynb_checkpoints/)
- Virtual environment folder (venv/ or .venv/)
- IDE settings (.vscode/, .idea/)
- Trained model weights (.pt, .pth files)
- TensorBoard log directories

Write a comprehensive .gitignore file and explain:
1. Which of these should ALWAYS be gitignored and why
2. Should I track result CSVs and report PDFs? Pros and cons.
3. Should I track config.yaml? (Yes, explain why)
4. How to use .gitkeep to preserve empty directories in Git
5. Any ML-specific gitignore patterns I might be missing
```

**Why this is useful:**
Avoids bloating the repository, prevents committing sensitive data, and keeps history clean.

---

### Prompt 9: Planning the Data Pipeline

**Goal:**
Design the data loading and preprocessing architecture before implementation.

**Prompt:**
```
My project loads datasets from the UCI ML Repository using the `ucimlrepo` library.
I need to support 3 datasets:
1. Bike Sharing (regression, 13 features, mix of numeric & categorical)
2. Covertype (7-class classification, 11 features)
3. Adult Income (binary classification, 13 features, mix of numeric & categorical)

Design the data pipeline:
1. How should I structure the loader functions? One per dataset or a generic loader?
2. What preprocessing steps are needed? (scaling, one-hot encoding, handling missing values)
3. How to split into train/val/test with configurable ratios and stratification for classification?
4. How to handle the feature groups concept (grouping one-hot columns back to their original feature)?
5. What class or function interfaces would make this extensible for new datasets?

Focus on architecture decisions, not implementation details.
```

**Why this is useful:**
Planning data pipelines prevents rework. Bad data preprocessing is the most common source of bugs in ML projects.

---

### Prompt 10: Setting Up Reproducibility Controls

**Goal:**
Ensure experiments are deterministic and reproducible.

**Prompt:**
```
My PyTorch ML project needs to be fully reproducible. Explain and show me how to:
1. Set seeds for Python's random, NumPy, and PyTorch (CPU + CUDA)
2. Configure CuDNN for deterministic behavior (torch.backends.cudnn.deterministic)
3. Handle the performance trade-off of deterministic mode
4. Auto-detect whether to use CPU or CUDA
5. Save and restore random states for interrupted experiments
6. Use stratified splits for classification datasets

Write a reproducibility.py utility module with:
- set_global_seed(seed: int)
- resolve_device() -> torch.device
- A docstring explaining what each function does and why
```

**Why this is useful:**
Reproducibility is a cornerstone of scientific computing. Reviewers and collaborators must be able to replicate your results exactly.

---

### Prompt 11: Understanding the Research Paper's Mathematical Framework

**Goal:**
Translate the mathematical notation from the paper into code design decisions.

**Prompt:**
```
The InstaSHAP paper defines the following key equations:
- Equation 1: Standard Shapley value definition with coalitional game
- Equation 20: The core InstaSHAP training objective (masked additive model approximation)
- The GAM decomposition: f(x) = Σ f_j(x_j) + Σ f_{jk}(x_j, x_k)

Help me understand:
1. What does each equation mean intuitively (not just mathematically)?
2. How does each equation translate to a neural network architecture component?
3. What are the inputs, outputs, and training targets for each model?
4. What is the masked objective and why is it needed?
5. How does the additive structure enable single-pass SHAP value extraction?

Explain as if I understand Python and basic ML but need help with the XAI theory.
```

**Why this is useful:**
You can't implement what you don't understand. This bridges the gap between math and code.

---

### Prompt 12: Choosing Between MLP Architectures

**Goal:**
Understand how different hidden layer configurations affect model capacity.

**Prompt:**
```
I'm building multiple MLP models for my project:
1. Black-box MLP: hidden_dims=[256, 128], dropout=0.10
2. GAM subnetworks: hidden_dims=[96, 64], dropout=0.05
3. Surrogate MLP: hidden_dims=[256, 128], dropout=0.10
4. InstaSHAP subnetworks: hidden_dims=[96, 64], dropout=0.05

Explain:
1. Why are the black-box and surrogate larger than the GAM/InstaSHAP subnetworks?
2. How does the number of subnetworks in a GAM affect total parameter count?
3. Why is dropout lower for GAM/InstaSHAP? (Hint: additive structure already regularizes)
4. How would I decide to increase or decrease these dimensions for a new dataset?
5. What are the signs that my model is too large or too small?
```

**Why this is useful:**
Teaches architectural intuition rather than blindly copying hyperparameters.

---

### Prompt 13: Creating a Project README from Scratch

**Goal:**
Write a professional README that serves as the project's landing page.

**Prompt:**
```
Help me write a comprehensive README.md for my research project. The project:
- Reproduces results from [paper name] at [venue]
- Uses [tech stack: Python, PyTorch, scikit-learn, SHAP, etc.]
- Supports [N] datasets and [M] model types
- Has a CLI interface with multiple run modes
- Generates tables, plots, and PDF reports

The README should include:
1. Title with badges (Python version, license, framework)
2. One-paragraph overview
3. Key capabilities (bullet list)
4. Project structure (directory tree with descriptions)
5. Quick start (prerequisites, install, run commands)
6. How it works (pipeline diagram in ASCII or mermaid)
7. Configuration, outputs, and documentation sections
8. Reproducibility guarantees

Make it look professional and be specific — not generic placeholder text.
```

**Why this is useful:**
The README is the first thing reviewers, collaborators, and future-you will see. A great README saves hours of confusion.

---

### Prompt 14: Setting Up the Testing Framework

**Goal:**
Choose and configure a testing strategy before writing tests.

**Prompt:**
```
I'm planning the testing strategy for my Python ML project. The project has:
- Data loaders that fetch from UCI and preprocess
- Multiple neural network model classes (MLP, GAM, InstaSHAP)
- Training loops with early stopping
- SHAP computation wrappers
- Visualization functions that produce matplotlib figures
- PDF report generators

Help me plan:
1. Which testing framework to use (pytest, unittest) and why
2. What kinds of tests I need (unit, integration, smoke tests)
3. How to test ML models without training them for minutes (fixtures, small data)
4. How to test visualization functions (snapshot testing? checking figure properties?)
5. How to organize tests/ directory structure mirroring src/
6. What to test first (highest risk areas)
7. A sample conftest.py with useful fixtures

Don't write the tests yet — just the plan and structure.
```

**Why this is useful:**
Planning the test strategy prevents ad-hoc, incomplete testing later.

---

### Prompt 15: Understanding PyTorch's nn.Module Pattern

**Goal:**
Learn the PyTorch model-building pattern used throughout the project.

**Prompt:**
```
I'm new to PyTorch and need to understand the nn.Module pattern. Using a simple MLP as an example, explain:

1. Why every model class inherits from nn.Module
2. What __init__ should contain (layers, not data)
3. What forward() does and why it's called automatically
4. How nn.Sequential works vs. defining layers individually
5. Where activation functions go (in __init__ or forward?)
6. How to add dropout and what it does during training vs. eval
7. The difference between model.train() and model.eval()
8. How to move a model to GPU with .to(device)
9. How parameters() is used by the optimizer

Show a concrete example: an MLP with configurable hidden_dims, dropout, and output size.
Please explain each line, not just show the code.
```

**Why this is useful:**
nn.Module is the foundation of every PyTorch project. Deep understanding here prevents dozens of bugs later.

---

### Prompt 16: Setting Up TensorBoard Logging

**Goal:**
Integrate TensorBoard for training visualization.

**Prompt:**
```
I'm training multiple models (MLP, GAM, InstaSHAP) and want to use TensorBoard to visualize:
- Training loss per epoch
- Validation loss per epoch
- Learning rate schedule
- Model-specific metrics (R², accuracy, F1)

Show me:
1. How to set up a SummaryWriter with organized log directories (e.g., runs/{dataset}/{model_name})
2. How to log scalar values during the training loop
3. How to compare multiple runs in the TensorBoard UI
4. How to properly close the writer
5. The command to launch TensorBoard
6. Best practices for organizing runs so they don't pollute each other

Explain the trade-off between logging everything vs. logging selectively.
```

**Why this is useful:**
TensorBoard is invaluable for diagnosing training issues (underfitting, overfitting, learning rate problems).

---

### Prompt 17: Handling Mixed Feature Types (Numeric + Categorical)

**Goal:**
Understand preprocessing strategies for datasets with both numeric and categorical features.

**Prompt:**
```
My dataset has 13 features: 5 numeric (e.g., temperature, humidity) and 8 categorical (e.g., season, weather).

Explain and compare these preprocessing approaches:
1. One-hot encoding all categorical features
2. Label encoding + embedding layers
3. Target encoding

For my approach (one-hot + standard scaling of numerics):
1. Why scale numeric features but not one-hot columns?
2. How to keep track of which one-hot columns map to which original feature (feature groups)?
3. Why is feature group tracking critical for SHAP value aggregation?
4. How to implement a TabularPreprocessor class that handles this cleanly
5. What to watch out for with high-cardinality categorical features
```

**Why this is useful:**
Incorrect preprocessing is the most common source of silent errors in ML pipelines.

---

### Prompt 18: Understanding Train/Validation/Test Splits

**Goal:**
Learn proper data splitting methodology for ML experiments.

**Prompt:**
```
My ML project splits data into train (70%), validation (10%), and test (20%) sets.

Explain:
1. Why do we need THREE sets, not just train and test?
2. What is the validation set used for specifically? (early stopping, hyperparameter selection)
3. Why must I use stratified splitting for classification datasets?
4. How does scikit-learn's train_test_split handle stratification?
5. How to do a two-step split: first split off test, then split remainder into train/val
6. Why the split ratios from the config (test_size: 0.20, val_size: 0.10) are applied sequentially, not simultaneously
7. Common mistakes students make with data splitting (data leakage, applying preprocessing before splitting)
```

**Why this is useful:**
Improper splitting leads to overly optimistic results and is a common academic integrity issue.

---

### Prompt 19: Setting Up the Results Directory Structure

**Goal:**
Design an organized output directory for experiment results.

**Prompt:**
```
My ML project generates many output files during experiments:
- CSV tables: model metrics, paper comparison tables, explanation comparisons
- Plots: training curves (.png), shape functions (.png), heatmaps (.png), bar charts (.png)
- Artifacts: JSON summaries, TensorBoard log directories
- Reports: multi-page PDF, one-page summary PDF

Design the results/ directory structure:
1. How to organize by artifact type (tables/, plots/, artifacts/) vs. by dataset
2. How to handle plots being per-dataset (results/plots/bike/, results/plots/covertype/)
3. How to name files consistently (e.g., {dataset}_{model}_{metric}.csv)
4. Where to put the combined/aggregate results
5. How to ensure the directories exist before writing (os.makedirs with exist_ok)
6. Should results be tracked in Git? Why or why not?
```

**Why this is useful:**
Organized outputs prevent the chaos of files scattered randomly. It also makes report generation much easier.

---

### Prompt 20: Designing the Experiment Orchestrator

**Goal:**
Plan how to wire together data loading, training, evaluation, and reporting.

**Prompt:**
```
My project needs an experiment orchestrator that:
1. Loads a dataset (Bike, Covertype, or Adult) using the data module
2. Preprocesses features (scaling, one-hot encoding)
3. Trains multiple models in sequence: black-box → surrogate → GAM → InstaSHAP
4. Evaluates each model on the test set
5. Computes permutation SHAP values on the black-box
6. Computes InstaSHAP explanations in a single pass
7. Compares SHAP vs InstaSHAP explanations (correlation, MSE)
8. Generates visualizations (training curves, shape functions, heatmaps)
9. Saves all results to the results/ directory
10. Returns a summary object with paths to all outputs

Design the function signature and flow. Should this be:
- One monolithic function?
- A class with methods for each stage?
- Separate functions composed in main()?

Explain the trade-offs, especially regarding error handling if one stage fails.
```

**Why this is useful:**
The orchestrator is the backbone of the project. Getting its design right prevents cascading refactors.

---

### Prompt 21: Understanding the Additive Model Concept (GAM)

**Goal:**
Build intuition about Generalized Additive Models before implementing them.

**Prompt:**
```
My project implements a GAM (Generalized Additive Model) as a neural network. Explain:

1. What is the core idea of a GAM? (f(x) = Σ f_j(x_j))
2. How does this differ from a standard MLP that sees all features at once?
3. What is a "subnetwork" in the context of a neural GAM?
4. How does GAM-1 (univariate only) differ from GAM-2 (with pairwise interactions)?
5. Why does additive structure make the model inherently interpretable?
6. How are SHAP values related to GAM components?
7. What are "shape functions" and how do you visualize them?
8. What's the trade-off: a standard MLP likely has higher accuracy, so why use a GAM?

Use concrete examples — e.g., how "hour" and "workingday" might interact in bike rental predictions.
```

**Why this is useful:**
Understanding GAMs is essential for this project and for XAI research in general.

---

### Prompt 22: Planning the Masked Surrogate Model

**Goal:**
Understand the masked surrogate concept before coding it.

**Prompt:**
```
The InstaSHAP pipeline trains a "masked surrogate" model. Explain in detail:

1. What does "masking" mean in this context? (Setting subsets of features to zero/baseline)
2. Why can't we just train InstaSHAP directly on labels?
3. What is the surrogate's training target? (The black-box's output on masked inputs)
4. How are binary masks generated? (Random subsets with edge probabilities)
5. What is masks_per_sample and why use multiple masks per data point?
6. What is edge_mask_probability and how does it affect mask diversity?
7. How does this relate to the Shapley value's marginal contribution formula?
8. What happens if the surrogate is poorly trained?

This is conceptual — I want to understand before implementing.
```

**Why this is useful:**
The masked surrogate is the most conceptually complex component. Implementing without understanding leads to subtle bugs.

---

### Prompt 23: Writing a Good __init__.py for Python Packages

**Goal:**
Understand Python package initialization and import management.

**Prompt:**
```
My project has this structure:
instashap_project/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── blackbox_model.py
│   ├── gam.py
│   └── instashap.py

What should each __init__.py contain? Explain:
1. The purpose of __init__.py (making directories into packages)
2. When to use it for re-exporting (from .blackbox_model import TabularMLP)
3. When to leave it empty (or nearly empty)
4. How __all__ works and when to use it
5. How to avoid circular imports between packages (e.g., models importing from training)
6. The difference between relative imports (from .gam import GAMModel) and absolute imports
7. Best practices for a research project vs. a published library
```

**Why this is useful:**
Most import errors in Python projects trace back to misunderstanding __init__.py and package structure.

---

### Prompt 24: Setting Up the Device Resolution Logic

**Goal:**
Write portable code that automatically selects CPU or GPU.

**Prompt:**
```
My project should run on both CPU and NVIDIA GPUs. The config.yaml has:
  device: auto  # Can be "auto", "cpu", or "cuda"

Write a resolve_device() function that:
1. If "auto": checks torch.cuda.is_available() and returns cuda if available, else cpu
2. If "cpu" or "cuda": uses the specified device
3. Logs which device was selected
4. Handles the case where user specifies "cuda" but no GPU is available (graceful fallback with warning)

Then explain:
1. How to use the resolved device throughout the codebase (pass it or use a global?)
2. How .to(device) works for both models and tensors
3. Common device mismatch errors and how to debug them
4. The MPS device for Apple Silicon — should I support it?
```

**Why this is useful:**
Device management is a common source of runtime errors, especially for students who develop on CPU but deploy on GPU.

---

### Prompt 25: Planning Feature Groups for SHAP Aggregation

**Goal:**
Design the feature grouping system needed for proper SHAP value interpretation.

**Prompt:**
```
After one-hot encoding, a categorical feature like "season" (4 categories) becomes 4 binary columns. But when I compute SHAP values, I want the importance of "season" as a single feature, not 4 separate columns.

Design a feature groups system:
1. How to represent the mapping: {"season": [0, 1, 2, 3], "weather": [4, 5, 6, 7], "temp": [8], ...}
2. When in the pipeline should this mapping be created? (During preprocessing)
3. How to aggregate SHAP values for grouped features (sum the absolute SHAP values?)
4. Should the GAM subnetworks operate on original features or one-hot features?
5. How do interaction pairs map to feature groups (e.g., "hour × workingday")?
6. What data structure should I use to store this mapping efficiently?

Think through edge cases: what if a categorical feature has 50 categories?
```

**Why this is useful:**
Feature groups are critical for interpretability. Without them, SHAP results are fragmented and misleading.

---

### Prompt 26: Setting Up Early Stopping

**Goal:**
Understand and implement early stopping for training loops.

**Prompt:**
```
My training loops use early stopping to prevent overfitting. Explain:

1. What is early stopping and why is it important?
2. What metric should I monitor? (Validation loss, validation accuracy, etc.)
3. What is "patience" (e.g., patience=5 means stop if no improvement for 5 epochs)?
4. Should I track the best model weights and restore them after stopping?
5. How to implement this cleanly without cluttering the training loop
6. The difference between monitoring loss (lower is better) and accuracy (higher is better)
7. Should I use a fixed patience for all models or customize it?

My config has:
- blackbox: patience=5, epochs=25
- gam: patience=6, epochs=35
- instashap: patience=6, epochs=35

Why might GAM/InstaSHAP need more patience than the black-box?
```

**Why this is useful:**
Early stopping is simple in concept but tricky in implementation. Wrong monitoring leads to underfitting or overfitting.

---

### Prompt 27: Evaluating Project Risks and Failure Points

**Goal:**
Identify what could go wrong before it does.

**Prompt:**
```
I'm starting an ML research reproducibility project. Help me create a risk register:

For each risk, provide: Description, Probability (H/M/L), Impact (H/M/L), Mitigation.

Consider these risk categories:
1. Data: UCI API downtime, data preprocessing bugs, data leakage
2. Models: Training instability, NaN losses, slow convergence
3. Dependencies: SHAP library version incompatibilities, PyTorch breaking changes
4. Reproducibility: Non-deterministic results across runs, platform differences (Linux vs Windows)
5. Performance: SHAP computation taking too long, memory issues with large datasets
6. Scope: Feature creep, trying to reproduce too many experiments
7. Documentation: Results not matching paper exactly (expected variance)

Give me the top 10 risks and concrete mitigation strategies.
```

**Why this is useful:**
Risk assessment prevents panic when things inevitably go wrong. Planning mitigations in advance saves time.

---

### Prompt 28: Choosing a License for Your Project

**Goal:**
Understand open-source licenses and pick the right one.

**Prompt:**
```
I'm publishing a research reproducibility project on GitHub. Help me choose a license:

1. Compare MIT, Apache 2.0, GPL v3, and BSD 3-Clause for a research project
2. Which license allows others to use my code in their papers with minimal restrictions?
3. Which license requires derivative works to also be open source?
4. Does the license of my dependencies (PyTorch=BSD, scikit-learn=BSD, SHAP=MIT) affect my choice?
5. If my university has an IP policy, how does that interact with open-source licensing?
6. Where do I put the LICENSE file in my repository?
7. How do I add the license badge to my README?

Recommend the most appropriate license for an academic reproducibility project and explain why.
```

**Why this is useful:**
Licensing mistakes can prevent collaboration or even get you in legal trouble. Understanding this once saves headaches forever.

---

### Prompt 29: Setting Up Pre-commit Hooks for Code Quality

**Goal:**
Automate code quality checks before every commit.

**Prompt:**
```
I want to set up pre-commit hooks for my Python ML project to enforce code quality. Help me:

1. Explain what pre-commit hooks are and why they matter
2. Write a .pre-commit-config.yaml with these hooks:
   - black (code formatting)
   - ruff or flake8 (linting)
   - isort (import sorting)
   - trailing whitespace and end-of-file fixer
   - YAML syntax checker
3. Show the commands to install and run pre-commit
4. Explain how these hooks interact with Git workflows
5. Should I add type checking (mypy) as a hook for a research project?
6. How strict should I be? (Balance between quality and productivity)

What's the minimum viable .pre-commit-config.yaml that still adds real value?
```

**Why this is useful:**
Automated code quality prevents style debates and catches common errors before they enter the codebase.

---

### Prompt 30: Creating a Development Workflow Checklist

**Goal:**
Establish a systematic development workflow for the project.

**Prompt:**
```
I'm about to start implementing my ML research project from scratch. Create a development workflow checklist in order:

Phase 1 - Foundation:
- [ ] Set up repository, .gitignore, README, LICENSE
- [ ] Create virtual environment and install dependencies
- [ ] Set up project folder structure (empty modules with docstrings)
- [ ] Implement config.yaml loader
- [ ] Implement reproducibility utilities (seeding, device resolution)
- [ ] Implement structured logging

Phase 2 - Data:
- [ ] Implement dataset loaders
- [ ] Implement preprocessing pipeline
- [ ] Write smoke tests for data loading

Phase 3 - Models:
- [ ] Implement models one by one (blackbox → GAM → surrogate → InstaSHAP)
- [ ] Test each model with a tiny dataset

Phase 4 - Training:
- [ ] Implement training loops with early stopping
- [ ] Add TensorBoard logging
- [ ] Run first end-to-end experiment

Phase 5 - Explainability:
- [ ] Implement SHAP wrapper
- [ ] Implement InstaSHAP explainer
- [ ] Compare explanations

Phase 6 - Polish:
- [ ] Implement visualizations
- [ ] Implement PDF report generation
- [ ] Write comprehensive tests
- [ ] Write documentation

For each phase, explain what to validate before moving on.
```

**Why this is useful:**
Having a clear workflow prevents the common student mistake of trying to build everything at once and getting overwhelmed.
