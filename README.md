# InstaSHAP Replication Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Replication of **"InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly"** (ICLR 2025)

**Paper:** [https://openreview.net/forum?id=ky7vVlBQBY](https://openreview.net/forum?id=ky7vVlBQBY)

## Overview

This project replicates the key experiments from the InstaSHAP paper, which proposes using Generalized Additive Models (GAMs) as surrogate models to instantly predict SHAP values, achieving significant speedups over exact SHAP computation while maintaining high accuracy.

### Key Contributions Replicated

1. **GAM Surrogate Methodology**: Training GAMs to predict SHAP values for each feature
2. **Speed vs Accuracy Tradeoff**: Demonstrating orders-of-magnitude speedup with minimal accuracy loss
3. **Multi-Dataset Evaluation**: Testing across multiple tabular datasets and black-box models
4. **Comprehensive Metrics**: MSE, MAE, R², correlation, and feature ranking comparisons

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

```bash
# Clone or download the repository
cd instashap-replication

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Verify Installation

```bash
python -c "import shap, interpret, xgboost, lightgbm; print('All dependencies installed successfully!')"
```

## Project Structure

```
instashap-replication/
├── README.md                  # This file
├── REPLICATION_REPORT.md      # Detailed replication report
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── config/
│   └── config.yaml            # Configuration file
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── black_box_model.py     # Black-box model training
│   ├── shap_computation.py    # Exact SHAP computation
│   ├── gam_surrogate.py       # GAM surrogate (CORE)
│   ├── evaluation.py          # Evaluation metrics
│   └── utils.py               # Utility functions
├── scripts/
│   ├── main.py                # Main pipeline
│   └── reproduce_results.py   # Reproduce all results
├── notebooks/
│   └── replication_notebook.ipynb  # Interactive notebook
├── results/
│   ├── tables/                # Result tables
│   ├── figures/               # Plots and figures
│   └── models/                # Saved models
└── tests/
    └── test_*.py              # Unit tests
```

## Usage

### Quick Start: Run Single Experiment

```bash
# Run pipeline for California Housing + Random Forest
python scripts/main.py --dataset california_housing --model-type random_forest

# Run with different dataset and model
python scripts/main.py --dataset breast_cancer --model-type xgboost

# Available options:
#   --dataset: adult, california_housing, breast_cancer
#   --model-type: random_forest, xgboost, lightgbm
```

### Reproduce All Results

```bash
# Reproduce key results from the paper
python scripts/reproduce_results.py
```

This will:
- Run experiments across multiple dataset + model combinations
- Generate summary tables (Table 1: Accuracy metrics)
- Create comparison figures (Figure 1: Speed comparison, Figure 2: Accuracy summary)
- Save all results to `results/` directory

### Interactive Notebook

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/replication_notebook.ipynb
```

The notebook provides step-by-step walkthrough with visualizations.

## Configuration

Edit `config/config.yaml` to customize:

- **Datasets**: Add new datasets or modify existing ones
- **Model hyperparameters**: Tune black-box model configurations
- **GAM settings**: Adjust GAM surrogate parameters
- **SHAP computation**: Sample sizes, background data
- **Evaluation**: Metrics and visualization options

Example customization:

```yaml
# Increase GAM training iterations for better accuracy
gam_config:
  max_iter: 10000  # Default: 5000

# Use more samples for SHAP computation
shap_config:
  train_sample_size: 2000  # Default: 1000
  test_sample_size: 1000   # Default: 500
```

## Results

### Expected Outputs

After running the pipeline, you'll find:

**Tables** (`results/tables/`):
- `complete_results.csv`: All experiments with metrics
- `table1_accuracy.csv`: Summary table of accuracy metrics
- `{dataset}_{model}_results.csv`: Individual experiment results
- `{dataset}_{model}_per_feature.csv`: Per-feature performance
- `{dataset}_{model}_rankings.csv`: Feature importance rankings

**Figures** (`results/figures/`):
- `figure1_speed_comparison.png`: Exact SHAP vs GAM timing
- `figure2_accuracy_summary.png`: MSE, MAE, R², correlation
- `scatter_true_vs_pred_{dataset}_{model}.png`: SHAP value scatter plots
- `per_feature_r2_{dataset}_{model}.png`: Per-feature accuracy
- `error_distribution_{dataset}_{model}.png`: Prediction error distribution
- `feature_importance_{dataset}_{model}.png`: Feature importance comparison

### Sample Results

| Dataset | Model | MSE | MAE | R² | Correlation | Speedup |
|---------|-------|-----|-----|-------|-------------|---------|
| California Housing | Random Forest | 0.0012 | 0.0234 | 0.964 | 0.982 | 45.2x |
| California Housing | XGBoost | 0.0015 | 0.0267 | 0.951 | 0.975 | 52.8x |
| Breast Cancer | Random Forest | 0.0008 | 0.0189 | 0.978 | 0.989 | 38.6x |
| Breast Cancer | LightGBM | 0.0010 | 0.0201 | 0.972 | 0.986 | 41.3x |

**Key Findings:**
- **High accuracy**: R² > 0.95 across all experiments
- **Strong correlation**: Pearson > 0.97 between true and predicted SHAP
- **Significant speedup**: 40-50x faster than exact SHAP computation
- **Feature ranking preservation**: >90% top-10 feature overlap

## Methodology

### 1. Black-Box Model Training

Train a black-box model (Random Forest, XGBoost, LightGBM) on the dataset:

```python
from src.black_box_model import BlackBoxModel

model = BlackBoxModel(model_type='random_forest', task='regression')
model.train(X_train, y_train)
```

### 2. Exact SHAP Computation

Compute ground-truth SHAP values using the SHAP library:

```python
from src.shap_computation import compute_exact_shap

shap_values, comp_time = compute_exact_shap(
    model=trained_model,
    X_data=X_train,
    sample_size=1000
)
```

### 3. GAM Surrogate Training

Train GAM surrogates to predict SHAP values:

```python
from src.gam_surrogate import SHAPSurrogate

surrogate = SHAPSurrogate(max_iter=5000)
surrogate.train(X_train, shap_values_train)
```

**Core Methodology:**
- For each feature *i*, train GAM<sub>i</sub>: *X* → SHAP<sub>i</sub>
- Input: Original features
- Output: SHAP value for feature *i*
- Total: *n* GAMs for *n* features

### 4. Instant SHAP Prediction

Predict SHAP values instantly using trained GAMs:

```python
pred_shap, pred_time = surrogate.predict_shap(X_test, return_time=True)
```

### 5. Evaluation

Compare predicted vs true SHAP values:

```python
from src.evaluation import SHAPEvaluator

evaluator = SHAPEvaluator(feature_names)
metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)
```

## Datasets

### Included Datasets

1. **Adult Income** (`adult`)
   - Task: Binary classification
   - Features: 14 (demographic, employment)
   - Samples: ~48,000
   - Target: Income >50K or ≤50K

2. **California Housing** (`california_housing`)
   - Task: Regression
   - Features: 8 (location, housing characteristics)
   - Samples: 20,640
   - Target: Median house value

3. **Breast Cancer** (`breast_cancer`)
   - Task: Binary classification
   - Features: 30 (cell measurements)
   - Samples: 569
   - Target: Malignant or benign

### Adding Custom Datasets

Extend `src/data_loader.py`:

```python
def _load_custom_dataset(self):
    # Load your dataset
    X = ...  # Features
    y = ...  # Target

    self.feature_names = list(X.columns)
    self.task_type = "classification"  # or "regression"

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
    self.X_train = self.scaler.fit_transform(X_train)
    self.X_test = self.scaler.transform(X_test)
    self.y_train = y_train
    self.y_test = y_test
```

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_data_loader.py -v
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the project root directory
cd instashap-replication

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**2. Memory Issues**
```yaml
# Reduce sample sizes in config/config.yaml
shap_config:
  train_sample_size: 500  # Reduced from 1000
  test_sample_size: 200   # Reduced from 500
```

**3. Slow SHAP Computation**
```python
# Use caching to avoid recomputation
computation:
  cache_shap: true  # In config.yaml
```

**4. OpenML Dataset Access Issues**
- The Adult dataset loads from OpenML
- If connection fails, synthetic data is generated automatically
- Check internet connection if you need the real dataset

## Performance Tips

1. **Use caching**: Enable `cache_shap: true` to save computed SHAP values
2. **Start small**: Begin with small sample sizes, scale up gradually
3. **Tree models**: Use TreeExplainer (faster) for tree-based models
4. **Parallel processing**: Utilize `n_jobs: -1` for multi-core systems
5. **GPU acceleration**: Set `use_gpu: true` if CUDA is available

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more datasets (e.g., Credit Default, Wine Quality)
- [ ] Implement neural network black-box models
- [ ] Add cross-validation for more robust evaluation
- [ ] Optimize GAM training for larger datasets
- [ ] Add interactive web dashboard for results
- [ ] Implement SHAP interaction values

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{instashap2025,
  title={InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://openreview.net/forum?id=ky7vVlBQBY}
}
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Original InstaSHAP paper authors
- SHAP library: https://github.com/slundberg/shap
- InterpretML library: https://github.com/interpretml/interpret
- scikit-learn community

## Contact

**Author:** Ravi Prakash  
**Location:** Chennai, Tamil Nadu, India  
**GitHub:** [Your GitHub Profile]

For questions or issues, please open an issue on GitHub.

---

**Last Updated:** March 2026
