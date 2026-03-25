# Usage Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Training Models](#training-models)
- [Running Inference](#running-inference)
- [Example Workflows](#example-workflows)
- [Jupyter Notebook Usage](#jupyter-notebook-usage)
- [Advanced Usage](#advanced-usage)

---

## Quick Start

### 5-Minute Quickstart

```bash
# 1. Activate environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 2. Run single experiment
python scripts/main.py --dataset california_housing --model-type xgboost

# 3. View results
ls results/california_housing_xgboost/
```

**Output:**
- Trained models saved to `models/`
- Metrics saved to `results/california_housing_xgboost/metrics.csv`
- Figures saved to `results/california_housing_xgboost/figures/`

---

## Running Experiments

### Basic Usage

**Command Structure:**
```bash
python scripts/main.py --dataset <DATASET> --model-type <MODEL> [OPTIONS]
```

**Required Arguments:**
- `--dataset`: Dataset name (`california_housing`, `breast_cancer`, `adult`)
- `--model-type`: Model type (`random_forest`, `xgboost`, `lightgbm`)

**Optional Arguments:**
- `--config`: Path to config file (default: `config/config.yaml`)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--output-dir`: Custom output directory

### Example Commands

**Example 1: California Housing + XGBoost**
```bash
python scripts/main.py \
    --dataset california_housing \
    --model-type xgboost \
    --log-level INFO
```

**Example 2: Breast Cancer + Random Forest**
```bash
python scripts/main.py \
    --dataset breast_cancer \
    --model-type random_forest
```

**Example 3: Adult Income + LightGBM (Debug Mode)**
```bash
python scripts/main.py \
    --dataset adult \
    --model-type lightgbm \
    --log-level DEBUG
```

### Reproduce All Results

```bash
# Run all experiments from the paper
python scripts/reproduce_results.py
```

**What This Does:**
- Runs 6 experiments (3 datasets × 2 models each)
- Generates comparison tables
- Creates publication-ready figures
- Saves to `results/replication/`

**Time:** ~15-30 minutes

---

## Training Models

### Train Black-Box Model Only

```python
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel

# Load data
loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, feature_names = loader.get_data()

# Train model
model = BlackBoxModel(model_type='xgboost', task='regression')
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(f"Test R²: {metrics['r2']:.4f}")

# Save
model.save('models/my_xgboost_model.pkl')
```

### Train GAM Surrogate

```python
from src.shap_computation import SHAPComputer
from src.gam_surrogate import GAMSurrogate

# Compute SHAP values
shap_computer = SHAPComputer(model, X_train, task='regression')
shap_train = shap_computer.compute_shap_values(X_train[:1000])

# Train GAM
gam = GAMSurrogate(feature_names=feature_names)
gam.train(X_train[:1000], shap_train)

# Save
gam.save('models/my_gam_surrogate.pkl')
```

### End-to-End Training Script

```python
# Complete workflow in one script
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel
from src.shap_computation import SHAPComputer
from src.gam_surrogate import GAMSurrogate
from src.evaluation import Evaluator

# 1. Load data
loader = DataLoader('breast_cancer')
X_train, X_test, y_train, y_test, features = loader.get_data()

# 2. Train black-box
model = BlackBoxModel('random_forest', 'classification')
model.train(X_train, y_train)
print(f"Black-box accuracy: {model.evaluate(X_test, y_test)['accuracy']:.3f}")

# 3. Compute SHAP
shap_computer = SHAPComputer(model, X_train, 'classification')
shap_train = shap_computer.compute_shap_values(X_train[:1000])
shap_test = shap_computer.compute_shap_values(X_test[:500])

# 4. Train GAM
gam = GAMSurrogate(feature_names=features)
gam.train(X_train[:1000], shap_train)

# 5. Predict and evaluate
shap_pred = gam.predict(X_test[:500])
evaluator = Evaluator(shap_test, shap_pred, features)
metrics = evaluator.evaluate()
print(f"GAM R²: {metrics['r2']:.4f}")
```

---

## Running Inference

### Predict with Black-Box Model

```python
from src.black_box_model import BlackBoxModel
import numpy as np

# Load trained model
model = BlackBoxModel.load('models/california_housing_xgboost.pkl')

# Single prediction
x_new = np.array([[3.5, 25.0, 5.2, 1.1, 1500, 3.0, 37.5, -122.3]])  # 8 features
prediction = model.predict(x_new)
print(f"Predicted house value: ${prediction[0] * 100000:.0f}")

# Batch prediction
X_batch = np.random.randn(100, 8)  # 100 samples
predictions = model.predict(X_batch)
print(f"Mean prediction: ${predictions.mean() * 100000:.0f}")
```

### Generate SHAP Explanations (Instant)

```python
from src.gam_surrogate import GAMSurrogate

# Load GAM surrogate
gam = GAMSurrogate.load('models/gam_surrogate_california_housing.pkl')

# Instant SHAP prediction
x_new = np.array([[3.5, 25.0, 5.2, 1.1, 1500, 3.0, 37.5, -122.3]])
shap_values = gam.predict(x_new)

# Display explanation
for feature, shap_val in zip(feature_names, shap_values[0]):
    print(f"{feature:15s}: {shap_val:+.4f}")
```

**Output:**
```
MedInc         : +0.4523
HouseAge       : -0.0234
AveRooms       : +0.0456
AveBedrms      : -0.0123
Population     : -0.0089
AveOccup       : +0.0012
Latitude       : +0.1234
Longitude      : -0.0567
```

### Explain Single Prediction

```python
import matplotlib.pyplot as plt
from src.gam_surrogate import GAMSurrogate

# Load GAM
gam = GAMSurrogate.load('models/gam_surrogate_breast_cancer.pkl')

# New patient data
x_patient = X_test[0:1]  # First test sample
shap_patient = gam.predict(x_patient)[0]

# Waterfall plot
import shap
shap.waterfall_plot(
    shap.Explanation(
        values=shap_patient,
        base_values=0.0,
        data=x_patient[0],
        feature_names=feature_names
    )
)
plt.title("Patient Diagnosis Explanation")
plt.show()
```

---

## Example Workflows

### Workflow 1: Compare Models

```python
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel

# Load data
loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, _ = loader.get_data()

# Train multiple models
models = ['random_forest', 'xgboost', 'lightgbm']
results = {}

for model_type in models:
    model = BlackBoxModel(model_type, 'regression')
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    results[model_type] = metrics['r2']
    print(f"{model_type:15s}: R² = {metrics['r2']:.4f}")

# Best model
best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} (R² = {results[best_model]:.4f})")
```

### Workflow 2: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from src.data_loader import DataLoader

# Load data
loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, _ = loader.get_data()

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Grid search
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R²: {grid_search.best_score_:.4f}")

# Test performance
best_model = grid_search.best_estimator_
test_r2 = best_model.score(X_test, y_test)
print(f"Test R²: {test_r2:.4f}")
```

### Workflow 3: Feature Importance Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from src.shap_computation import SHAPComputer

# Compute SHAP values
shap_computer = SHAPComputer(model, X_train, 'regression')
shap_values = shap_computer.compute_shap_values(X_test)

# Global feature importance
importance = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(importance)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Mean |SHAP value|')
plt.title('Global Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
```

### Workflow 4: Batch Explanation Generation

```python
from src.gam_surrogate import GAMSurrogate
import pandas as pd

# Load GAM
gam = GAMSurrogate.load('models/gam_surrogate.pkl')

# Batch predict SHAP
shap_batch = gam.predict(X_test)  # Instant for all test samples

# Create explanation DataFrame
df_explanations = pd.DataFrame(
    shap_batch,
    columns=feature_names
)
df_explanations['prediction'] = model.predict(X_test)

# Save to CSV
df_explanations.to_csv('batch_explanations.csv', index=False)
print(f"Generated {len(df_explanations)} explanations")
```

### Workflow 5: Real-Time Explanation API

```python
from flask import Flask, request, jsonify
from src.black_box_model import BlackBoxModel
from src.gam_surrogate import GAMSurrogate
import numpy as np

app = Flask(__name__)

# Load models at startup
model = BlackBoxModel.load('models/my_model.pkl')
gam = GAMSurrogate.load('models/my_gam.pkl')

@app.route('/predict', methods=['POST'])
def predict_and_explain():
    # Parse input
    data = request.json
    x = np.array(data['features']).reshape(1, -1)
    
    # Predict
    prediction = model.predict(x)[0]
    
    # Explain (instant!)
    shap_values = gam.predict(x)[0]
    
    # Format response
    explanation = {
        feat: float(shap)
        for feat, shap in zip(feature_names, shap_values)
    }
    
    return jsonify({
        'prediction': float(prediction),
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Test API:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [3.5, 25.0, 5.2, 1.1, 1500, 3.0, 37.5, -122.3]}'
```

---

## Jupyter Notebook Usage

### Launch Notebook

```bash
jupyter notebook notebooks/replication_notebook.ipynb
```

### Interactive Exploration

**Cell 1: Load Data**
```python
from src.data_loader import DataLoader

loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, feature_names = loader.get_data()

print(f"Training samples: {len(X_train)}")
print(f"Features: {feature_names}")
```

**Cell 2: Train Model**
```python
from src.black_box_model import BlackBoxModel

model = BlackBoxModel('xgboost', 'regression')
model.train(X_train, y_train)

metrics = model.evaluate(X_test, y_test)
print(f"Test R²: {metrics['r2']:.4f}")
```

**Cell 3: Visualize SHAP**
```python
import shap
from src.shap_computation import SHAPComputer

shap_computer = SHAPComputer(model, X_train, 'regression')
shap_values = shap_computer.compute_shap_values(X_test[:100])

# Summary plot
shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names)
```

**Cell 4: Train and Evaluate GAM**
```python
from src.gam_surrogate import GAMSurrogate
from src.evaluation import Evaluator

# Train
gam = GAMSurrogate(feature_names=feature_names)
shap_train = shap_computer.compute_shap_values(X_train[:1000])
gam.train(X_train[:1000], shap_train)

# Evaluate
shap_pred = gam.predict(X_test[:100])
evaluator = Evaluator(shap_values, shap_pred, feature_names)
evaluator.plot_true_vs_pred()
```

---

## Advanced Usage

### Custom Dataset

```python
from src.black_box_model import BlackBoxModel
from src.gam_surrogate import GAMSurrogate
import pandas as pd

# Load your dataset
df = pd.read_csv('my_dataset.csv')
X = df.drop('target', axis=1).values
y = df['target'].values
feature_names = df.drop('target', axis=1).columns.tolist()

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocess
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = BlackBoxModel('xgboost', 'regression')
model.train(X_train, y_train)

# Continue with SHAP and GAM training...
```

### Custom Configuration

```python
from src.utils import load_config
import yaml

# Load default config
config = load_config('config/config.yaml')

# Modify settings
config['gam_config']['max_iter'] = 10000
config['shap_config']['train_sample_size'] = 2000

# Save custom config
with open('config/custom_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Use custom config
python scripts/main.py --config config/custom_config.yaml --dataset california_housing --model-type xgboost
```

### Parallel Experiments

```bash
# Run multiple experiments in parallel (Unix/Mac)
python scripts/main.py --dataset california_housing --model-type xgboost &
python scripts/main.py --dataset breast_cancer --model-type random_forest &
python scripts/main.py --dataset adult --model-type lightgbm &
wait
echo "All experiments complete!"
```

---

## Command Reference

### Main Script Options

```
usage: main.py [-h] --dataset {california_housing,breast_cancer,adult}
               --model-type {random_forest,xgboost,lightgbm}
               [--config CONFIG] [--log-level {DEBUG,INFO,WARNING,ERROR}]
               [--output-dir OUTPUT_DIR]

required arguments:
  --dataset             Dataset to use
  --model-type          Model type to train

optional arguments:
  -h, --help           Show help message
  --config             Path to config file (default: config/config.yaml)
  --log-level          Logging verbosity (default: INFO)
  --output-dir         Custom output directory (default: results/)
```

### Reproduce Script

```bash
python scripts/reproduce_results.py [--output-dir DIR]
```

**Runs:**
1. California Housing + Random Forest
2. California Housing + XGBoost
3. Breast Cancer + Random Forest
4. Breast Cancer + LightGBM
5. Adult Income + XGBoost
6. Adult Income + LightGBM

---

## Tips and Best Practices

### Performance Tips

1. **Use caching:** SHAP values are cached automatically (saves time)
2. **Start small:** Test with `breast_cancer` (smallest dataset)
3. **Use LightGBM:** Fastest training for large datasets
4. **Reduce sample sizes:** Edit `config.yaml` for faster iteration

### Reproducibility Tips

1. **Always set random seed:** Already configured in `config.yaml`
2. **Document changes:** Keep notes on configuration modifications
3. **Version control:** Commit code and config before experiments
4. **Save everything:** Models, SHAP values, results are all saved

### Debugging Tips

1. **Use DEBUG logging:** `--log-level DEBUG`
2. **Start with small samples:** Reduce `train_sample_size` in config
3. **Check data shapes:** Print `X_train.shape`, `y_train.shape`
4. **Test components individually:** Use Python REPL or Jupyter

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Related Docs:** [INSTALLATION.md](INSTALLATION.md), [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
