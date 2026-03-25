# Installation Guide

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Step-by-Step Installation](#step-by-step-installation)
- [Environment Setup](#environment-setup)
- [Verification](#verification)
- [Troubleshooting Installation](#troubleshooting-installation)

---

## Prerequisites

### System Requirements

**Minimum:**
- OS: Windows 10, macOS 10.13+, or Linux (Ubuntu 18.04+)
- Python: 3.8 or higher
- RAM: 4 GB
- Storage: 2 GB free space
- Internet: For downloading packages and datasets

**Recommended:**
- Python: 3.9 - 3.11
- RAM: 8 GB+
- Storage: 5 GB+ SSD
- CPU: 4+ cores

### Software Prerequisites

1. **Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify installation: `python --version`

2. **pip (Python Package Manager)**
   - Usually included with Python
   - Verify: `pip --version`
   - Upgrade: `python -m pip install --upgrade pip`

3. **Git (Optional, for cloning repository)**
   - Download from [git-scm.com](https://git-scm.com/)
   - Verify: `git --version`

---

## Installation Methods

### Method 1: Quick Install (Recommended)

**For most users:**

```bash
# Clone repository
git clone https://github.com/your-repo/instashap-replication.git
cd instashap-replication

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import shap, interpret; print('Installation successful!')"
```

### Method 2: Manual Installation

**If you don't have Git:**

1. Download project as ZIP from GitHub
2. Extract to desired location
3. Follow steps from Method 1 (starting from "Create virtual environment")

### Method 3: Development Installation

**For contributors:**

```bash
# Clone and enter directory
git clone https://github.com/your-repo/instashap-replication.git
cd instashap-replication

# Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

---

## Step-by-Step Installation

### Step 1: Clone or Download Project

**Option A: Using Git**
```bash
git clone https://github.com/your-repo/instashap-replication.git
cd instashap-replication
```

**Option B: Download ZIP**
1. Go to GitHub repository
2. Click "Code" → "Download ZIP"
3. Extract ZIP file
4. Open terminal in extracted folder

### Step 2: Create Virtual Environment

**Why Virtual Environment?**
- Isolates project dependencies
- Prevents version conflicts
- Easy to recreate

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Verify Activation:**
- Command prompt should show `(.venv)` prefix
- Check: `which python` (should point to .venv directory)

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

**Why?** Newer pip versions have better dependency resolution

### Step 4: Install Dependencies

**All Dependencies:**
```bash
pip install -r requirements.txt
```

**This installs:**
- Core ML: numpy, pandas, scikit-learn
- SHAP: shap library
- GAM: interpret (EBM implementation)
- Boosting: xgboost, lightgbm
- Visualization: matplotlib, seaborn, plotly
- Utils: tqdm, pyyaml, joblib
- Jupyter: jupyter, ipykernel
- Testing: pytest

**Installation Time:** 3-10 minutes (depends on internet speed)

**Monitor Progress:**
```
Collecting numpy>=1.26
  Downloading numpy-1.26.0-cp310-cp310-win_amd64.whl (15.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.8/15.8 MB 5.2 MB/s
Installing collected packages: numpy, pandas, ...
Successfully installed numpy-1.26.0 pandas-2.2.0 ...
```

### Step 5: Verify Installation

**Quick Test:**
```bash
python -c "import numpy, pandas, sklearn, shap, interpret, xgboost, lightgbm; print('All packages installed!')"
```

**Detailed Test:**
```bash
python -c "
import sys
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import pandas as pd
print(f'Pandas: {pd.__version__}')

import sklearn
print(f'scikit-learn: {sklearn.__version__}')

import shap
print(f'SHAP: {shap.__version__}')

import interpret
print(f'InterpretML: {interpret.__version__}')

import xgboost as xgb
print(f'XGBoost: {xgb.__version__}')

import lightgbm as lgb
print(f'LightGBM: {lgb.__version__}')

print('\n✓ All packages successfully imported!')
"
```

**Expected Output:**
```
Python: 3.10.0
NumPy: 1.26.0
Pandas: 2.2.0
scikit-learn: 1.4.0
SHAP: 0.45.0
InterpretML: 0.5.0
XGBoost: 2.0.0
LightGBM: 4.3.0

✓ All packages successfully imported!
```

### Step 6: Run Test Script

```bash
# Run quick test
python scripts/main.py --dataset breast_cancer --model-type random_forest
```

**Expected:**
- Script runs without errors
- Outputs metrics and results
- Creates `results/` directory with outputs

---

## Environment Setup

### Python Version Management

**Using pyenv (Recommended for macOS/Linux):**

```bash
# Install pyenv
curl https://pyenv.run | bash

# Install Python 3.10
pyenv install 3.10.0
pyenv local 3.10.0

# Verify
python --version  # Should show 3.10.0
```

**Using Anaconda:**

```bash
# Create conda environment
conda create -n instashap python=3.10
conda activate instashap

# Install dependencies
pip install -r requirements.txt
```

### IDE Setup

**VS Code:**

1. Install Python extension
2. Select interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv`
3. Install recommended extensions:
   - Python
   - Jupyter
   - Pylance

**PyCharm:**

1. Open project
2. File → Settings → Project → Python Interpreter
3. Add Interpreter → Existing → Select `.venv/Scripts/python.exe`

**Jupyter Notebook:**

```bash
# Install Jupyter kernel
python -m ipykernel install --user --name=instashap --display-name="InstaSHAP"

# Launch Jupyter
jupyter notebook

# Open: notebooks/replication_notebook.ipynb
```

### Environment Variables (Optional)

**For custom configurations:**

```bash
# Windows
set INSTASHAP_DATA_DIR=C:\data\instashap
set INSTASHAP_RESULTS_DIR=C:\results\instashap

# macOS/Linux
export INSTASHAP_DATA_DIR=/data/instashap
export INSTASHAP_RESULTS_DIR=/results/instashap
```

---

## Verification

### Comprehensive Test Suite

**Run all tests:**
```bash
pytest tests/ -v
```

**Expected Output:**
```
tests/test_data_loader.py::test_load_california_housing PASSED
tests/test_data_loader.py::test_load_breast_cancer PASSED
tests/test_black_box_model.py::test_train_random_forest PASSED
tests/test_black_box_model.py::test_train_xgboost PASSED
tests/test_gam_surrogate.py::test_train_gam PASSED
tests/test_gam_surrogate.py::test_predict_gam PASSED
tests/test_evaluation.py::test_compute_metrics PASSED

==================== 15 passed in 45.2s ====================
```

### Quick Functionality Test

**Test 1: Data Loading**
```bash
python -c "from src.data_loader import DataLoader; loader = DataLoader('california_housing'); X_train, X_test, y_train, y_test, names = loader.get_data(); print(f'Loaded {len(X_train)} training samples')"
```

**Test 2: Model Training**
```bash
python -c "
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel

loader = DataLoader('breast_cancer')
X_train, X_test, y_train, y_test, names = loader.get_data()

model = BlackBoxModel('random_forest', 'classification')
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
print(f\"Test Accuracy: {metrics['accuracy']:.3f}\")
"
```

**Test 3: SHAP Computation**
```bash
python -c "
from src.data_loader import DataLoader
from src.black_box_model import BlackBoxModel
from src.shap_computation import SHAPComputer

loader = DataLoader('california_housing')
X_train, X_test, y_train, y_test, names = loader.get_data()

model = BlackBoxModel('xgboost', 'regression')
model.train(X_train, y_train)

shap_computer = SHAPComputer(model, X_train, 'regression')
shap_values = shap_computer.compute_shap_values(X_test[:100])
print(f'Computed SHAP for {len(shap_values)} samples')
"
```

### Check Directory Structure

```bash
# Verify all necessary directories exist
ls -la

# Should see:
# config/
# data/
# docs/
# notebooks/
# scripts/
# src/
# tests/
# requirements.txt
# setup.py
```

---

## Troubleshooting Installation

### Issue 1: pip install fails

**Error:** `ERROR: Could not find a version that satisfies the requirement`

**Solutions:**

1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Check Python version:**
   ```bash
   python --version  # Must be 3.8+
   ```

3. **Install packages individually:**
   ```bash
   pip install numpy pandas scikit-learn
   pip install shap interpret
   pip install xgboost lightgbm
   ```

### Issue 2: SHAP installation fails

**Error:** `Building wheel for shap ... error`

**Solutions (Windows):**

1. **Install Visual C++ Build Tools:**
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install "Desktop development with C++"

2. **Use prebuilt wheel:**
   ```bash
   pip install shap --no-build-isolation
   ```

**Solutions (macOS):**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Then retry
pip install shap
```

### Issue 3: XGBoost/LightGBM installation fails

**Error:** `OSError: cannot open shared object file`

**Linux Solution:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential libgomp1
pip install xgboost lightgbm
```

**macOS Solution:**
```bash
brew install libomp
pip install xgboost lightgbm
```

### Issue 4: Import errors after installation

**Error:** `ModuleNotFoundError: No module named 'src'`

**Solutions:**

1. **Ensure virtual environment is activated:**
   ```bash
   # Check for (.venv) prefix in terminal
   which python  # Should point to .venv
   ```

2. **Run from project root:**
   ```bash
   cd /path/to/instashap-replication
   python scripts/main.py
   ```

3. **Add project to PYTHONPATH:**
   ```bash
   # Windows
   set PYTHONPATH=%PYTHONPATH%;C:\path\to\instashap-replication
   
   # macOS/Linux
   export PYTHONPATH=$PYTHONPATH:/path/to/instashap-replication
   ```

### Issue 5: Memory errors during installation

**Error:** `MemoryError` or system freezes

**Solutions:**

1. **Install packages one at a time:**
   ```bash
   pip install numpy
   pip install pandas
   pip install scikit-learn
   # etc.
   ```

2. **Increase virtual memory (Windows):**
   - System Properties → Advanced → Performance Settings → Advanced → Virtual Memory

3. **Close other applications** during installation

### Issue 6: Permission denied errors

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solutions:**

1. **Don't use sudo** (creates permission issues)
2. **Use virtual environment** (already recommended)
3. **Windows: Run terminal as administrator** (only if necessary)

### Issue 7: SSL certificate errors

**Error:** `SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]`

**Solutions:**

1. **Update certifi:**
   ```bash
   pip install --upgrade certifi
   ```

2. **Temporary workaround (not recommended for production):**
   ```bash
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

---

## Platform-Specific Notes

### Windows

**PowerShell Execution Policy:**
```powershell
# If activation fails, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Long Path Support:**
```
# Enable long paths (Windows 10+)
# Run as Administrator:
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### macOS

**Xcode Command Line Tools:**
```bash
xcode-select --install
```

**Homebrew (for libomp):**
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install libomp
```

### Linux (Ubuntu/Debian)

**System dependencies:**
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv build-essential libgomp1
```

**For headless servers (no GUI):**
```bash
# Install matplotlib backend
sudo apt-get install -y python3-tk
```

---

## Docker Installation (Alternative)

**Coming Soon:** Docker container for reproducible environment

**Preview:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "scripts/main.py"]
```

---

## Next Steps

After successful installation:

1. **Read Usage Guide:** `docs/USAGE_GUIDE.md`
2. **Run Quick Example:** `python scripts/main.py --dataset breast_cancer --model-type random_forest`
3. **Explore Notebook:** `jupyter notebook notebooks/replication_notebook.ipynb`
4. **Configure Project:** Edit `config/config.yaml`

---

## Getting Help

**If installation fails:**

1. Check this troubleshooting section
2. Review error messages carefully
3. Search issues on GitHub repository
4. Create new issue with:
   - Operating system and version
   - Python version
   - Full error message
   - Steps to reproduce

**Useful diagnostic command:**
```bash
python -c "
import sys, platform
print(f'OS: {platform.system()} {platform.release()}')
print(f'Python: {sys.version}')
print(f'pip: {__import__(\"pip\").__version__}')
"
```

---

**Last Updated:** March 2026  
**Version:** 1.0.0  
**Tested On:** Windows 10/11, macOS 12+, Ubuntu 20.04/22.04
