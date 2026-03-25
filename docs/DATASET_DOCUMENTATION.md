# Dataset Documentation

## Table of Contents
- [Overview](#overview)
- [Dataset Catalog](#dataset-catalog)
- [California Housing Dataset](#california-housing-dataset)
- [Breast Cancer Dataset](#breast-cancer-dataset)
- [Adult Income Dataset](#adult-income-dataset)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Train/Validation/Test Split](#trainvalidationtest-split)
- [Data Statistics](#data-statistics)
- [Data Quality and Limitations](#data-quality-and-limitations)

---

## Overview

This project uses three well-established benchmark datasets from the machine learning community to validate the InstaSHAP methodology. These datasets were chosen to:

1. **Cover Multiple Task Types:** Regression (California Housing) and Binary Classification (Breast Cancer, Adult Income)
2. **Vary in Scale:** From small (569 samples) to medium (~48,000 samples)
3. **Diverse Feature Types:** Numeric, categorical, geographic
4. **Real-World Relevance:** Healthcare, economics, real estate applications
5. **Reproducibility:** Publicly available through sklearn and OpenML

All datasets are automatically downloaded and preprocessed by the `src/data_loader.py` module.

---

## Dataset Catalog

| Dataset | Task | Samples | Features | Target Variable | Source | File Location |
|---------|------|---------|----------|-----------------|--------|---------------|
| **California Housing** | Regression | 20,640 | 8 (all numeric) | Median house value ($100k) | sklearn.datasets | `data/california_housing/` |
| **Breast Cancer** | Binary Classification | 569 | 30 (all numeric) | Malignant (1) / Benign (0) | sklearn.datasets | `data/breast_cancer/` |
| **Adult Income** | Binary Classification | ~48,842 | 14 (mixed) | Income >$50K (1) / ≤$50K (0) | OpenML | `data/adult/` |

---

## California Housing Dataset

### Description

The California Housing dataset is derived from the 1990 U.S. Census and contains information about housing in California districts. The goal is to predict the **median house value** for California districts based on various demographic and geographic features.

### Task Type
**Regression** - Continuous target variable

### Dataset Statistics

- **Total Samples:** 20,640
- **Training Samples:** 16,512 (80%)
- **Test Samples:** 4,128 (20%)
- **Features:** 8 (all continuous)
- **Target Range:** $14,999 - $500,001
- **Missing Values:** None
- **Duplicate Rows:** None

### Features

| Feature Name | Type | Description | Range | Mean | Std Dev |
|--------------|------|-------------|-------|------|---------|
| `MedInc` | Continuous | Median income in block group (in $10,000s) | 0.5 - 15.0 | 3.87 | 1.90 |
| `HouseAge` | Continuous | Median house age in block group (years) | 1.0 - 52.0 | 28.64 | 12.59 |
| `AveRooms` | Continuous | Average number of rooms per household | 0.8 - 141.9 | 5.43 | 2.47 |
| `AveBedrms` | Continuous | Average number of bedrooms per household | 0.3 - 34.1 | 1.10 | 0.47 |
| `Population` | Continuous | Block group population | 3.0 - 35682.0 | 1425.48 | 1132.46 |
| `AveOccup` | Continuous | Average number of household members | 0.7 - 1243.3 | 3.07 | 10.39 |
| `Latitude` | Continuous | Latitude of block group | 32.5 - 41.9 | 35.63 | 2.14 |
| `Longitude` | Continuous | Longitude of block group | -124.3 - -114.3 | -119.57 | 2.00 |

### Target Variable

- **Name:** `MedHouseVal` (Median House Value)
- **Type:** Continuous
- **Unit:** Hundreds of thousands of dollars ($100,000s)
- **Range:** 0.15 - 5.00 (representing $15,000 - $500,000)
- **Mean:** 2.07 ($207,000)
- **Median:** 1.80 ($180,000)
- **Distribution:** Right-skewed with a ceiling at $500,000

### Data Source

- **Origin:** U.S. Census Bureau (1990)
- **Access:** `sklearn.datasets.fetch_california_housing()`
- **License:** Public domain
- **Reference:** Pace, R. Kelley and Ronald Barry. "Sparse Spatial Autoregressions." Statistics & Probability Letters, Volume 33, Number 3, May 5 1997, p. 291-297.

### Use Case

Predict housing prices based on neighborhood characteristics for:
- Real estate valuation
- Investment decision support
- Urban planning insights
- Understanding housing market factors

### Preprocessing Applied

1. **Feature Scaling:** StandardScaler (zero mean, unit variance)
2. **Train/Test Split:** 80/20 random split (seed=42)
3. **No Missing Values:** Dataset is complete
4. **No Encoding Required:** All features are numeric

---

## Breast Cancer Dataset

### Description

The Breast Cancer Wisconsin (Diagnostic) dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. The features describe characteristics of cell nuclei present in the images. The goal is to classify tumors as **malignant (cancerous)** or **benign (non-cancerous)**.

### Task Type
**Binary Classification**

### Dataset Statistics

- **Total Samples:** 569
- **Training Samples:** 455 (80%)
- **Test Samples:** 114 (20%)
- **Features:** 30 (all continuous)
- **Classes:** 2 (Malignant: 212, Benign: 357)
- **Class Distribution:** 37.3% malignant, 62.7% benign
- **Missing Values:** None
- **Duplicate Rows:** None

### Feature Groups

The 30 features are organized into three groups (mean, standard error, worst) of 10 measurements each:

#### Mean Features (10)
Computed for each cell nucleus:

| Feature | Description | Example Range |
|---------|-------------|---------------|
| `radius_mean` | Mean of distances from center to points on perimeter | 6.98 - 28.11 |
| `texture_mean` | Standard deviation of gray-scale values | 9.71 - 39.28 |
| `perimeter_mean` | Mean perimeter of nucleus | 43.79 - 188.50 |
| `area_mean` | Mean area of nucleus | 143.5 - 2501.0 |
| `smoothness_mean` | Local variation in radius lengths | 0.053 - 0.163 |
| `compactness_mean` | (perimeter² / area - 1.0) | 0.019 - 0.345 |
| `concavity_mean` | Severity of concave portions | 0.000 - 0.427 |
| `concave_points_mean` | Number of concave portions | 0.000 - 0.201 |
| `symmetry_mean` | Symmetry of nucleus | 0.106 - 0.304 |
| `fractal_dimension_mean` | "Coastline approximation" - 1 | 0.050 - 0.097 |

#### Standard Error Features (10)
Standard error of measurements above (e.g., `radius_se`, `texture_se`, etc.)

#### Worst Features (10)
Mean of the three largest values of measurements (e.g., `radius_worst`, `texture_worst`, etc.)

### Target Variable

- **Name:** `target`
- **Type:** Binary
- **Classes:**
  - `0`: Malignant (cancerous) - 212 samples (37.3%)
  - `1`: Benign (non-cancerous) - 357 samples (62.7%)
- **Evaluation Focus:** High recall for malignant cases (minimize false negatives)

### Data Source

- **Origin:** University of Wisconsin Hospitals, Madison
- **Created by:** Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian
- **Year:** 1995
- **Access:** `sklearn.datasets.load_breast_cancer()`
- **License:** Public domain
- **Reference:** 
  - W.N. Street, W.H. Wolberg and O.L. Mangasarian. "Nuclear feature extraction for breast tumor diagnosis." IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology, volume 1905, pages 861-870, San Jose, CA, 1993.

### Use Case

Medical diagnosis support for:
- Early detection of breast cancer
- Reducing unnecessary biopsies
- Supporting radiologist decision-making
- Research on tumor characteristics

### Preprocessing Applied

1. **Feature Scaling:** StandardScaler (critical due to varying feature magnitudes)
2. **Train/Test Split:** 80/20 stratified split (maintains class balance)
3. **No Missing Values:** Dataset is complete
4. **No Encoding Required:** All features are numeric, target is already binary

### Class Balance Consideration

The dataset has a **1.68:1 ratio** (benign:malignant). This is reasonably balanced and doesn't require special sampling techniques, though evaluation metrics should include precision, recall, and F1-score (not just accuracy).

---

## Adult Income Dataset

### Description

The Adult Income dataset (also known as "Census Income") contains demographic data from the 1994 U.S. Census. The goal is to predict whether an individual's annual income exceeds **$50,000** based on census attributes.

### Task Type
**Binary Classification**

### Dataset Statistics

- **Total Samples:** ~48,842 (after removing missing values)
- **Training Samples:** ~39,074 (80%)
- **Test Samples:** ~9,768 (20%)
- **Features:** 14 (6 continuous, 8 categorical)
- **Classes:** 2 (>50K: 24.1%, ≤50K: 75.9%)
- **Missing Values:** ~7.4% of samples (handled during preprocessing)
- **Duplicate Rows:** Some present (handled during loading)

### Features

#### Continuous Features (6)

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `age` | Integer | Age in years | 17 - 90 |
| `fnlwgt` | Integer | Final sampling weight (census weight) | 12,285 - 1,484,705 |
| `education-num` | Integer | Number of years of education | 1 - 16 |
| `capital-gain` | Integer | Capital gains ($) | 0 - 99,999 |
| `capital-loss` | Integer | Capital losses ($) | 0 - 4,356 |
| `hours-per-week` | Integer | Hours worked per week | 1 - 99 |

#### Categorical Features (8)

| Feature | Type | Unique Values | Example Categories |
|---------|------|---------------|-------------------|
| `workclass` | Categorical | 9 | Private, Self-emp-not-inc, Local-gov, etc. |
| `education` | Categorical | 16 | Bachelors, HS-grad, 11th, Masters, etc. |
| `marital-status` | Categorical | 7 | Married-civ-spouse, Never-married, Divorced, etc. |
| `occupation` | Categorical | 15 | Tech-support, Craft-repair, Sales, etc. |
| `relationship` | Categorical | 6 | Wife, Own-child, Husband, Not-in-family, etc. |
| `race` | Categorical | 5 | White, Asian-Pac-Islander, Black, etc. |
| `sex` | Categorical | 2 | Male, Female |
| `native-country` | Categorical | 42 | United-States, Cambodia, England, etc. |

### Target Variable

- **Name:** `income`
- **Type:** Binary
- **Classes:**
  - `0`: Income ≤ $50,000 - 36,548 samples (75.9%)
  - `1`: Income > $50,000 - 11,687 samples (24.1%)
- **Class Imbalance:** 3.13:1 ratio (requires stratified splitting)

### Data Source

- **Origin:** U.S. Census Bureau (1994)
- **Donor:** Ronny Kohavi and Barry Becker (Silicon Graphics)
- **Access:** OpenML dataset ID 1590
- **License:** Public domain
- **Reference:**
  - Kohavi, R. "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid." Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996.

### Use Case

Socioeconomic analysis and prediction for:
- Targeted marketing campaigns
- Credit risk assessment
- Policy impact analysis
- Understanding income determinants

### Missing Values

The dataset contains missing values denoted by `?`:
- **workclass:** ~5.6% missing
- **occupation:** ~5.7% missing
- **native-country:** ~1.8% missing

**Handling Strategy:** Rows with missing values are dropped during preprocessing (reduces dataset from ~48,842 to ~45,222 samples).

### Preprocessing Applied

1. **Missing Value Removal:** Drop rows with `?` values
2. **Categorical Encoding:** LabelEncoder for all categorical features
3. **Feature Scaling:** StandardScaler (after encoding)
4. **Train/Test Split:** 80/20 stratified split (maintains class imbalance)
5. **Class Imbalance:** Noted but not artificially balanced (stratification ensures both sets have same ratio)

### Ethical Considerations

This dataset contains sensitive attributes (race, sex) that could be used for discriminatory purposes. When using this dataset:

- **Be Transparent:** Clearly document use of sensitive features
- **Fairness Analysis:** Check for bias across protected groups
- **Legal Compliance:** Ensure compliance with anti-discrimination laws
- **Research Context:** Use primarily for understanding and mitigating bias

---

## Data Preprocessing Pipeline

All datasets undergo a consistent preprocessing pipeline implemented in `src/data_loader.py:DataLoader`.

### Step 1: Data Loading

```python
data_loader = DataLoader(dataset_name='california_housing')
X_train, X_test, y_train, y_test, feature_names = data_loader.get_data()
```

**Process:**
1. Fetch from sklearn or OpenML
2. Extract features and target
3. Identify feature types (numeric, categorical)

### Step 2: Handling Missing Values

**Strategy:**
- **Numerical Features:** Median imputation
- **Categorical Features:** Mode imputation (or drop rows if excessive)

**Implementation:**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_numeric = imputer.fit_transform(X_numeric)
```

### Step 3: Categorical Encoding

**Strategy:** LabelEncoder for ordinal/nominal features

**Implementation:**
```python
from sklearn.preprocessing import LabelEncoder

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
```

**Why LabelEncoder:**
- Tree-based models handle ordinal encoding well
- Maintains single-column representation
- Avoids curse of dimensionality from one-hot encoding

**Note:** For models sensitive to ordinal assumptions, consider one-hot encoding (not used in this project).

### Step 4: Feature Scaling

**Strategy:** StandardScaler (z-score normalization)

**Formula:**
```
X_scaled = (X - mean(X)) / std(X)
```

**Implementation:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training statistics
```

**Why Scaling:**
- Ensures features contribute equally to SHAP computation
- Required for distance-based methods (though not critical for tree models)
- Standardizes interpretation across features

### Step 5: Train/Test Splitting

**Strategy:**
- **Classification:** Stratified split (maintains class distribution)
- **Regression:** Random split
- **Ratio:** 80% training, 20% testing
- **Random Seed:** 42 (for reproducibility)

**Implementation:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y if classification else None,
    random_state=42
)
```

---

## Train/Validation/Test Split

### Split Strategy

This project uses a **two-way split** (train/test only), as validation is primarily used for hyperparameter tuning, which is not the focus of this research.

**Split Ratios:**
- **Training Set:** 80% - Used for training black-box models and computing training SHAP values
- **Test Set:** 20% - Used for final evaluation and computing test SHAP values

**No Validation Set Because:**
1. Hyperparameters are pre-defined (from literature)
2. Focus is on methodology validation, not model optimization
3. Simplifies reproducibility across datasets

### SHAP Computation Subsets

Due to computational cost, not all training/test samples are used for SHAP computation:

**Training SHAP:**
- **Samples Used:** 1,000 (randomly sampled from training set)
- **Purpose:** Train GAM surrogates
- **Rationale:** Sufficient for learning SHAP patterns

**Test SHAP:**
- **Samples Used:** 500 (randomly sampled from test set)
- **Purpose:** Evaluate surrogate accuracy
- **Rationale:** Balance between evaluation robustness and computation time

**Background Dataset:**
- **Samples Used:** 100 (randomly sampled from training set)
- **Purpose:** SHAP explainer background distribution
- **Rationale:** Sufficient for KernelSHAP approximation

### Data Flow Diagram

```
Full Dataset (N samples)
    |
    |-- 80% Train (0.8N samples)
    |      |
    |      |-- Random 1,000 samples → Compute Exact SHAP → Train GAM Surrogates
    |      |-- Random 100 samples → Background Dataset for SHAP
    |      |-- All samples → Train Black-Box Model
    |
    |-- 20% Test (0.2N samples)
           |
           |-- Random 500 samples → Compute Exact SHAP → Evaluate GAM Surrogates
           |-- All samples → Evaluate Black-Box Model
```

---

## Data Statistics

### Summary Table

| Metric | California Housing | Breast Cancer | Adult Income |
|--------|-------------------|---------------|--------------|
| **Training Samples** | 16,512 | 455 | ~39,074 |
| **Test Samples** | 4,128 | 114 | ~9,768 |
| **Total Features** | 8 | 30 | 14 |
| **Numeric Features** | 8 | 30 | 6 |
| **Categorical Features** | 0 | 0 | 8 |
| **Missing Values** | 0% | 0% | ~7.4% (dropped) |
| **Class Balance (if classification)** | N/A | 1.68:1 | 3.13:1 |
| **Target Type** | Continuous | Binary | Binary |
| **Feature Correlations** | Moderate | High (by design) | Low-Moderate |

### Feature Importance (from Black-Box Models)

**Top 3 Most Important Features per Dataset:**

1. **California Housing:**
   - MedInc (median income) - 45-50% importance
   - Latitude - 15-20%
   - Longitude - 10-15%

2. **Breast Cancer:**
   - worst_perimeter - 20-25%
   - worst_concave_points - 15-20%
   - worst_radius - 10-15%

3. **Adult Income:**
   - capital-gain - 30-35%
   - education-num - 15-20%
   - age - 10-15%

---

## Data Quality and Limitations

### Strengths

1. **Well-Curated:** Standard benchmark datasets with known properties
2. **Complete Documentation:** Extensive literature and references
3. **Reproducible:** Consistent access through sklearn/OpenML
4. **Diverse:** Covers different domains and problem types
5. **Real-World:** Based on actual census and medical data

### Limitations

1. **Temporal:** Data is from 1990s (Adult, Housing) or 1995 (Breast Cancer)
2. **Geographic:** Limited to U.S. (housing, census) or Wisconsin (breast cancer)
3. **Scale:** Small to medium datasets (not "big data")
4. **Simplicity:** Relatively few features compared to modern datasets
5. **Privacy:** Some datasets (Adult) contain sensitive attributes

### Recommendations for Extension

**To Scale to Larger Datasets:**
1. Use sampling for SHAP computation (already implemented)
2. Parallelize GAM training across features
3. Use incremental learning for surrogates
4. Consider distributed computing frameworks

**To Handle More Complex Data:**
1. Add support for text features (NLP preprocessing)
2. Add support for image features (embeddings)
3. Implement one-hot encoding option for high-cardinality categoricals
4. Add time-series preprocessing

---

## Data Access Code Examples

### Example 1: Load California Housing

```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader('california_housing')

# Get preprocessed data
X_train, X_test, y_train, y_test, feature_names = loader.get_data()

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Feature names: {feature_names}")
```

### Example 2: Load with Custom Split

```python
from src.data_loader import DataLoader

# Custom test size
loader = DataLoader('breast_cancer', test_size=0.3)
X_train, X_test, y_train, y_test, feature_names = loader.get_data()
```

### Example 3: Access Raw Data (Before Preprocessing)

```python
from sklearn.datasets import fetch_california_housing

# Load raw data
data = fetch_california_housing(as_frame=True)
df = data.frame

print(df.head())
print(df.describe())
```

---

## Data Storage Structure

```
data/
├── california_housing/
│   ├── raw/                    # Raw downloaded data (cached)
│   ├── processed/              # Preprocessed data (scaled, split)
│   └── shap_values/            # Cached SHAP computations
│       ├── train_shap.npy
│       └── test_shap.npy
├── breast_cancer/
│   └── [same structure]
└── adult/
    └── [same structure]
```

**Caching Benefits:**
- Avoid re-downloading datasets
- Avoid re-computing expensive SHAP values
- Faster experiment iteration

---

**Last Updated:** March 2026  
**Version:** 1.0.0
