# 📁 02 — Development Prompts

> **Total Prompts in this file:** 30
> **Category:** Feature Implementation, Code Writing, Refactoring, Code Explanation, Readability



### Prompt 1: Building a Tabular MLP from Scratch

**Goal:**
Implement a configurable Multi-Layer Perceptron for tabular data using PyTorch.

**Prompt:**
```
I need to build a TabularMLP class in PyTorch for tabular data classification/regression. Requirements:
- Accept configurable parameters: input_dim, hidden_dims (list of ints), output_dim, dropout rate
- Use ReLU activation between hidden layers
- Apply dropout after each hidden layer
- Support both regression (output_dim=1) and classification (output_dim=num_classes)
- The model should NOT include the final activation (no sigmoid/softmax) — that should be handled by the loss function

Write the class step by step:
1. First show the __init__ with layer construction
2. Then show the forward method
3. Add type hints and docstrings
4. Explain why we build layers using nn.Sequential or a loop
5. Explain why the final layer has no activation or dropout

After the implementation, show me how to instantiate it for:
- A regression task with 45 input features, hidden=[256,128], output=1
- A 7-class classification task with 11 features, hidden=[256,128], output=7
```

**Why this is useful:**
Building a model from scratch teaches PyTorch fundamentals far better than copying a snippet.

**Example Use Case:**
The `blackbox_model.py` in the project uses this exact pattern for the black-box MLP baseline.

---

### Prompt 2: Implementing a GAM as a Neural Network

**Goal:**
Build an additive model where each feature has its own subnetwork.

**Prompt:**
```
I need to implement a GAM (Generalized Additive Model) as a PyTorch neural network.

Architecture:
- For each of N features, create a separate small MLP subnetwork (e.g., hidden_dims=[96,64])
- Each subnetwork i takes ONLY feature i as input and produces a scalar output
- The final prediction is the SUM of all subnetwork outputs (plus an optional bias term)
- For GAM-2: also include pairwise interaction subnetworks that take 2 features as input

Implement a GAMModel class with:
1. __init__ that creates nn.ModuleList of subnetworks
2. forward() that feeds each feature to its subnetwork and sums the outputs
3. A method to extract individual shape functions: get_shape_function(feature_idx, x_values)
4. Support for both GAM-1 (univariate only) and GAM-2 (with interaction pairs)

Explain:
- Why nn.ModuleList is necessary (not a regular Python list)
- How the additive structure makes the model interpretable
- How shape functions relate to SHAP values
- How to specify which pairs to include for GAM-2
```

**Why this is useful:**
Implementing a GAM teaches you about model architecture, interpretability, and the connection between structure and explainability.

---

### Prompt 3: Writing a Training Loop with Progress Bars and Logging

**Goal:**
Implement a professional training loop, not a minimal one.

**Prompt:**
```
I need a production-quality training loop for my PyTorch models. The current loop is basic:
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

Improve it to include:
1. tqdm progress bar showing epoch, batch, loss, and learning rate
2. Validation evaluation after each epoch
3. Early stopping based on validation loss (with configurable patience)
4. Best model checkpoint saving (in memory, not to disk)
5. Structured logging of epoch results
6. Proper model.train() / model.eval() switching
7. torch.no_grad() for validation
8. Return a history dict with training curves (train_loss, val_loss per epoch)
9. Optional TensorBoard logging

Write this as a reusable function:
train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device, logger)

Explain WHY each improvement matters, not just what it does.
```

**Why this is useful:**
The training loop is where most ML bugs hide. A robust loop prevents subtle errors.

---

### Prompt 4: Implementing Dataset Loaders with Error Handling

**Goal:**
Write robust data loading functions that handle real-world issues.

**Prompt:**
```
I'm loading datasets from the UCI ML Repository using the `ucimlrepo` library.

Here's a basic version:
from ucimlrepo import fetch_ucirepo
data = fetch_ucirepo(id=275)
X = data.data.features
y = data.data.targets

This is fragile. Improve the loader to:
1. Handle API failures gracefully (try/except with informative error messages)
2. Handle missing values (detect and report, then impute or drop)
3. Log the dataset shape, feature types, and target distribution
4. Validate that the data looks correct (expected number of features, no NaN in targets)
5. Support a max_rows parameter to subsample large datasets
6. Return a clean (X, y, feature_names, categorical_columns) tuple
7. Add a fast_dev_run mode that loads only 500 rows

Write the function for the Bike Sharing dataset (UCI ID 275, regression).
Then explain how to generalize it for classification datasets (Covertype, Adult).
Show me what error handling I should never skip.
```

**Why this is useful:**
Real data is messy. Robust loaders save hours of debugging mysterious downstream errors.

---

### Prompt 5: Building the TabularPreprocessor Class

**Goal:**
Implement a reusable preprocessing pipeline for mixed-type tabular data.

**Prompt:**
```
I need a TabularPreprocessor class that:
1. Identifies numeric vs. categorical columns automatically
2. Standard-scales numeric columns (fit on train, transform on val/test)
3. One-hot encodes categorical columns
4. Tracks feature groups: which output columns map to which original feature
5. Converts pandas DataFrames to PyTorch tensors
6. Has fit(X_train), transform(X), and fit_transform(X_train) methods
7. Stores the fitted scalers/encoders so they can be applied to new data

Important design decisions to explain:
- Why fit ONLY on training data and transform all sets?
- Why use StandardScaler (zero mean, unit variance) instead of MinMaxScaler?
- How to handle unknown categories in the test set
- How to return feature_groups as a dict like {"temp": [0], "season": [1,2,3,4]}
- Error handling: what if a column has only one unique value?

Show the full class with type hints and docstrings.
```

**Why this is useful:**
The preprocessor is used by every experiment. Getting it right prevents data leakage and ensures consistent transformations.

---

### Prompt 6: Implementing SHAP Value Computation Wrapper

**Goal:**
Wrap the SHAP library for permutation-based explanations with feature group aggregation.

**Prompt:**
```
I need a SHAP wrapper that:
1. Takes a trained PyTorch model and creates a SHAP-compatible predict function
2. Uses PermutationExplainer (not KernelExplainer) for model-agnostic explanations
3. Handles the model being on GPU (moves data to CPU for SHAP, result back for model)
4. Aggregates SHAP values for one-hot encoded features using feature_groups
5. Returns numpy arrays of SHAP values with one column per original feature

Key challenges to address:
- How to wrap a PyTorch model as a callable for SHAP (model.eval(), torch.no_grad())
- Why SHAP needs a background dataset and how to choose its size (64 samples)
- How to limit eval_samples and max_evals for faster computation
- How to aggregate: for feature group "season" with columns [1,2,3,4], sum their absolute SHAP values
- How to handle classification output (multi-class SHAP has shape differences)

Show the implementation of:
compute_shap_values(model, X_background, X_eval, feature_groups, device, config)

Explain each parameter and return value explicitly.
```

**Why this is useful:**
The SHAP wrapper bridges the gap between PyTorch models and the SHAP library, which operates on numpy arrays.

---

### Prompt 7: Implementing the InstaSHAP Explainer (Single-Pass SHAP)

**Goal:**
Build the key contribution of the project — instant SHAP values from a single forward pass.

**Prompt:**
```
The InstaSHAP model is a GAM trained with a masked objective. After training, SHAP values are extracted in a single forward pass.

Explain and implement the InstaSHAP explainer:
1. The InstaSHAP model structure: additive model with subnetworks per feature
2. After training, each subnetwork's output for feature i IS the SHAP value for feature i
3. The explainer just runs a forward pass and returns each subnetwork's output separately
4. No need for expensive permutation sampling

Implement:
class InstaSHAPExplainer:
    def __init__(self, model: InstaSHAPModel):
        ...
    
    def explain(self, X: torch.Tensor) -> np.ndarray:
        """Return SHAP values for each sample. Shape: (n_samples, n_features)"""
        ...

Explain:
- Why this is orders of magnitude faster than permutation SHAP
- How the masked training objective ensures the subnetwork outputs approximate true SHAP values
- What accuracy trade-offs exist (approximate vs. exact SHAP)
- How to validate the InstaSHAP explanations against permutation SHAP
```

**Why this is useful:**
This is the core research contribution. Understanding it deeply is essential for the project and for learning about XAI.

---

### Prompt 8: Implementing the Masked Forward Pass

**Goal:**
Code the feature masking mechanism used in surrogate and InstaSHAP training.

**Prompt:**
```
During training of the masked surrogate and InstaSHAP models, we need to:
1. Generate random binary masks (which features to include/exclude)
2. Apply masks to input features (zero out excluded features)
3. Forward the masked input through the black-box to get the target output
4. Train the surrogate/InstaSHAP to predict this masked output

Implement the masking logic:
1. Function to generate random binary masks: (batch_size, n_features) with configurable edge probability
2. What edge_mask_probability=0.10 means (10% chance each feature is included)
3. Why we need masks_per_sample=2 (multiple masks per data point per epoch)
4. How to apply the mask: X_masked = X * mask (element-wise)
5. How to handle feature groups in masking (mask the entire group, not individual one-hot columns)

Then show how this fits into the training loop:
for X_batch in dataloader:
    masks = generate_masks(X_batch.shape[0], n_features, edge_prob)
    X_masked = apply_mask(X_batch, masks, feature_groups)
    target = blackbox_model(X_masked)
    pred = surrogate_model(X_masked)
    loss = criterion(pred, target)

Explain the intuition: why training on masked inputs teaches the model about feature contributions.
```

**Why this is useful:**
The masking mechanism is mathematically subtle. Implementing it correctly is essential for the InstaSHAP method to work.

---

### Prompt 9: Writing Evaluation Functions for Regression and Classification

**Goal:**
Implement comprehensive model evaluation with proper metrics.

**Prompt:**
```
I need evaluation functions for both regression and classification tasks.

For regression (e.g., Bike Sharing):
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (coefficient of determination)

For classification (e.g., Covertype, Adult):
- Accuracy
- F1 score (macro and weighted)
- Precision and Recall (macro)
- Confusion matrix

Implement:
1. evaluate_regression(y_true, y_pred) -> dict of metrics
2. evaluate_classification(y_true, y_pred, n_classes) -> dict of metrics
3. A unified evaluate(y_true, y_pred, task_type) dispatcher

Also:
- explain_error(y_true_shap, y_pred_shap) -> MSE and correlation between SHAP and InstaSHAP values

For each metric:
- Write a one-line explanation of what it measures
- When is each metric most useful?
- Which metric is the "primary" one for each task type?

Use scikit-learn's implementations where possible.
Show proper handling of multi-class edge cases.
```

**Why this is useful:**
Understanding metrics deeply prevents reporting misleading results — a critical research skill.

---

### Prompt 10: Implementing Training Curve Visualization

**Goal:**
Create publication-quality training curve plots.

**Prompt:**
```
After training, I have a history dictionary:
history = {
    "train_loss": [0.85, 0.62, 0.45, ...],
    "val_loss": [0.90, 0.68, 0.52, ...],
    "epochs": [1, 2, 3, ...]
}

Create a function plot_training_curves(history, title, save_path) that:
1. Plots train and val loss on the same axes
2. Uses different colors and line styles for train vs val
3. Marks the best epoch (lowest val_loss) with a vertical dashed line and annotation
4. Adds proper labels, title, legend, and grid
5. Uses a clean, publication-ready style (seaborn's "whitegrid" or similar)
6. Saves the figure as PNG at 150 DPI
7. Closes the figure after saving (matplotlib memory leak prevention)

Also explain:
- How to interpret training curves (overfitting, underfitting, good fit)
- What it means if train_loss keeps decreasing but val_loss plateaus
- How to tell if you need more epochs or more regularization
- Why plt.close() is important in scripts that generate many plots
```

**Why this is useful:**
Training curves are the primary diagnostic tool for ML. Creating professional plots is a transferable skill.

---

### Prompt 11: Implementing Shape Function Visualization

**Goal:**
Plot how individual features contribute to predictions in a GAM.

**Prompt:**
```
My GAM model has individual subnetworks for each feature. A "shape function" shows how the subnetwork's output varies as the feature value changes.

Implement plot_shape_functions(model, feature_names, X_sample, save_path):
1. For each feature, sort X_sample by that feature's values
2. Feed sorted values through the corresponding subnetwork
3. Plot feature value (x-axis) vs. subnetwork output (y-axis)
4. Create a grid of subplots (e.g., 3 columns, as many rows as needed)
5. Add feature name as subplot title
6. Add a horizontal line at y=0 (baseline reference)
7. Use consistent y-axis scaling across subplots for comparison

Questions to address:
- How do shape functions relate to partial dependence plots?
- What does a monotonically increasing shape function mean?
- What does a flat shape function tell us?
- How to handle categorical features (bar plot instead of line?)
- Why is this possible with a GAM but not with a standard MLP?
```

**Why this is useful:**
Shape functions are the key visualization for additive models and directly demonstrate interpretability.

---

### Prompt 12: Building the Interaction Heatmap Visualization

**Goal:**
Visualize pairwise feature interactions in GAM-2 models.

**Prompt:**
```
My GAM-2 model has pairwise interaction components (e.g., hour × workingday).
I want to create heatmaps showing how two features jointly affect predictions.

Implement plot_interaction_heatmap(model, feature_i, feature_j, X_sample, save_path):
1. Create a grid of (feature_i_values, feature_j_values)
2. Feed each grid point through the interaction subnetwork
3. Plot as a 2D heatmap using seaborn.heatmap or matplotlib.pcolormesh
4. Add proper axis labels with feature names
5. Add a colorbar showing the contribution magnitude
6. Use a diverging colormap (e.g., RdBu_r) centered at zero

Explain:
- What does a synergistic interaction look like on the heatmap?
- What does a redundant interaction look like?
- Why are interaction heatmaps important for the InstaSHAP paper's experiments?
- How to interpret the Bike Sharing "hour × workingday" interaction
  (e.g., rush hours matter more on working days)
```

**Why this is useful:**
Feature interaction analysis is central to the InstaSHAP paper's experiments and is a valuable XAI skill.

---

### Prompt 13: Refactoring Repeated Training Code

**Goal:**
Identify and eliminate code duplication in training functions.

**Prompt:**
```
I have four training functions with very similar structure:
- train_blackbox(model, train_loader, val_loader, config, ...)
- train_surrogate(model, blackbox, train_loader, val_loader, config, ...)
- train_gam(model, train_loader, val_loader, config, ...)
- train_instashap(model, surrogate, train_loader, val_loader, config, ...)

The common pattern is:
- Set up optimizer and loss function
- Training loop with forward pass, backward pass, optimizer step
- Validation evaluation after each epoch
- Early stopping
- Learning rate scheduling
- Return training history

The differences are:
- Surrogate and InstaSHAP need masked inputs and a reference model for targets
- Different models use different config sections for hyperparameters
- The loss function varies (MSE for regression, CrossEntropy for classification)

Help me refactor this:
1. Extract the common training loop into a base function
2. Use callbacks or strategy pattern for the model-specific parts
3. Keep the code readable — don't over-engineer
4. Show before/after for one function

Explain: when is duplication acceptable and when must it be removed?
```

**Why this is useful:**
Refactoring is a core development skill. It teaches you to balance DRY principles with readability.

---

### Prompt 14: Type Hinting a Python Module

**Goal:**
Add type hints to a module to improve code quality and IDE support.

**Prompt:**
```
Here's a Python function from my project without type hints:

def evaluate_model(model, dataloader, criterion, device, task_type):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            all_preds.append(output.cpu())
            all_targets.append(y_batch.cpu())
    ...

Add comprehensive type hints:
1. Function parameters and return type
2. Use Union, Optional, List from typing where appropriate
3. Should model be typed as nn.Module or a custom protocol?
4. How to type hint torch.Tensor vs numpy.ndarray
5. How to type hint the return dict (TypedDict or plain dict?)
6. When are type hints worth the effort in a research project?
7. How do type hints help with debugging and IDE autocomplete?

Show the fully typed version and explain each annotation.
```

**Why this is useful:**
Type hints catch bugs before runtime and make code self-documenting — critical for collaborative research.

---

### Prompt 15: Implementing Metrics Comparison Tables

**Goal:**
Generate formatted tables comparing model performance against paper-reported results.

**Prompt:**
```
After running experiments, I have metrics like:
{
    "bike": {"blackbox_rmse": 45.2, "gam1_rmse": 48.1, "instashap_rmse": 47.8},
    "covertype": {"blackbox_acc": 0.89, "gam1_acc": 0.85, "instashap_acc": 0.84}
}

And the paper reports:
{
    "bike": {"paper_blackbox_rmse": 43.5, "paper_gam1_rmse": 46.0},
    "covertype": {"paper_blackbox_acc": 0.91, "paper_gam1_acc": 0.87}
}

Implement generate_comparison_table(reproduced, paper_values, save_path):
1. Create a pandas DataFrame with columns: Model, Metric, Our Result, Paper Result, Difference
2. Calculate absolute and percentage differences
3. Highlight results that are within 5% of paper values (acceptable reproduction)
4. Flag results that differ by more than 10% (needs investigation)
5. Save as CSV and return formatted console output
6. Handle missing paper values gracefully (show "N/A")

Explain:
- Why exact reproduction is often impossible (hardware, library versions, unreported details)
- What constitutes a "successful" reproduction
- How to present these results in your report honestly
```

**Why this is useful:**
Comparing against paper results is the core validation of a reproducibility project. Honest reporting is an ethical imperative.

---

### Prompt 16: Creating DataLoaders for PyTorch

**Goal:**
Convert preprocessed data into PyTorch DataLoaders with proper batching.

**Prompt:**
```
After preprocessing, I have numpy arrays: X_train, y_train, X_val, y_val, X_test, y_test.

Show me how to:
1. Convert numpy arrays to PyTorch tensors (correct dtypes: float32 for X, appropriate for y)
2. Create TensorDataset from (X, y) pairs
3. Create DataLoaders with:
   - Configurable batch_size (from config.yaml)
   - Shuffling only for training data
   - num_workers=0 (for Windows compatibility)
   - drop_last=False (keep partial final batch)
4. Handle different y dtypes for regression (float32) vs classification (long/int64)

Explain:
- Why float32 for features, not float64? (Memory + GPU performance)
- Why shuffle training data but not validation/test?
- What does num_workers do and why 0 on Windows?
- What is pin_memory and when should I use it?
- How batch_size affects training (memory, gradient noise, convergence speed)

Show the complete data preparation pipeline from numpy to DataLoader.
```

**Why this is useful:**
Data type and DataLoader misconfigurations are among the most common PyTorch errors students encounter.

---

### Prompt 17: Implementing the Experiment Runner for a Dataset

**Goal:**
Wire together all components for a complete dataset experiment.

**Prompt:**
```
I need to implement a run(config, selected_model) function for the Bike Sharing experiment.

The function should:
1. Load the Bike Sharing dataset from UCI
2. Preprocess: scale numerics, one-hot encode categoricals
3. Split into train/val/test
4. Create DataLoaders
5. Train each selected model:
   a. Black-box MLP → evaluate predictions
   b. Masked Surrogate → train to mimic black-box on masked inputs
   c. GAM (1 & 2) → train and extract shape functions
   d. InstaSHAP → train with surrogate targets and extract instant SHAP
6. Compute permutation SHAP on black-box
7. Compare SHAP vs InstaSHAP (MSE, correlation)
8. Generate all plots (training curves, shape functions, heatmaps)
9. Save metrics to CSV tables
10. Return a summary object with paths to all outputs

Show the function step by step. After each step, explain:
- What can go wrong
- How to validate the step succeeded
- What to log for debugging

Keep the function under 100 lines by delegating to helper functions.
```

**Why this is useful:**
The experiment runner is the heart of the project. Well-structured orchestration code makes experiments reproducible and debuggable.

---

### Prompt 18: Implementing PDF Report Generation

**Goal:**
Create a multi-page PDF report summarizing experiment results.

**Prompt:**
```
I need a Python function that generates a multi-page PDF report containing:
- Title page with project name, date, and configuration used
- For each dataset: metrics table, training curves, shape function plots
- SHAP vs InstaSHAP comparison section with correlation plots
- Summary section with key findings

Options for PDF generation:
1. matplotlib's PdfPages (direct from plots)
2. reportlab (programmatic PDF creation)
3. LaTeX via subprocess (high quality but needs pdflatex)
4. WeasyPrint (HTML to PDF)

For each option:
- Pros and cons
- When to use it
- Does it need extra system dependencies?

Then implement the report using matplotlib PdfPages (most portable):
generate_full_report(results_dir, output_path):
1. Create a PdfPages context
2. Add a title page
3. For each dataset: add a results page with embedded plots and text
4. Save and close

Show how to add text, tables, and images to matplotlib figures for report pages.
```

**Why this is useful:**
Generating polished reports programmatically is a valuable skill for ML research and industry data science.

---

### Prompt 19: Implementing the One-Page Summary

**Goal:**
Create a condensed one-page summary of all results.

**Prompt:**
```
I need a one-page summary PDF that fits on a single sheet and contains:
- Project title and date
- A small table with key metrics for all datasets and models
- One or two key plots (e.g., SHAP vs InstaSHAP correlation)
- A brief text summary of findings (2-3 sentences)

Implement generate_one_page_summary(results_dir, output_path) using matplotlib:
1. Create a single figure with a specific size (8.5 x 11 inches) and tight layout
2. Use subplots and gridspec for layout control
3. Embed metrics as a table using ax.table()
4. Embed a plot using ax.imshow() or by re-plotting
5. Add text annotations for the summary

Explain:
- How to fit everything without overlapping
- How to adjust font sizes for readability
- How to handle cases where some datasets haven't been run yet
- Tips for making the summary look professional (consistent spacing, alignment)
```

**Why this is useful:**
Stakeholders (professors, reviewers) often want a quick overview. A well-designed one-pager is extremely valuable.

---

### Prompt 20: Implementing Explanation Comparison Logic

**Goal:**
Build the comparison between permutation SHAP and InstaSHAP explanations.

**Prompt:**
```
I have two sets of feature attributions for the same test samples:
- shap_values: (n_samples, n_features) from PermutationExplainer
- instashap_values: (n_samples, n_features) from InstaSHAP's single-pass explainer

Implement compare_explanations(shap_values, instashap_values, feature_names):
1. Per-feature MSE: how well does InstaSHAP approximate SHAP for each feature?
2. Overall MSE: average across all features and samples
3. Per-feature Pearson correlation: rank agreement for each feature
4. Overall correlation: global agreement
5. Top-K agreement: for the top-K most important features per sample, what percentage overlap?
6. Generate a comparison DataFrame with these metrics per feature

Also implement a bar chart showing per-feature InstaSHAP vs SHAP correlation.

Explain:
- What's a "good" correlation? (>0.9 is strong, >0.95 is excellent)
- When MSE matters vs when rank correlation matters
- Why InstaSHAP might disagree with SHAP on redundant features
- How to interpret cases where one feature has high MSE but high correlation
```

**Why this is useful:**
The SHAP vs InstaSHAP comparison is the key result of the project. Getting the comparison metrics right is essential.

---

### Prompt 21: Implementing the Masked Surrogate Training Loop

**Goal:**
Code the specialized training loop for the masked surrogate model.

**Prompt:**
```
The masked surrogate training is different from standard training:
- Input: masked version of X (random binary mask applied)
- Target: black-box model's output on the SAME masked input (not ground truth labels)
- Multiple masks per sample per epoch (masks_per_sample: 2)

Implement train_surrogate(surrogate, blackbox, train_loader, config, device):
1. Generate random masks for each batch
2. Apply masks to input features
3. Get black-box predictions on masked inputs (detached — no gradients through blackbox)
4. Train surrogate to predict black-box outputs
5. Handle edge_mask_probability correctly
6. Track training and validation loss

Key implementation details:
- blackbox.eval() with torch.no_grad() when generating targets
- Why we detach() the black-box outputs (we don't want to update the blackbox)
- How to repeat each sample for masks_per_sample (expand the batch)
- How masks work at the feature group level for one-hot encoded features

Show the complete training loop with comments explaining each step.
```

**Why this is useful:**
The surrogate training has non-obvious details (detaching targets, feature group masking) that are easy to get wrong.

---

### Prompt 22: Understanding and Implementing Random Forest as Black-Box

**Goal:**
Add an alternative black-box model for comparison.

**Prompt:**
```
My project uses an MLP as the black-box model, but I also want to support Random Forest as an alternative. This is useful because:
1. RF doesn't need GPU or training loops
2. It provides a different inductive bias (ensemble of trees vs neural network)
3. It can serve as a sanity check for SHAP values

Implement a RandomForestBlackBox wrapper class that:
1. Has the same interface as TabularMLP: fit(X_train, y_train), predict(X) 
2. Uses scikit-learn's RandomForestClassifier/Regressor
3. Automatically selects classifier vs regressor based on task type
4. Wraps predict() to return torch tensors (for compatibility with rest of pipeline)
5. Has a SHAP-compatible predict function (numpy in, numpy out)

Explain:
- Why does RF work well without preprocessing (no scaling needed)?
- When would RF outperform MLP on tabular data?
- How does feature importance from RF compare to SHAP importance?
- Design pattern: how to make MLP and RF interchangeable (duck typing vs ABC)
```

**Why this is useful:**
Supporting multiple model types teaches good software design and provides scientific comparison.

---

### Prompt 23: Writing Docstrings Following NumPy Convention

**Goal:**
Add comprehensive documentation to functions and classes.

**Prompt:**
```
My project functions lack proper docstrings. Teach me the NumPy docstring convention and rewrite this function with a proper docstring:

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device, logger=None):
    # ... training code ...
    return history

The docstring should include:
1. One-line summary
2. Extended description if needed
3. Parameters section with name, type, and description for each
4. Returns section with type and description
5. Raises section for possible exceptions
6. Examples section showing basic usage
7. Notes section for implementation details

Show both the docstring and explain:
- Why NumPy convention over Google convention?
- How docstrings enable IDE tooltips and auto-generated documentation
- What level of detail is appropriate (too much vs too little)
- Should I document private/internal functions the same way?
```

**Why this is useful:**
Good docstrings make code self-documenting and are essential for collaborative projects.

---

### Prompt 24: Implementing Stratified Data Splitting

**Goal:**
Correctly split classification datasets while preserving class distribution.

**Prompt:**
```
For classification datasets (Covertype with 7 classes, Adult with 2 classes), I need stratified splitting.

My current code:
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

Improvements needed:
1. Add stratify=y for classification tasks
2. Explain why 0.125 for val_size (0.10 of total from the remaining 0.80)
3. Handle the edge case where a class has very few samples (stratification fails)
4. Verify the splits worked: log actual class distributions in each set
5. Make this work for regression too (no stratification, or binned stratification)

Implement a generic split_data(X, y, test_size, val_size, task_type, seed) function.

Explain common mistakes:
- Using stratification for regression (usually wrong)
- Forgetting to pass the same random_state to both splits
- Not checking that resulting set sizes are reasonable
```

**Why this is useful:**
Incorrect splitting is a subtle but serious bug. Unstratified classification splits can create biased training sets.

---

### Prompt 25: Building a Configuration-Driven Training Pipeline

**Goal:**
Wire config.yaml values into training functions cleanly.

**Prompt:**
```
My config.yaml has per-model training settings:
training:
  blackbox:
    hidden_dims: [256, 128]
    dropout: 0.10
    lr: 0.001
    epochs: 25
    patience: 5
  gam:
    hidden_dims: [96, 64]
    dropout: 0.05
    lr: 0.001
    epochs: 35
    patience: 6

Currently, I'm extracting values manually:
bb_config = config["training"]["blackbox"]
model = TabularMLP(input_dim, bb_config["hidden_dims"], output_dim, bb_config["dropout"])
optimizer = torch.optim.Adam(model.parameters(), lr=bb_config["lr"])

This works but is verbose and error-prone. Help me:
1. Create a helper function that extracts model config: get_model_config(config, model_type)
2. Create a builder function: build_model(model_type, input_dim, output_dim, config)
3. Handle missing config keys with sensible defaults
4. Validate config values (e.g., dropout must be between 0 and 1, epochs must be positive)
5. Log the configuration being used for each training run

Explain: when is this level of abstraction worth it vs. just accessing the dict?
```

**Why this is useful:**
Configuration-driven code reduces hard-coded magic numbers and makes experiments reproducible.

---

### Prompt 26: Implementing Loss Functions for Different Tasks

**Goal:**
Choose and configure the right loss function for each task type.

**Prompt:**
```
My project handles regression and classification tasks. I need to set up loss functions correctly.

Explain and show code for:
1. Regression: nn.MSELoss — why MSE and not MAE?
2. Binary classification: nn.BCEWithLogitsLoss — why logits, not probabilities?
3. Multi-class classification: nn.CrossEntropyLoss — why does it expect raw logits?
4. How to create the right loss function based on task_type and output_dim

Build a function get_criterion(task_type, output_dim):
- "regression" → nn.MSELoss()
- "binary_classification" → nn.BCEWithLogitsLoss()
- "classification" → nn.CrossEntropyLoss()

Explain:
- Why not apply sigmoid/softmax in the model when using these losses?
- How CrossEntropyLoss combines log_softmax + NLLLoss
- Target shape and dtype requirements for each loss
- The difference between reduction='mean' and reduction='sum'
- Common error: "Expected Float but got Long" — what causes it and how to fix
```

**Why this is useful:**
Loss function misconfiguration is one of the top causes of training failures and confusing error messages.

---

### Prompt 27: Implementing Progress Tracking with tqdm

**Goal:**
Add informative progress bars to long-running operations.

**Prompt:**
```
My training loops and SHAP computations can take minutes. I want to add tqdm progress bars.

Show me how to use tqdm in these contexts:
1. Epoch loop: showing current epoch, total epochs, latest train/val loss
2. Batch loop within an epoch: showing batch progress with loss
3. Dataset loading: showing download/preprocessing progress
4. SHAP computation: showing progress over eval samples
5. Multi-dataset experiment: showing which dataset is being processed

Best practices:
- How to nest tqdm bars (epoch + batch) without messy output
- How to update the description mid-loop: pbar.set_postfix(loss=0.42)
- How to use tqdm.write() instead of print() to avoid display conflicts
- The leave=False parameter for inner loops
- How to conditionally disable tqdm (for non-interactive environments)

Show a complete example for a training loop with nested progress bars.
```

**Why this is useful:**
Good progress indication reduces anxiety about long-running processes and helps spot stuck computations.

---

### Prompt 28: Implementing JSON Artifact Saving

**Goal:**
Save experiment results as structured JSON for later analysis.

**Prompt:**
```
After running experiments, I want to save structured artifacts as JSON:
{
    "dataset": "bike",
    "timestamp": "2024-03-15T10:30:00",
    "config": { ... },
    "models": {
        "blackbox": {"rmse": 45.2, "r2": 0.92, "training_time": 120.5},
        "gam1": {"rmse": 48.1, "r2": 0.89},
        "instashap": {"rmse": 47.8, "r2": 0.88}
    },
    "explanation_comparison": {
        "mse": 0.023,
        "correlation": 0.94,
        "per_feature": { ... }
    }
}

Implement:
1. save_json(data, path) — with proper datetime serialization
2. load_json(path) — with error handling
3. A results dataclass or dict schema that standardizes what gets saved
4. How to handle numpy types (np.float64 aren't JSON serializable by default)
5. How to make the JSON pretty-printed and human-readable

Show the custom JSONEncoder for numpy/datetime types.

Explain:
- Why JSON over pickle? (Portability, readability, safety)
- When to use JSON vs CSV vs Parquet for different data types
- How to include config snapshots in artifacts for full reproducibility
```

**Why this is useful:**
Structured artifact saving enables programmatic analysis and comparison of experiment results.

---

### Prompt 29: Implementing Learning Rate Scheduling

**Goal:**
Add learning rate warmup and decay to the training loop.

**Prompt:**
```
My training uses a constant learning rate (lr=0.001). I want to add scheduling:

1. ReduceLROnPlateau: lower LR when val_loss plateaus
2. CosineAnnealingLR: gradual decay following a cosine curve
3. One-cycle policy: warmup then decay

For my project (training 4 different models for 25-35 epochs):
1. Which scheduler is most appropriate and why?
2. How to integrate it into the existing training loop
3. Where does scheduler.step() go? (After each epoch for plateau, after each batch for one-cycle)
4. How to log the current learning rate for debugging
5. How to configure the scheduler from config.yaml

Implement the scheduler integration:
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
# Where does this go in the loop?

Explain the interaction between early stopping patience and LR scheduler patience.
```

**Why this is useful:**
Learning rate scheduling can significantly improve convergence and final model quality.

---

### Prompt 30: Code Review Checklist for ML Projects

**Goal:**
Learn to review your own code systematically before committing.

**Prompt:**
```
I've finished implementing a module in my ML project and want to self-review before committing.

Create a comprehensive code review checklist for ML Python projects:

1. **Correctness**:
   - Are tensor shapes correct at each computation step?
   - Is the loss function appropriate for the task?
   - Are gradients flowing where expected (no accidental detach/no_grad)?
   - Is data leakage prevented (preprocessing fit only on train)?

2. **Robustness**:
   - Are inputs validated?
   - Is error handling appropriate (not too broad, not too narrow)?
   - Are edge cases handled (empty datasets, single-class splits)?

3. **Code Quality**:
   - Type hints on all public functions?
   - Docstrings on all public functions?
   - No magic numbers (use config or named constants)?
   - Consistent naming (snake_case, descriptive names)?

4. **ML-Specific**:
   - model.train() before training, model.eval() before inference?
   - torch.no_grad() during evaluation?
   - Proper device management (all tensors on same device)?
   - Reproducibility controls (seeds set)?

5. **Performance**:
   - No unnecessary data copies (tensor.clone() vs tensor)?
   - Figures properly closed after saving?
   - Large tensors moved to CPU after use?

For each item, give an example of the bug it catches.
```

**Why this is useful:**
Self-code-review catches bugs before they become difficult to debug. It builds discipline for professional development.
