# 📁 03 — Debugging & Error Analysis Prompts



### Prompt 1: Decoding a PyTorch RuntimeError (Device Mismatch)

**Goal:**
Understand and fix the most common PyTorch error: tensors on different devices.

**Prompt:**
```
I'm getting this PyTorch error:

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

The stack trace points to this line in my training loop:
loss = criterion(model(X_batch), y_batch)

My code:
model = TabularMLP(...).to(device)  # device = cuda:0
for X_batch, y_batch in train_loader:
    output = model(X_batch)
    loss = criterion(output, y_batch)

Help me:
1. Explain exactly what this error means
2. Identify which tensor is on the wrong device
3. Show the fix (moving tensors to device before computation)
4. Explain why DataLoader doesn't automatically put data on GPU
5. Show a pattern to prevent this error everywhere
6. How to debug: print(X_batch.device, y_batch.device, next(model.parameters()).device)
```

**Why this is useful:**
This is the #1 most common PyTorch error. Understanding device management prevents hours of debugging.

**Example Use Case:**
In the InstaSHAP project, the training loop in `train.py` must move both X_batch and y_batch to the configured device before passing them through the model.

---

### Prompt 2: Debugging NaN Loss During Training

**Goal:**
Diagnose and fix the dreaded NaN loss issue.

**Prompt:**
```
My model's training loss becomes NaN after a few epochs:

Epoch 1: train_loss=0.85, val_loss=0.90
Epoch 2: train_loss=0.62, val_loss=0.68
Epoch 3: train_loss=nan, val_loss=nan

My setup:
- Model: TabularMLP with hidden_dims=[256, 128], dropout=0.10
- Optimizer: Adam, lr=0.001
- Loss: MSELoss for regression
- Data: Bike Sharing dataset, preprocessed with StandardScaler

Help me debug this systematically:
1. What are the common causes of NaN loss? (gradient explosion, bad data, bad LR, log(0), etc.)
2. How to add NaN checks: torch.isnan(loss).any()
3. How to check for extreme values in data: X.max(), X.min()
4. How to check gradient magnitudes: torch.nn.utils.clip_grad_norm_
5. Could my preprocessing be the problem? (missing values becoming NaN)
6. How to use torch.autograd.set_detect_anomaly(True) for detailed debugging
7. Step-by-step debugging procedure for NaN loss

Don't just tell me the answer — teach me the debugging methodology.
```

**Why this is useful:**
NaN debugging is a rite of passage in ML. The systematic approach here applies to any training issue.

---

### Prompt 3: Understanding ImportError and ModuleNotFoundError

**Goal:**
Fix Python import errors caused by project structure issues.

**Prompt:**
```
I'm getting this error when running my project:

ModuleNotFoundError: No module named 'instashap_project'

I run: python instashap_project/main.py
From the project root directory.

My project structure:
X_AI-Project/
└── instashap_project/
    ├── __init__.py
    ├── main.py
    ├── models/
    │   ├── __init__.py
    │   └── blackbox_model.py

And main.py has: from instashap_project.models.blackbox_model import TabularMLP

Help me understand:
1. Why this error occurs (Python's import system and sys.path)
2. The difference between running as a script vs as a module
3. How to fix it: python -m instashap_project.main
4. The alternative fix in main.py: sys.path manipulation
5. Why the __init__.py files are necessary
6. How relative vs absolute imports work in this structure
7. Best practice: which approach should I use?

Explain the Python import system conceptually, not just the fix.
```

**Why this is useful:**
Import errors are the #1 source of confusion for Python beginners. Understanding the import system prevents recurring issues.

---

### Prompt 4: Debugging Silent Data Leakage

**Goal:**
Detect and fix data leakage that inflates model performance.

**Prompt:**
```
My model has suspiciously high accuracy: 99.5% on test set for a task where the paper reports 91%.

I suspect data leakage. Help me investigate:

My preprocessing code:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit on ALL data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

1. What is data leakage and why is this code causing it?
2. Show the correct order: split FIRST, then fit scaler on train ONLY
3. What other forms of data leakage exist?
   - Feature leakage: including the target variable (or proxy) as a feature
   - Temporal leakage: using future data to predict the past
   - Preprocessing leakage: fitting encoders/scalers on all data
4. How to audit for data leakage systematically
5. What metrics patterns suggest leakage? (unrealistically high scores)
6. How to validate: train on dataset A, test on dataset B (should generalize)

This is critical — data leakage makes research results meaningless.
```

**Why this is useful:**
Data leakage is a silent bug that can invalidate entire experiments. Recognizing it is an essential ML skill.

---

### Prompt 5: Fixing Shape Mismatch Errors in Training

**Goal:**
Debug tensor shape mismatches that cause runtime errors.

**Prompt:**
```
I'm getting this error during training:

RuntimeError: output with shape [512, 7] doesn't match the broadcast shape [512, 1, 7]

Or sometimes:
RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x13 and 45x256)

My code:
output = model(X_batch)  # Shape: (batch_size, output_dim)
loss = criterion(output, y_batch)  # y_batch shape might be wrong

Help me debug shape issues:
1. How to systematically trace tensor shapes through the network
2. Common causes: wrong input_dim (not accounting for one-hot expansion)
3. The squeeze/unsqueeze problem: y having shape (N,) vs (N,1) for regression
4. How to check: print(f"X shape: {X_batch.shape}, y shape: {y_batch.shape}, output shape: {output.shape}")
5. How CrossEntropyLoss expects (N, C) predictions and (N,) targets (no one-hot for targets!)
6. How MSELoss expects matching shapes (both (N,1) or both (N,))
7. The debugging pattern: add shape assertions at key points

Then provide a shape debugging cheatsheet for common PyTorch operations.
```

**Why this is useful:**
Shape errors are the second most common PyTorch error. A systematic approach saves enormous debugging time.

---

### Prompt 6: Diagnosing Model Not Learning (Constant Loss)

**Goal:**
Figure out why a model's training loss isn't decreasing.

**Prompt:**
```
My model's loss stays constant during training:

Epoch 1: train_loss=2.31, val_loss=2.31
Epoch 2: train_loss=2.31, val_loss=2.31
...
Epoch 25: train_loss=2.31, val_loss=2.31

(Note: 2.31 ≈ log(10)/10 ≈ what random guessing gives for 10-class classification)

My setup:
- Model: GAMModel with hidden_dims=[96, 64]
- Optimizer: Adam(model.parameters(), lr=0.001)
- Loss: CrossEntropyLoss

Help me investigate step by step:
1. Is this the "random baseline" loss? (What loss values does random guessing produce?)
2. Check gradient flow: are any gradients zero?
   for name, param in model.named_parameters():
       print(name, param.grad.abs().mean() if param.grad is not None else "NO GRAD")
3. Check if all parameters are actually registered (model.parameters())
4. Check if the optimizer received all parameters
5. Common causes: forgetting to call loss.backward(), wrong criterion, learning rate too small
6. GAM-specific: are subnetworks properly connected in the computation graph?
7. Sanity check: can the model overfit a tiny dataset (10 samples)?

Walk me through the systematic debugging process.
```

**Why this is useful:**
A model that doesn't learn is useless. This systematic approach identifies the root cause efficiently.

---

### Prompt 7: Debugging SHAP Library Errors

**Goal:**
Fix common errors when using the SHAP library with PyTorch models.

**Prompt:**
```
I'm getting errors when computing SHAP values on my PyTorch model:

Error 1: TypeError: 'Tensor' object does not support item assignment
Error 2: ValueError: The model produced NaN values when applied to the background dataset
Error 3: The SHAP computation takes 30+ minutes for just 24 samples

My code:
def model_predict(X):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    return model(X_tensor).detach().numpy()

explainer = shap.PermutationExplainer(model_predict, X_background)
shap_values = explainer(X_eval)

Help me fix each issue:
1. Error 1: SHAP tries to modify the input — need to return numpy, not tensor
2. Error 2: Model produces NaN on some inputs — check model_predict wrapper
3. Slow computation: reduce max_evals, background size, or use batch prediction

Show the corrected model_predict wrapper:
- model.eval() and torch.no_grad()
- Handle CPU/GPU transfers properly  
- Return numpy array, not tensor
- Handle multi-class output shape
- Add error checking inside the wrapper

Also explain PermutationExplainer parameters: max_evals, seed, and their effect on runtime and accuracy.
```

**Why this is useful:**
SHAP with PyTorch is a common pain point. These specific errors affect many students.

---

### Prompt 8: Debugging Memory Errors (CUDA Out of Memory)

**Goal:**
Diagnose and fix GPU memory issues during training.

**Prompt:**
```
I'm getting: RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB

This happens when:
1. Training the black-box MLP on the Covertype dataset (60,000 rows)
2. Computing SHAP values (creates many forward passes)
3. Training GAM with many subnetworks

Help me debug and fix:
1. How to check current GPU memory usage: torch.cuda.memory_summary()
2. How to profile memory: torch.cuda.max_memory_allocated()
3. Quick fixes:
   - Reduce batch_size (512 → 256 → 128)
   - Reduce model size (hidden_dims=[256,128] → [128,64])
   - Use torch.no_grad() during evaluation
   - Delete unnecessary tensors and call torch.cuda.empty_cache()
4. Common memory leaks:
   - Storing loss tensors in a list (stores entire computation graph!)
   - Not detaching when accumulating metrics: total_loss += loss.item() (not loss)
   - Creating new tensors in a loop without cleanup
5. How to move computation to CPU when GPU memory is insufficient
6. The gradient accumulation pattern for effectively larger batch sizes

Show the correct and incorrect patterns side by side.
```

**Why this is useful:**
Memory management is critical for ML projects. One bad pattern can make an otherwise working project crash.

---

### Prompt 9: Root Cause Analysis — Wrong Predictions Despite Low Loss

**Goal:**
Investigate when metrics look good but actual predictions are wrong.

**Prompt:**
```
My model reports low training loss and high R² (0.92) on the test set, but when I look at actual predictions:
- The predictions are all clustered around the mean value
- The predicted range is [350, 450] while actual range is [0, 1000]
- The model seems to have learned the average, not the pattern

This is for the Bike Sharing regression task.

Help me investigate:
1. What does "predicting the mean" look like in terms of R² and MSE?
2. How can R² be high even with this behavior? (If variation is large)
3. Check: plot predicted vs actual values (scatter plot) — what should it look like?
4. Possible causes:
   - Target not properly scaled (very large values dominate loss)
   - Target variable has extreme skew
   - Model architecture too simple for the number of features
   - Learning rate too high (oscillating around mean)
5. How to diagnose: residual plots, prediction distribution plots
6. How to fix: check target preprocessing, try different loss functions

Show me the diagnostic code and visualization to identify this issue.
```

**Why this is useful:**
Numerical metrics can be misleading. Visual inspection of predictions catches problems that metrics miss.

---

### Prompt 10: Debugging Configuration Loading Errors

**Goal:**
Fix YAML parsing errors and missing configuration keys.

**Prompt:**
```
I'm getting errors when loading my config.yaml:

Error 1: yaml.scanner.ScannerError: mapping values are not allowed here
Error 2: KeyError: 'instashap' when accessing config["training"]["instashap"]
Error 3: TypeError: 'NoneType' object is not subscriptable

My config.yaml snippet that might have issues:
training:
  blackbox:
    hidden_dims: [256, 128]
    dropout: 0.10
  instashap:
    hidden_dims [96, 64]  # <-- missing colon?
    dropout: 0.05

Help me:
1. Fix the YAML syntax error (missing colon after hidden_dims)
2. Explain YAML syntax gotchas (indentation, colons, lists)
3. Add defensive config access: config.get("training", {}).get("instashap", {})
4. Write a validate_config(config) function that checks for required keys
5. How to set defaults for optional keys
6. Best practice: parse config into a dataclass or namedtuple for type safety
7. How to use YAML anchors (&alias) and merge keys (<<:) to reduce duplication

Show the validated config loading pattern that never crashes on missing keys.
```

**Why this is useful:**
Config errors cause crashes at startup, wasting time. Robust config handling prevents frustration.

---

### Prompt 11: Debugging Early Stopping Not Working Correctly

**Goal:**
Fix issues where early stopping triggers too early or not at all.

**Prompt:**
```
My early stopping has two problems:

Problem A: It stops after just 3 epochs even though the model could improve more
- patience=5 but it stopped at epoch 3
- val_loss was: [0.85, 0.84, 0.83, STOPPED] — still decreasing!

Problem B: In a different run, the model trained for all 35 epochs without stopping
- val_loss clearly plateaued at epoch 15
- My patience is 6, so it should have stopped around epoch 21

My early stopping code:
best_loss = float('inf')
patience_counter = 0
for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss = evaluate(...)
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
    if patience_counter > patience:  # <-- Bug: should be >= ?
        break

Debug both problems:
1. Problem A: Am I comparing against the wrong loss? (float comparison issues?)
2. Problem B: Is the counter off by one? (> vs >=)
3. Should I use a minimum improvement threshold (min_delta)?
4. Should I restore best model weights after stopping?
5. Common mistakes: comparing train loss instead of val loss, wrong counter logic

Show the corrected implementation with all edge cases handled.
```

**Why this is useful:**
Early stopping bugs silently cause either underfitting or wasted computation time.

---

### Prompt 12: Debugging Visualization Issues (matplotlib)

**Goal:**
Fix common matplotlib problems that produce wrong or ugly plots.

**Prompt:**
```
I'm having several matplotlib issues:

Issue 1: Plots show blank white images when saved (but display fine interactively)
Issue 2: Subplot labels overlap and are cut off
Issue 3: Memory keeps growing when generating many plots in a loop
Issue 4: Heatmap color scale is wrong (all one color)
Issue 5: Legend covers the data

Show me the fix for each:
1. Blank plots: plt.savefig() before plt.show(), or use fig.savefig()
2. Overlapping labels: plt.tight_layout() or fig.set_constrained_layout(True)
3. Memory leak: plt.close(fig) after saving — explain the figure lifecycle
4. Single-color heatmap: data might have no variance, or vmin/vmax are wrong
5. Legend placement: bbox_to_anchor, loc, and ncol parameters

For my project (generating 10+ plots per dataset, 3 datasets):
- Show the correct pattern for generating and saving multiple figures in a loop
- How to set a global matplotlib style at project startup
- How to ensure consistent figure sizes across all plots
- How to handle the "Matplotlib is currently using agg backend" warning
```

**Why this is useful:**
Visualization bugs waste time and produce unprofessional results. These specific issues affect every matplotlib user.

---

### Prompt 13: Debugging Feature Groups Mismatch

**Goal:**
Fix the alignment between pre-processing feature groups and SHAP computation.

**Prompt:**
```
I'm getting wrong SHAP aggregation results. The feature importance values don't make sense:
- "season" shows near-zero importance, but it's clearly important for bike rentals
- One-hot features of "season" individually show some importance
- The aggregation seems broken

My feature_groups dict:
{"temp": [0], "humidity": [1], "season": [2, 3, 4, 5], "weather": [6, 7, 8, 9], ...}

My aggregation code:
aggregated_shap = {}
for feature, indices in feature_groups.items():
    aggregated_shap[feature] = shap_values[:, indices].sum(axis=1)

Debug this:
1. Am I summing correctly? Should I sum absolute values or raw values?
2. Is the feature_groups index mapping correct after preprocessing?
3. What happens when the preprocessor drops rare categories?
4. How to verify: compare len(feature_groups[all_indices]) == shap_values.shape[1]
5. Is the order of operations correct: preprocess → train → SHAP → aggregate?
6. How to add validation checks that catch index mismatches early

Show me a debugging approach to verify the feature groups mapping is correct.
```

**Why this is useful:**
Feature group alignment bugs produce silently wrong results — the code runs fine but conclusions are invalid.

---

### Prompt 14: Diagnosing Slow Training (Performance Profiling)

**Goal:**
Find and fix performance bottlenecks in the training pipeline.

**Prompt:**
```
My training is much slower than expected:
- Bike Sharing (17,000 rows): 15 minutes per epoch
- Expected: ~30 seconds per epoch

I'm using: batch_size=512, device=cuda, hidden_dims=[256,128]

Help me profile and fix:
1. Is the bottleneck in data loading, forward pass, backward pass, or transfer?
   - Add timing code: time.time() around each section
2. Check DataLoader performance:
   - Is num_workers=0 the problem? (Windows limitation)
   - Should I use pin_memory=True?
3. Check device transfers:
   - Am I moving data to GPU each batch? (Can I create datasets already on GPU?)
4. Check for Python-level slowness:
   - Am I using Python loops where vectorized operations would work?
   - Am I creating new tensors unnecessarily each batch?
5. Check for debugging code left in:
   - print statements in the training loop
   - torch.autograd.set_detect_anomaly(True) (massive slowdown!)
6. Profile with torch.profiler or cProfile

Show me a timing wrapper for the training loop sections.
```

**Why this is useful:**
Slow training wastes hours. Systematic profiling identifies the bottleneck quickly.

---

### Prompt 15: Debugging Inconsistent Results Across Runs

**Goal:**
Fix non-deterministic behavior when reproducibility is expected.

**Prompt:**
```
I've set random seed to 42, but my results change slightly between runs:
- Run 1: RMSE = 45.23
- Run 2: RMSE = 44.89
- Run 3: RMSE = 45.67

My seeding code:
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

What am I missing?
1. CUDA-specific seeding: torch.cuda.manual_seed_all(42)
2. CuDNN determinism: torch.backends.cudnn.deterministic = True
3. CuDNN benchmark: torch.backends.cudnn.benchmark = False
4. DataLoader worker seeds: generator=torch.Generator().manual_seed(42)
5. HashSeed: os.environ["PYTHONHASHSEED"] = "42"
6. Are there operations that are fundamentally non-deterministic on GPU?
   (torch.use_deterministic_algorithms(True) — catches these)

For my project:
- Show the complete set_global_seed() function
- Explain the performance cost of full determinism
- When is approximate reproducibility (±1%) acceptable?
- How to document the expected variance in results
```

**Why this is useful:**
Reproducibility is a core requirement of research. Understanding all sources of randomness is essential.

---

### Prompt 16: Debugging DataLoader Worker Errors on Windows

**Goal:**
Fix the common Windows multiprocessing issue with DataLoaders.

**Prompt:**
```
When I set num_workers > 0 in my DataLoader on Windows, I get:

RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase.

Or sometimes:
BrokenPipeError: [Errno 32] Broken pipe

My code:
train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

Help me understand and fix:
1. Why this happens on Windows but not Linux (fork vs spawn)
2. The Windows-specific fix: if __name__ == "__main__" guard
3. Alternative: set num_workers=0 (what's the performance impact?)
4. When num_workers > 0 actually helps (large datasets, heavy preprocessing)
5. The persistent_workers=True option and when to use it
6. How the config.yaml num_workers: 0 setting handles this

Should I always use num_workers=0 for a project that needs to run cross-platform?
```

**Why this is useful:**
This Windows-specific issue trips up many students. Understanding the root cause prevents repeated frustration.

---

### Prompt 17: Debugging Incorrect SHAP Value Signs

**Goal:**
Understand why SHAP values have unexpected positive/negative signs.

**Prompt:**
```
My SHAP values for the Bike Sharing dataset show that "temperature" has negative SHAP contribution, meaning higher temperature → fewer bike rentals. But intuitively, warm weather should increase rentals.

Before saying "the data knows best," help me investigate:
1. Is the preprocessing inverting the relationship? (Check: does StandardScaler flip signs?)
2. Am I confusing absolute SHAP values with signed SHAP values?
3. Is the model actually using this signal correctly? (Check: partial dependence plot)
4. Am I reading the SHAP values correctly? (SHAP output shape for regression vs classification)
5. Is the SHAP background dataset representative? (Biased background → biased attributions)
6. Could there be multicollinearity? (temp correlates with season, confounding SHAP)

Walk me through the diagnostic steps:
a. Plot raw data: temperature vs bike count (scatter)
b. Check model predictions: does model output increase with temperature?
c. Verify SHAP computation parameters
d. Compare permutation SHAP with exact SHAP (if feasible)

Don't just explain — show me the Python code to diagnose this.
```

**Why this is useful:**
Interpreting SHAP values incorrectly leads to wrong conclusions. This teaches critical evaluation of XAI outputs.

---

### Prompt 18: Debugging InstaSHAP vs SHAP Discrepancy

**Goal:**
Investigate when InstaSHAP explanations disagree significantly with permutation SHAP.

**Prompt:**
```
My InstaSHAP explanation correlation with permutation SHAP is only 0.65 (paper reports >0.90).

Metrics:
- Overall MSE: 0.15 (too high)
- Feature "hour": correlation 0.92 ✓
- Feature "season": correlation 0.43 ✗
- Feature "workingday": correlation 0.51 ✗

Help me diagnose:
1. Poor surrogate quality: if the surrogate doesn't approximate the black-box well, InstaSHAP inherits that error. Check surrogate R² on masked test inputs.
2. Insufficient training: did InstaSHAP converge? Check training curves.
3. Masked training configuration:
   - Is masks_per_sample too low? (Try 4 instead of 2)
   - Is edge_mask_probability appropriate? (0.10 → try 0.20)
4. Feature group handling in masking: are one-hot features masked as groups or individually?
5. Categorical features (season, workingday) may need different masking treatment
6. Fast-dev-run mode: am I training on too little data?

For each hypothesis, show the specific check I should perform.
Prioritize the checks from most likely to least likely cause.
```

**Why this is useful:**
This is the core debugging scenario specific to InstaSHAP. Understanding it demonstrates mastery of the method.

---

### Prompt 19: Debugging PDF Report Generation Failures

**Goal:**
Fix crashes in the report generation pipeline.

**Prompt:**
```
My PDF report generation crashes with:

Case 1: FileNotFoundError: results/plots/bike/training_curves.png
  - Experiments ran but maybe not all plots were saved?

Case 2: matplotlib.backends.backend_pdf.PdfPages gives: OSError: [Errno 22] Invalid argument
  - Maybe invalid characters in the output path?

Case 3: Report generates but images are missing/blank

Help me debug:
1. Case 1: How to check which files exist before trying to embed them
   - Use Path.exists() checks before opening
   - Implement graceful fallback: skip missing plots, note in report
2. Case 2: Path issues on Windows (backslash, long paths, special characters)
   - Use pathlib.Path instead of string concatenation
3. Case 3: Matplotlib figure lifecycle issues
   - Are figures being closed before saving to PDF?
   - Is the DPI/resolution setting correct for embedding?

Show me a robust report generation pattern:
check_files → embed_what_exists → log_what_is_missing → generate_report
```

**Why this is useful:**
Report generation often fails because it depends on many upstream outputs. Defensive programming is essential.

---

### Prompt 20: Debugging One-Hot Encoding Misalignment

**Goal:**
Fix the bug where different datasets produce different one-hot column orders.

**Prompt:**
```
My model was trained on the Bike Sharing dataset where one-hot encoding produced:
[season_1, season_2, season_3, season_4, weather_1, weather_2, weather_3, ...]

But when I apply the same preprocessor to new data (or a different split), I get:
[season_2, season_1, season_4, season_3, weather_2, weather_1, ...]

The column order changed! This would silently produce wrong predictions.

Help me:
1. Why does this happen? (set-based operations, random hash ordering)
2. How scikit-learn's OneHotEncoder handles this (fit on train, categories_ stored)
3. How to ensure consistent column order: fit preprocessor on train, transform test
4. What happens with categories in test that weren't in train? (handle_unknown='ignore')
5. How to detect this bug: add an assertion checking column names match
6. How feature_groups mapping breaks when columns shuffle

Show the safe preprocessing pattern:
preprocessor.fit(X_train)  # Learn column order
X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)  # Same order guaranteed
```

**Why this is useful:**
Column order misalignment is a silent bug that causes models to produce garbage without any error message.

---

### Prompt 21: Debugging the Masked Surrogate Training

**Goal:**
Fix issues in the surrogate training where the surrogate doesn't learn to approximate the blackbox.

**Prompt:**
```
My masked surrogate has very high loss that doesn't improve:

Surrogate training loss per epoch: [1.85, 1.84, 1.83, 1.82, 1.82, 1.82, ...]

The black-box achieves MSE=0.12 on unmasked test data, but the surrogate can barely approximate it even on unmasked inputs.

Investigate:
1. Am I generating masks correctly?
   - Print a few masks: are they all zeros? All ones? Reasonable mix?
   - Are masks applied at the feature group level (not individual one-hot columns)?
2. Am I getting targets correctly?
   - targets = blackbox(X_masked).detach()  # Must detach!
   - Are targets reasonable values? (Not NaN, not all same)
3. Is the surrogate architecture appropriate?
   - Should it have the same capacity as the black-box? (Yes, usually)
4. Is edge_mask_probability appropriate?
   - Too low (0.01): most inputs are fully masked → target ≈ constant
   - Too high (0.90): most inputs are nearly unmasked → doesn't learn masking behavior
   - Sweet spot: 0.10 - 0.30
5. Are train/val splits consistent between black-box and surrogate training?

Add diagnostic logging to the surrogate training to identify the issue.
```

**Why this is useful:**
The surrogate is a critical intermediate step. If it fails, InstaSHAP will also fail, and the error might not be obvious.

---

### Prompt 22: Debugging Python TypeError and AttributeError

**Goal:**
Quickly resolve common Python runtime errors.

**Prompt:**
```
I keep encountering these errors in my ML project:

Error 1: TypeError: 'dict' object is not callable
  config = load_config(path)
  hidden = config("training")("blackbox")("hidden_dims")  # Wrong: () not []

Error 2: AttributeError: 'NoneType' object has no attribute 'shape'
  X_test = preprocessor.transform(X_test)
  print(X_test.shape)  # But transform returned None!

Error 3: TypeError: unsupported operand type(s) for +: 'int' and 'str'
  result = "Accuracy: " + accuracy  # accuracy is a float

For EACH error:
1. Explain what the error message means in plain English
2. Show the exact cause in the code
3. Show the fix
4. Explain the pattern to prevent this (type awareness)
5. How type hints would have caught this before runtime

Also: what's the difference between TypeError and AttributeError?
When is each raised?
```

**Why this is useful:**
These errors are basic but frequent. Understanding Python's type system prevents time wasted on trivial bugs.

---

### Prompt 23: Debugging scikit-learn Pipeline Errors

**Goal:**
Fix errors when using scikit-learn transformers incorrectly.

**Prompt:**
```
I'm getting errors with scikit-learn preprocessing:

Error 1: NotFittedError: This StandardScaler instance is not fitted yet.
  value = scaler.transform(new_data)  # Forgot to fit first!

Error 2: ValueError: X has 13 features, but StandardScaler is expecting 45 features.
  scaler.fit(X_train)  # X_train was already one-hot encoded (45 cols)
  scaler.transform(X_raw)  # X_raw is original (13 cols) — mismatch!

Error 3: ValueError: Found unknown categories during transform.
  encoder.fit(X_train[['season']])  # Train has seasons 1,2,3,4
  encoder.transform(X_test[['season']])  # Test has season 5 somehow

Fix each and explain:
1. Error 1: The fit/transform lifecycle — when to fit vs transform vs fit_transform
2. Error 2: The order of operations — one-hot encode BEFORE or AFTER scaling?
3. Error 3: Set handle_unknown='ignore' or handle_unknown='infrequent_if_exist'
4. Design pattern: when to use sklearn Pipeline vs manual preprocessing
5. How to save fitted preprocessors (pickle or joblib) for production use

Show the correct preprocessing order for mixed-type data.
```

**Why this is useful:**
scikit-learn preprocessing bugs are tricky because the error messages are often about shapes rather than logic.

---

### Prompt 24: Debugging Training Loop Off-By-One Errors

**Goal:**
Find subtle index/counter bugs in training and evaluation loops.

**Prompt:**
```
My training metrics don't match what I expect. Possible off-by-one errors:

Scenario 1: val_loss is always one epoch behind (reporting epoch N-1's loss for epoch N)
Scenario 2: The last batch's loss is not included in the epoch average
Scenario 3: Predictions include duplicates from the DataLoader

My average loss computation:
total_loss = 0
for X, y in loader:
    loss = criterion(model(X), y)
    total_loss += loss.item()
avg_loss = total_loss / len(loader)  # Is this right?

Investigate:
1. len(loader) returns number of batches — is that what I want for averaging?
2. Should I weight by batch size? (Last batch might be smaller)
3. Am I evaluating AFTER the optimizer.step() or BEFORE in the training loop?
4. The DataLoader drop_last=True issue: am I losing data points?
5. Common: accumulating loss as tensor (keeps computation graph) vs loss.item() (scalar)

Show the correct loss accumulation and averaging pattern for both training and eval.
```

**Why this is useful:**
Off-by-one errors in ML training are subtle and can cause misleading metrics.

---

### Prompt 25: Debugging the Experiment Orchestrator Logic

**Goal:**
Fix control flow issues when running multi-model, multi-dataset experiments.

**Prompt:**
```
My experiment runner has logic bugs:

Bug 1: When running --model all, it fails on InstaSHAP because the surrogate hasn't been trained yet.
The order matters: blackbox → surrogate → GAM → InstaSHAP

Bug 2: When running --dataset all, the second dataset uses the first dataset's preprocessor.
Each dataset needs its own preprocessing.

Bug 3: When one model fails, the entire experiment crashes instead of continuing with the remaining models.

My runner:
for dataset in datasets:
    for model_type in models:
        train_and_evaluate(dataset, model_type, config)

Fix these:
1. Bug 1: Enforce training order (ensure dependencies are trained first)
2. Bug 2: Ensure all state (preprocessor, dataloaders) is scoped within the dataset loop
3. Bug 3: Add try/except with logging — continue on failure, report at end
4. How to handle partial results (some models succeeded, some failed)
5. Return type: a results summary indicating which experiments succeeded/failed

Show the corrected orchestrator with proper error handling and dependency management.
```

**Why this is useful:**
Orchestration bugs waste time by requiring full re-runs. Defensive error handling saves hours.

---

### Prompt 26: Debugging torch.no_grad() Misuse

**Goal:**
Understand when torch.no_grad() is required and when it causes bugs.

**Prompt:**
```
I'm confused about when to use torch.no_grad(). I have bugs related to it:

Bug 1: Validation loss keeps decreasing but training loss stays flat
  - I accidentally wrapped the TRAINING loop in torch.no_grad()
  - No gradients → optimizer.step() does nothing

Bug 2: Memory keeps growing during evaluation
  - I forgot torch.no_grad() during evaluation
  - Computation graphs are accumulating

Bug 3: SHAP computation fails with "leaf variable has been moved into the graph interior"
  - torch.no_grad() was used inside the model wrapper but affects SHAP internals

For each scenario, explain:
1. What torch.no_grad() actually does (prevents gradient computation)
2. When it's REQUIRED: evaluation, inference, generating surrogate targets
3. When it's HARMFUL: training forward pass (need gradients!)
4. When it's TRICKY: SHAP/XAI wrappers
5. The alternative: tensor.detach() — when to use each
6. with torch.no_grad(): vs @torch.no_grad() decorator

Show a cheatsheet: "use no_grad here / don't use no_grad here"
```

**Why this is useful:**
Misusing no_grad causes silent training failures or memory leaks — two of the most frustrating bugs to find.

---

### Prompt 27: Debugging Cross-Entropy Loss Input Requirements

**Goal:**
Fix the specific gotchas of PyTorch's CrossEntropyLoss.

**Prompt:**
```
I keep getting errors with CrossEntropyLoss:

Error: RuntimeError: expected scalar type Long but found Float
My code: loss = criterion(output, y_batch)
y_batch is float32 but CrossEntropyLoss needs int64 (Long)

Error 2: The loss is negative (how can loss be negative?)
My code: loss = criterion(torch.softmax(output, dim=1), y_batch)
I applied softmax manually AND CrossEntropyLoss applies log_softmax internally = double activation!

Error 3: RuntimeError: 0D or 1D target tensor expected for multi-target, but got [512, 7]
I one-hot encoded my targets but CrossEntropyLoss expects class indices, not one-hot

Fix each and create a comprehensive reference:
1. CrossEntropyLoss input requirements:
   - predictions: (N, C) raw logits (NO softmax/sigmoid applied)
   - targets: (N,) class indices as Long/int64 (NOT one-hot)
2. BCEWithLogitsLoss requirements:
   - predictions: (N,) raw logits
   - targets: (N,) float32 in [0, 1]
3. MSELoss requirements:
   - predictions and targets: same shape, float32

Show a decision table: task → loss function → input/target requirements → dtype.
```

**Why this is useful:**
CrossEntropyLoss has unintuitive requirements that confuse almost every PyTorch beginner.

---

### Prompt 28: Debugging pandas DataFrame Processing Issues

**Goal:**
Fix common pandas issues in the data loading pipeline.

**Prompt:**
```
I'm encountering pandas-related bugs in my data loading:

Bug 1: SettingWithCopyWarning: A value is trying to be set on a copy of a slice
  df['new_col'] = df['old_col'] * 2  # After filtering: df = df[df['age'] > 0]

Bug 2: Missing values not detected
  df.isnull().sum() returns 0, but I see '?' characters in the data (Adult dataset)
  
Bug 3: Categorical column has dtype 'object' but contains numbers as strings
  Bike Sharing 'season' column: ['1', '2', '3', '4'] — string, not int

Bug 4: merge/concat produces duplicate columns with suffixes (_x, _y)

Fix each:
1. Bug 1: Use .copy() or .loc[] properly
2. Bug 2: Replace sentinel values before null checks: df.replace('?', np.nan)
3. Bug 3: Explicit dtype conversion: df['season'] = df['season'].astype(int)
4. Bug 4: Clean join keys and handle overlapping column names

Show the defensive data loading pattern:
load → replace sentinels → check nulls → enforce dtypes → validate → return clean DataFrame
```

**Why this is useful:**
Data loading bugs are the most dangerous because they silently corrupt all downstream results.

---

### Prompt 29: Debugging Multi-Class SHAP Output Shape

**Goal:**
Handle the shape difference between single-output and multi-class SHAP values.

**Prompt:**
```
For regression (Bike Sharing):
  shap_values.shape = (n_samples, n_features) ✓ Simple 2D array

For classification (Covertype, 7 classes):
  shap_values.shape = (n_samples, n_features, n_classes) — 3D array!

My code assumed 2D and breaks:
aggregated = shap_values[:, feature_indices].sum(axis=1)  # Wrong for 3D!

Help me fix this:
1. Explain why multi-class SHAP has an extra dimension (per-class attributions)
2. How to reduce to 2D: take absolute mean over classes? Pick specific class?
3. Paper convention: which reduction does InstaSHAP use?
4. How does this affect the comparison between SHAP and InstaSHAP?
5. Fix the aggregation to handle both regression and classification:
   if len(shap_values.shape) == 3:
       shap_values = ???  # What goes here?

Show the complete shape-aware explanation processing pipeline.
```

**Why this is useful:**
Multi-class SHAP is a common stumbling block. Getting the shapes wrong silently produces nonsense.

---

### Prompt 30: Performance Bottleneck — SHAP Computation Too Slow

**Goal:**
Speed up permutation SHAP computation without sacrificing accuracy.

**Prompt:**
```
Computing permutation SHAP on 100 test samples with 45 features takes 45 minutes.
For my project, I need to evaluate 3 datasets × multiple runs. This is too slow.

Current settings:
- shap_background_size: 64
- shap_eval_samples: 24
- shap_max_evals: 256

Help me speed this up:
1. How does each parameter affect runtime and accuracy?
   - background_size: more → better baselines, linearly slower
   - eval_samples: more → more robust estimates, linearly slower
   - max_evals: more → more feature permutations evaluated
2. What are sensible minimums that still give reliable attributions?
3. Can I batch the forward passes? (Currently one-sample-at-a-time)
4. Can I cache intermediate results?
5. Alternative approaches:
   - TreeSHAP (only for tree models, but exact and fast)
   - DeepSHAP (for neural nets, but makes linearity assumptions)
   - GradientSHAP (fast but approximate)
6. When is the speed vs accuracy trade-off acceptable?

Show the optimized configuration and explain the accuracy impact.
```

**Why this is useful:**
SHAP computation is the biggest bottleneck in XAI projects. Understanding the speed/accuracy trade-offs is essential.
