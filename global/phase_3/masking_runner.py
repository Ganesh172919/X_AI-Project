"""
Steps 3-4: Masking Strategies and InstaSHAP Runner.

Implements:
- zero_mask: sets all columns for hidden feature groups to 0.0
- empirical_background: copies hidden group columns from a real training row
- Coalition sampling + surrogate training for both strategies
- MLP black-box model and linear surrogate
"""

from __future__ import annotations

import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from scipy.stats import spearmanr
from typing import Optional

from preprocessing import SyntheticPreprocessor

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════════════════
# BLACK-BOX MODEL (MLP Classifier)
# ═══════════════════════════════════════════════════════════════════════

class BlackBoxMLP(nn.Module):
    """Simple feed-forward MLP for binary classification."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = [64, 32], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 2))  # 2 classes
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_blackbox(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 50, lr: float = 0.001,
                   batch_size: int = 32, patience: int = 10) -> BlackBoxMLP:
    """Train a black-box MLP classifier."""
    input_dim = X_train.shape[1]
    model = BlackBoxMLP(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_train).to(DEVICE)
    y_t = torch.LongTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).to(DEVICE)
    y_v = torch.LongTensor(y_val).to(DEVICE)

    best_loss = float('inf')
    patience_ctr = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out, y_v).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# MASKING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def apply_zero_mask(inputs: np.ndarray, feature_mask: np.ndarray,
                    preprocessor: SyntheticPreprocessor) -> np.ndarray:
    """
    Zero-masking: hidden features become 0.0 in transformed space.

    For categorical one-hot groups: all columns → 0, creating an impossible state.
    For numeric columns: value → 0.0 (which is NOT the mean after standardization).
    """
    expanded = preprocessor.expand_feature_mask(feature_mask)
    return inputs * expanded


def apply_empirical_background(inputs: np.ndarray, feature_mask: np.ndarray,
                               preprocessor: SyntheticPreprocessor,
                               background_bank: np.ndarray,
                               rng: np.random.RandomState) -> np.ndarray:
    """
    Empirical background masking: hidden features get values from a real training row.

    For categorical one-hot groups: copies a valid one-hot vector from the donor row.
    For numeric columns: copies the real (standardized) value from the donor row.
    """
    batch_size = inputs.shape[0]
    bg_indices = rng.randint(0, len(background_bank), size=batch_size)
    bg_rows = background_bank[bg_indices].astype(np.float32)

    expanded = preprocessor.expand_feature_mask(feature_mask)
    masked = inputs * expanded + bg_rows * (1.0 - expanded)
    return masked


# ═══════════════════════════════════════════════════════════════════════
# COALITION SAMPLING
# ═══════════════════════════════════════════════════════════════════════

def sample_coalitions(n_features: int, n_coalitions: int,
                      rng: np.random.RandomState) -> np.ndarray:
    """
    Sample binary feature masks uniformly. Each mask is a binary vector of
    length n_features, where 1 = visible, 0 = hidden.

    We exclude all-zeros and all-ones masks.
    """
    masks = rng.binomial(1, 0.5, size=(n_coalitions, n_features)).astype(np.float32)
    # Re-sample any all-zeros or all-ones
    for i in range(len(masks)):
        while masks[i].sum() == 0 or masks[i].sum() == n_features:
            masks[i] = rng.binomial(1, 0.5, size=n_features).astype(np.float32)
    return masks


# ═══════════════════════════════════════════════════════════════════════
# SURROGATE TRAINING
# ═══════════════════════════════════════════════════════════════════════

class LinearSurrogate(nn.Module):
    """Simple linear surrogate for SHAP-like decomposition."""

    def __init__(self, n_features: int, n_outputs: int = 2):
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, mask):
        return self.linear(mask) + self.bias


def train_surrogate_for_row(
    row_data: np.ndarray,
    X_train: np.ndarray,
    blackbox: BlackBoxMLP,
    preprocessor: SyntheticPreprocessor,
    strategy: str,  # 'zero_mask' or 'empirical_background'
    n_coalitions: int = 500,
    rng: np.random.RandomState = None,
    epochs: int = 200,
    lr: float = 0.01,
) -> tuple[LinearSurrogate, np.ndarray, np.ndarray]:
    """
    Train a per-row linear surrogate using coalition sampling.

    Returns:
        surrogate: trained LinearSurrogate
        masks: coalition masks used (n_coalitions, n_features)
        masked_data: the actual masked inputs sent to the blackbox (n_coalitions, input_dim)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    n_features = preprocessor.num_original_features
    masks = sample_coalitions(n_features, n_coalitions, rng)

    # Replicate the row for all coalitions
    row_batch = np.tile(row_data, (n_coalitions, 1))

    # Apply masking strategy
    if strategy == 'zero_mask':
        masked_data = apply_zero_mask(row_batch, masks, preprocessor)
    elif strategy == 'empirical_background':
        masked_data = apply_empirical_background(
            row_batch, masks, preprocessor, X_train, rng
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Get blackbox predictions for masked data
    blackbox.eval()
    with torch.no_grad():
        masked_t = torch.FloatTensor(masked_data).to(DEVICE)
        targets = F.softmax(blackbox(masked_t), dim=1).cpu().numpy()  # (n_coalitions, 2)

    # Train linear surrogate: predict blackbox output from mask
    surrogate = LinearSurrogate(n_features, 2).to(DEVICE)
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)
    criterion = nn.MSELoss()

    masks_t = torch.FloatTensor(masks).to(DEVICE)
    targets_t = torch.FloatTensor(targets).to(DEVICE)

    for _ in range(epochs):
        pred = surrogate(masks_t)
        loss = criterion(pred, targets_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    surrogate.eval()
    return surrogate, masks, masked_data


# ═══════════════════════════════════════════════════════════════════════
# FULL PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_instashap_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    blackbox: BlackBoxMLP,
    preprocessor: SyntheticPreprocessor,
    strategy: str,
    n_coalitions: int = 500,
    n_eval_rows: int = 40,
    random_state: int = 42,
) -> dict:
    """
    Run full InstaSHAP pipeline with specified masking strategy.

    Steps:
    1. For each training row (up to n_eval_rows), sample coalitions
    2. Apply masking strategy, get blackbox predictions
    3. Train per-row linear surrogate
    4. Collect SHAP values (surrogate weights), predictions, and coalition data

    Returns dict with all metrics data.
    """
    rng = np.random.RandomState(random_state)
    n_features = preprocessor.num_original_features

    start_time = time.perf_counter()

    # Storage
    all_shap_values = []
    all_surrogate_preds = []
    all_blackbox_preds = []
    all_masked_data = []
    all_masks = []
    surrogate_losses = []

    eval_rows = min(n_eval_rows, len(X_train))

    for i in range(eval_rows):
        row = X_train[i:i+1]

        surrogate, masks, masked_data = train_surrogate_for_row(
            row_data=row,
            X_train=X_train,
            blackbox=blackbox,
            preprocessor=preprocessor,
            strategy=strategy,
            n_coalitions=n_coalitions,
            rng=rng,
        )

        # SHAP values = surrogate weights
        with torch.no_grad():
            shap_vals = surrogate.linear.weight.cpu().numpy()  # (2, n_features)
        all_shap_values.append(shap_vals)

        # Surrogate predictions on test masks
        test_masks = sample_coalitions(n_features, 50, rng)
        with torch.no_grad():
            s_pred = surrogate(torch.FloatTensor(test_masks).to(DEVICE)).cpu().numpy()

        # Blackbox predictions on test-masked data
        test_row_batch = np.tile(row, (50, 1))
        if strategy == 'zero_mask':
            test_masked = apply_zero_mask(test_row_batch, test_masks, preprocessor)
        else:
            test_masked = apply_empirical_background(
                test_row_batch, test_masks, preprocessor, X_train, rng
            )
        with torch.no_grad():
            bb_pred = F.softmax(
                blackbox(torch.FloatTensor(test_masked).to(DEVICE)), dim=1
            ).cpu().numpy()

        all_surrogate_preds.append(s_pred)
        all_blackbox_preds.append(bb_pred)
        all_masked_data.append(masked_data)
        all_masks.append(masks)

        surrogate_loss = np.mean((s_pred - bb_pred) ** 2)
        surrogate_losses.append(surrogate_loss)

    wall_time = time.perf_counter() - start_time

    return {
        'strategy': strategy,
        'shap_values': np.array(all_shap_values),       # (eval_rows, 2, n_features)
        'surrogate_preds': all_surrogate_preds,
        'blackbox_preds': all_blackbox_preds,
        'masked_data': all_masked_data,
        'masks': all_masks,
        'surrogate_mse_list': surrogate_losses,
        'surrogate_mse_mean': float(np.mean(surrogate_losses)),
        'wall_time': wall_time,
    }


# ═══════════════════════════════════════════════════════════════════════
# COALITION VALIDITY METRICS (Level 1)
# ═══════════════════════════════════════════════════════════════════════

def compute_coalition_validity(
    masked_data_list: list[np.ndarray],
    masks_list: list[np.ndarray],
    preprocessor: SyntheticPreprocessor,
    X_train: np.ndarray,
) -> dict:
    """
    Measure coalition validity at the structural level.

    Metrics:
    - hidden_categorical_valid_rate: fraction of hidden one-hot groups where exactly one column is 1
    - hidden_categorical_invalid_rate: complement
    - hidden_numeric_exact_zero_rate: fraction of hidden numeric values that are exactly 0.0
    - nearest_train_distance_mean: mean L2 distance from each masked vector to nearest training row
    """
    cat_groups = preprocessor.categorical_group_names()
    num_groups = preprocessor.numeric_group_names()

    total_cat_hidden = 0
    valid_cat_hidden = 0
    total_num_hidden = 0
    exact_zero_num = 0
    all_distances = []

    for masked_data, masks in zip(masked_data_list, masks_list):
        expanded = preprocessor.expand_feature_mask(masks)

        for row_idx in range(len(masked_data)):
            # Check each categorical group
            for cat_name in cat_groups:
                grp = preprocessor.group(cat_name)
                feat_idx = preprocessor.feature_index(cat_name)

                if masks[row_idx, feat_idx] == 0:  # hidden
                    total_cat_hidden += 1
                    cat_vals = masked_data[row_idx, grp.start:grp.end]
                    # Valid = exactly one 1 and rest 0
                    if np.sum(cat_vals) == 1.0 and np.max(cat_vals) == 1.0 and np.min(cat_vals) == 0.0:
                        valid_cat_hidden += 1

            # Check numeric groups
            for num_name in num_groups:
                grp = preprocessor.group(num_name)
                feat_idx = preprocessor.feature_index(num_name)

                if masks[row_idx, feat_idx] == 0:  # hidden
                    total_num_hidden += 1
                    val = masked_data[row_idx, grp.start]
                    if abs(val) < 1e-10:
                        exact_zero_num += 1

            # Nearest train distance
            diffs = X_train - masked_data[row_idx:row_idx+1]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            all_distances.append(float(np.min(dists)))

    cat_valid_rate = valid_cat_hidden / total_cat_hidden if total_cat_hidden > 0 else 0.0
    cat_invalid_rate = 1.0 - cat_valid_rate
    num_zero_rate = exact_zero_num / total_num_hidden if total_num_hidden > 0 else 0.0
    dist_mean = float(np.mean(all_distances)) if all_distances else 0.0

    return {
        'hidden_categorical_groups_evaluated': total_cat_hidden,
        'hidden_categorical_valid_rate': cat_valid_rate,
        'hidden_categorical_invalid_rate': cat_invalid_rate,
        'hidden_numeric_entries_evaluated': total_num_hidden,
        'hidden_numeric_exact_zero_rate': num_zero_rate,
        'nearest_train_distance_mean': dist_mean,
    }


# ═══════════════════════════════════════════════════════════════════════
# PREDICTIVE METRICS (Level 2)
# ═══════════════════════════════════════════════════════════════════════

def compute_predictive_metrics(result: dict) -> dict:
    """Compute surrogate accuracy, F1, and prediction MSE against blackbox."""
    all_s = np.concatenate(result['surrogate_preds'], axis=0)
    all_b = np.concatenate(result['blackbox_preds'], axis=0)

    s_labels = np.argmax(all_s, axis=1)
    b_labels = np.argmax(all_b, axis=1)

    return {
        'surrogate_accuracy': float(accuracy_score(b_labels, s_labels)),
        'surrogate_f1': float(f1_score(b_labels, s_labels, average='weighted', zero_division=0)),
        'surrogate_mse_vs_blackbox': float(np.mean((all_s - all_b) ** 2)),
    }


# ═══════════════════════════════════════════════════════════════════════
# EXPLANATION METRICS (Level 3)
# ═══════════════════════════════════════════════════════════════════════

def compute_explanation_metrics(result_zero: dict, result_emp: dict,
                                reference_shap: Optional[np.ndarray] = None) -> dict:
    """
    Compute SHAP quality metrics.

    Uses surrogate SHAP from empirical_background as working reference
    if no external reference is provided.
    """
    shap_zero = result_zero['shap_values']      # (N, 2, n_features)
    shap_emp = result_emp['shap_values']

    n = min(len(shap_zero), len(shap_emp))

    if reference_shap is None:
        reference_shap = shap_emp[:n]

    # Spearman rank correlation
    corrs_zero = []
    corrs_emp = []
    for i in range(n):
        ref = reference_shap[i].ravel()
        z = shap_zero[i].ravel()
        e = shap_emp[i].ravel()

        if np.std(ref) > 1e-12:
            rho_z, _ = spearmanr(ref, z)
            rho_e, _ = spearmanr(ref, e)
            corrs_zero.append(float(rho_z) if not np.isnan(rho_z) else 0.0)
            corrs_emp.append(float(rho_e) if not np.isnan(rho_e) else 0.0)

    # MAE
    mae_zero = float(np.mean(np.abs(shap_zero[:n] - reference_shap[:n])))
    mae_emp = float(np.mean(np.abs(shap_emp[:n] - reference_shap[:n])))

    return {
        'zero_mask_spearman_corr': float(np.mean(corrs_zero)) if corrs_zero else 0.0,
        'empirical_spearman_corr': float(np.mean(corrs_emp)) if corrs_emp else 1.0,
        'zero_mask_explanation_mae': mae_zero,
        'empirical_explanation_mae': mae_emp,
    }
