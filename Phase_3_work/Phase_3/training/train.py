"""Training loops for all 4 stages — extended with 3 innovations.

Innovations integrated:
  1. Empirical-background masking in surrogate/InstaSHAP training
  2. Curriculum-weighted Shapley mask scheduling
  3. Multi-surrogate ensemble training
"""

from __future__ import annotations

import copy
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.preprocessing import TabularPreprocessor
from masking.config import MaskingConfig
from masking.zero_mask import apply_zero_mask, expand_feature_mask_torch
from masking.background_mask import apply_background_mask, compute_background_targets
from masking.curriculum import curriculum_shapley_masks, standard_shapley_masks
from models.blackbox_model import MaskedSurrogateMLP, SurrogateEnsemble, TabularMLP
from models.gam import GAMModel
from models.instashap import InstaSHAPModel
from training.evaluate import predict_raw_outputs
from utils.logging_utils import get_logger

log = get_logger("training")


# ── Helpers ─────────────────────────────────────────────────────────────

def _np_to_tensor(X: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(X.astype(np.float32)).to(device)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64) if y.dtype in (np.int32, np.int64, int) else y.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _loss_fn(task: str) -> nn.Module:
    return nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()


# ── Stage 1: Black-Box Training ─────────────────────────────────────────

def train_blackbox(
    model: TabularMLP,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: torch.device,
    task: str,
    cfg: dict,
) -> tuple[TabularMLP, list[dict[str, float]]]:
    """Standard supervised training for the black-box MLP."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = _loss_fn(task)
    train_ld = _make_loader(X_train, y_train, cfg["batch_size"])
    val_ld = _make_loader(X_val, y_val, cfg["batch_size"], shuffle=False)
    best_loss, best_state, patience_cnt = float("inf"), None, 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_losses.append(loss.item())

        tl, vl = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"train_loss": tl, "val_loss": vl})

        if vl < best_loss:
            best_loss, best_state, patience_cnt = vl, copy.deepcopy(model.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ── Stage 2: GAM Training ───────────────────────────────────────────────

def train_gam(
    model: GAMModel,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    device: torch.device,
    task: str,
    cfg: dict,
) -> tuple[GAMModel, list[dict[str, float]]]:
    """Standard supervised training for GAM-1 and GAM-2."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = _loss_fn(task)
    train_ld = _make_loader(X_train, y_train, cfg["batch_size"])
    val_ld = _make_loader(X_val, y_val, cfg["batch_size"], shuffle=False)
    best_loss, best_state, patience_cnt = float("inf"), None, 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg["epochs"]):
        model.train()
        train_losses = []
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_losses.append(loss.item())

        tl, vl = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"train_loss": tl, "val_loss": vl})

        if vl < best_loss:
            best_loss, best_state, patience_cnt = vl, copy.deepcopy(model.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ── Stage 3: Masked Surrogate Training ───────────────────────────────────

def train_masked_surrogate(
    surrogate: MaskedSurrogateMLP,
    blackbox_model: nn.Module,
    X_train: np.ndarray, X_val: np.ndarray,
    preprocessor: TabularPreprocessor,
    device: torch.device,
    cfg: dict,
    masking_config: MaskingConfig,
    background_bank: np.ndarray | None = None,
) -> tuple[MaskedSurrogateMLP, list[dict[str, float]]]:
    """Train the masked surrogate with configurable masking strategy.

    Supports:
      - Zero-masking (baseline)
      - Empirical-background masking (Innovation 1)
      - Curriculum scheduling (Innovation 2)
    """
    surrogate = surrogate.to(device)
    blackbox_model.eval()

    # Pre-compute blackbox raw outputs for zero-mask training targets
    bb_train_raw = predict_raw_outputs(blackbox_model, X_train, device)
    bb_val_raw = predict_raw_outputs(blackbox_model, X_val, device)

    n_features = preprocessor.num_original_features
    masks_per_sample = cfg.get("masks_per_sample", 2)
    edge_prob = cfg.get("edge_mask_probability", 0.10)
    total_epochs = cfg["epochs"]

    opt = torch.optim.AdamW(surrogate.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss()
    rng = np.random.default_rng(masking_config.seed)
    best_loss, best_state, patience_cnt = float("inf"), None, 0
    history: list[dict[str, float]] = []
    use_bg = masking_config.strategy == "empirical_background"

    for epoch in range(total_epochs):
        surrogate.train()
        train_losses = []

        # Create shuffled mini-batches
        n = len(X_train)
        indices = rng.permutation(n)
        bs = cfg["batch_size"]

        for batch_start in range(0, n, bs):
            batch_idx = indices[batch_start:batch_start + bs]
            actual_bs = len(batch_idx)

            x_batch = _np_to_tensor(X_train[batch_idx], device)
            bb_raw_batch = _np_to_tensor(bb_train_raw[batch_idx], device)

            # Sample masks (Innovation 2: curriculum scheduling when enabled)
            effective_bs = actual_bs * masks_per_sample
            if masking_config.use_curriculum:
                mask_np = curriculum_shapley_masks(
                    effective_bs, n_features, rng, epoch, total_epochs,
                    masking_config.curriculum_warmup_frac,
                    masking_config.curriculum_standard_frac,
                    edge_prob,
                )
            else:
                mask_np = standard_shapley_masks(effective_bs, n_features, rng, edge_prob)

            # Repeat inputs for multiple masks per sample
            x_repeated = x_batch.repeat(masks_per_sample, 1)
            bb_repeated = bb_raw_batch.repeat(masks_per_sample, 1)
            feature_mask = _np_to_tensor(mask_np, device)

            if use_bg and background_bank is not None:
                # Innovation 1: Background masking
                masked_inputs = apply_background_mask(
                    x_repeated, feature_mask, preprocessor, background_bank, rng,
                    K=masking_config.background_samples_train,
                )
                # Re-compute targets using background-averaged blackbox
                targets = compute_background_targets(
                    blackbox_model, x_repeated, feature_mask,
                    preprocessor, background_bank, rng, device,
                    K=masking_config.background_samples_train,
                )
            else:
                # Baseline: zero-masking
                masked_inputs = apply_zero_mask(x_repeated, feature_mask, preprocessor)
                targets = bb_repeated

            predictions = surrogate(masked_inputs, feature_mask)
            loss = criterion(predictions, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())

        # Validation
        surrogate.eval()
        val_losses = []
        n_val = len(X_val)
        val_indices = np.arange(n_val)
        with torch.no_grad():
            for vstart in range(0, n_val, bs):
                vidx = val_indices[vstart:vstart + bs]
                actual_vbs = len(vidx)
                x_vb = _np_to_tensor(X_val[vidx], device)
                bb_vb = _np_to_tensor(bb_val_raw[vidx], device)

                vmask_np = standard_shapley_masks(actual_vbs, n_features, rng, edge_prob)
                vmask = _np_to_tensor(vmask_np, device)

                if use_bg and background_bank is not None:
                    vmasked = apply_background_mask(x_vb, vmask, preprocessor, background_bank, rng, K=1)
                    vtargets = compute_background_targets(
                        blackbox_model, x_vb, vmask, preprocessor, background_bank, rng, device, K=1)
                else:
                    vmasked = apply_zero_mask(x_vb, vmask, preprocessor)
                    vtargets = bb_vb

                vpred = surrogate(vmasked, vmask)
                vloss = criterion(vpred, vtargets)
                val_losses.append(vloss.item())

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        history.append({"train_loss": tl, "val_loss": vl})

        if vl < best_loss:
            best_loss, best_state, patience_cnt = vl, copy.deepcopy(surrogate.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    if best_state:
        surrogate.load_state_dict(best_state)
    return surrogate, history


# ── Innovation 3: Multi-Surrogate Ensemble Training ──────────────────────

def train_surrogate_ensemble(
    blackbox_model: nn.Module,
    X_train: np.ndarray, X_val: np.ndarray,
    preprocessor: TabularPreprocessor,
    device: torch.device,
    cfg: dict,
    masking_config: MaskingConfig,
    background_bank: np.ndarray | None,
    ensemble_size: int = 3,
    base_seed: int = 42,
) -> tuple[SurrogateEnsemble, list[list[dict[str, float]]]]:
    """Train M independent surrogates and wrap in ensemble (Innovation 3)."""
    surrogates: list[MaskedSurrogateMLP] = []
    all_histories: list[list[dict[str, float]]] = []

    for i in range(ensemble_size):
        sub_seed = base_seed + i * 1000 + 777
        sub_config = MaskingConfig(
            strategy=masking_config.strategy,
            background_bank_size=masking_config.background_bank_size,
            background_samples_train=masking_config.background_samples_train,
            background_samples_eval=masking_config.background_samples_eval,
            use_curriculum=masking_config.use_curriculum,
            curriculum_warmup_frac=masking_config.curriculum_warmup_frac,
            curriculum_standard_frac=masking_config.curriculum_standard_frac,
            seed=sub_seed,
        )

        import torch as _torch
        _torch.manual_seed(sub_seed)
        np.random.seed(sub_seed)

        surr = MaskedSurrogateMLP(
            feature_dim=preprocessor.input_dim,
            num_original_features=preprocessor.num_original_features,
            output_dim=cfg["output_dim"],
            hidden_dims=cfg["hidden_dims"],
            dropout=cfg["dropout"],
        ).to(device)

        log.info(f"  Training surrogate {i+1}/{ensemble_size} (seed={sub_seed})")
        surr, hist = train_masked_surrogate(
            surr, blackbox_model, X_train, X_val,
            preprocessor, device, cfg, sub_config, background_bank,
        )
        surrogates.append(surr)
        all_histories.append(hist)

    ensemble = SurrogateEnsemble(surrogates).to(device)
    return ensemble, all_histories


# ── Stage 4: InstaSHAP Training ──────────────────────────────────────────

def train_instashap_model(
    instashap: InstaSHAPModel,
    surrogate: nn.Module,
    X_train: np.ndarray, X_val: np.ndarray,
    preprocessor: TabularPreprocessor,
    device: torch.device,
    cfg: dict,
    masking_config: MaskingConfig,
    background_bank: np.ndarray | None = None,
) -> tuple[InstaSHAPModel, list[dict[str, float]]]:
    """Train InstaSHAP against frozen surrogate under Shapley masks.

    The surrogate can be a single model or SurrogateEnsemble (Innovation 3).
    """
    instashap = instashap.to(device)
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    n_features = preprocessor.num_original_features
    edge_prob = cfg.get("edge_mask_probability", 0.10)
    masks_per_sample = cfg.get("masks_per_sample", 2)
    total_epochs = cfg["epochs"]

    opt = torch.optim.AdamW(instashap.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.MSELoss()
    rng = np.random.default_rng(masking_config.seed + 999)
    best_loss, best_state, patience_cnt = float("inf"), None, 0
    history: list[dict[str, float]] = []
    use_bg = masking_config.strategy == "empirical_background"

    for epoch in range(total_epochs):
        instashap.train()
        train_losses = []
        n = len(X_train)
        indices = rng.permutation(n)
        bs = cfg["batch_size"]

        for batch_start in range(0, n, bs):
            batch_idx = indices[batch_start:batch_start + bs]
            actual_bs = len(batch_idx)
            x_batch = _np_to_tensor(X_train[batch_idx], device)

            effective_bs = actual_bs * masks_per_sample
            mask_np = standard_shapley_masks(effective_bs, n_features, rng, edge_prob)
            x_repeated = x_batch.repeat(masks_per_sample, 1)
            feature_mask = _np_to_tensor(mask_np, device)

            # Compute surrogate targets (frozen)
            with torch.no_grad():
                if use_bg and background_bank is not None:
                    masked_inputs = apply_background_mask(
                        x_repeated, feature_mask, preprocessor, background_bank, rng, K=1)
                else:
                    masked_inputs = apply_zero_mask(x_repeated, feature_mask, preprocessor)
                targets = surrogate(masked_inputs, feature_mask)

            # InstaSHAP forward (with gradient)
            predictions = instashap.masked_forward(x_repeated, feature_mask)
            loss = criterion(predictions, targets)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())

        # Validation
        instashap.eval()
        val_losses = []
        n_val = len(X_val)
        with torch.no_grad():
            for vstart in range(0, n_val, bs):
                vidx = np.arange(vstart, min(vstart + bs, n_val))
                x_vb = _np_to_tensor(X_val[vidx], device)
                actual_vbs = len(vidx)
                vmask_np = standard_shapley_masks(actual_vbs, n_features, rng, edge_prob)
                vmask = _np_to_tensor(vmask_np, device)

                if use_bg and background_bank is not None:
                    vmasked = apply_background_mask(x_vb, vmask, preprocessor, background_bank, rng, K=1)
                else:
                    vmasked = apply_zero_mask(x_vb, vmask, preprocessor)
                vtargets = surrogate(vmasked, vmask)
                vpred = instashap.masked_forward(x_vb, vmask)
                vloss = criterion(vpred, vtargets)
                val_losses.append(vloss.item())

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        history.append({"train_loss": tl, "val_loss": vl})

        if vl < best_loss:
            best_loss, best_state, patience_cnt = vl, copy.deepcopy(instashap.state_dict()), 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["patience"]:
                break

    if best_state:
        instashap.load_state_dict(best_state)
    return instashap, history
