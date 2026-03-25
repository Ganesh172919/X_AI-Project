"""Training helpers for black-box, surrogate, GAM, and InstaSHAP models."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from instashap_project.data.preprocessing import TabularPreprocessor
from instashap_project.models.blackbox_model import MaskedSurrogateMLP, RandomForestBlackBox, TabularMLP
from instashap_project.models.gam import GAMModel
from instashap_project.models.instashap import InstaSHAPModel
from instashap_project.utils.reproducibility import ensure_dir

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency at runtime
    SummaryWriter = None


@dataclass(slots=True)
class TrainingResult:
    """Bundle together a trained model and its learning curves."""

    model: object
    history: list[dict[str, float]]


def _make_tensor_loader(X: np.ndarray, y: np.ndarray | None, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = torch.from_numpy(X.astype(np.float32))
    if y is None:
        dataset = TensorDataset(x_tensor)
    else:
        dataset = TensorDataset(x_tensor, torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _expand_feature_mask_torch(preprocessor: TabularPreprocessor, feature_mask: torch.Tensor) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for feature_index, feature_name in enumerate(preprocessor.feature_order):
        width = preprocessor.group(feature_name).width
        parts.append(feature_mask[:, [feature_index]].repeat(1, width))
    return torch.cat(parts, dim=1)


def _shapley_size_distribution(num_features: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.arange(1, num_features, dtype=int)
    weights = np.asarray(
        [1.0 / (comb(num_features, int(size)) * size * (num_features - int(size))) for size in sizes],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    return sizes, weights


def sample_shapley_feature_masks(
    batch_size: int,
    num_features: int,
    rng: np.random.Generator,
    edge_mask_probability: float = 0.0,
) -> np.ndarray:
    """Draw masks from the Shapley kernel, with optional empty/full edge masks for stability."""

    sizes, probabilities = _shapley_size_distribution(num_features)
    masks = np.zeros((batch_size, num_features), dtype=np.float32)
    for row in range(batch_size):
        coin = rng.random()
        if coin < edge_mask_probability / 2.0:
            masks[row, :] = 0.0
            continue
        if coin < edge_mask_probability:
            masks[row, :] = 1.0
            continue
        subset_size = int(rng.choice(sizes, p=probabilities))
        chosen = rng.choice(num_features, size=subset_size, replace=False)
        masks[row, chosen] = 1.0
    return masks


def _create_writer(log_dir: str | Path | None) -> Any:
    if log_dir is None or SummaryWriter is None:
        return None
    ensure_dir(log_dir)
    return SummaryWriter(log_dir=str(log_dir))


def _supervised_loss(task: str, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if task == "regression":
        return nn.functional.mse_loss(outputs, targets)
    return nn.functional.cross_entropy(outputs, targets.squeeze(-1).long())


def _validation_loss(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        loader = _make_tensor_loader(X_val, y_val, batch_size=batch_size, shuffle=False)
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            losses.append(float(_supervised_loss(task, model(inputs), targets).item()))
    return float(np.mean(losses))


def train_blackbox_model(
    task: str,
    input_dim: int,
    output_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    log_dir: str | Path | None = None,
) -> TrainingResult:
    """Train the baseline black-box model."""

    model_type = str(config.get("model_type", "mlp")).lower()
    if model_type == "random_forest":
        forest = RandomForestBlackBox(task=task, random_state=seed)
        forest.fit(X_train, y_train.reshape(-1))
        return TrainingResult(model=forest, history=[])

    model = TabularMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=list(config["hidden_dims"]),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    loader = _make_tensor_loader(X_train, y_train, batch_size=int(config["batch_size"]), shuffle=True)
    history: list[dict[str, float]] = []
    writer = _create_writer(log_dir)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = int(config.get("patience", 5))
    patience_counter = 0

    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _supervised_loss(task, model(inputs), targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_loss = _validation_loss(model, X_val, y_val, task, device, int(config["batch_size"]))
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "val_loss": val_loss}
        history.append(record)
        if writer is not None:
            writer.add_scalar("loss/train", record["train_loss"], epoch + 1)
            writer.add_scalar("loss/val", record["val_loss"], epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if writer is not None:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(model=model, history=history)


def _raw_outputs(model: object, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    if isinstance(model, RandomForestBlackBox):
        return model.predict_raw(X)

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size].astype(np.float32)).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_masked_surrogate(
    blackbox_model: object,
    preprocessor: TabularPreprocessor,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
    log_dir: str | Path | None = None,
) -> TrainingResult:
    """Train a mask-aware surrogate approximating the masked black-box function f(x; S)."""

    output_dim = _raw_outputs(blackbox_model, X_train[:16], device).shape[1]
    model = MaskedSurrogateMLP(
        feature_dim=preprocessor.input_dim,
        num_original_features=preprocessor.num_original_features,
        output_dim=output_dim,
        hidden_dims=list(config["hidden_dims"]),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    train_targets = _raw_outputs(blackbox_model, X_train, device)
    val_targets = _raw_outputs(blackbox_model, X_val, device)
    train_loader = _make_tensor_loader(X_train, train_targets, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = _make_tensor_loader(X_val, val_targets, batch_size=int(config["batch_size"]), shuffle=False)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    writer = _create_writer(log_dir)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = int(config.get("patience", 5))
    patience_counter = 0

    for epoch in range(int(config["epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            feature_mask = torch.from_numpy(
                sample_shapley_feature_masks(
                    batch_size=len(inputs),
                    num_features=preprocessor.num_original_features,
                    rng=rng,
                    edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
                )
            ).to(device)
            expanded_mask = _expand_feature_mask_torch(preprocessor, feature_mask)
            predictions = model(inputs * expanded_mask, feature_mask)
            loss = nn.functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                feature_mask = torch.from_numpy(
                    sample_shapley_feature_masks(
                        batch_size=len(inputs),
                        num_features=preprocessor.num_original_features,
                        rng=rng,
                        edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
                    )
                ).to(device)
                expanded_mask = _expand_feature_mask_torch(preprocessor, feature_mask)
                predictions = model(inputs * expanded_mask, feature_mask)
                val_losses.append(float(nn.functional.mse_loss(predictions, targets).item()))

        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        if writer is not None:
            writer.add_scalar("loss/train", record["train_loss"], epoch + 1)
            writer.add_scalar("loss/val", record["val_loss"], epoch + 1)

        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if writer is not None:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(model=model, history=history)


def train_gam_model(
    task: str,
    preprocessor: TabularPreprocessor,
    output_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    interactions: list[tuple[str, str]],
    device: torch.device,
    log_dir: str | Path | None = None,
) -> TrainingResult:
    """Train a vanilla GAM or GAM-2 model on the original labels."""

    model = GAMModel(
        preprocessor=preprocessor,
        output_dim=output_dim,
        hidden_dims=list(config["hidden_dims"]),
        interactions=interactions,
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    train_loader = _make_tensor_loader(X_train, y_train, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = _make_tensor_loader(X_val, y_val, batch_size=int(config["batch_size"]), shuffle=False)

    history: list[dict[str, float]] = []
    writer = _create_writer(log_dir)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = int(config.get("patience", 6))
    patience_counter = 0

    for epoch in range(int(config["epochs"])):
        model.train()
        train_losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _supervised_loss(task, model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                val_losses.append(float(_supervised_loss(task, model(inputs), targets).item()))

        record = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        if writer is not None:
            writer.add_scalar("loss/train", record["train_loss"], epoch + 1)
            writer.add_scalar("loss/val", record["val_loss"], epoch + 1)
        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if writer is not None:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(model=model, history=history)


def train_instashap_model(
    preprocessor: TabularPreprocessor,
    surrogate_model: nn.Module,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict[str, Any],
    interactions: list[tuple[str, str]],
    device: torch.device,
    seed: int,
    log_dir: str | Path | None = None,
) -> TrainingResult:
    """Train the masked additive InstaSHAP model against the surrogate's masked outputs."""

    surrogate_model.eval()
    with torch.no_grad():
        probe_inputs = torch.from_numpy(X_train[: min(len(X_train), 16)].astype(np.float32)).to(device)
        probe_mask = torch.ones(probe_inputs.shape[0], preprocessor.num_original_features, device=device)
        output_dim = int(surrogate_model(probe_inputs, probe_mask).shape[1])
    model = InstaSHAPModel(
        preprocessor=preprocessor,
        output_dim=output_dim,
        hidden_dims=list(config["hidden_dims"]),
        interactions=interactions,
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    train_loader = _make_tensor_loader(X_train, None, int(config["batch_size"]), True)
    val_loader = _make_tensor_loader(X_val, None, int(config["batch_size"]), False)
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    writer = _create_writer(log_dir)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    patience = int(config.get("patience", 6))
    patience_counter = 0

    for epoch in range(int(config["epochs"])):
        model.train()
        train_losses: list[float] = []
        for (inputs,) in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad(set_to_none=True)
            feature_mask = torch.from_numpy(
                sample_shapley_feature_masks(
                    batch_size=len(inputs),
                    num_features=preprocessor.num_original_features,
                    rng=rng,
                    edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
                )
            ).to(device)
            expanded_mask = _expand_feature_mask_torch(preprocessor, feature_mask)
            with torch.no_grad():
                targets = surrogate_model(inputs * expanded_mask, feature_mask)
            predictions = model.masked_forward(inputs, feature_mask)
            loss = nn.functional.mse_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (inputs,) in val_loader:
                inputs = inputs.to(device)
                feature_mask = torch.from_numpy(
                    sample_shapley_feature_masks(
                        batch_size=len(inputs),
                        num_features=preprocessor.num_original_features,
                        rng=rng,
                        edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
                    )
                ).to(device)
                expanded_mask = _expand_feature_mask_torch(preprocessor, feature_mask)
                targets = surrogate_model(inputs * expanded_mask, feature_mask)
                predictions = model.masked_forward(inputs, feature_mask)
                val_losses.append(float(nn.functional.mse_loss(predictions, targets).item()))

        record = {"epoch": epoch + 1, "train_loss": float(np.mean(train_losses)), "val_loss": float(np.mean(val_losses))}
        history.append(record)
        if writer is not None:
            writer.add_scalar("loss/train", record["train_loss"], epoch + 1)
            writer.add_scalar("loss/val", record["val_loss"], epoch + 1)
        if record["val_loss"] < best_val:
            best_val = record["val_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if writer is not None:
        writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainingResult(model=model, history=history)
