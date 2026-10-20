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
from instashap_project.masking import MaskingConfig, build_masked_batch
from instashap_project.models.blackbox_model import MaskedSurrogateMLP, RandomForestBlackBox, TabularMLP
from instashap_project.models.gam import GAMModel
from instashap_project.models.instashap import InstaSHAPModel
from instashap_project.utils.reproducibility import ensure_dir

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
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
    for feature_name in preprocessor.feature_order:
        width = preprocessor.group(feature_name).width
        index = preprocessor.feature_index(feature_name)
        parts.append(feature_mask[:, [index]].repeat(1, width))
    return torch.cat(parts, dim=1)


def _repeat_feature_mask_torch(feature_mask: torch.Tensor, repeats: int) -> torch.Tensor:
    return feature_mask.unsqueeze(1).repeat(1, repeats, 1).reshape(-1, feature_mask.shape[1])


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
    """Draw masks from the Shapley kernel, with optional empty/full edge masks."""

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


def mean_blackbox_outputs(
    model: object,
    masked_inputs: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    batch_size_outer, background_samples, input_dim = masked_inputs.shape
    flat = masked_inputs.reshape(batch_size_outer * background_samples, input_dim)
    outputs = _raw_outputs(model, flat, device, batch_size=batch_size)
    return outputs.reshape(batch_size_outer, background_samples, -1).mean(axis=1)


def mean_surrogate_outputs(
    model: nn.Module,
    masked_inputs: np.ndarray,
    feature_mask: np.ndarray,
    original_inputs: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    batch_outer, background_samples, input_dim = masked_inputs.shape
    flat_inputs = masked_inputs.reshape(batch_outer * background_samples, input_dim).astype(np.float32)
    repeated_feature_mask = np.repeat(feature_mask, background_samples, axis=0).astype(np.float32)
    repeated_original_inputs = np.repeat(original_inputs, background_samples, axis=0).astype(np.float32)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(flat_inputs), batch_size):
            inputs_batch = torch.from_numpy(flat_inputs[start : start + batch_size]).to(device)
            mask_batch = torch.from_numpy(repeated_feature_mask[start : start + batch_size]).to(device)
            original_batch = torch.from_numpy(repeated_original_inputs[start : start + batch_size]).to(device)
            outputs.append(model(inputs_batch, mask_batch, original_batch).cpu().numpy())
    merged = np.concatenate(outputs, axis=0)
    return merged.reshape(batch_outer, background_samples, -1).mean(axis=1)


def _mean_surrogate_outputs_per_realization(
    model: nn.Module,
    masked_inputs: np.ndarray,
    feature_mask: np.ndarray,
    original_inputs: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return surrogate outputs for each masked realization without averaging."""

    model.eval()
    batch_outer, background_samples, input_dim = masked_inputs.shape
    flat_inputs = masked_inputs.reshape(batch_outer * background_samples, input_dim).astype(np.float32)
    repeated_feature_mask = np.repeat(feature_mask, background_samples, axis=0).astype(np.float32)
    repeated_original_inputs = np.repeat(original_inputs, background_samples, axis=0).astype(np.float32)

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(flat_inputs), batch_size):
            inputs_batch = torch.from_numpy(flat_inputs[start : start + batch_size]).to(device)
            mask_batch = torch.from_numpy(repeated_feature_mask[start : start + batch_size]).to(device)
            original_batch = torch.from_numpy(repeated_original_inputs[start : start + batch_size]).to(device)
            outputs.append(model(inputs_batch, mask_batch, original_batch).cpu().numpy())
    merged = np.concatenate(outputs, axis=0)
    return merged.reshape(batch_outer, background_samples, -1)


def train_masked_surrogate(
    blackbox_model: object,
    preprocessor: TabularPreprocessor,
    X_train: np.ndarray,
    X_val: np.ndarray,
    config: dict[str, Any],
    masking_config: MaskingConfig,
    background_bank: np.ndarray | None,
    device: torch.device,
    seed: int,
    log_dir: str | Path | None = None,
) -> TrainingResult:
    """Train a mask-aware surrogate approximating the masked black-box function."""

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

    train_loader = _make_tensor_loader(X_train, None, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = _make_tensor_loader(X_val, None, batch_size=int(config["batch_size"]), shuffle=False)
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
        for (inputs,) in train_loader:
            inputs_np = inputs.numpy().astype(np.float32)
            feature_mask_np = sample_shapley_feature_masks(
                batch_size=len(inputs_np),
                num_features=preprocessor.num_original_features,
                rng=rng,
                edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
            )
            masked_inputs = build_masked_batch(
                preprocessor=preprocessor,
                transformed_inputs=inputs_np,
                feature_mask=feature_mask_np,
                strategy=masking_config.strategy,
                rng=rng,
                background_bank=background_bank,
                background_samples=masking_config.background_samples_train,
            )
            flat_targets = _raw_outputs(
                blackbox_model,
                masked_inputs.reshape(-1, masked_inputs.shape[-1]),
                device,
            ).reshape(len(inputs_np), masked_inputs.shape[1], -1)
            targets = flat_targets.mean(axis=1)

            flat_inputs = masked_inputs.reshape(-1, masked_inputs.shape[-1])
            repeated_original_inputs = np.repeat(inputs_np, masked_inputs.shape[1], axis=0).astype(np.float32)
            feature_mask = torch.from_numpy(feature_mask_np).to(device)
            repeated_feature_mask = _repeat_feature_mask_torch(feature_mask, masked_inputs.shape[1])
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                torch.from_numpy(flat_inputs).to(device),
                repeated_feature_mask,
                torch.from_numpy(repeated_original_inputs).to(device),
            ).reshape(len(inputs_np), masked_inputs.shape[1], -1)
            prediction_mean = predictions.mean(dim=1)
            target_tensor = torch.from_numpy(flat_targets.astype(np.float32)).to(device)
            target_mean_tensor = torch.from_numpy(targets.astype(np.float32)).to(device)
            loss = (
                nn.functional.mse_loss(predictions, target_tensor)
                + nn.functional.mse_loss(prediction_mean, target_mean_tensor)
            ) / 2.0
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (inputs,) in val_loader:
                inputs_np = inputs.numpy().astype(np.float32)
                feature_mask_np = sample_shapley_feature_masks(
                    batch_size=len(inputs_np),
                    num_features=preprocessor.num_original_features,
                    rng=rng,
                    edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
                )
                masked_inputs = build_masked_batch(
                    preprocessor=preprocessor,
                    transformed_inputs=inputs_np,
                    feature_mask=feature_mask_np,
                    strategy=masking_config.strategy,
                    rng=rng,
                    background_bank=background_bank,
                    background_samples=masking_config.background_samples_eval,
                )
                flat_targets = _raw_outputs(
                    blackbox_model,
                    masked_inputs.reshape(-1, masked_inputs.shape[-1]),
                    device,
                ).reshape(len(inputs_np), masked_inputs.shape[1], -1)
                targets = flat_targets.mean(axis=1)
                predictions = mean_surrogate_outputs(
                    model,
                    masked_inputs,
                    feature_mask_np,
                    inputs_np,
                    device=device,
                )
                per_realization = _mean_surrogate_outputs_per_realization(
                    model,
                    masked_inputs,
                    feature_mask_np,
                    inputs_np,
                    device=device,
                )
                val_losses.append(
                    float(
                        (
                            np.mean(np.square(predictions - targets))
                            + np.mean(np.square(per_realization - flat_targets))
                        )
                        / 2.0
                    )
                )

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
    masking_config: MaskingConfig,
    background_bank: np.ndarray | None,
    device: torch.device,
    seed: int,
    log_dir: str | Path | None = None,
) -> TrainingResult:
    """Train the additive InstaSHAP model against the surrogate's coalition outputs."""

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
            inputs_np = inputs.numpy().astype(np.float32)
            feature_mask_np = sample_shapley_feature_masks(
                batch_size=len(inputs_np),
                num_features=preprocessor.num_original_features,
                rng=rng,
                edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
            )
            masked_inputs = build_masked_batch(
                preprocessor=preprocessor,
                transformed_inputs=inputs_np,
                feature_mask=feature_mask_np,
                strategy=masking_config.strategy,
                rng=rng,
                background_bank=background_bank,
                background_samples=masking_config.background_samples_train,
            )
            targets = mean_surrogate_outputs(
                surrogate_model,
                masked_inputs,
                feature_mask_np,
                inputs_np,
                device=device,
            )

            optimizer.zero_grad(set_to_none=True)
            predictions = model.masked_forward(
                torch.from_numpy(inputs_np).to(device),
                torch.from_numpy(feature_mask_np).to(device),
            )
            loss = nn.functional.mse_loss(predictions, torch.from_numpy(targets.astype(np.float32)).to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for (inputs,) in val_loader:
                inputs_np = inputs.numpy().astype(np.float32)
                feature_mask_np = sample_shapley_feature_masks(
                    batch_size=len(inputs_np),
                    num_features=preprocessor.num_original_features,
                    rng=rng,
                    edge_mask_probability=float(config.get("edge_mask_probability", 0.0)),
                )
                masked_inputs = build_masked_batch(
                    preprocessor=preprocessor,
                    transformed_inputs=inputs_np,
                    feature_mask=feature_mask_np,
                    strategy=masking_config.strategy,
                    rng=rng,
                    background_bank=background_bank,
                    background_samples=masking_config.background_samples_eval,
                )
                targets = mean_surrogate_outputs(
                    surrogate_model,
                    masked_inputs,
                    feature_mask_np,
                    inputs_np,
                    device=device,
                )
                predictions = model.masked_forward(
                    torch.from_numpy(inputs_np).to(device),
                    torch.from_numpy(feature_mask_np).to(device),
                ).cpu().numpy()
                val_losses.append(float(np.mean(np.square(predictions - targets))))

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
