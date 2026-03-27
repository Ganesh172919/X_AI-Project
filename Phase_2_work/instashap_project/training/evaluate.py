"""Evaluation helpers shared by experiments and explainers."""

from __future__ import annotations

import numpy as np
import torch

from instashap_project.models.blackbox_model import RandomForestBlackBox
from instashap_project.utils.metrics import classification_metrics, regression_metrics


def predict_raw_outputs(model: object, X: np.ndarray, device: str | torch.device, batch_size: int = 1024) -> np.ndarray:
    """Return regression outputs or logits for any supported model."""

    if isinstance(model, RandomForestBlackBox):
        return model.predict_raw(X)

    torch_device = torch.device(device)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size].astype(np.float32)).to(torch_device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_targets(task: str, model: object, X: np.ndarray, device: str | torch.device) -> dict[str, np.ndarray]:
    """Get predictions in a task-aware format."""

    raw_outputs = predict_raw_outputs(model, X, device)
    if task == "regression":
        return {"raw": raw_outputs.reshape(-1), "predictions": raw_outputs.reshape(-1)}

    probabilities = torch.softmax(torch.from_numpy(raw_outputs), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {"raw": raw_outputs, "probabilities": probabilities, "predictions": predictions}


def evaluate_supervised_model(
    task: str,
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    device: str | torch.device,
) -> dict[str, float]:
    """Evaluate a trained model on regression or classification metrics."""

    outputs = predict_targets(task, model, X, device)
    if task == "regression":
        return regression_metrics(y, outputs["predictions"]).to_dict()
    return classification_metrics(y, outputs["probabilities"], outputs["predictions"]).to_dict()
