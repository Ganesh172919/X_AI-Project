"""Evaluation helpers for Phase 3."""

from __future__ import annotations
import numpy as np
import torch
from torch import nn


def predict_raw_outputs(
    model: nn.Module | object,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Get raw (logit/regression) outputs from a PyTorch model or sklearn wrapper."""
    if hasattr(model, "predict_raw"):
        return model.predict_raw(X)

    model.eval()
    outputs: list[np.ndarray] = []
    n = len(X)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
            out = model(batch)
            outputs.append(out.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def predict_classes(
    model: nn.Module | object,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict class labels."""
    raw = predict_raw_outputs(model, X, device, batch_size)
    if raw.shape[1] == 1:
        return (raw[:, 0] > 0).astype(int)
    return raw.argmax(axis=1)


def predict_probabilities(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predict class probabilities via softmax."""
    raw = predict_raw_outputs(model, X, device, batch_size)
    raw_t = torch.from_numpy(raw)
    probs = torch.softmax(raw_t, dim=1).numpy()
    return probs
