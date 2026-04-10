"""InstaSHAP explainer — single forward pass attribution."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch

from models.instashap import InstaSHAPModel


@dataclass
class InstaSHAPExplanationResult:
    grouped_values: np.ndarray   # (n_samples, n_features, n_outputs)
    feature_names: list[str]


def explain_instashap(
    model: InstaSHAPModel,
    X: np.ndarray,
    device: torch.device,
    feature_names: list[str],
    batch_size: int = 1024,
) -> InstaSHAPExplanationResult:
    """Generate attributions with a single forward pass per sample."""
    model.eval()
    all_attrs: list[np.ndarray] = []
    n = len(X)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = torch.from_numpy(X[start:start + batch_size].astype(np.float32)).to(device)
            attrs = model.explain(batch)  # (bs, n_features, n_outputs)
            all_attrs.append(attrs.cpu().numpy())
    return InstaSHAPExplanationResult(
        grouped_values=np.concatenate(all_attrs, axis=0),
        feature_names=feature_names,
    )
