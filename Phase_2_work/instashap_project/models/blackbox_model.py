"""Black-box and surrogate model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import torch
from torch import nn


def _build_mlp_layers(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class TabularMLP(nn.Module):
    """Simple feed-forward network used as the black-box baseline."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.network = _build_mlp_layers(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MaskedSurrogateMLP(nn.Module):
    """Mask-aware surrogate approximating f(x; S) for the InstaSHAP objective."""

    def __init__(
        self,
        feature_dim: int,
        num_original_features: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = _build_mlp_layers(feature_dim + num_original_features, hidden_dims, output_dim, dropout)

    def forward(self, masked_inputs: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([masked_inputs, feature_mask], dim=1))


@dataclass(slots=True)
class RandomForestBlackBox:
    """Sklearn random forest wrapper with a black-box style interface."""

    task: Literal["regression", "classification"]
    random_state: int
    n_estimators: int = 300
    max_depth: int | None = None

    def __post_init__(self) -> None:
        if self.task == "regression":
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBlackBox":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba is only defined for classification tasks.")
        return self.model.predict_proba(X)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return np.asarray(self.model.predict(X), dtype=np.float32).reshape(-1, 1)
        probabilities = np.clip(self.model.predict_proba(X), 1e-8, 1.0)
        return np.log(probabilities).astype(np.float32)

