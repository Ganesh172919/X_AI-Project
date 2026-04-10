"""Black-box MLP and surrogate model definitions."""

from __future__ import annotations

import torch
from torch import nn


def _build_mlp_layers(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class TabularMLP(nn.Module):
    """Feed-forward network used as the black-box baseline."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.network = _build_mlp_layers(input_dim, hidden_dims, output_dim, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class MaskedSurrogateMLP(nn.Module):
    """Mask-aware surrogate approximating f(x; S)."""
    def __init__(self, feature_dim: int, num_original_features: int, output_dim: int,
                 hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        self.network = _build_mlp_layers(feature_dim + num_original_features, hidden_dims, output_dim, dropout)

    def forward(self, masked_inputs: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([masked_inputs, feature_mask], dim=1))


class SurrogateEnsemble(nn.Module):
    """Innovation 3: Ensemble of surrogates — averaged predictions reduce noise."""
    def __init__(self, surrogates: list[MaskedSurrogateMLP]) -> None:
        super().__init__()
        self.surrogates = nn.ModuleList(surrogates)

    def forward(self, masked_inputs: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        outputs = [s(masked_inputs, feature_mask) for s in self.surrogates]
        return torch.stack(outputs, dim=0).mean(dim=0)

    def forward_all(self, masked_inputs: torch.Tensor, feature_mask: torch.Tensor) -> list[torch.Tensor]:
        """Return individual surrogate outputs for stability analysis."""
        return [s(masked_inputs, feature_mask) for s in self.surrogates]
