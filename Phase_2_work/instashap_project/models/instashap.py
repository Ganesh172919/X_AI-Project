"""InstaSHAP additive architecture following Equation (20) of the paper."""

from __future__ import annotations

import torch

from instashap_project.data.preprocessing import TabularPreprocessor
from instashap_project.models.gam import GAMModel


class InstaSHAPModel(GAMModel):
    """Masked additive model whose full-pass component outputs serve as SHAP attributions."""

    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        output_dim: int,
        hidden_dims: list[int],
        interactions: list[tuple[str, str]] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            preprocessor=preprocessor,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            interactions=interactions,
            dropout=dropout,
        )

    def masked_forward(self, inputs: torch.Tensor, feature_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the paper's masked objective."""

        return super().forward(inputs, feature_mask=feature_mask)

    def explain(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return direct per-feature attributions in one forward pass."""

        return self.feature_attributions(inputs)
