"""Generalized additive neural networks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from instashap_project.data.preprocessing import TabularPreprocessor


def component_name(features: tuple[str, ...]) -> str:
    """Create a stable module key for an additive component."""

    return "__".join(features)


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Specification for one additive component."""

    features: tuple[str, ...]
    input_dim: int


class ComponentMLP(nn.Module):
    """Small MLP used for a single additive component."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class GAMModel(nn.Module):
    """Neural GAM supporting both univariate and pairwise interactions."""

    def __init__(
        self,
        preprocessor: TabularPreprocessor,
        output_dim: int,
        hidden_dims: list[int],
        interactions: list[tuple[str, str]] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.feature_order = list(preprocessor.feature_order)
        self.output_dim = output_dim
        self.interactions = [tuple(pair) for pair in (interactions or [])]
        self.bias = nn.Parameter(torch.zeros(output_dim))

        specs: list[ComponentSpec] = []
        for feature_name in self.feature_order:
            specs.append(ComponentSpec(features=(feature_name,), input_dim=preprocessor.group(feature_name).width))
        for first_feature, second_feature in self.interactions:
            specs.append(
                ComponentSpec(
                    features=(first_feature, second_feature),
                    input_dim=len(preprocessor.slices_for((first_feature, second_feature))),
                )
            )
        self.component_specs = specs
        self.components = nn.ModuleDict(
            {
                component_name(spec.features): ComponentMLP(
                    input_dim=spec.input_dim,
                    output_dim=output_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                )
                for spec in self.component_specs
            }
        )

    def _gate(self, features: tuple[str, ...], feature_mask: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if feature_mask is None:
            return torch.ones(batch_size, 1, device=device)
        indices = [self.feature_order.index(feature_name) for feature_name in features]
        return feature_mask[:, indices].prod(dim=1, keepdim=True)

    def _component_inputs(self, inputs: torch.Tensor, features: tuple[str, ...]) -> torch.Tensor:
        indices = self.preprocessor.slices_for(features)
        return inputs[:, indices]

    def component_contributions(
        self,
        inputs: torch.Tensor,
        feature_mask: torch.Tensor | None = None,
    ) -> dict[tuple[str, ...], torch.Tensor]:
        """Return additive component outputs, optionally gated by a feature mask."""

        contributions: dict[tuple[str, ...], torch.Tensor] = {}
        batch_size = inputs.shape[0]
        for spec in self.component_specs:
            network = self.components[component_name(spec.features)]
            component_output = network(self._component_inputs(inputs, spec.features))
            component_output = component_output * self._gate(spec.features, feature_mask, batch_size, inputs.device)
            contributions[spec.features] = component_output
        return contributions

    def forward(self, inputs: torch.Tensor, feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        total = self.bias.unsqueeze(0).expand(inputs.shape[0], -1)
        for contribution in self.component_contributions(inputs, feature_mask).values():
            total = total + contribution
        return total

    def feature_attributions(self, inputs: torch.Tensor) -> torch.Tensor:
        """Recover per-feature SHAP-style attributions from the purified additive components."""

        contributions = self.component_contributions(inputs, feature_mask=None)
        attributions = torch.zeros(inputs.shape[0], len(self.feature_order), self.output_dim, device=inputs.device)
        for feature_index, feature_name in enumerate(self.feature_order):
            attributions[:, feature_index, :] += contributions[(feature_name,)]
        for pair in self.interactions:
            shared = contributions[pair] / 2.0
            first_idx = self.feature_order.index(pair[0])
            second_idx = self.feature_order.index(pair[1])
            attributions[:, first_idx, :] += shared
            attributions[:, second_idx, :] += shared
        return attributions

    def single_component(self, inputs: torch.Tensor, features: tuple[str, ...]) -> torch.Tensor:
        """Evaluate one specific component without applying a mask."""

        return self.components[component_name(features)](self._component_inputs(inputs, features))

