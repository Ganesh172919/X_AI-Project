"""Generalized additive neural networks."""

from __future__ import annotations
from dataclasses import dataclass

import torch
from torch import nn
from data.preprocessing import TabularPreprocessor


def component_name(features: tuple[str, ...]) -> str:
    return "__".join(features)


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    features: tuple[str, ...]
    input_dim: int


class ComponentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class GAMModel(nn.Module):
    """Neural GAM supporting univariate and pairwise interactions."""
    def __init__(self, preprocessor: TabularPreprocessor, output_dim: int, hidden_dims: list[int],
                 interactions: list[tuple[str, str]] | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.feature_order = list(preprocessor.feature_order)
        self.output_dim = output_dim
        self.interactions = [tuple(p) for p in (interactions or [])]
        self.bias = nn.Parameter(torch.zeros(output_dim))

        specs: list[ComponentSpec] = []
        for fn in self.feature_order:
            specs.append(ComponentSpec(features=(fn,), input_dim=preprocessor.group(fn).width))
        for f1, f2 in self.interactions:
            specs.append(ComponentSpec(features=(f1, f2), input_dim=len(preprocessor.slices_for((f1, f2)))))

        self.component_specs = specs
        self.components = nn.ModuleDict({
            component_name(s.features): ComponentMLP(s.input_dim, output_dim, hidden_dims, dropout)
            for s in specs
        })

    def _gate(self, features: tuple[str, ...], feature_mask: torch.Tensor | None, bs: int, device: torch.device) -> torch.Tensor:
        if feature_mask is None:
            return torch.ones(bs, 1, device=device)
        indices = [self.feature_order.index(f) for f in features]
        return feature_mask[:, indices].prod(dim=1, keepdim=True)

    def _component_inputs(self, inputs: torch.Tensor, features: tuple[str, ...]) -> torch.Tensor:
        return inputs[:, self.preprocessor.slices_for(features)]

    def component_contributions(self, inputs: torch.Tensor, feature_mask: torch.Tensor | None = None) -> dict[tuple[str, ...], torch.Tensor]:
        contribs: dict[tuple[str, ...], torch.Tensor] = {}
        bs = inputs.shape[0]
        for spec in self.component_specs:
            net = self.components[component_name(spec.features)]
            out = net(self._component_inputs(inputs, spec.features))
            out = out * self._gate(spec.features, feature_mask, bs, inputs.device)
            contribs[spec.features] = out
        return contribs

    def forward(self, inputs: torch.Tensor, feature_mask: torch.Tensor | None = None) -> torch.Tensor:
        total = self.bias.unsqueeze(0).expand(inputs.shape[0], -1)
        for c in self.component_contributions(inputs, feature_mask).values():
            total = total + c
        return total

    def feature_attributions(self, inputs: torch.Tensor) -> torch.Tensor:
        contribs = self.component_contributions(inputs, feature_mask=None)
        attrs = torch.zeros(inputs.shape[0], len(self.feature_order), self.output_dim, device=inputs.device)
        for fi, fn in enumerate(self.feature_order):
            attrs[:, fi, :] += contribs[(fn,)]
        for pair in self.interactions:
            shared = contribs[pair] / 2.0
            attrs[:, self.feature_order.index(pair[0]), :] += shared
            attrs[:, self.feature_order.index(pair[1]), :] += shared
        return attrs

    def single_component(self, inputs: torch.Tensor, features: tuple[str, ...]) -> torch.Tensor:
        return self.components[component_name(features)](self._component_inputs(inputs, features))
