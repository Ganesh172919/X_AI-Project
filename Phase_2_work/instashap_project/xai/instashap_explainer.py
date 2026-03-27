"""Direct explainer for trained InstaSHAP models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class InstaSHAPExplanationResult:
    """Result object containing feature attributions."""

    grouped_values: np.ndarray


class InstaSHAPExplainer:
    """One-pass explanation wrapper."""

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = torch.device(device)

    def explain(self, transformed_inputs: np.ndarray, batch_size: int = 1024) -> InstaSHAPExplanationResult:
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(transformed_inputs), batch_size):
                batch = torch.from_numpy(transformed_inputs[start : start + batch_size].astype(np.float32)).to(self.device)
                outputs.append(self.model.explain(batch).cpu().numpy())
        return InstaSHAPExplanationResult(grouped_values=np.concatenate(outputs, axis=0))
