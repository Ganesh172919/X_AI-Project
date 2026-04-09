"""Baseline zero-masking strategy (reproduces Phase 2 behavior)."""

from __future__ import annotations
import numpy as np
import torch

from data.preprocessing import TabularPreprocessor


def expand_feature_mask_torch(preprocessor: TabularPreprocessor, feature_mask: torch.Tensor) -> torch.Tensor:
    """Expand original-feature mask to transformed-column mask."""
    parts: list[torch.Tensor] = []
    for fi, fn in enumerate(preprocessor.feature_order):
        width = preprocessor.group(fn).width
        parts.append(feature_mask[:, [fi]].repeat(1, width))
    return torch.cat(parts, dim=1)


def apply_zero_mask(
    inputs: torch.Tensor,
    feature_mask: torch.Tensor,
    preprocessor: TabularPreprocessor,
) -> torch.Tensor:
    """Apply zero-masking: absent features become 0 in transformed space.

    This is the exact Phase 2 behavior: ``inputs * expanded_mask``.
    """
    expanded = expand_feature_mask_torch(preprocessor, feature_mask)
    return inputs * expanded
