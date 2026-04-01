"""Masking strategies for surrogate and InstaSHAP training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from instashap_project.data.preprocessing import TabularPreprocessor


MaskingStrategyName = Literal["zero_mask", "empirical_background"]


@dataclass(slots=True)
class MaskingConfig:
    """Configuration for coalition masking."""

    strategy: MaskingStrategyName
    background_bank_size: int
    background_samples_train: int
    background_samples_eval: int
    seed: int


def build_background_bank(
    transformed_train: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """Create a fixed bank of real transformed training rows used for masking."""

    if len(transformed_train) == 0:
        raise ValueError("Cannot build a background bank from an empty training matrix.")
    if max_rows <= 0:
        raise ValueError("background bank size must be positive.")

    if len(transformed_train) <= max_rows:
        return np.asarray(transformed_train, dtype=np.float32)

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(transformed_train), size=max_rows, replace=False)
    return np.asarray(transformed_train[selected], dtype=np.float32)


def build_masked_batch(
    *,
    preprocessor: TabularPreprocessor,
    transformed_inputs: np.ndarray,
    feature_mask: np.ndarray,
    strategy: MaskingStrategyName,
    rng: np.random.Generator,
    background_bank: np.ndarray | None,
    background_samples: int,
) -> np.ndarray:
    """Construct masked coalition inputs for one batch.

    Returns an array of shape ``[batch, samples, input_dim]`` so callers can
    average model outputs across multiple empirical background draws.
    """

    expanded_mask = preprocessor.expand_feature_mask(feature_mask)
    base = np.asarray(transformed_inputs, dtype=np.float32)

    if strategy == "zero_mask":
        masked = base * expanded_mask
        return masked[:, None, :].astype(np.float32)

    if background_bank is None:
        raise ValueError("background_bank is required for empirical_background masking.")
    if background_samples <= 0:
        raise ValueError("background_samples must be positive.")

    batch_size, input_dim = base.shape
    sampled_background = np.empty((batch_size, background_samples, input_dim), dtype=np.float32)
    for row_index in range(batch_size):
        visible = expanded_mask[row_index] > 0.5
        if visible.any():
            distances = np.square(background_bank - base[row_index])[:, visible].sum(axis=1)
            nearest_indices = np.argsort(distances)[:background_samples]
            if len(nearest_indices) < background_samples:
                chosen_indices = rng.choice(nearest_indices, size=background_samples, replace=True)
            else:
                chosen_indices = nearest_indices
        else:
            chosen_indices = rng.integers(0, len(background_bank), size=background_samples)
        sampled_background[row_index] = background_bank[chosen_indices]
    inputs_3d = np.repeat(base[:, None, :], background_samples, axis=1)
    expanded_mask_3d = np.repeat(expanded_mask[:, None, :], background_samples, axis=1)
    masked = expanded_mask_3d * inputs_3d + (1.0 - expanded_mask_3d) * sampled_background
    return masked.reshape(batch_size, background_samples, input_dim).astype(np.float32)
