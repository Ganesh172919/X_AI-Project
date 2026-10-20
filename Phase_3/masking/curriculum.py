"""Innovation 2: Curriculum-weighted Shapley mask sampling."""

from __future__ import annotations

from math import comb
import numpy as np


def _shapley_weights(num_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute the theoretical Shapley kernel weights for coalition sizes 1..n-1."""
    sizes = np.arange(1, num_features, dtype=int)
    weights = np.asarray(
        [1.0 / (comb(num_features, int(s)) * s * (num_features - int(s))) for s in sizes],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    return sizes, weights


def curriculum_shapley_masks(
    batch_size: int,
    num_features: int,
    rng: np.random.Generator,
    epoch: int,
    total_epochs: int,
    warmup_frac: float = 0.25,
    standard_frac: float = 0.40,
    edge_mask_probability: float = 0.10,
) -> np.ndarray:
    """Sample masks with curriculum-based temperature scheduling.

    Three phases:
      - Warm-up (0 to warmup_frac): favor large coalitions (temperature=3.0)
      - Standard (warmup_frac to warmup_frac+standard_frac): Shapley kernel (temperature=1.0)
      - Hard (remaining): emphasize sparse coalitions (temperature=0.3)
    """
    progress = epoch / max(total_epochs, 1)

    if progress < warmup_frac:
        temperature = 3.0   # Flatten → favor larger coalition sizes
    elif progress < warmup_frac + standard_frac:
        temperature = 1.0   # Standard Shapley kernel
    else:
        temperature = 0.3   # Sharpen → emphasize extreme (small) sizes

    sizes, base_weights = _shapley_weights(num_features)

    # Apply temperature
    tempered = base_weights ** (1.0 / temperature)
    tempered = tempered / tempered.sum()

    masks = np.zeros((batch_size, num_features), dtype=np.float32)
    for row in range(batch_size):
        coin = rng.random()
        if coin < edge_mask_probability / 2.0:
            masks[row, :] = 0.0
            continue
        if coin < edge_mask_probability:
            masks[row, :] = 1.0
            continue
        subset_size = int(rng.choice(sizes, p=tempered))
        chosen = rng.choice(num_features, size=subset_size, replace=False)
        masks[row, chosen] = 1.0

    return masks


def standard_shapley_masks(
    batch_size: int,
    num_features: int,
    rng: np.random.Generator,
    edge_mask_probability: float = 0.0,
) -> np.ndarray:
    """Standard Shapley kernel masks (Phase 2 baseline, temperature=1.0)."""
    sizes, weights = _shapley_weights(num_features)
    masks = np.zeros((batch_size, num_features), dtype=np.float32)
    for row in range(batch_size):
        coin = rng.random()
        if coin < edge_mask_probability / 2.0:
            continue
        if coin < edge_mask_probability:
            masks[row, :] = 1.0
            continue
        subset_size = int(rng.choice(sizes, p=weights))
        chosen = rng.choice(num_features, size=subset_size, replace=False)
        masks[row, chosen] = 1.0
    return masks
