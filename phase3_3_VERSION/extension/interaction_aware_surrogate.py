"""Interaction-aware surrogate model utilities for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from phase2.models.gam_surrogate import (
    SurrogateBundle,
    evaluate_surrogate_fidelity,
    train_gam_surrogate,
)
from phase2.utils import SEED


@dataclass
class InteractionAwareSurrogateBundle(SurrogateBundle):
    """Specialized metadata wrapper for GA²M-style surrogates."""

    interaction_terms: list[str] | None = None


def _normalize_interaction_pairs(
    feature_names: list[str],
    interaction_pairs: list[tuple[int | str, int | str]] | None,
) -> list[tuple[int, int]] | None:
    """Convert user-friendly feature names into EBM term indices."""
    if interaction_pairs is None:
        return None
    index_lookup = {name: idx for idx, name in enumerate(feature_names)}
    normalized: list[tuple[int, int]] = []
    for left, right in interaction_pairs:
        left_idx = index_lookup[left] if isinstance(left, str) else int(left)
        right_idx = index_lookup[right] if isinstance(right, str) else int(right)
        normalized.append((left_idx, right_idx))
    return normalized


def train_interaction_aware_surrogate(
    X_train: pd.DataFrame,
    black_box_predictions,
    feature_names: list[str],
    interaction_pairs: list[tuple[int | str, int | str]] | None = None,
    interaction_count: int = 5,
    random_state: int = SEED,
    save_dir: Path | str | None = None,
) -> InteractionAwareSurrogateBundle:
    """Fit a GA²M-style surrogate with selected or automatically discovered interactions."""
    interactions = _normalize_interaction_pairs(feature_names, interaction_pairs)
    if interactions is None:
        interactions = min(interaction_count, max(1, len(feature_names) // 2))

    bundle = train_gam_surrogate(
        X_train=X_train,
        black_box_predictions=black_box_predictions,
        feature_names=feature_names,
        interactions=interactions,
        random_state=random_state,
        save_dir=save_dir,
    )
    interaction_terms = [
        term_name
        for term_name, feature_group in zip(bundle.surrogate.term_names_, bundle.surrogate.term_features_)
        if len(feature_group) > 1
    ]
    return InteractionAwareSurrogateBundle(
        surrogate=bundle.surrogate,
        feature_names=feature_names,
        interactions=interactions,
        artifact_path=bundle.artifact_path,
        interaction_terms=interaction_terms,
    )


def interaction_surrogate_fidelity(
    surrogate,
    X_eval: pd.DataFrame,
    black_box_predictions,
) -> dict[str, float]:
    """Alias to keep the interaction-aware API readable at call sites."""
    return evaluate_surrogate_fidelity(surrogate, X_eval, black_box_predictions)
