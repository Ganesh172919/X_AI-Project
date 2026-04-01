"""Adaptive strategy that upgrades the surrogate only when interactions are needed."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from phase2.models.gam_surrogate import evaluate_surrogate_fidelity, train_gam_surrogate
from phase3.extension.interaction_aware_surrogate import train_interaction_aware_surrogate


@dataclass
class AdaptiveSurrogateResult:
    """Decision record for the adaptive surrogate strategy."""

    surrogate: object
    chosen_mode: str
    additive_metrics: dict[str, float]
    final_metrics: dict[str, float]


def fit_adaptive_surrogate(
    X_train: pd.DataFrame,
    train_predictions,
    X_validation: pd.DataFrame,
    validation_predictions,
    feature_names: list[str],
    fidelity_threshold: float = 0.95,
    interaction_pairs: list[tuple[int | str, int | str]] | None = None,
    interaction_count: int = 5,
) -> AdaptiveSurrogateResult:
    """Fit an additive surrogate first, then upgrade if fidelity is too low."""
    additive_bundle = train_gam_surrogate(
        X_train=X_train,
        black_box_predictions=train_predictions,
        feature_names=feature_names,
        interactions=0,
    )
    additive_metrics = evaluate_surrogate_fidelity(
        surrogate=additive_bundle.surrogate,
        X_eval=X_validation,
        black_box_predictions=validation_predictions,
    )
    if additive_metrics["r2"] >= fidelity_threshold:
        return AdaptiveSurrogateResult(
            surrogate=additive_bundle.surrogate,
            chosen_mode="additive",
            additive_metrics=additive_metrics,
            final_metrics=additive_metrics,
        )

    interaction_bundle = train_interaction_aware_surrogate(
        X_train=X_train,
        black_box_predictions=train_predictions,
        feature_names=feature_names,
        interaction_pairs=interaction_pairs,
        interaction_count=interaction_count,
    )
    interaction_metrics = evaluate_surrogate_fidelity(
        surrogate=interaction_bundle.surrogate,
        X_eval=X_validation,
        black_box_predictions=validation_predictions,
    )
    if interaction_metrics["r2"] <= additive_metrics["r2"]:
        return AdaptiveSurrogateResult(
            surrogate=additive_bundle.surrogate,
            chosen_mode="additive",
            additive_metrics=additive_metrics,
            final_metrics=additive_metrics,
        )

    return AdaptiveSurrogateResult(
        surrogate=interaction_bundle.surrogate,
        chosen_mode="interaction_aware",
        additive_metrics=additive_metrics,
        final_metrics=interaction_metrics,
    )
