"""Interaction-aware analytical attribution for GA²M-style surrogates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from phase2.models.instashap import InstaShapOutput, compute_instashap_values


@dataclass
class EnhancedInstaShapOutput:
    """Return type for the interaction-aware InstaSHAP extension."""

    values: pd.DataFrame
    base_value: float
    centered_term_values: pd.DataFrame
    reference_term_means: pd.Series
    interaction_breakdown: pd.DataFrame


def compute_interaction_aware_instashap(
    surrogate,
    X,
    reference_data,
    feature_names: list[str] | None = None,
) -> EnhancedInstaShapOutput:
    """Compute InstaSHAP values while fairly allocating interaction terms.

    The underlying implementation reuses the general Phase 2 term allocator, which
    already centers every surrogate term over the reference distribution and divides
    each non-additive term equally across the features in that term. For pairwise
    terms this is the standard 50/50 split.
    """

    output: InstaShapOutput = compute_instashap_values(
        surrogate=surrogate,
        X=X,
        reference_data=reference_data,
        feature_names=feature_names,
    )
    interaction_columns = [
        term_name
        for term_name, feature_group in zip(surrogate.term_names_, surrogate.term_features_)
        if len(feature_group) > 1
    ]
    interaction_breakdown = output.centered_term_values[interaction_columns].copy()
    return EnhancedInstaShapOutput(
        values=output.values,
        base_value=output.base_value,
        centered_term_values=output.centered_term_values,
        reference_term_means=output.reference_term_means,
        interaction_breakdown=interaction_breakdown,
    )
