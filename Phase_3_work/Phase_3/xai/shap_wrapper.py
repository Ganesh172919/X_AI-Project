"""Permutation SHAP wrapper with feature-group aggregation."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import shap
import torch

from data.preprocessing import TabularPreprocessor


@dataclass
class ShapExplanationResult:
    grouped_values: np.ndarray   # (n_samples, n_original_features, n_outputs)
    base_values: np.ndarray
    transformed_values: np.ndarray
    feature_names: list[str]


def compute_shap_values(
    model_fn,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    preprocessor: TabularPreprocessor,
    max_evals: int = 256,
) -> ShapExplanationResult:
    """Compute permutation SHAP with feature-group aggregation."""
    explainer = shap.Explainer(model_fn, X_background, algorithm="permutation")
    explanation = explainer(X_explain, max_evals=max_evals)

    raw_values = np.asarray(explanation.values, dtype=np.float32)
    base_values = np.asarray(explanation.base_values, dtype=np.float32)

    if raw_values.ndim == 2:
        raw_values = raw_values[:, :, np.newaxis]

    n_samples = raw_values.shape[0]
    n_outputs = raw_values.shape[2]
    n_orig = preprocessor.num_original_features
    grouped = np.zeros((n_samples, n_orig, n_outputs), dtype=np.float32)

    for fi, fn in enumerate(preprocessor.feature_order):
        grp = preprocessor.feature_groups[fn]
        grouped[:, fi, :] = raw_values[:, grp.start:grp.end, :].sum(axis=1)

    return ShapExplanationResult(
        grouped_values=grouped,
        base_values=base_values,
        transformed_values=raw_values,
        feature_names=list(preprocessor.feature_order),
    )
