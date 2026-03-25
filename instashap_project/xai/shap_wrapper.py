"""SHAP baseline wrapper with aggregation back to original features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap

from instashap_project.data.preprocessing import TabularPreprocessor
from instashap_project.training.evaluate import predict_raw_outputs


@dataclass(slots=True)
class ShapExplanationResult:
    """Container for grouped SHAP values."""

    grouped_values: np.ndarray
    base_values: np.ndarray
    transformed_values: np.ndarray


class ShapBaselineExplainer:
    """Compute SHAP values on transformed inputs and aggregate them to original features."""

    def __init__(
        self,
        model: object,
        preprocessor: TabularPreprocessor,
        device: str,
        max_evals: int = 256,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.max_evals = max_evals

    def _model_fn(self, transformed_inputs: np.ndarray) -> np.ndarray:
        outputs = predict_raw_outputs(self.model, np.asarray(transformed_inputs, dtype=np.float32), self.device)
        if outputs.ndim == 1:
            return outputs.reshape(-1, 1)
        return outputs

    def _aggregate(self, values: np.ndarray) -> np.ndarray:
        if values.ndim == 2:
            values = values[:, :, None]
        grouped = np.zeros(
            (values.shape[0], self.preprocessor.num_original_features, values.shape[2]),
            dtype=np.float32,
        )
        for feature_index, feature_name in enumerate(self.preprocessor.feature_order):
            indices = self.preprocessor.group(feature_name).indices
            grouped[:, feature_index, :] = values[:, indices, :].sum(axis=1)
        return grouped

    def explain(self, background: np.ndarray, evaluation_inputs: np.ndarray) -> ShapExplanationResult:
        minimum_evals = 2 * evaluation_inputs.shape[1] + 1
        explainer = shap.Explainer(self._model_fn, background, algorithm="permutation")
        explanation = explainer(
            evaluation_inputs,
            max_evals=max(self.max_evals, minimum_evals),
            silent=True,
        )
        return ShapExplanationResult(
            grouped_values=self._aggregate(np.asarray(explanation.values)),
            base_values=np.asarray(explanation.base_values),
            transformed_values=np.asarray(explanation.values),
        )

