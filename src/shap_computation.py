"""
SHAP value computation module.

This module handles computing exact SHAP values using the SHAP library,
which serve as ground truth for evaluating GAM surrogate predictions.
It supports multiple explainer types and includes caching functionality.

Supported Explainers:
    - tree: TreeExplainer for tree-based models (RF, XGBoost, LightGBM) - fastest
    - kernel: KernelExplainer - model-agnostic but slower
    - linear: LinearExplainer for linear models

Features:
    - Automatic explainer selection based on model type
    - Configurable background dataset sampling
    - Result caching to avoid recomputation
    - SHAP visualization support (summary, bar, waterfall plots)

Example:
    >>> from src.shap_computation import SHAPComputer, compute_exact_shap
    >>> # Using the class
    >>> computer = SHAPComputer(trained_model, model_type='tree')
    >>> shap_values, comp_time = computer.compute_shap_values(X_test, sample_size=500)
    >>> # Using the convenience function
    >>> shap_values, time = compute_exact_shap(model, X_data, cache_path='shap_cache.pkl')
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import shap
import logging
import time
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class SHAPComputer:
    """
    Compute and cache SHAP values for black-box models.
    """

    def __init__(
        self,
        model,
        model_type: str = "tree",
        background_size: int = 100,
        check_additivity: bool = False,
    ):
        """
        Initialize SHAPComputer.

        Args:
            model: Trained model
            model_type: Type of model ('tree', 'kernel', 'linear')
            background_size: Size of background dataset for SHAP
            check_additivity: Whether to check SHAP additivity
        """
        self.model = model
        self.model_type = model_type.lower()
        self.background_size = background_size
        self.check_additivity = check_additivity
        self.explainer = None

    def _initialize_explainer(self, X_background: np.ndarray) -> None:
        """
        Initialize SHAP explainer.

        Args:
            X_background: Background dataset for SHAP
        """
        logger.info(f"Initializing {self.model_type} SHAP explainer")

        if self.model_type == "tree":
            # For tree-based models (RF, XGBoost, LightGBM)
            self.explainer = shap.TreeExplainer(self.model)

        elif self.model_type == "kernel":
            # For any model using KernelSHAP (slower but universal)
            if X_background.shape[0] > self.background_size:
                background_sample = shap.sample(X_background, self.background_size)
            else:
                background_sample = X_background

            self.explainer = shap.KernelExplainer(self.model.predict, background_sample)

        elif self.model_type == "linear":
            # For linear models
            self.explainer = shap.LinearExplainer(self.model, X_background)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def compute_shap_values(
        self,
        X: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        sample_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute SHAP values for given data.

        Args:
            X: Input data
            X_background: Background dataset (required for kernel explainer)
            sample_size: Optional sample size (if X is large)

        Returns:
            Tuple of (SHAP values array, computation time)
        """
        # Sample data if needed
        if sample_size is not None and X.shape[0] > sample_size:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            logger.info(f"Sampled {sample_size} instances from {X.shape[0]} total")
        else:
            X_sample = X
            indices = np.arange(X.shape[0])

        # Initialize explainer if not already done
        if self.explainer is None:
            if X_background is None:
                X_background = X_sample[: self.background_size]
            self._initialize_explainer(X_background)

        # Compute SHAP values
        logger.info(f"Computing SHAP values for {X_sample.shape[0]} samples...")
        start_time = time.time()

        shap_values = self.explainer.shap_values(X_sample)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For multi-class classification, use the positive class
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]

        computation_time = time.time() - start_time
        logger.info(f"SHAP computation completed in {computation_time:.2f}s")

        return shap_values, computation_time

    def save_shap_values(
        self,
        shap_values: np.ndarray,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save computed SHAP values to disk.

        Args:
            shap_values: SHAP values array
            filepath: Path to save file
            metadata: Optional metadata dictionary
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {"shap_values": shap_values, "metadata": metadata or {}}

        joblib.dump(save_dict, filepath)
        logger.info(f"SHAP values saved to {filepath}")

    def load_shap_values(self, filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load SHAP values from disk.

        Args:
            filepath: Path to saved file

        Returns:
            Tuple of (SHAP values, metadata)
        """
        save_dict = joblib.load(filepath)
        logger.info(f"SHAP values loaded from {filepath}")
        return save_dict["shap_values"], save_dict.get("metadata", {})

    def visualize_shap(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: list,
        plot_type: str = "summary",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create SHAP visualizations.

        Args:
            shap_values: SHAP values array
            X: Input data
            feature_names: Feature names
            plot_type: Type of plot ('summary', 'bar', 'waterfall')
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        if plot_type == "summary":
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        elif plot_type == "bar":
            shap.summary_plot(
                shap_values, X, feature_names=feature_names, plot_type="bar", show=False
            )
        elif plot_type == "waterfall":
            # Show waterfall for first instance
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=0,
                    data=X[0],
                    feature_names=feature_names,
                ),
                show=False,
            )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def compute_exact_shap(
    model,
    X_data: np.ndarray,
    model_type: str = "tree",
    sample_size: Optional[int] = 1000,
    background_size: int = 100,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to compute exact SHAP values.

    Args:
        model: Trained model
        X_data: Input data
        model_type: Model type ('tree', 'kernel', 'linear')
        sample_size: Number of samples to compute SHAP for
        background_size: Background dataset size
        cache_path: Optional path to cache results

    Returns:
        Tuple of (SHAP values, computation time)
    """
    # Check if cached
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached SHAP values from {cache_path}")
        computer = SHAPComputer(model, model_type, background_size)
        shap_values, metadata = computer.load_shap_values(cache_path)
        return shap_values, metadata.get("computation_time", 0.0)

    # Compute SHAP values
    computer = SHAPComputer(model, model_type, background_size)
    shap_values, comp_time = computer.compute_shap_values(
        X_data, sample_size=sample_size
    )

    # Cache if path provided
    if cache_path:
        metadata = {
            "computation_time": comp_time,
            "sample_size": shap_values.shape[0],
            "n_features": shap_values.shape[1],
        }
        computer.save_shap_values(shap_values, cache_path, metadata)

    return shap_values, comp_time
