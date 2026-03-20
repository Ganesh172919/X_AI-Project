"""
GAM surrogate model for instant SHAP prediction.

This is the core implementation of the InstaSHAP methodology. For each
feature *i*, a separate Generalized Additive Model (GAM) is trained to
predict the SHAP value of that feature given the input features:

    GAM_i : X -> SHAP_i

At prediction time, all GAMs are evaluated in parallel to produce the full
SHAP value matrix, yielding orders-of-magnitude speedup over exact SHAP
computation (e.g., KernelSHAP or TreeSHAP) while preserving high fidelity.

The GAMs are implemented using Explainable Boosting Machines (EBMs) from
the InterpretML library, which provide interpretable shape functions for
each input feature.

References
----------
- Nori et al., "InstaSHAP: Interpretable Additive Models Explain Shapley
  Values Instantly", ICLR 2025.
- Caruana et al., "Intelligible Models for Classification and Regression",
  KDD 2015.

Example
-------
>>> from src.gam_surrogate import SHAPSurrogate
>>> surrogate = SHAPSurrogate(max_iter=5000, learning_rate=0.01)
>>> surrogate.train(X_train, shap_values_train, feature_names=feature_names)
>>> pred_shap, pred_time = surrogate.predict_shap(X_test, return_time=True)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from interpret.glassbox import ExplainableBoostingRegressor
import logging
import time
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class SHAPSurrogate:
    """
    Generalized Additive Model (GAM) surrogate for instant SHAP prediction.

    Trains one Explainable Boosting Machine (EBM) per feature to approximate
    the mapping from input features to SHAP values for that feature. At
    prediction time, all EBMs are evaluated to produce the full SHAP matrix.

    Attributes
    ----------
    max_iter : int
        Maximum boosting rounds per GAM.
    max_bins : int
        Number of bins for feature discretization.
    interactions : int
        Number of pairwise interaction terms (0 for pure additive GAM).
    learning_rate : float
        Step size for gradient boosting.
    min_samples_leaf : int
        Minimum samples in each leaf node.
    random_state : int
        Random seed for reproducibility.
    gam_models : dict[int, ExplainableBoostingRegressor]
        Trained GAM models, keyed by feature index.
    n_features : int or None
        Number of features (set after training).
    feature_names : list of str or None
        Feature names (set after training).
    is_fitted : bool
        Whether the surrogates have been trained.

    Example
    -------
    >>> surrogate = SHAPSurrogate(max_iter=5000)
    >>> surrogate.train(X_train, shap_values_train)
    >>> pred_shap = surrogate.predict_shap(X_test)
    >>> metrics = surrogate.evaluate(X_test, true_shap_values)
    """

    def __init__(
        self,
        max_iter: int = 5000,
        max_bins: int = 256,
        interactions: int = 0,
        learning_rate: float = 0.01,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize SHAP surrogate model.

        Args:
            max_iter: Maximum boosting iterations
            max_bins: Maximum number of bins for discretization
            interactions: Number of interaction terms (0 for pure GAM)
            learning_rate: Learning rate for boosting
            min_samples_leaf: Minimum samples per leaf
            random_state: Random seed
        """
        self.max_iter = max_iter
        self.max_bins = max_bins
        self.interactions = interactions
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # Dictionary to store one GAM per feature
        self.gam_models: Dict[int, ExplainableBoostingRegressor] = {}
        self.n_features = None
        self.feature_names = None
        self.is_fitted = False

    def train(
        self,
        X_train: np.ndarray,
        shap_values_train: np.ndarray,
        feature_names: Optional[list] = None,
        verbose: bool = True,
    ) -> "SHAPSurrogate":
        """
        Train GAM surrogates to predict SHAP values.

        For each feature i, we train:
            GAM_i: X -> SHAP_i

        Args:
            X_train: Training features (n_samples, n_features)
            shap_values_train: True SHAP values (n_samples, n_features)
            feature_names: Optional feature names
            verbose: Whether to print progress

        Returns:
            Self (for chaining)
        """
        self.n_features = X_train.shape[1]
        self.feature_names = feature_names or [
            f"Feature_{i}" for i in range(self.n_features)
        ]

        if X_train.shape[0] != shap_values_train.shape[0]:
            raise ValueError(
                "X_train and shap_values_train must have same number of samples"
            )

        if X_train.shape[1] != shap_values_train.shape[1]:
            raise ValueError(
                "X_train and shap_values_train must have same number of features"
            )

        logger.info(f"Training {self.n_features} GAM surrogates...")
        logger.info(
            f"Training data: {X_train.shape[0]} samples, {self.n_features} features"
        )

        start_time = time.time()

        # Train one GAM per feature
        for feature_idx in range(self.n_features):
            if verbose and (feature_idx + 1) % 5 == 0:
                logger.info(f"Training GAM {feature_idx + 1}/{self.n_features}")

            # Target is SHAP values for this feature
            y_target = shap_values_train[:, feature_idx]

            # Initialize GAM
            gam = ExplainableBoostingRegressor(
                max_rounds=self.max_iter,
                max_bins=self.max_bins,
                interactions=self.interactions,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )

            # Train GAM to predict SHAP values for feature i
            gam.fit(X_train, y_target)

            # Store trained model
            self.gam_models[feature_idx] = gam

        training_time = time.time() - start_time
        self.is_fitted = True

        logger.info(f"GAM training completed in {training_time:.2f}s")
        logger.info(f"Average time per GAM: {training_time / self.n_features:.3f}s")

        return self

    def predict_shap(self, X_test: np.ndarray, return_time: bool = False) -> np.ndarray:
        """
        Predict SHAP values using trained GAM surrogates.

        Args:
            X_test: Test features (n_samples, n_features)
            return_time: Whether to return prediction time

        Returns:
            Predicted SHAP values (n_samples, n_features)
            Or tuple of (predicted SHAP values, prediction time) if return_time=True
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if X_test.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X_test.shape[1]}"
            )

        start_time = time.time()

        # Initialize output array
        shap_predictions = np.zeros((X_test.shape[0], self.n_features))

        # Predict SHAP values for each feature
        for feature_idx in range(self.n_features):
            gam = self.gam_models[feature_idx]
            shap_predictions[:, feature_idx] = gam.predict(X_test)

        prediction_time = time.time() - start_time

        if return_time:
            return shap_predictions, prediction_time
        return shap_predictions

    def evaluate(
        self, X_test: np.ndarray, true_shap_values: np.ndarray, verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate GAM surrogate predictions against true SHAP values.

        Args:
            X_test: Test features
            true_shap_values: True SHAP values
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import pearsonr, spearmanr

        # Predict SHAP values
        pred_shap_values, pred_time = self.predict_shap(X_test, return_time=True)

        # Compute metrics
        mse = mean_squared_error(true_shap_values.flatten(), pred_shap_values.flatten())
        mae = mean_absolute_error(
            true_shap_values.flatten(), pred_shap_values.flatten()
        )
        r2 = r2_score(true_shap_values.flatten(), pred_shap_values.flatten())

        # Correlation metrics
        pearson_corr, _ = pearsonr(
            true_shap_values.flatten(), pred_shap_values.flatten()
        )
        spearman_corr, _ = spearmanr(
            true_shap_values.flatten(), pred_shap_values.flatten()
        )

        # Per-feature metrics
        per_feature_mse = []
        per_feature_r2 = []
        for feature_idx in range(self.n_features):
            f_mse = mean_squared_error(
                true_shap_values[:, feature_idx], pred_shap_values[:, feature_idx]
            )
            f_r2 = r2_score(
                true_shap_values[:, feature_idx], pred_shap_values[:, feature_idx]
            )
            per_feature_mse.append(f_mse)
            per_feature_r2.append(f_r2)

        metrics = {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
            "r2": r2,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "prediction_time": pred_time,
            "per_feature_mse": per_feature_mse,
            "per_feature_r2": per_feature_r2,
            "mean_per_feature_r2": np.mean(per_feature_r2),
        }

        if verbose:
            logger.info("\n=== GAM Surrogate Evaluation ===")
            logger.info(f"MSE: {metrics['mse']:.6f}")
            logger.info(f"MAE: {metrics['mae']:.6f}")
            logger.info(f"RMSE: {metrics['rmse']:.6f}")
            logger.info(f"R²: {metrics['r2']:.4f}")
            logger.info(f"Pearson Correlation: {metrics['pearson_correlation']:.4f}")
            logger.info(f"Spearman Correlation: {metrics['spearman_correlation']:.4f}")
            logger.info(f"Mean Per-Feature R²: {metrics['mean_per_feature_r2']:.4f}")
            logger.info(f"Prediction Time: {metrics['prediction_time']:.4f}s")

        return metrics

    def get_feature_importance(self, feature_idx: int) -> np.ndarray:
        """
        Get feature importance from GAM for a specific feature's SHAP predictor.

        Args:
            feature_idx: Index of feature

        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        gam = self.gam_models[feature_idx]
        return gam.term_importances()

    def save_model(self, filepath: str) -> None:
        """
        Save trained GAM surrogates to disk.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "gam_models": self.gam_models,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "config": {
                "max_iter": self.max_iter,
                "max_bins": self.max_bins,
                "interactions": self.interactions,
                "learning_rate": self.learning_rate,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
            },
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"GAM surrogate saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load trained GAM surrogates from disk.

        Args:
            filepath: Path to saved file
        """
        save_dict = joblib.load(filepath)

        self.gam_models = save_dict["gam_models"]
        self.n_features = save_dict["n_features"]
        self.feature_names = save_dict["feature_names"]

        config = save_dict["config"]
        self.max_iter = config["max_iter"]
        self.max_bins = config["max_bins"]
        self.interactions = config["interactions"]
        self.learning_rate = config["learning_rate"]
        self.min_samples_leaf = config["min_samples_leaf"]
        self.random_state = config["random_state"]

        self.is_fitted = True
        logger.info(f"GAM surrogate loaded from {filepath}")
