"""
Black-box model training and evaluation module.

Wraps sklearn, XGBoost, and LightGBM models with a unified interface for
training, prediction, evaluation, and serialization. These models serve as
the targets whose predictions are explained via SHAP values.

Supported Models
----------------
random_forest
    sklearn RandomForestClassifier / RandomForestRegressor.
xgboost
    XGBClassifier / XGBRegressor with optional early stopping.
lightgbm
    LGBMClassifier / LGBMRegressor with optional early stopping.

Example
-------
>>> from src.black_box_model import BlackBoxModel
>>> model = BlackBoxModel("xgboost", task="classification", n_estimators=200)
>>> model.train(X_train, y_train, X_val, y_val)
>>> metrics = model.evaluate(X_test, y_test)
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    classification_report,
    roc_auc_score,
)
import xgboost as xgb
import lightgbm as lgb
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class BlackBoxModel:
    """
    Wrapper for training and evaluating black-box machine learning models.

    Attributes
    ----------
    model_type : str
        Type of model (``'random_forest'``, ``'xgboost'``, or ``'lightgbm'``).
    task : str
        Task type (``'classification'`` or ``'regression'``).
    model_params : dict
        Additional keyword arguments passed to the underlying estimator.
    model : estimator
        The fitted scikit-learn compatible estimator.

    Example
    -------
    >>> bb = BlackBoxModel("random_forest", task="regression", n_estimators=100)
    >>> bb.train(X_train, y_train)
    >>> bb.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        task: str = "classification",
        **model_params,
    ):
        """
        Initialize BlackBoxModel.

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            task: Task type ('classification' or 'regression')
            **model_params: Additional model parameters
        """
        self.model_type = model_type.lower()
        self.task = task.lower()
        self.model_params = model_params
        self.model = None

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the appropriate model."""
        logger.info(f"Initializing {self.model_type} for {self.task}")

        if self.model_type == "random_forest":
            if self.task == "classification":
                self.model = RandomForestClassifier(**self.model_params)
            else:
                self.model = RandomForestRegressor(**self.model_params)

        elif self.model_type == "xgboost":
            if self.task == "classification":
                self.model = xgb.XGBClassifier(**self.model_params)
            else:
                self.model = xgb.XGBRegressor(**self.model_params)

        elif self.model_type == "lightgbm":
            if self.task == "classification":
                self.model = lgb.LGBMClassifier(**self.model_params)
            else:
                self.model = lgb.LGBMRegressor(**self.model_params)

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BlackBoxModel":
        """
        Train the black-box model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features
            y_val: Optional validation targets

        Returns:
            Self (for chaining)
        """
        logger.info(f"Training {self.model_type} on {X_train.shape[0]} samples")

        if X_val is not None and y_val is not None:
            if self.model_type in ["xgboost", "lightgbm"]:
                self.model.fit(
                    X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        logger.info("Training completed")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification).

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification tasks")
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)

        metrics = {}

        if self.task == "classification":
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["f1_score"] = f1_score(y_test, y_pred, average="weighted")

            # Add AUC if binary classification
            if len(np.unique(y_test)) == 2:
                y_proba = self.predict_proba(X_test)[:, 1]
                metrics["auc_roc"] = roc_auc_score(y_test, y_proba)

            if verbose:
                logger.info(f"\nClassification Metrics:")
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
                if "auc_roc" in metrics:
                    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

        else:  # regression
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2"] = r2_score(y_test, y_pred)

            if verbose:
                logger.info(f"\nRegression Metrics:")
                logger.info(f"  MSE: {metrics['mse']:.4f}")
                logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                logger.info(f"  R²: {metrics['r2']:.4f}")

        return metrics

    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.

        Args:
            filepath: Path to saved model
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")

    def get_model(self):
        """
        Get the underlying model object.

        Returns:
            The trained model
        """
        return self.model
