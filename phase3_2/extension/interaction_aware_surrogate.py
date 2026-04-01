"""
Interaction-Aware Surrogate Module

This module implements the GA²M (Generalized Additive Model with pairwise interactions)
surrogate that captures feature interactions while maintaining interpretability.

Key Features:
- EBM with pairwise interaction terms enabled
- Automatic interaction detection
- Configurable number of interactions

Author: DS357 Course Project Team
"""

import numpy as np
import pandas as pd
from interpret.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from typing import Dict, Any, Optional, Union, Tuple, List
import joblib
import os
import warnings
from itertools import combinations

# Set random seed for reproducibility
RANDOM_SEED = 42


class InteractionAwareSurrogate:
    """
    GA²M Surrogate model using InterpretML's EBM with pairwise interactions.

    This extends the purely additive GAM to capture important feature interactions
    while maintaining the closed-form Shapley value computation capability.
    """

    def __init__(
        self,
        task_type: str = "regression",
        max_bins: int = 256,
        max_rounds: int = 5000,
        learning_rate: float = 0.01,
        interactions: int = 10,  # Number of pairwise interactions
        outer_bags: int = 8,
        inner_bags: int = 0,
        interaction_detection: str = "auto",  # 'auto', 'all', or list of pairs
    ):
        """
        Initialize the interaction-aware surrogate.

        Args:
            task_type: 'regression' or 'classification'
            max_bins: Maximum number of bins for discretization
            max_rounds: Maximum number of boosting rounds
            learning_rate: Learning rate for boosting
            interactions: Number of interaction terms to include
            outer_bags: Number of outer bags for bagging
            inner_bags: Number of inner bags
            interaction_detection: How to select interactions
        """
        self.task_type = task_type
        self.max_bins = max_bins
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.interactions = interactions
        self.outer_bags = outer_bags
        self.inner_bags = inner_bags
        self.interaction_detection = interaction_detection
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.interaction_pairs = []
        self.interaction_indices = {}

    def _init_model(self) -> None:
        """Initialize the EBM model with interactions enabled."""

        common_params = {
            "max_bins": self.max_bins,
            "max_rounds": self.max_rounds,
            "learning_rate": self.learning_rate,
            "interactions": self.interactions,  # Enable interactions
            "outer_bags": self.outer_bags,
            "inner_bags": self.inner_bags,
            "random_state": RANDOM_SEED,
        }

        if self.task_type == "regression":
            self.model = ExplainableBoostingRegressor(**common_params)
        else:
            self.model = ExplainableBoostingClassifier(**common_params)

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "InteractionAwareSurrogate":
        """
        Fit the GA²M surrogate with interactions.

        Args:
            X: Feature matrix
            y: Target values
            feature_names: Optional list of feature names

        Returns:
            self
        """
        print("Fitting Interaction-Aware GA²M surrogate model...")

        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Initialize the model
        self._init_model()

        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y)

        # Extract interaction information
        self._extract_interaction_info()

        self.is_fitted = True
        print(
            f"GA²M surrogate fitted with {len(self.interaction_pairs)} interaction terms."
        )
        return self

    def _extract_interaction_info(self) -> None:
        """Extract information about learned interactions from the model."""

        # Get global explanation to find interaction terms
        global_exp = self.model.explain_global()
        term_names = global_exp.data()["names"]

        self.interaction_pairs = []
        self.interaction_indices = {}

        for idx, name in enumerate(term_names):
            # Interaction terms contain ' x ' in their name
            if " x " in str(name):
                parts = str(name).split(" x ")
                if len(parts) == 2:
                    feat1, feat2 = parts[0].strip(), parts[1].strip()

                    # Find indices in feature_names
                    if feat1 in self.feature_names and feat2 in self.feature_names:
                        idx1 = self.feature_names.index(feat1)
                        idx2 = self.feature_names.index(feat2)

                        self.interaction_pairs.append((idx1, idx2))
                        self.interaction_indices[(idx1, idx2)] = idx

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the fitted GA²M surrogate."""
        if not self.is_fitted:
            raise RuntimeError(
                "GA²M surrogate must be fitted before making predictions."
            )
        return self.model.predict(X)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Get probability predictions for classification tasks."""
        if self.task_type != "classification":
            raise RuntimeError(
                "predict_proba is only available for classification tasks."
            )
        if not self.is_fitted:
            raise RuntimeError(
                "GA²M surrogate must be fitted before making predictions."
            )
        return self.model.predict_proba(X)

    def get_feature_contributions(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get individual feature contributions AND interaction contributions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Tuple of:
                - main_contributions: Array of shape (n_samples, n_features)
                - interaction_contributions: Array of shape (n_samples, n_interactions)
                - intercept: The global intercept f_0
        """
        if not self.is_fitted:
            raise RuntimeError(
                "GA²M surrogate must be fitted before getting contributions."
            )

        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        n_samples = len(X)
        n_features = len(self.feature_names)
        n_interactions = len(self.interaction_pairs)

        # Get local explanations from EBM
        local_explanations = self.model.explain_local(X)

        # Initialize contribution arrays
        main_contributions = np.zeros((n_samples, n_features))
        interaction_contributions = np.zeros((n_samples, n_interactions))

        for i in range(n_samples):
            exp_data = local_explanations.data(i)

            for j, name in enumerate(exp_data["names"]):
                score = exp_data["scores"][j]

                # Check if this is an interaction term
                if " x " in str(name):
                    parts = str(name).split(" x ")
                    if len(parts) == 2:
                        feat1, feat2 = parts[0].strip(), parts[1].strip()
                        if feat1 in self.feature_names and feat2 in self.feature_names:
                            idx1 = self.feature_names.index(feat1)
                            idx2 = self.feature_names.index(feat2)

                            # Find interaction index
                            if (idx1, idx2) in self.interaction_pairs:
                                int_idx = self.interaction_pairs.index((idx1, idx2))
                                interaction_contributions[i, int_idx] = score
                            elif (idx2, idx1) in self.interaction_pairs:
                                int_idx = self.interaction_pairs.index((idx2, idx1))
                                interaction_contributions[i, int_idx] = score
                else:
                    # Main effect
                    if name in self.feature_names:
                        feat_idx = self.feature_names.index(name)
                        main_contributions[i, feat_idx] = score

        # Get intercept
        intercept = self.model.intercept_
        if isinstance(intercept, np.ndarray):
            intercept = intercept[0]

        return main_contributions, interaction_contributions, intercept

    def get_interaction_pairs(self) -> List[Tuple[int, int]]:
        """Get the list of interaction pairs (feature indices)."""
        return self.interaction_pairs

    def evaluate_fidelity(
        self, X: Union[pd.DataFrame, np.ndarray], y_blackbox: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate how well the surrogate approximates the black-box model.

        Args:
            X: Feature matrix
            y_blackbox: Black-box model predictions

        Returns:
            Dictionary of fidelity metrics
        """
        y_surrogate = self.predict(X)

        if self.task_type == "regression":
            metrics = {
                "r2": r2_score(y_blackbox, y_surrogate),
                "mse": mean_squared_error(y_blackbox, y_surrogate),
                "rmse": np.sqrt(mean_squared_error(y_blackbox, y_surrogate)),
            }
        else:
            metrics = {
                "accuracy": accuracy_score(y_blackbox, y_surrogate),
                "agreement_rate": np.mean(y_blackbox == y_surrogate),
            }

        return metrics

    def save(self, filepath: str) -> None:
        """Save the trained GA²M surrogate to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(
            filepath
        ) else None
        joblib.dump(
            {
                "model": self.model,
                "task_type": self.task_type,
                "feature_names": self.feature_names,
                "is_fitted": self.is_fitted,
                "interaction_pairs": self.interaction_pairs,
                "interaction_indices": self.interaction_indices,
                "interactions": self.interactions,
            },
            filepath,
        )
        print(f"GA²M surrogate saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "InteractionAwareSurrogate":
        """Load a trained GA²M surrogate from disk."""
        data = joblib.load(filepath)
        surrogate = cls(
            task_type=data["task_type"], interactions=data.get("interactions", 10)
        )
        surrogate.model = data["model"]
        surrogate.feature_names = data["feature_names"]
        surrogate.is_fitted = data["is_fitted"]
        surrogate.interaction_pairs = data["interaction_pairs"]
        surrogate.interaction_indices = data["interaction_indices"]
        print(f"GA²M surrogate loaded from {filepath}")
        return surrogate


def train_interaction_surrogate_for_blackbox(
    blackbox_model,
    X_train: Union[pd.DataFrame, np.ndarray],
    task_type: str = "regression",
    use_proba: bool = False,
    n_interactions: int = 10,
) -> InteractionAwareSurrogate:
    """
    Train a GA²M surrogate with interactions to approximate a black-box model.

    Args:
        blackbox_model: Trained black-box model with predict method
        X_train: Training feature matrix
        task_type: 'regression' or 'classification'
        use_proba: For classification, use predict_proba instead of predict
        n_interactions: Number of pairwise interactions to include

    Returns:
        Fitted InteractionAwareSurrogate instance
    """
    print("Training GA²M surrogate with interactions...")

    # Get black-box predictions as training targets
    if task_type == "classification" and use_proba:
        y_blackbox = blackbox_model.predict_proba(X_train)[:, 1]
        surrogate_task = "regression"
    else:
        y_blackbox = blackbox_model.predict(X_train)
        surrogate_task = task_type

    # Create and fit surrogate
    surrogate = InteractionAwareSurrogate(
        task_type=surrogate_task, interactions=n_interactions
    )
    surrogate.fit(X_train, y_blackbox)

    # Evaluate fidelity
    fidelity = surrogate.evaluate_fidelity(X_train, y_blackbox)
    print(f"\nGA²M Surrogate Fidelity (on training data):")
    for metric_name, value in fidelity.items():
        print(f"  {metric_name}: {value:.4f}")

    return surrogate


if __name__ == "__main__":
    # Test the module
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "phase2"
        )
    )

    from phase2.data.data_loader import load_california_housing
    from phase2.models.base_model import train_model_for_dataset

    print("=" * 60)
    print("Testing Interaction-Aware Surrogate Module")
    print("=" * 60)

    # Load data and train black-box model
    cal_data = load_california_housing()
    blackbox = train_model_for_dataset(cal_data, model_type="xgboost")

    # Train GA²M surrogate with interactions
    surrogate = train_interaction_surrogate_for_blackbox(
        blackbox.model, cal_data["X_train"], task_type="regression", n_interactions=5
    )

    # Test getting feature contributions
    print("\nTesting feature contributions extraction...")
    X_sample = cal_data["X_test"].iloc[:5]
    main_contrib, interact_contrib, intercept = surrogate.get_feature_contributions(
        X_sample
    )

    print(f"Intercept: {intercept:.4f}")
    print(f"Main contributions shape: {main_contrib.shape}")
    print(f"Interaction contributions shape: {interact_contrib.shape}")
    print(f"Detected interaction pairs: {surrogate.get_interaction_pairs()}")

    print("\n" + "=" * 60)
    print("Interaction-Aware Surrogate module test complete!")
    print("=" * 60)
