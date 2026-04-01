"""
Adaptive Surrogate Module

This module implements an adaptive strategy that automatically selects
between a purely additive GAM and a GA²M with interactions based on
surrogate fidelity.

Strategy:
1. Fit additive surrogate first (fast)
2. Evaluate fidelity (R² with black-box)
3. If fidelity < threshold, upgrade to GA²M with interactions
4. Return the best surrogate

This balances speed and accuracy automatically.

Author: DS357 Course Project Team
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "phase2"
    )
)

from interaction_aware_surrogate import InteractionAwareSurrogate
from phase2.models.gam_surrogate import GAMSurrogate

# Set random seed for reproducibility
RANDOM_SEED = 42


class AdaptiveSurrogate:
    """
    Adaptive surrogate selection between GAM and GA²M.

    Automatically determines whether interactions are needed based on
    how well the additive surrogate approximates the black-box model.
    """

    def __init__(
        self,
        task_type: str = "regression",
        fidelity_threshold: float = 0.90,
        n_interactions: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize the adaptive surrogate.

        Args:
            task_type: 'regression' or 'classification'
            fidelity_threshold: R² threshold below which to add interactions
            n_interactions: Number of interactions to add if needed
            verbose: Whether to print progress messages
        """
        self.task_type = task_type
        self.fidelity_threshold = fidelity_threshold
        self.n_interactions = n_interactions
        self.verbose = verbose

        self.surrogate = None
        self.surrogate_type = None  # 'additive' or 'interaction'
        self.is_fitted = False
        self.additive_fidelity = None
        self.interaction_fidelity = None
        self.feature_names = None

    def fit(
        self,
        blackbox_model,
        X_train: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[List[str]] = None,
    ) -> "AdaptiveSurrogate":
        """
        Fit the adaptive surrogate, choosing between GAM and GA²M.

        Args:
            blackbox_model: Trained black-box model
            X_train: Training feature matrix
            feature_names: Optional feature names

        Returns:
            self
        """
        if self.verbose:
            print("=" * 60)
            print("Adaptive Surrogate Selection")
            print("=" * 60)

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        elif feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Get black-box predictions
        y_blackbox = blackbox_model.predict(X_train)

        # Step 1: Try additive surrogate first
        if self.verbose:
            print("\nStep 1: Fitting additive GAM surrogate...")

        additive_surrogate = GAMSurrogate(
            task_type=self.task_type
            if self.task_type == "regression"
            else "regression",
            interactions=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            additive_surrogate.fit(X_train, y_blackbox, self.feature_names)

        # Evaluate additive fidelity
        additive_preds = additive_surrogate.predict(X_train)
        self.additive_fidelity = r2_score(y_blackbox, additive_preds)

        if self.verbose:
            print(f"Additive surrogate R²: {self.additive_fidelity:.4f}")

        # Step 2: Check if we need interactions
        if self.additive_fidelity >= self.fidelity_threshold:
            if self.verbose:
                print(
                    f"\nFidelity {self.additive_fidelity:.4f} >= threshold {self.fidelity_threshold}"
                )
                print("Using additive surrogate (no interactions needed)")

            self.surrogate = additive_surrogate
            self.surrogate_type = "additive"
            self.interaction_fidelity = None

        else:
            if self.verbose:
                print(
                    f"\nFidelity {self.additive_fidelity:.4f} < threshold {self.fidelity_threshold}"
                )
                print("Step 2: Fitting GA²M with interactions...")

            # Fit interaction-aware surrogate
            interaction_surrogate = InteractionAwareSurrogate(
                task_type=self.task_type
                if self.task_type == "regression"
                else "regression",
                interactions=self.n_interactions,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                interaction_surrogate.fit(X_train, y_blackbox, self.feature_names)

            # Evaluate interaction fidelity
            interaction_preds = interaction_surrogate.predict(X_train)
            self.interaction_fidelity = r2_score(y_blackbox, interaction_preds)

            if self.verbose:
                print(f"GA²M surrogate R²: {self.interaction_fidelity:.4f}")
                print(
                    f"Improvement: {self.interaction_fidelity - self.additive_fidelity:.4f}"
                )

            # Use interaction surrogate if it improves fidelity
            if self.interaction_fidelity > self.additive_fidelity:
                self.surrogate = interaction_surrogate
                self.surrogate_type = "interaction"
                if self.verbose:
                    print("Using GA²M surrogate (interactions help)")
            else:
                self.surrogate = additive_surrogate
                self.surrogate_type = "additive"
                if self.verbose:
                    print("Using additive surrogate (interactions don't help)")

        self.is_fitted = True

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Final surrogate type: {self.surrogate_type}")
            print("=" * 60)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Make predictions using the selected surrogate."""
        if not self.is_fitted:
            raise RuntimeError("Adaptive surrogate must be fitted first.")
        return self.surrogate.predict(X)

    def get_feature_contributions(self, X: Union[pd.DataFrame, np.ndarray]) -> Tuple:
        """
        Get feature contributions from the selected surrogate.

        Returns format depends on surrogate type:
        - Additive: (contributions, intercept)
        - Interaction: (main_contrib, interaction_contrib, intercept)
        """
        if not self.is_fitted:
            raise RuntimeError("Adaptive surrogate must be fitted first.")

        return self.surrogate.get_feature_contributions(X)

    def get_surrogate_type(self) -> str:
        """Get the type of surrogate that was selected."""
        return self.surrogate_type

    def get_fidelity_metrics(self) -> Dict[str, float]:
        """Get fidelity metrics from the fitting process."""
        metrics = {
            "additive_r2": self.additive_fidelity,
            "selected_type": self.surrogate_type,
            "threshold": self.fidelity_threshold,
        }
        if self.interaction_fidelity is not None:
            metrics["interaction_r2"] = self.interaction_fidelity
            metrics["improvement"] = self.interaction_fidelity - self.additive_fidelity
        return metrics

    def uses_interactions(self) -> bool:
        """Check if the selected surrogate uses interactions."""
        return self.surrogate_type == "interaction"

    def get_interaction_pairs(self) -> List[Tuple[int, int]]:
        """Get interaction pairs if using interaction surrogate."""
        if self.surrogate_type == "interaction":
            return self.surrogate.get_interaction_pairs()
        return []


class AdaptiveInstaSHAPExplainer:
    """
    Complete adaptive InstaSHAP explainer that automatically selects
    between standard InstaSHAP and Enhanced InstaSHAP based on data.
    """

    def __init__(
        self,
        task_type: str = "regression",
        fidelity_threshold: float = 0.90,
        n_interactions: int = 10,
    ):
        """
        Initialize the adaptive explainer.

        Args:
            task_type: 'regression' or 'classification'
            fidelity_threshold: R² threshold for interaction decision
            n_interactions: Number of interactions if needed
        """
        self.task_type = task_type
        self.fidelity_threshold = fidelity_threshold
        self.n_interactions = n_interactions

        self.adaptive_surrogate = None
        self.instashap_explainer = None
        self.is_fitted = False

    def fit(
        self, blackbox_model, X_train: Union[pd.DataFrame, np.ndarray]
    ) -> "AdaptiveInstaSHAPExplainer":
        """
        Fit the adaptive explainer.

        Args:
            blackbox_model: Trained black-box model
            X_train: Training data

        Returns:
            self
        """
        from enhanced_instashap import EnhancedInstaSHAP
        from phase2.models.instashap import InstaSHAP

        # Fit adaptive surrogate
        self.adaptive_surrogate = AdaptiveSurrogate(
            task_type=self.task_type,
            fidelity_threshold=self.fidelity_threshold,
            n_interactions=self.n_interactions,
        )
        self.adaptive_surrogate.fit(blackbox_model, X_train)

        # Create appropriate InstaSHAP explainer
        if self.adaptive_surrogate.uses_interactions():
            self.instashap_explainer = EnhancedInstaSHAP(
                self.adaptive_surrogate.surrogate, X_train
            )
        else:
            self.instashap_explainer = InstaSHAP(
                self.adaptive_surrogate.surrogate, X_train
            )

        self.is_fitted = True
        return self

    def explain(
        self, X: Union[pd.DataFrame, np.ndarray], return_dataframe: bool = True
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Compute Shapley values using the appropriate method.

        Args:
            X: Samples to explain
            return_dataframe: Return as DataFrame

        Returns:
            Shapley values
        """
        if not self.is_fitted:
            raise RuntimeError("Explainer must be fitted first.")

        return self.instashap_explainer.explain(X, return_dataframe)

    def get_base_value(self) -> float:
        """Get the base value."""
        return self.instashap_explainer.get_base_value()

    def get_decision_info(self) -> Dict:
        """Get information about the adaptive decision."""
        return {
            "surrogate_type": self.adaptive_surrogate.get_surrogate_type(),
            "uses_interactions": self.adaptive_surrogate.uses_interactions(),
            "fidelity_metrics": self.adaptive_surrogate.get_fidelity_metrics(),
            "interaction_pairs": self.adaptive_surrogate.get_interaction_pairs(),
        }


if __name__ == "__main__":
    # Test the module
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "phase2"
        )
    )

    from phase2.data.data_loader import (
        load_california_housing,
        create_synthetic_interaction_dataset,
    )
    from phase2.models.base_model import train_model_for_dataset

    print("=" * 60)
    print("Testing Adaptive Surrogate Module")
    print("=" * 60)

    # Test 1: Dataset without strong interactions
    print("\n" + "=" * 60)
    print("Test 1: California Housing (weak interactions expected)")
    print("=" * 60)

    cal_data = load_california_housing()
    blackbox1 = train_model_for_dataset(cal_data, model_type="xgboost")

    adaptive1 = AdaptiveSurrogate(fidelity_threshold=0.85)
    adaptive1.fit(blackbox1.model, cal_data["X_train"])

    print(f"\nSelected: {adaptive1.get_surrogate_type()}")

    # Test 2: Synthetic dataset with strong interactions
    print("\n" + "=" * 60)
    print("Test 2: Synthetic data (strong interactions)")
    print("=" * 60)

    synth_data = create_synthetic_interaction_dataset(
        n_samples=2000, interaction_strength=2.0
    )
    blackbox2 = train_model_for_dataset(synth_data, model_type="xgboost")

    adaptive2 = AdaptiveSurrogate(fidelity_threshold=0.90)
    adaptive2.fit(blackbox2.model, synth_data["X_train"])

    print(f"\nSelected: {adaptive2.get_surrogate_type()}")
    print(f"Metrics: {adaptive2.get_fidelity_metrics()}")

    print("\n" + "=" * 60)
    print("Adaptive Surrogate module test complete!")
    print("=" * 60)
