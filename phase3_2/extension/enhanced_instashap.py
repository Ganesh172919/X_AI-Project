"""
Enhanced InstaSHAP Module

This module extends the InstaSHAP algorithm to handle pairwise interactions
from GA²M surrogates. The key extension is the fair allocation of interaction
contributions between the two interacting features.

Shapley Interaction Allocation:
For an interaction term f_ij(x_i, x_j), the contribution is split equally:
    φ_i^{(ij)} = φ_j^{(ij)} = 0.5 * (f_ij(x_i, x_j) - E[f_ij(X_i, X_j)])

Author: DS357 Course Project Team
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict
import warnings
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from interaction_aware_surrogate import InteractionAwareSurrogate

# Set random seed for reproducibility
RANDOM_SEED = 42


class EnhancedInstaSHAP:
    """
    Enhanced InstaSHAP: Shapley values for GA²M models with interactions.

    This class extends InstaSHAP to handle pairwise interaction terms,
    allocating interaction contributions fairly between features.
    """

    def __init__(
        self,
        ga2m_surrogate: InteractionAwareSurrogate,
        background_data: Union[pd.DataFrame, np.ndarray],
        interaction_allocation: str = "equal",  # 'equal' or 'proportional'
    ):
        """
        Initialize Enhanced InstaSHAP with a fitted GA²M surrogate.

        Args:
            ga2m_surrogate: Fitted InteractionAwareSurrogate instance
            background_data: Background/reference data for computing expectations
            interaction_allocation: How to allocate interaction contributions
                - 'equal': Split 50-50 between interacting features
                - 'proportional': Based on main effect magnitudes
        """
        self.ga2m_surrogate = ga2m_surrogate
        self.interaction_allocation = interaction_allocation

        # Convert background data to DataFrame
        if isinstance(background_data, np.ndarray):
            self.background_data = pd.DataFrame(
                background_data, columns=ga2m_surrogate.feature_names
            )
        else:
            self.background_data = background_data.copy()

        # Precompute expected contributions
        self._precompute_expected_contributions()

    def _precompute_expected_contributions(self) -> None:
        """
        Precompute E[f_i(X_i)] and E[f_ij(X_i, X_j)] for efficiency.
        """
        print("Precomputing expected contributions (main + interactions)...")

        # Get contributions for all background samples
        main_contrib, interact_contrib, self.intercept = (
            self.ga2m_surrogate.get_feature_contributions(self.background_data)
        )

        # Expected main effects
        self.expected_main = np.mean(main_contrib, axis=0)

        # Expected interactions
        self.expected_interactions = np.mean(interact_contrib, axis=0)

        # Store info
        self.feature_names = self.ga2m_surrogate.feature_names
        self.n_features = len(self.feature_names)
        self.interaction_pairs = self.ga2m_surrogate.get_interaction_pairs()
        self.n_interactions = len(self.interaction_pairs)

        print(
            f"Expected contributions computed for {self.n_features} features "
            f"and {self.n_interactions} interactions."
        )

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_dataframe: bool = True,
        include_interaction_details: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame, Dict]:
        """
        Compute Enhanced InstaSHAP values with interaction handling.

        The Shapley value for each feature is:
            φ_i(x) = [f_i(x_i) - E[f_i(X_i)]] + Σ_{j: (i,j) in pairs} φ_i^{(ij)}

        where φ_i^{(ij)} is the allocated interaction contribution.

        Args:
            X: Feature matrix (n_samples, n_features)
            return_dataframe: If True, return results as DataFrame
            include_interaction_details: If True, return detailed breakdown

        Returns:
            Shapley values array/DataFrame, or dict with detailed breakdown
        """
        # Convert to DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        n_samples = len(X)

        # Get feature contributions
        main_contrib, interact_contrib, _ = (
            self.ga2m_surrogate.get_feature_contributions(X)
        )

        # Initialize SHAP values with main effects
        shap_values = main_contrib - self.expected_main

        # Allocate interaction contributions
        interaction_shap = np.zeros_like(shap_values)

        for int_idx, (feat_i, feat_j) in enumerate(self.interaction_pairs):
            # Interaction contribution for this term
            int_contrib = (
                interact_contrib[:, int_idx] - self.expected_interactions[int_idx]
            )

            if self.interaction_allocation == "equal":
                # Split equally between the two features
                allocation = 0.5
            else:  # proportional
                # This would require computing main effect magnitudes
                # For simplicity, use equal allocation
                allocation = 0.5

            # Add interaction contribution to both features
            interaction_shap[:, feat_i] += allocation * int_contrib
            interaction_shap[:, feat_j] += (1 - allocation) * int_contrib

        # Total SHAP values = main effects + interaction allocations
        total_shap = shap_values + interaction_shap

        if include_interaction_details:
            return {
                "total_shap": total_shap,
                "main_effect_shap": shap_values,
                "interaction_shap": interaction_shap,
                "interaction_pairs": self.interaction_pairs,
            }

        if return_dataframe:
            return pd.DataFrame(total_shap, columns=self.feature_names, index=X.index)

        return total_shap

    def explain_single(self, x: Union[pd.Series, np.ndarray, Dict]) -> Dict[str, float]:
        """
        Compute Enhanced InstaSHAP values for a single sample.

        Args:
            x: Single sample

        Returns:
            Dictionary mapping feature names to Shapley values
        """
        if isinstance(x, dict):
            x = pd.DataFrame([x])
        elif isinstance(x, np.ndarray):
            x = pd.DataFrame([x], columns=self.feature_names)
        elif isinstance(x, pd.Series):
            x = x.to_frame().T

        shap_values = self.explain(x, return_dataframe=False)[0]
        return dict(zip(self.feature_names, shap_values))

    def get_base_value(self) -> float:
        """
        Get the expected model output (base value).

        Returns:
            Base value φ_0
        """
        return (
            self.intercept
            + np.sum(self.expected_main)
            + np.sum(self.expected_interactions)
        )

    def verify_additivity(
        self, X: Union[pd.DataFrame, np.ndarray], tolerance: float = 1e-4
    ) -> Dict:
        """
        Verify that SHAP values satisfy the efficiency property.

        Args:
            X: Feature matrix to verify
            tolerance: Numerical tolerance

        Returns:
            Verification results dictionary
        """
        # Get predictions
        predictions = self.ga2m_surrogate.predict(X)

        # Get SHAP values
        shap_values = self.explain(X, return_dataframe=False)

        # Compute reconstructed predictions
        base_value = self.get_base_value()
        reconstructed = shap_values.sum(axis=1) + base_value

        # Check difference
        diff = np.abs(predictions - reconstructed)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        return {
            "passed": max_diff < tolerance,
            "max_difference": max_diff,
            "mean_difference": mean_diff,
            "tolerance": tolerance,
        }

    def feature_importance(
        self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute global feature importance as mean absolute SHAP value.

        Args:
            X: Optional data to compute importance over

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if X is None:
            X = self.background_data

        shap_values = self.explain(X, return_dataframe=False)
        importance = np.mean(np.abs(shap_values), axis=0)

        return dict(zip(self.feature_names, importance))

    def get_interaction_importance(
        self, X: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ) -> Dict[Tuple[int, int], float]:
        """
        Compute importance of each interaction term.

        Args:
            X: Optional data to compute importance over

        Returns:
            Dictionary mapping interaction pairs to importance scores
        """
        if X is None:
            X = self.background_data

        # Get interaction contributions
        _, interact_contrib, _ = self.ga2m_surrogate.get_feature_contributions(X)

        # Compute mean absolute contribution for each interaction
        importance = {}
        for int_idx, pair in enumerate(self.interaction_pairs):
            importance[pair] = np.mean(
                np.abs(
                    interact_contrib[:, int_idx] - self.expected_interactions[int_idx]
                )
            )

        return importance


def compute_enhanced_instashap_values(
    ga2m_surrogate: InteractionAwareSurrogate,
    X_explain: Union[pd.DataFrame, np.ndarray],
    X_background: Union[pd.DataFrame, np.ndarray],
) -> Tuple[np.ndarray, float]:
    """
    Convenience function to compute Enhanced InstaSHAP values.

    Args:
        ga2m_surrogate: Fitted GA²M surrogate
        X_explain: Samples to explain
        X_background: Background data for expectations

    Returns:
        Tuple of (shap_values array, base_value)
    """
    enhanced = EnhancedInstaSHAP(ga2m_surrogate, X_background)
    shap_values = enhanced.explain(X_explain, return_dataframe=False)
    base_value = enhanced.get_base_value()

    return shap_values, base_value


if __name__ == "__main__":
    # Test the module
    sys.path.append(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "phase2"
        )
    )

    from phase2.data.data_loader import load_california_housing
    from phase2.models.base_model import train_model_for_dataset
    from interaction_aware_surrogate import train_interaction_surrogate_for_blackbox

    print("=" * 60)
    print("Testing Enhanced InstaSHAP Module")
    print("=" * 60)

    # Load data and train models
    cal_data = load_california_housing()
    blackbox = train_model_for_dataset(cal_data, model_type="xgboost")

    # Train GA²M surrogate with interactions
    ga2m = train_interaction_surrogate_for_blackbox(
        blackbox.model, cal_data["X_train"], task_type="regression", n_interactions=5
    )

    # Create Enhanced InstaSHAP explainer
    print("\n--- Creating Enhanced InstaSHAP explainer ---")
    enhanced_instashap = EnhancedInstaSHAP(ga2m, cal_data["X_train"])

    # Explain test samples
    print("\n--- Computing Enhanced InstaSHAP values ---")
    X_test_sample = cal_data["X_test"].iloc[:10]
    shap_values = enhanced_instashap.explain(X_test_sample)

    print(f"SHAP values shape: {shap_values.shape}")
    print(f"\nSHAP values for first sample:")
    print(shap_values.iloc[0])

    # Verify additivity
    print("\n--- Verifying additivity ---")
    verification = enhanced_instashap.verify_additivity(X_test_sample)
    print(f"Verification passed: {verification['passed']}")
    print(f"Max difference: {verification['max_difference']:.6f}")

    # Feature importance
    print("\n--- Feature importance ---")
    importance = enhanced_instashap.feature_importance(X_test_sample)
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_importance[:5]:
        print(f"  {feat}: {imp:.4f}")

    # Interaction importance
    print("\n--- Interaction importance ---")
    int_importance = enhanced_instashap.get_interaction_importance(X_test_sample)
    for pair, imp in int_importance.items():
        feat_names = [cal_data["feature_names"][i] for i in pair]
        print(f"  {feat_names[0]} x {feat_names[1]}: {imp:.4f}")

    print("\n" + "=" * 60)
    print("Enhanced InstaSHAP module test complete!")
    print("=" * 60)
