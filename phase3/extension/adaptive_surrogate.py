"""
adaptive_surrogate.py — Adaptive strategy that automatically selects between
a purely additive GAM and a GA²M with interactions based on surrogate fidelity.

If the additive surrogate's R² exceeds a threshold, use it (faster).
Otherwise, upgrade to GA²M with interactions (more accurate).
"""

import numpy as np
from phase3.extension.interaction_aware_surrogate import fit_interaction_ebm
from phase2.models.gam_surrogate import fit_ebm_surrogate, surrogate_fidelity
from phase2.models.instashap import instashap_from_ebm
from phase3.extension.enhanced_instashap import enhanced_instashap_from_ebm


class AdaptiveInstaSHAP:
    """Automatically choose between additive and interaction-aware surrogates.

    Parameters
    ----------
    blackbox_model : fitted estimator
    task : 'regression' or 'classification'
    fidelity_threshold : float
        Minimum acceptable R² for the additive surrogate.
        If R² < threshold, upgrade to GA²M.
    n_interactions : int
        Number of interaction terms for the GA²M fallback.
    feature_names : list of str
    """

    def __init__(
        self,
        blackbox_model,
        task="regression",
        fidelity_threshold=0.95,
        n_interactions=10,
        feature_names=None,
        random_state=42,
    ):
        self.blackbox_model = blackbox_model
        self.task = task
        self.fidelity_threshold = fidelity_threshold
        self.n_interactions = n_interactions
        self.feature_names = feature_names
        self.random_state = random_state
        self.surrogate_ = None
        self.mode_ = None  # 'additive' or 'interaction'

    def fit(self, X_train):
        """Fit the adaptive surrogate.

        1. Try additive EBM.
        2. Check fidelity.
        3. If insufficient, upgrade to GA²M.
        """
        from sklearn.metrics import r2_score

        # Step 1: Try additive
        ebm_additive = fit_ebm_surrogate(
            self.blackbox_model,
            X_train,
            task=self.task,
            interactions=0,
            random_state=self.random_state,
        )

        # Check fidelity
        fid = surrogate_fidelity(
            self.blackbox_model, ebm_additive, X_train, task=self.task
        )

        if fid["r2"] >= self.fidelity_threshold:
            self.surrogate_ = ebm_additive
            self.mode_ = "additive"
            print(f"  Adaptive: using ADDITIVE surrogate (R²={fid['r2']:.4f})")
        else:
            # Upgrade to GA²M
            ebm_interaction = fit_interaction_ebm(
                self.blackbox_model,
                X_train,
                task=self.task,
                n_interactions=self.n_interactions,
                random_state=self.random_state,
            )
            fid2 = surrogate_fidelity(
                self.blackbox_model, ebm_interaction, X_train, task=self.task
            )
            self.surrogate_ = ebm_interaction
            self.mode_ = "interaction"
            print(
                f"  Adaptive: upgraded to INTERACTION surrogate "
                f"(additive R²={fid['r2']:.4f} → interaction R²={fid2['r2']:.4f})"
            )

        if self.feature_names is None and hasattr(X_train, "columns"):
            self.feature_names = list(X_train.columns)

        return self

    def explain(self, X_explain):
        """Compute Shapley values using the selected surrogate.

        Returns
        -------
        shap_values : np.ndarray or DataFrame
        base_value : float
        info : dict with mode and fidelity details
        """
        if self.surrogate_ is None:
            raise RuntimeError("Call .fit(X_train) first.")

        X_np = np.asarray(X_explain)

        if self.mode_ == "additive":
            shap_values, base_value = instashap_from_ebm(
                self.surrogate_, X_np, self.feature_names
            )
            info = {"mode": "additive", "n_interactions": 0}
        else:
            shap_values, base_value, ext_info = enhanced_instashap_from_ebm(
                self.surrogate_, X_np, self.feature_names
            )
            info = {
                "mode": "interaction",
                "n_interactions": ext_info.get("n_interaction_terms", 0),
                "interaction_pairs": ext_info.get("interaction_pairs", []),
            }

        return shap_values, base_value, info

    def fit_and_explain(self, X_train, X_explain):
        """Convenience: fit then explain."""
        self.fit(X_train)
        return self.explain(X_explain)
