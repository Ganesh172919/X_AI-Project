"""
enhanced_instashap.py — Extended InstaSHAP for GA²M models with pairwise
interactions.

For a GA²M model:
    f(x) = f_0 + Σ_j f_j(x_j) + Σ_{(j,k)} f_{jk}(x_j, x_k)

The Shapley value for feature j is:
    φ_j = (f_j(x_j) - E[f_j])
         + Σ_{k != j} 0.5 * (f_{jk}(x_j, x_k) - E[f_{jk}])

Each pairwise interaction is split equally between the two interacting
features. This is consistent with the Shapley interaction index for
pairwise terms.
"""

import numpy as np
import pandas as pd


def enhanced_instashap_ga2m(additive_components, interaction_components, baseline=None):
    """Compute enhanced InstaSHAP values for a GA²M model.

    Parameters
    ----------
    additive_components : np.ndarray, shape (n_samples, n_features)
        Per-feature additive component values f_j(x_j).
    interaction_components : dict
        Mapping (i, j) -> np.ndarray of shape (n_samples,)
        Interaction component values f_{ij}(x_i, x_j).
    baseline : float or None
        Base / intercept value. If None, computed from data.

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_features)
    base_value : float
    """
    additive_components = np.asarray(additive_components)
    n_samples, n_features = additive_components.shape

    # Mean additive components
    mean_additive = np.mean(additive_components, axis=0)

    # Start with additive Shapley values
    shap_values = additive_components - mean_additive[np.newaxis, :]

    # Add interaction contributions (split equally)
    for (i, j), interaction_vals in interaction_components.items():
        interaction_vals = np.asarray(interaction_vals)
        mean_interaction = np.mean(interaction_vals)
        centered = interaction_vals - mean_interaction

        # Split equally between features i and j
        shap_values[:, i] += 0.5 * centered
        shap_values[:, j] += 0.5 * centered

    # Base value
    if baseline is None:
        total_additive = np.sum(additive_components, axis=1)
        total_interaction = sum(np.asarray(v) for v in interaction_components.values())
        total = total_additive + total_interaction
        base_value = float(np.mean(total))
    else:
        base_value = float(baseline)

    return shap_values, base_value


def enhanced_instashap_from_ebm(ebm, X, feature_names=None):
    """Compute enhanced InstaSHAP from a fitted EBM (with interactions).

    Parameters
    ----------
    ebm : fitted EBM (InterpretML)
    X : array-like
    feature_names : list of str, optional

    Returns
    -------
    shap_values : np.ndarray or pd.DataFrame
    base_value : float
    info : dict with additive/interaction component details
    """
    X_np = np.asarray(X)

    # Extract components
    if hasattr(ebm, "term_features") and hasattr(ebm, "predict_and_contrib"):
        preds, contrib = ebm.predict_and_contrib(X_np)
        contrib = np.asarray(contrib)

        # Determine original feature count
        all_features = set()
        for tf in ebm.term_features:
            all_features.update(tf)
        n_features = max(all_features) + 1

        additive = np.zeros((X_np.shape[0], n_features))
        interactions = {}

        for i, term_feat in enumerate(ebm.term_features):
            if len(term_feat) == 1:
                additive[:, term_feat[0]] += contrib[:, i]
            elif len(term_feat) == 2:
                fi, fj = term_feat
                interactions[(fi, fj)] = contrib[:, i]
    else:
        # Fallback: treat as purely additive
        from phase2.models.instashap import instashap_from_ebm as _additive

        shap_vals, base_val = _additive(ebm, X_np, feature_names)
        return shap_vals, base_val, {"mode": "additive_fallback"}

    shap_values, base_value = enhanced_instashap_ga2m(additive, interactions)

    if feature_names is not None:
        if len(feature_names) == shap_values.shape[1]:
            shap_values = pd.DataFrame(shap_values, columns=feature_names)

    info = {
        "mode": "ga2m",
        "n_additive_terms": n_features,
        "n_interaction_terms": len(interactions),
        "interaction_pairs": list(interactions.keys()),
        "additive_components": additive,
        "interaction_components": interactions,
    }

    return shap_values, base_value, info


def enhanced_instashap_explain(
    blackbox_model,
    X_train,
    X_explain,
    task="regression",
    feature_names=None,
    n_interactions=10,
    **ebm_kwargs,
):
    """One-shot: fit GA²M surrogate then compute enhanced InstaSHAP.

    Parameters
    ----------
    blackbox_model : fitted estimator
    X_train : array-like
    X_explain : array-like
    task : 'regression' or 'classification'
    feature_names : list of str
    n_interactions : int or 'auto'
    **ebm_kwargs : forwarded to fit_interaction_ebm

    Returns
    -------
    shap_values : np.ndarray or DataFrame
    base_value : float
    ebm : fitted GA²M surrogate
    info : dict
    """
    from phase3.extension.interaction_aware_surrogate import fit_interaction_ebm

    ebm = fit_interaction_ebm(
        blackbox_model, X_train, task=task, n_interactions=n_interactions, **ebm_kwargs
    )

    shap_values, base_value, info = enhanced_instashap_from_ebm(
        ebm, X_explain, feature_names
    )

    return shap_values, base_value, ebm, info
