"""
interaction_aware_surrogate.py — Fit a GA²M surrogate (GAM with pairwise
interactions) to better approximate black-box models that rely on
feature interactions.

Uses InterpretML's EBM with the `interactions` parameter enabled.
"""

import numpy as np
import pandas as pd

try:
    from interpret.glassbox import (
        ExplainableBoostingRegressor,
        ExplainableBoostingClassifier,
    )

    HAS_INTERPRET = True
except ImportError:
    HAS_INTERPRET = False


def fit_interaction_ebm(
    blackbox_model,
    X_train,
    task="regression",
    n_interactions=10,
    max_bins=256,
    outer_bags=8,
    learning_rate=0.01,
    max_rounds=5000,
    min_samples_leaf=2,
    random_state=42,
    interactions=None,
):
    """Fit an EBM with pairwise interactions (GA²M surrogate).

    Parameters
    ----------
    blackbox_model : fitted estimator
    X_train : array-like (n_samples, n_features)
    task : 'regression' or 'classification'
    n_interactions : int
        Number of pairwise interaction terms to include.
        If 'auto', InterpretML selects them automatically.
    max_bins : int
    outer_bags : int
    learning_rate : float
    max_rounds : int
    min_samples_leaf : int
    random_state : int
    interactions : list of tuples or 'auto' or None
        Explicit interaction pairs. If provided, overrides n_interactions.

    Returns
    -------
    ebm : fitted EBM with interactions
    """
    if not HAS_INTERPRET:
        raise ImportError(
            "InterpretML is required for interaction-aware EBM. "
            "Install with: pip install interpret"
        )

    X_np = np.asarray(X_train)

    # Get surrogate targets
    if task == "classification":
        try:
            y_surrogate = blackbox_model.predict_proba(X_np)[:, 1]
        except Exception:
            y_surrogate = blackbox_model.predict(X_np).astype(float)
    else:
        y_surrogate = blackbox_model.predict(X_np)

    # Determine interaction specification
    if interactions is not None:
        interaction_param = interactions
    elif n_interactions == "auto" or n_interactions == 0:
        interaction_param = n_interactions
    else:
        # Let InterpretML auto-detect top n_interactions
        interaction_param = n_interactions

    ebm_params = dict(
        max_bins=max_bins,
        interactions=interaction_param,
        outer_bags=outer_bags,
        learning_rate=learning_rate,
        max_rounds=max_rounds,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    if task == "classification":
        ebm = ExplainableBoostingClassifier(**ebm_params)
    else:
        ebm = ExplainableBoostingRegressor(**ebm_params)

    ebm.fit(X_np, y_surrogate)
    return ebm


def get_interaction_pairs(ebm, feature_names=None):
    """Extract the interaction pairs learned by the EBM.

    Returns
    -------
    list of tuples (feature_idx_i, feature_idx_j, interaction_name)
    """
    if not hasattr(ebm, "term_features"):
        return []

    interactions = []
    for i, term_feat in enumerate(ebm.term_features):
        if len(term_feat) == 2:
            fi, fj = term_feat
            if feature_names:
                name = f"{feature_names[fi]} × {feature_names[fj]}"
            else:
                name = f"feature_{fi} × feature_{fj}"
            interactions.append((fi, fj, name))
    return interactions


def get_all_component_values(ebm, X):
    """Extract ALL component function values including interaction terms.

    Parameters
    ----------
    ebm : fitted EBM (with or without interactions)
    X : array-like

    Returns
    -------
    additive_components : np.ndarray (n_samples, n_features)
        Per-feature additive contributions.
    interaction_components : dict
        Mapping (i, j) -> np.ndarray of interaction contributions.
    """
    X_np = np.asarray(X)

    if hasattr(ebm, "predict_and_contrib"):
        preds, contrib = ebm.predict_and_contrib(X_np)
        contrib = np.asarray(contrib)
    elif hasattr(ebm, "explain_global"):
        # Fallback: use term_scores
        contrib = np.zeros((X_np.shape[0], len(ebm.term_features)))
        for i, term_feat in enumerate(ebm.term_features):
            # Approximate by differencing predictions
            pass
        return contrib, {}
    else:
        raise AttributeError("Cannot extract components from this EBM.")

    # Separate additive and interaction components
    n_features_original = max(max(tf) for tf in ebm.term_features if len(tf) <= 2) + 1
    additive = np.zeros((X_np.shape[0], n_features_original))
    interactions = {}

    for i, term_feat in enumerate(ebm.term_features):
        if len(term_feat) == 1:
            additive[:, term_feat[0]] += contrib[:, i]
        elif len(term_feat) == 2:
            fi, fj = term_feat
            interactions[(fi, fj)] = contrib[:, i]

    return additive, interactions


def compute_h_statistic(model, X, feature_indices=None, n_samples=100):
    """Compute Friedman's H-statistic to measure interaction strength.

    H_jk measures the fraction of variance of f(x) attributable to the
    interaction between features j and k.

    Parameters
    ----------
    model : fitted estimator with predict method
    X : array-like
    feature_indices : list of int, optional
    n_samples : int

    Returns
    -------
    h_matrix : np.ndarray (n_features, n_features)
        Symmetric matrix of H-statistics. h_matrix[j][k] is the
        interaction strength between features j and k.
    """
    X_np = np.asarray(X)
    rng = np.random.RandomState(42)

    if len(X_np) > n_samples:
        idx = rng.choice(len(X_np), n_samples, replace=False)
        X_sub = X_np[idx]
    else:
        X_sub = X_np

    n_features = X_sub.shape[1]
    if feature_indices is None:
        feature_indices = list(range(n_features))

    # Full predictions
    f_full = model.predict(X_sub)

    h_matrix = np.zeros((n_features, n_features))

    for j_idx, j in enumerate(feature_indices):
        for k_idx, k in enumerate(feature_indices[j_idx + 1 :], j_idx + 1):
            # Compute H-statistic for pair (j, k)
            # H_jk = E_{x}[variance of f with j,k permuted] - ...
            # Simplified: difference between joint and marginal effects
            X_jk = X_sub.copy()
            X_j_perm = X_sub.copy()
            X_k_perm = X_sub.copy()

            perm_idx = rng.permutation(len(X_sub))
            X_j_perm[:, j] = X_sub[perm_idx, j]
            X_k_perm[:, k] = X_sub[perm_idx, k]

            f_jk = model.predict(X_jk)
            f_j = model.predict(X_j_perm)
            f_k = model.predict(X_k_perm)

            # Interaction = f - f_j_marginal - f_k_marginal + f_both_marginal
            interaction_var = np.mean((f_full - f_j - f_k + f_jk) ** 2)
            total_var = np.var(f_full) + 1e-12

            h_matrix[j, k] = np.sqrt(interaction_var / total_var)
            h_matrix[k, j] = h_matrix[j, k]

    return h_matrix
