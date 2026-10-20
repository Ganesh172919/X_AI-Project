"""Extended metrics for Phase 3 — includes innovation-specific metrics."""

from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn import metrics as skmetrics


# ── Predictive Performance ──────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(skmetrics.mean_squared_error(y_true, y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skmetrics.mean_squared_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skmetrics.r2_score(y_true, y_pred))


def nmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((1.0 - skmetrics.r2_score(y_true, y_pred)) * 100)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skmetrics.accuracy_score(y_true, y_pred))


def log_loss(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    return float(skmetrics.log_loss(y_true, y_pred_prob, labels=np.arange(y_pred_prob.shape[1])))


# ── Explanation Fidelity ────────────────────────────────────────────────

def explanation_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean((reference - candidate) ** 2))


def explanation_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.mean(np.abs(reference - candidate)))


def spearman_rank_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Feature-level Spearman rank correlation averaged across samples."""
    n_samples = reference.shape[0]
    corrs = []
    for i in range(n_samples):
        ref_flat = reference[i].ravel()
        cand_flat = candidate[i].ravel()
        if np.std(ref_flat) < 1e-12 or np.std(cand_flat) < 1e-12:
            corrs.append(0.0)
        else:
            rho, _ = stats.spearmanr(ref_flat, cand_flat)
            corrs.append(float(rho) if not np.isnan(rho) else 0.0)
    return float(np.mean(corrs))


# ── Innovation 1: Coalition Fidelity ────────────────────────────────────

def coalition_fidelity_mse(surrogate_pred: np.ndarray, true_pred: np.ndarray) -> float:
    """MSE between surrogate and true background-averaged blackbox outputs."""
    return float(np.mean((surrogate_pred - true_pred) ** 2))


# ── Innovation 2: Convergence Speed ────────────────────────────────────

def convergence_epoch(val_losses: list[float], threshold_ratio: float = 0.95) -> int:
    """Epoch at which val loss first reaches threshold_ratio * final best loss."""
    if not val_losses:
        return 0
    best = min(val_losses)
    target = best / threshold_ratio  # ceiling threshold
    for ep, loss in enumerate(val_losses):
        if loss <= target:
            return ep
    return len(val_losses) - 1


# ── Innovation 3: Explanation Stability ─────────────────────────────────

def explanation_stability_score(attributions_list: list[np.ndarray], eps: float = 1e-8) -> float:
    """1 - mean coefficient of variation across ensemble attributions.

    attributions_list: list of (n_samples, n_features, n_outputs) arrays
    Returns a score in [0, 1] where 1 = perfectly stable.
    """
    stacked = np.stack(attributions_list, axis=0)  # (M, n_samples, n_features, n_outputs)
    stds = np.std(stacked, axis=0)
    means = np.mean(stacked, axis=0)
    cv = stds / (np.abs(means) + eps)
    return float(1.0 - np.mean(cv))


def per_feature_confidence(attributions_list: list[np.ndarray], eps: float = 1e-8) -> np.ndarray:
    """Per-feature confidence from ensemble variance. Shape: (n_features,)."""
    stacked = np.stack(attributions_list, axis=0)
    stds = np.std(stacked, axis=0).mean(axis=(0, -1))  # avg over samples, outputs
    means = np.abs(np.mean(stacked, axis=0)).mean(axis=(0, -1))
    return 1.0 - stds / (means + eps)


# ── Latency benchmark ──────────────────────────────────────────────────

import time

def benchmark_callable(fn, *args, n_runs: int = 5, **kwargs) -> dict[str, float]:
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}
