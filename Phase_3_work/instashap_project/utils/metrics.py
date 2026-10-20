"""Metric helpers used across experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score


@dataclass(slots=True)
class RegressionMetrics:
    rmse: float
    mse: float
    r2: float
    nmse_pct: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    log_loss: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Compute regression metrics used in the paper-style tables."""

    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    nmse_pct = (1.0 - r2) * 100.0
    return RegressionMetrics(
        rmse=float(np.sqrt(mse)),
        mse=mse,
        r2=r2,
        nmse_pct=nmse_pct,
    )


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute classification accuracy and log-loss."""

    if predictions is None:
        predictions = probabilities.argmax(axis=1)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, predictions)),
        log_loss=float(log_loss(y_true, probabilities)),
    )


def explanation_error(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compare explanation tensors with mean-squared and mean-absolute error."""

    difference = np.asarray(reference) - np.asarray(candidate)
    return {
        "mse": float(np.mean(np.square(difference))),
        "mae": float(np.mean(np.abs(difference))),
    }


def explanation_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Compute explanation fidelity metrics against a reference explainer."""

    summary = explanation_error(reference, candidate)
    correlation = spearmanr(np.asarray(reference).reshape(-1), np.asarray(candidate).reshape(-1)).correlation
    summary["spearman"] = 0.0 if correlation is None or np.isnan(correlation) else float(correlation)
    return summary


def benchmark_callable(callable_fn: Callable[[], np.ndarray], repeats: int = 5) -> dict[str, float]:
    """Estimate latency statistics of a callable."""

    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        callable_fn()
        timings.append(perf_counter() - start)
    timings_array = np.asarray(timings, dtype=float)
    return {
        "seconds_mean": float(timings_array.mean()),
        "seconds_std": float(timings_array.std()),
        "seconds_min": float(timings_array.min()),
        "seconds_max": float(timings_array.max()),
    }
