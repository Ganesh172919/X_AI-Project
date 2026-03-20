"""
Comprehensive evaluation module for SHAP prediction accuracy.

Provides metrics and visualizations for comparing true (exact) SHAP values
against GAM surrogate predictions. Covers global accuracy, per-feature
accuracy, speed benchmarks, and feature ranking preservation.

Metrics Computed
----------------
Global:
    MSE, MAE, RMSE, R², MAPE, Pearson correlation, Spearman correlation.

Per-Feature:
    MSE, MAE, R², Pearson correlation (returned as a DataFrame).

Speed:
    Exact vs. surrogate computation time, speedup factor, per-sample latency.

Ranking:
    Top-k feature overlap ratio, Spearman correlation of importance rankings.

Example
-------
>>> from src.evaluation import SHAPEvaluator
>>> evaluator = SHAPEvaluator(feature_names=["age", "income", "education"])
>>> metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)
>>> speed = evaluator.compute_speed_metrics(exact_time=120.0, surrogate_time=0.5, n_samples=500)
>>> evaluator.generate_comparison_plots(true_shap, pred_shap, save_dir="results/figures")
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SHAPEvaluator:
    """
    Evaluate and compare true SHAP vs predicted SHAP values.

    Attributes
    ----------
    feature_names : list of str
        Names of the features corresponding to SHAP columns.
    n_features : int
        Number of features.

    Example
    -------
    >>> evaluator = SHAPEvaluator(feature_names)
    >>> acc = evaluator.compute_accuracy_metrics(true_shap, pred_shap)
    >>> per_feat = evaluator.compute_per_feature_metrics(true_shap, pred_shap)
    >>> rankings = evaluator.compare_feature_rankings(true_shap, pred_shap, top_k=10)
    """

    def __init__(self, feature_names: List[str]):
        """
        Initialize evaluator.

        Args:
            feature_names: List of feature names
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)

    def compute_accuracy_metrics(
        self, true_shap: np.ndarray, pred_shap: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics between true and predicted SHAP values.

        Args:
            true_shap: True SHAP values (n_samples, n_features)
            pred_shap: Predicted SHAP values (n_samples, n_features)

        Returns:
            Dictionary of metrics
        """
        # Flatten arrays for global metrics
        true_flat = true_shap.flatten()
        pred_flat = pred_shap.flatten()

        # Global metrics
        mse = mean_squared_error(true_flat, pred_flat)
        mae = mean_absolute_error(true_flat, pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_flat, pred_flat)

        pearson_corr, pearson_pval = pearsonr(true_flat, pred_flat)
        spearman_corr, spearman_pval = spearmanr(true_flat, pred_flat)

        # Compute MAPE (handling near-zero values)
        epsilon = 1e-10
        mape = (
            np.mean(np.abs((true_flat - pred_flat) / (np.abs(true_flat) + epsilon)))
            * 100
        )

        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "mape": float(mape),
            "pearson_correlation": float(pearson_corr),
            "pearson_pvalue": float(pearson_pval),
            "spearman_correlation": float(spearman_corr),
            "spearman_pvalue": float(spearman_pval),
        }

        return metrics

    def compute_per_feature_metrics(
        self, true_shap: np.ndarray, pred_shap: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute metrics for each feature separately.

        Args:
            true_shap: True SHAP values
            pred_shap: Predicted SHAP values

        Returns:
            DataFrame with per-feature metrics
        """
        per_feature_results = []

        for i, feature_name in enumerate(self.feature_names):
            true_col = true_shap[:, i]
            pred_col = pred_shap[:, i]

            mse = mean_squared_error(true_col, pred_col)
            mae = mean_absolute_error(true_col, pred_col)
            r2 = r2_score(true_col, pred_col)
            corr, _ = pearsonr(true_col, pred_col)

            per_feature_results.append(
                {
                    "feature": feature_name,
                    "mse": mse,
                    "mae": mae,
                    "r2": r2,
                    "correlation": corr,
                }
            )

        df_results = pd.DataFrame(per_feature_results)
        return df_results

    def compute_speed_metrics(
        self, exact_time: float, surrogate_time: float, n_samples: int
    ) -> Dict[str, float]:
        """
        Compute speed comparison metrics.

        Args:
            exact_time: Time for exact SHAP computation (seconds)
            surrogate_time: Time for surrogate prediction (seconds)
            n_samples: Number of samples

        Returns:
            Dictionary of speed metrics
        """
        speedup = exact_time / surrogate_time if surrogate_time > 0 else float("inf")

        metrics = {
            "exact_time_seconds": float(exact_time),
            "surrogate_time_seconds": float(surrogate_time),
            "speedup_factor": float(speedup),
            "exact_time_per_sample_ms": float(exact_time / n_samples * 1000),
            "surrogate_time_per_sample_ms": float(surrogate_time / n_samples * 1000),
        }

        return metrics

    def compare_feature_rankings(
        self, true_shap: np.ndarray, pred_shap: np.ndarray, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compare feature importance rankings.

        Args:
            true_shap: True SHAP values
            pred_shap: Predicted SHAP values
            top_k: Number of top features to compare

        Returns:
            Dictionary with ranking comparison metrics
        """
        # Compute mean absolute SHAP values per feature
        true_importance = np.mean(np.abs(true_shap), axis=0)
        pred_importance = np.mean(np.abs(pred_shap), axis=0)

        # Get rankings
        true_ranking = np.argsort(true_importance)[::-1]
        pred_ranking = np.argsort(pred_importance)[::-1]

        # Top-k overlap
        true_top_k = set(true_ranking[:top_k])
        pred_top_k = set(pred_ranking[:top_k])
        overlap = len(true_top_k.intersection(pred_top_k))
        overlap_ratio = overlap / top_k

        # Spearman correlation of rankings
        ranking_corr, _ = spearmanr(true_importance, pred_importance)

        # Create ranking DataFrame
        ranking_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "true_importance": true_importance,
                "pred_importance": pred_importance,
                "true_rank": [
                    np.where(true_ranking == i)[0][0] + 1
                    for i in range(len(self.feature_names))
                ],
                "pred_rank": [
                    np.where(pred_ranking == i)[0][0] + 1
                    for i in range(len(self.feature_names))
                ],
            }
        )

        ranking_df = ranking_df.sort_values("true_importance", ascending=False)

        results = {
            "top_k": top_k,
            "top_k_overlap": overlap,
            "top_k_overlap_ratio": float(overlap_ratio),
            "ranking_correlation": float(ranking_corr),
            "ranking_df": ranking_df,
        }

        return results

    def generate_comparison_plots(
        self,
        true_shap: np.ndarray,
        pred_shap: np.ndarray,
        save_dir: str,
        dataset_name: str = "dataset",
    ) -> None:
        """
        Generate comprehensive comparison plots.

        Args:
            true_shap: True SHAP values
            pred_shap: Predicted SHAP values
            save_dir: Directory to save plots
            dataset_name: Name of dataset for titles
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")

        # 1. Scatter plot: True vs Predicted SHAP
        self._plot_scatter(true_shap, pred_shap, save_dir, dataset_name)

        # 2. Per-feature comparison
        self._plot_per_feature_comparison(true_shap, pred_shap, save_dir, dataset_name)

        # 3. Error distribution
        self._plot_error_distribution(true_shap, pred_shap, save_dir, dataset_name)

        # 4. Feature importance comparison
        self._plot_feature_importance(true_shap, pred_shap, save_dir, dataset_name)

        logger.info(f"Comparison plots saved to {save_dir}")

    def _plot_scatter(
        self,
        true_shap: np.ndarray,
        pred_shap: np.ndarray,
        save_dir: str,
        dataset_name: str,
    ) -> None:
        """Generate scatter plot of true vs predicted SHAP."""
        plt.figure(figsize=(10, 8))

        true_flat = true_shap.flatten()
        pred_flat = pred_shap.flatten()

        # Sample if too many points
        if len(true_flat) > 10000:
            indices = np.random.choice(len(true_flat), 10000, replace=False)
            true_flat = true_flat[indices]
            pred_flat = pred_flat[indices]

        plt.scatter(true_flat, pred_flat, alpha=0.3, s=10)

        # Add diagonal line
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect prediction",
        )

        # Compute R²
        r2 = r2_score(true_flat, pred_flat)

        plt.xlabel("True SHAP Values", fontsize=12)
        plt.ylabel("Predicted SHAP Values", fontsize=12)
        plt.title(
            f"True vs Predicted SHAP Values\n{dataset_name} (R² = {r2:.4f})",
            fontsize=14,
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/scatter_true_vs_pred_{dataset_name}.png", dpi=300)
        plt.close()

    def _plot_per_feature_comparison(
        self,
        true_shap: np.ndarray,
        pred_shap: np.ndarray,
        save_dir: str,
        dataset_name: str,
    ) -> None:
        """Plot per-feature R² scores."""
        per_feature_r2 = []
        for i in range(self.n_features):
            r2 = r2_score(true_shap[:, i], pred_shap[:, i])
            per_feature_r2.append(r2)

        plt.figure(figsize=(12, 6))
        plt.bar(range(self.n_features), per_feature_r2, color="steelblue", alpha=0.7)
        plt.axhline(
            y=np.mean(per_feature_r2),
            color="r",
            linestyle="--",
            label=f"Mean R² = {np.mean(per_feature_r2):.4f}",
        )
        plt.xlabel("Feature Index", fontsize=12)
        plt.ylabel("R² Score", fontsize=12)
        plt.title(f"Per-Feature SHAP Prediction Accuracy\n{dataset_name}", fontsize=14)
        plt.xticks(range(0, self.n_features, max(1, self.n_features // 10)))
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/per_feature_r2_{dataset_name}.png", dpi=300)
        plt.close()

    def _plot_error_distribution(
        self,
        true_shap: np.ndarray,
        pred_shap: np.ndarray,
        save_dir: str,
        dataset_name: str,
    ) -> None:
        """Plot error distribution."""
        errors = (pred_shap - true_shap).flatten()

        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, color="steelblue", alpha=0.7, edgecolor="black")
        plt.axvline(x=0, color="r", linestyle="--", linewidth=2)
        plt.axvline(
            x=np.mean(errors),
            color="g",
            linestyle="--",
            linewidth=2,
            label=f"Mean Error = {np.mean(errors):.4f}",
        )
        plt.xlabel("Prediction Error", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"SHAP Prediction Error Distribution\n{dataset_name}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/error_distribution_{dataset_name}.png", dpi=300)
        plt.close()

    def _plot_feature_importance(
        self,
        true_shap: np.ndarray,
        pred_shap: np.ndarray,
        save_dir: str,
        dataset_name: str,
    ) -> None:
        """Plot feature importance comparison."""
        true_importance = np.mean(np.abs(true_shap), axis=0)
        pred_importance = np.mean(np.abs(pred_shap), axis=0)

        # Sort by true importance
        sorted_indices = np.argsort(true_importance)[::-1][:20]  # Top 20

        x = np.arange(len(sorted_indices))
        width = 0.35

        plt.figure(figsize=(12, 6))
        plt.bar(
            x - width / 2,
            true_importance[sorted_indices],
            width,
            label="True",
            color="steelblue",
            alpha=0.7,
        )
        plt.bar(
            x + width / 2,
            pred_importance[sorted_indices],
            width,
            label="Predicted",
            color="coral",
            alpha=0.7,
        )

        plt.xlabel("Feature", fontsize=12)
        plt.ylabel("Mean |SHAP|", fontsize=12)
        plt.title(
            f"Feature Importance Comparison (Top 20)\n{dataset_name}", fontsize=14
        )
        plt.xticks(
            x,
            [
                self.feature_names[i] if i < len(self.feature_names) else f"F{i}"
                for i in sorted_indices
            ],
            rotation=45,
            ha="right",
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/feature_importance_{dataset_name}.png", dpi=300)
        plt.close()
