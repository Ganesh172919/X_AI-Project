"""
Unit tests for evaluation module.
"""

import pytest
import numpy as np
import os
from src.evaluation import SHAPEvaluator


@pytest.fixture
def evaluator():
    """Create a SHAPEvaluator instance."""
    return SHAPEvaluator(feature_names=["feat_a", "feat_b", "feat_c"])


@pytest.fixture
def shap_data():
    """Generate synthetic true and predicted SHAP values."""
    np.random.seed(42)
    n_samples, n_features = 100, 3
    true_shap = np.random.randn(n_samples, n_features)
    # Predicted SHAP with small noise (high accuracy scenario)
    pred_shap = true_shap + np.random.randn(n_samples, n_features) * 0.1
    return true_shap, pred_shap


class TestAccuracyMetrics:
    """Tests for compute_accuracy_metrics."""

    def test_returns_expected_keys(self, evaluator, shap_data):
        """Test that all expected metric keys are returned."""
        true_shap, pred_shap = shap_data
        metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)

        expected_keys = [
            "mse",
            "mae",
            "rmse",
            "r2",
            "mape",
            "pearson_correlation",
            "pearson_pvalue",
            "spearman_correlation",
            "spearman_pvalue",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_types(self, evaluator, shap_data):
        """Test that all metrics are floats."""
        true_shap, pred_shap = shap_data
        metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)

        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is not a float"

    def test_perfect_prediction(self, evaluator):
        """Test metrics for perfect predictions."""
        shap = np.random.randn(50, 3)
        metrics = evaluator.compute_accuracy_metrics(shap, shap)

        assert metrics["mse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["r2"] == pytest.approx(1.0, abs=1e-6)
        assert metrics["pearson_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_mse_non_negative(self, evaluator, shap_data):
        """Test that MSE is always non-negative."""
        true_shap, pred_shap = shap_data
        metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)
        assert metrics["mse"] >= 0

    def test_correlation_range(self, evaluator, shap_data):
        """Test that correlations are in [-1, 1]."""
        true_shap, pred_shap = shap_data
        metrics = evaluator.compute_accuracy_metrics(true_shap, pred_shap)

        assert -1 <= metrics["pearson_correlation"] <= 1
        assert -1 <= metrics["spearman_correlation"] <= 1


class TestPerFeatureMetrics:
    """Tests for compute_per_feature_metrics."""

    def test_returns_dataframe(self, evaluator, shap_data):
        """Test that per-feature metrics returns a DataFrame."""
        import pandas as pd

        true_shap, pred_shap = shap_data
        df = evaluator.compute_per_feature_metrics(true_shap, pred_shap)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_dataframe_columns(self, evaluator, shap_data):
        """Test that DataFrame has expected columns."""
        true_shap, pred_shap = shap_data
        df = evaluator.compute_per_feature_metrics(true_shap, pred_shap)

        expected_cols = ["feature", "mse", "mae", "r2", "correlation"]
        for col in expected_cols:
            assert col in df.columns

    def test_feature_names_in_output(self, evaluator, shap_data):
        """Test that feature names appear in the output."""
        true_shap, pred_shap = shap_data
        df = evaluator.compute_per_feature_metrics(true_shap, pred_shap)

        assert list(df["feature"]) == ["feat_a", "feat_b", "feat_c"]


class TestSpeedMetrics:
    """Tests for compute_speed_metrics."""

    def test_basic_speed_metrics(self, evaluator):
        """Test basic speed metric computation."""
        metrics = evaluator.compute_speed_metrics(
            exact_time=100.0, surrogate_time=2.0, n_samples=500
        )

        assert metrics["exact_time_seconds"] == 100.0
        assert metrics["surrogate_time_seconds"] == 2.0
        assert metrics["speedup_factor"] == pytest.approx(50.0)
        assert metrics["exact_time_per_sample_ms"] == pytest.approx(200.0)
        assert metrics["surrogate_time_per_sample_ms"] == pytest.approx(4.0)

    def test_zero_surrogate_time(self, evaluator):
        """Test speed metrics when surrogate time is zero."""
        metrics = evaluator.compute_speed_metrics(
            exact_time=100.0, surrogate_time=0.0, n_samples=500
        )
        assert metrics["speedup_factor"] == float("inf")


class TestFeatureRankings:
    """Tests for compare_feature_rankings."""

    def test_returns_expected_keys(self, evaluator, shap_data):
        """Test that ranking comparison returns expected keys."""
        true_shap, pred_shap = shap_data
        results = evaluator.compare_feature_rankings(true_shap, pred_shap, top_k=2)

        assert "top_k" in results
        assert "top_k_overlap" in results
        assert "top_k_overlap_ratio" in results
        assert "ranking_correlation" in results
        assert "ranking_df" in results

    def test_perfect_ranking_overlap(self, evaluator):
        """Test ranking comparison with identical SHAP values."""
        shap = np.random.randn(100, 3)
        results = evaluator.compare_feature_rankings(shap, shap, top_k=2)

        assert results["top_k_overlap"] == 2
        assert results["top_k_overlap_ratio"] == pytest.approx(1.0)
        assert results["ranking_correlation"] == pytest.approx(1.0)

    def test_ranking_df_structure(self, evaluator, shap_data):
        """Test structure of the ranking DataFrame."""
        import pandas as pd

        true_shap, pred_shap = shap_data
        results = evaluator.compare_feature_rankings(true_shap, pred_shap, top_k=2)

        df = results["ranking_df"]
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "true_importance" in df.columns
        assert "pred_importance" in df.columns
        assert "true_rank" in df.columns
        assert "pred_rank" in df.columns


class TestVisualization:
    """Tests for plot generation."""

    def test_generate_plots_creates_files(self, evaluator, shap_data, tmp_path):
        """Test that comparison plots are generated."""
        true_shap, pred_shap = shap_data
        save_dir = str(tmp_path / "plots")
        evaluator.generate_comparison_plots(
            true_shap, pred_shap, save_dir, dataset_name="test"
        )

        expected_files = [
            "scatter_true_vs_pred_test.png",
            "per_feature_r2_test.png",
            "error_distribution_test.png",
            "feature_importance_test.png",
        ]
        for filename in expected_files:
            filepath = os.path.join(save_dir, filename)
            assert os.path.exists(filepath), f"Missing file: {filepath}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
