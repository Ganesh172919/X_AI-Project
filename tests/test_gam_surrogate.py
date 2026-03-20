"""
Unit tests for gam_surrogate module.
"""

import pytest
import numpy as np
from src.gam_surrogate import SHAPSurrogate


@pytest.fixture
def synthetic_data():
    """Generate synthetic features and SHAP values for testing."""
    np.random.seed(42)
    n_samples, n_features = 200, 5
    X = np.random.randn(n_samples, n_features)
    # Synthetic SHAP values: linear combination with noise
    shap_values = (
        X * np.array([1.5, -0.8, 0.3, 2.0, -1.2])
        + np.random.randn(n_samples, n_features) * 0.1
    )
    return X[:150], shap_values[:150], X[150:], shap_values[150:]


@pytest.fixture
def trained_surrogate(synthetic_data):
    """Create and train a SHAPSurrogate."""
    X_train, shap_train, _, _ = synthetic_data
    surrogate = SHAPSurrogate(
        max_iter=100, max_bins=64, learning_rate=0.1, random_state=42
    )
    surrogate.train(X_train, shap_train, verbose=False)
    return surrogate


class TestSHAPSurrogateTraining:
    """Tests for GAM surrogate training."""

    def test_train_basic(self, synthetic_data):
        """Test basic training completes without error."""
        X_train, shap_train, _, _ = synthetic_data
        surrogate = SHAPSurrogate(max_iter=50, random_state=42)
        surrogate.train(X_train, shap_train, verbose=False)

        assert surrogate.is_fitted
        assert surrogate.n_features == 5
        assert len(surrogate.gam_models) == 5

    def test_train_with_feature_names(self, synthetic_data):
        """Test training with custom feature names."""
        X_train, shap_train, _, _ = synthetic_data
        names = [f"feat_{i}" for i in range(5)]
        surrogate = SHAPSurrogate(max_iter=50, random_state=42)
        surrogate.train(X_train, shap_train, feature_names=names, verbose=False)

        assert surrogate.feature_names == names

    def test_train_default_feature_names(self, synthetic_data):
        """Test that default feature names are generated."""
        X_train, shap_train, _, _ = synthetic_data
        surrogate = SHAPSurrogate(max_iter=50, random_state=42)
        surrogate.train(X_train, shap_train, verbose=False)

        assert all(name.startswith("Feature_") for name in surrogate.feature_names)

    def test_train_shape_mismatch_raises(self):
        """Test that mismatched shapes raise ValueError."""
        X = np.random.randn(100, 5)
        shap = np.random.randn(100, 3)  # Wrong number of features
        surrogate = SHAPSurrogate(max_iter=10)

        with pytest.raises(ValueError, match="same number of features"):
            surrogate.train(X, shap, verbose=False)

    def test_train_sample_mismatch_raises(self):
        """Test that mismatched sample counts raise ValueError."""
        X = np.random.randn(100, 5)
        shap = np.random.randn(50, 5)  # Wrong number of samples
        surrogate = SHAPSurrogate(max_iter=10)

        with pytest.raises(ValueError, match="same number of samples"):
            surrogate.train(X, shap, verbose=False)

    def test_train_returns_self(self, synthetic_data):
        """Test that train() returns self for chaining."""
        X_train, shap_train, _, _ = synthetic_data
        surrogate = SHAPSurrogate(max_iter=50, random_state=42)
        result = surrogate.train(X_train, shap_train, verbose=False)

        assert result is surrogate


class TestSHAPSurrogatePrediction:
    """Tests for SHAP value prediction."""

    def test_predict_shap_shape(self, trained_surrogate, synthetic_data):
        """Test that predicted SHAP values have correct shape."""
        _, _, X_test, _ = synthetic_data
        pred_shap = trained_surrogate.predict_shap(X_test)

        assert pred_shap.shape == X_test.shape

    def test_predict_shap_with_time(self, trained_surrogate, synthetic_data):
        """Test predict_shap with return_time=True."""
        _, _, X_test, _ = synthetic_data
        pred_shap, pred_time = trained_surrogate.predict_shap(X_test, return_time=True)

        assert pred_shap.shape == X_test.shape
        assert isinstance(pred_time, float)
        assert pred_time >= 0

    def test_predict_before_train_raises(self, synthetic_data):
        """Test that predicting before training raises an error."""
        _, _, X_test, _ = synthetic_data
        surrogate = SHAPSurrogate(max_iter=50)

        with pytest.raises(ValueError, match="not fitted"):
            surrogate.predict_shap(X_test)

    def test_predict_feature_mismatch_raises(self, trained_surrogate):
        """Test that wrong number of features raises an error."""
        X_wrong = np.random.randn(10, 3)  # Should be 5 features

        with pytest.raises(ValueError, match="Expected 5 features"):
            trained_surrogate.predict_shap(X_wrong)


class TestSHAPSurrogateEvaluation:
    """Tests for surrogate evaluation."""

    def test_evaluate_returns_metrics(self, trained_surrogate, synthetic_data):
        """Test that evaluate returns expected metric keys."""
        _, _, X_test, shap_test = synthetic_data
        metrics = trained_surrogate.evaluate(X_test, shap_test, verbose=False)

        expected_keys = [
            "mse",
            "mae",
            "rmse",
            "r2",
            "pearson_correlation",
            "spearman_correlation",
            "prediction_time",
            "per_feature_mse",
            "per_feature_r2",
            "mean_per_feature_r2",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_evaluate_metrics_range(self, trained_surrogate, synthetic_data):
        """Test that evaluation metrics are in reasonable ranges."""
        _, _, X_test, shap_test = synthetic_data
        metrics = trained_surrogate.evaluate(X_test, shap_test, verbose=False)

        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert -1 <= metrics["r2"] <= 1
        assert -1 <= metrics["pearson_correlation"] <= 1

    def test_per_feature_metrics_length(self, trained_surrogate, synthetic_data):
        """Test per-feature metrics have correct length."""
        _, _, X_test, shap_test = synthetic_data
        metrics = trained_surrogate.evaluate(X_test, shap_test, verbose=False)

        assert len(metrics["per_feature_mse"]) == 5
        assert len(metrics["per_feature_r2"]) == 5


class TestSHAPSurrogateSerialization:
    """Tests for model save/load."""

    def test_save_and_load(self, trained_surrogate, tmp_path):
        """Test saving and loading a trained surrogate."""
        filepath = str(tmp_path / "surrogate.pkl")
        trained_surrogate.save_model(filepath)

        new_surrogate = SHAPSurrogate(max_iter=50)
        new_surrogate.load_model(filepath)

        assert new_surrogate.is_fitted
        assert new_surrogate.n_features == trained_surrogate.n_features
        assert new_surrogate.feature_names == trained_surrogate.feature_names

    def test_save_before_train_raises(self, tmp_path):
        """Test that saving before training raises an error."""
        surrogate = SHAPSurrogate(max_iter=50)
        filepath = str(tmp_path / "surrogate.pkl")

        with pytest.raises(ValueError, match="not fitted"):
            surrogate.save_model(filepath)

    def test_predictions_match_after_load(
        self, trained_surrogate, synthetic_data, tmp_path
    ):
        """Test that predictions are identical after save/load."""
        _, _, X_test, _ = synthetic_data

        filepath = str(tmp_path / "surrogate.pkl")
        trained_surrogate.save_model(filepath)

        new_surrogate = SHAPSurrogate(max_iter=50)
        new_surrogate.load_model(filepath)

        preds_original = trained_surrogate.predict_shap(X_test)
        preds_loaded = new_surrogate.predict_shap(X_test)

        np.testing.assert_array_almost_equal(preds_original, preds_loaded, decimal=6)


class TestSHAPSurrogateFeatureImportance:
    """Tests for feature importance extraction."""

    def test_get_feature_importance(self, trained_surrogate):
        """Test retrieving feature importance from a trained GAM."""
        importance = trained_surrogate.get_feature_importance(0)
        assert importance is not None
        assert len(importance) > 0

    def test_get_feature_importance_before_train_raises(self):
        """Test that getting importance before training raises an error."""
        surrogate = SHAPSurrogate(max_iter=50)
        with pytest.raises(ValueError, match="not fitted"):
            surrogate.get_feature_importance(0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
