"""
Unit tests for black_box_model module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from src.black_box_model import BlackBoxModel


@pytest.fixture
def classification_data():
    """Generate synthetic classification data."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_classes=2, random_state=42
    )
    return X[:150], y[:150], X[150:], y[150:]


@pytest.fixture
def regression_data():
    """Generate synthetic regression data."""
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return X[:150], y[:150], X[150:], y[150:]


class TestBlackBoxModelClassification:
    """Tests for classification tasks."""

    def test_random_forest_classification(self, classification_data):
        """Test Random Forest classifier training and prediction."""
        X_train, y_train, X_test, y_test = classification_data
        model = BlackBoxModel(
            model_type="random_forest", task="classification", n_estimators=10
        )
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert set(np.unique(preds)).issubset({0, 1})

    def test_xgboost_classification(self, classification_data):
        """Test XGBoost classifier training and prediction."""
        X_train, y_train, X_test, y_test = classification_data
        model = BlackBoxModel(
            model_type="xgboost", task="classification", n_estimators=10
        )
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_lightgbm_classification(self, classification_data):
        """Test LightGBM classifier training and prediction."""
        X_train, y_train, X_test, y_test = classification_data
        model = BlackBoxModel(
            model_type="lightgbm", task="classification", n_estimators=10, verbosity=-1
        )
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_predict_proba(self, classification_data):
        """Test probability predictions for classification."""
        X_train, y_train, X_test, y_test = classification_data
        model = BlackBoxModel(
            model_type="random_forest", task="classification", n_estimators=10
        )
        model.train(X_train, y_train)

        proba = model.predict_proba(X_test)
        assert proba.shape == (len(y_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_regression_raises(self, regression_data):
        """Test that predict_proba raises for regression tasks."""
        X_train, y_train, X_test, y_test = regression_data
        model = BlackBoxModel(
            model_type="random_forest", task="regression", n_estimators=10
        )
        model.train(X_train, y_train)

        with pytest.raises(ValueError, match="classification"):
            model.predict_proba(X_test)


class TestBlackBoxModelRegression:
    """Tests for regression tasks."""

    def test_random_forest_regression(self, regression_data):
        """Test Random Forest regressor training and prediction."""
        X_train, y_train, X_test, y_test = regression_data
        model = BlackBoxModel(
            model_type="random_forest", task="regression", n_estimators=10
        )
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert preds.dtype in [np.float32, np.float64]

    def test_xgboost_regression(self, regression_data):
        """Test XGBoost regressor training and prediction."""
        X_train, y_train, X_test, y_test = regression_data
        model = BlackBoxModel(model_type="xgboost", task="regression", n_estimators=10)
        model.train(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape == y_test.shape


class TestBlackBoxModelEvaluation:
    """Tests for model evaluation."""

    def test_evaluate_classification(self, classification_data):
        """Test evaluation metrics for classification."""
        X_train, y_train, X_test, y_test = classification_data
        model = BlackBoxModel(
            model_type="random_forest", task="classification", n_estimators=10
        )
        model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test, verbose=False)
        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_regression(self, regression_data):
        """Test evaluation metrics for regression."""
        X_train, y_train, X_test, y_test = regression_data
        model = BlackBoxModel(
            model_type="random_forest", task="regression", n_estimators=10
        )
        model.train(X_train, y_train)

        metrics = model.evaluate(X_test, y_test, verbose=False)
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mse"] >= 0


class TestBlackBoxModelSerialization:
    """Tests for model save/load."""

    def test_save_and_load(self, classification_data, tmp_path):
        """Test saving and loading a model."""
        X_train, y_train, X_test, y_test = classification_data
        model = BlackBoxModel(
            model_type="random_forest", task="classification", n_estimators=10
        )
        model.train(X_train, y_train)

        filepath = str(tmp_path / "model.pkl")
        model.save_model(filepath)

        new_model = BlackBoxModel(model_type="random_forest", task="classification")
        new_model.load_model(filepath)

        preds_original = model.predict(X_test)
        preds_loaded = new_model.predict(X_test)
        np.testing.assert_array_equal(preds_original, preds_loaded)


class TestBlackBoxModelEdgeCases:
    """Tests for edge cases and errors."""

    def test_predict_before_train_raises(self, classification_data):
        """Test that predicting before training raises an error."""
        _, _, X_test, _ = classification_data
        model = BlackBoxModel(model_type="random_forest", task="classification")

        with pytest.raises(ValueError, match="not trained"):
            model.predict(X_test)

    def test_invalid_model_type_raises(self):
        """Test that an unknown model type raises an error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            BlackBoxModel(model_type="invalid_model", task="classification")

    def test_get_model(self, classification_data):
        """Test retrieving the underlying model object."""
        X_train, y_train, _, _ = classification_data
        model = BlackBoxModel(
            model_type="random_forest", task="classification", n_estimators=10
        )
        model.train(X_train, y_train)

        underlying = model.get_model()
        assert underlying is not None
        assert hasattr(underlying, "predict")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
