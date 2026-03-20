"""
Unit tests for data_loader module
"""

import pytest
import numpy as np
from src.data_loader import DatasetLoader


def test_california_housing_loading():
    """Test loading California Housing dataset."""
    loader = DatasetLoader('california_housing', test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = loader.load_data()

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 8
    assert loader.task_type == 'regression'


def test_breast_cancer_loading():
    """Test loading Breast Cancer dataset."""
    loader = DatasetLoader('breast_cancer', test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = loader.load_data()

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert X_train.shape[1] == 30
    assert loader.task_type == 'classification'


def test_feature_names():
    """Test feature name retrieval."""
    loader = DatasetLoader('california_housing', random_state=42)
    loader.load_data()
    feature_names = loader.get_feature_names()

    assert len(feature_names) == 8
    assert all(isinstance(name, str) for name in feature_names)


def test_describe_data():
    """Test dataset statistics."""
    loader = DatasetLoader('california_housing', random_state=42)
    loader.load_data()
    stats = loader.describe_data()

    assert 'dataset_name' in stats
    assert 'n_features' in stats
    assert stats['n_features'] == 8


def test_train_test_split():
    """Test train/test split functionality."""
    loader = DatasetLoader('california_housing', test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = loader.load_data()

    total_samples = X_train.shape[0] + X_test.shape[0]
    test_ratio = X_test.shape[0] / total_samples

    assert abs(test_ratio - 0.3) < 0.01  # Allow small deviation


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
