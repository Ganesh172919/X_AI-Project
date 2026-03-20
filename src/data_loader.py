"""
Data loading and preprocessing module for InstaSHAP replication.

Provides a unified interface for loading and preprocessing tabular datasets
used in the InstaSHAP experiments. Handles categorical encoding, missing
value imputation, train/test splitting, and feature scaling.

Supported Datasets
------------------
adult
    Binary classification (~48K samples, 14 features).
    Predicts whether income exceeds $50K/yr based on census data.
    Loaded from OpenML with fallback to synthetic data.
california_housing
    Regression (20,640 samples, 8 features).
    Predicts median house value in California districts.
breast_cancer
    Binary classification (569 samples, 30 features).
    Classifies tumors as malignant or benign.

Example
-------
>>> from src.data_loader import DatasetLoader
>>> loader = DatasetLoader("california_housing", test_size=0.2)
>>> X_train, X_test, y_train, y_test = loader.load_data()
>>> print(loader.describe_data())
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_california_housing, load_breast_cancer, fetch_openml
import logging

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Unified interface for loading and preprocessing tabular datasets.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset to load.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.
    X_train, X_test : np.ndarray
        Scaled training and test feature matrices.
    y_train, y_test : np.ndarray
        Training and test target arrays.
    feature_names : list of str
        Names of features in the dataset.
    task_type : str
        Either ``"regression"`` or ``"classification"``.
    scaler : StandardScaler
        Fitted feature scaler.

    Example
    -------
    >>> loader = DatasetLoader("breast_cancer", test_size=0.2, random_state=42)
    >>> X_train, X_test, y_train, y_test = loader.load_data()
    >>> loader.get_feature_names()[:3]
    ['mean radius', 'mean texture', 'mean perimeter']
    """

    def __init__(
        self, dataset_name: str, test_size: float = 0.2, random_state: int = 42
    ):
        """
        Initialize DatasetLoader.

        Args:
            dataset_name: Name of dataset to load
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.test_size = test_size
        self.random_state = random_state

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.task_type = None
        self.scaler = StandardScaler()

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess dataset.

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading dataset: {self.dataset_name}")

        if self.dataset_name == "adult":
            self._load_adult()
        elif self.dataset_name == "california_housing":
            self._load_california_housing()
        elif self.dataset_name == "breast_cancer":
            self._load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        logger.info(
            f"Dataset loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples"
        )
        logger.info(f"Features: {self.X_train.shape[1]}, Task: {self.task_type}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def _load_adult(self) -> None:
        """Load and preprocess Adult Income dataset."""
        try:
            # Load from OpenML
            data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
            X = data.data
            y = data.target

            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=["category", "object"]).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

            # Convert to numeric
            X = X.apply(pd.to_numeric, errors="coerce")
            X = X.fillna(X.median())

            # Encode target
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

            self.feature_names = list(X.columns)
            self.task_type = "classification"

        except Exception as e:
            logger.warning(f"Could not load from OpenML: {e}. Using synthetic data.")
            # Create synthetic dataset
            from sklearn.datasets import make_classification

            X, y = make_classification(
                n_samples=5000,
                n_features=14,
                n_informative=10,
                n_redundant=2,
                random_state=self.random_state,
            )
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(14)])
            self.feature_names = list(X.columns)
            self.task_type = "classification"

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def _load_california_housing(self) -> None:
        """Load and preprocess California Housing dataset."""
        data = fetch_california_housing(as_frame=True)
        X = data.data
        y = data.target

        self.feature_names = list(X.columns)
        self.task_type = "regression"

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

    def _load_breast_cancer(self) -> None:
        """Load and preprocess Breast Cancer dataset."""
        data = load_breast_cancer(as_frame=True)
        X = data.data
        y = data.target

        self.feature_names = list(X.columns)
        self.task_type = "classification"

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train.values
        self.y_test = y_test.values

    def get_feature_names(self) -> List[str]:
        """
        Get feature names.

        Returns:
            List of feature names
        """
        return self.feature_names

    def describe_data(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "n_train_samples": self.X_train.shape[0],
            "n_test_samples": self.X_test.shape[0],
            "n_features": self.X_train.shape[1],
            "feature_names": self.feature_names,
        }

        if self.task_type == "classification":
            stats["n_classes"] = len(np.unique(self.y_train))
            stats["class_distribution_train"] = dict(
                zip(*np.unique(self.y_train, return_counts=True))
            )
            stats["class_distribution_test"] = dict(
                zip(*np.unique(self.y_test, return_counts=True))
            )
        else:
            stats["target_mean_train"] = float(np.mean(self.y_train))
            stats["target_std_train"] = float(np.std(self.y_train))
            stats["target_mean_test"] = float(np.mean(self.y_test))
            stats["target_std_test"] = float(np.std(self.y_test))

        return stats

    def get_train_test_split(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test split.

        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.X_train, self.X_test, self.y_train, self.y_test
