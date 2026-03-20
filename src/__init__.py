"""
InstaSHAP Replication Package
=============================

Replication of "InstaSHAP: Interpretable Additive Models Explain Shapley
Values Instantly" (ICLR 2025).

This package implements the core InstaSHAP methodology of using Generalized
Additive Models (GAMs) as surrogate models to predict SHAP values instantly,
achieving orders-of-magnitude speedup over exact SHAP computation.

Modules
-------
data_loader
    Dataset loading and preprocessing for Adult, California Housing, and
    Breast Cancer datasets.
black_box_model
    Training and evaluation of black-box models (Random Forest, XGBoost,
    LightGBM).
shap_computation
    Exact SHAP value computation with caching support.
gam_surrogate
    GAM surrogate training and instant SHAP prediction (core methodology).
evaluation
    Comprehensive evaluation metrics and visualization utilities.
utils
    Configuration loading, logging, and serialization helpers.

Example
-------
>>> from src import DatasetLoader, BlackBoxModel, SHAPSurrogate
>>> loader = DatasetLoader("california_housing")
>>> X_train, X_test, y_train, y_test = loader.load_data()
>>> model = BlackBoxModel("random_forest", task="regression")
>>> model.train(X_train, y_train)
"""

__version__ = "1.0.0"
__author__ = "Ravi Prakash"

from .data_loader import DatasetLoader
from .black_box_model import BlackBoxModel
from .shap_computation import SHAPComputer
from .gam_surrogate import SHAPSurrogate
from .evaluation import SHAPEvaluator
from .utils import load_config, set_random_seed, setup_logging

__all__ = [
    "DatasetLoader",
    "BlackBoxModel",
    "SHAPComputer",
    "SHAPSurrogate",
    "SHAPEvaluator",
    "load_config",
    "set_random_seed",
    "setup_logging",
]
