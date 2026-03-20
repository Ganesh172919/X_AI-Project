"""
InstaSHAP Replication Project
Setup script for installation
"""

from setuptools import setup, find_packages

setup(
    name="instashap-replication",
    version="1.0.0",
    description="Replication of InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly (ICLR 2025)",
    author="Ravi Prakash",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "shap>=0.42.1",
        "interpret>=0.4.3",
        "xgboost>=1.7.6",
        "lightgbm>=4.0.0",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
