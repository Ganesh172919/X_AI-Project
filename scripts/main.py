#!/usr/bin/env python3
"""
Main pipeline for InstaSHAP replication
Orchestrates the complete workflow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.data_loader import DatasetLoader
from src.black_box_model import BlackBoxModel
from src.shap_computation import compute_exact_shap
from src.gam_surrogate import SHAPSurrogate
from src.evaluation import SHAPEvaluator
from src.utils import set_random_seed, setup_logging, ensure_dir

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml"):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    """Main pipeline execution."""

    # Setup
    setup_logging(log_level=args.log_level)
    config = load_config(args.config)
    set_random_seed(config['random_seed'])

    logger.info("="*60)
    logger.info("InstaSHAP Replication Pipeline")
    logger.info("="*60)

    # Create output directories
    ensure_dir("results/tables")
    ensure_dir("results/figures")
    ensure_dir("results/models")

    # 1. Load Dataset
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 1: Loading {args.dataset} dataset")
    logger.info(f"{'='*60}")

    dataset_config = config['datasets'][args.dataset]
    loader = DatasetLoader(
        dataset_name=args.dataset,
        test_size=dataset_config['test_size'],
        random_state=config['random_seed']
    )

    X_train, X_test, y_train, y_test = loader.load_data()
    feature_names = loader.get_feature_names()
    task_type = loader.task_type

    # Print dataset statistics
    stats = loader.describe_data()
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        if key not in ['feature_names', 'class_distribution_train', 'class_distribution_test']:
            logger.info(f"  {key}: {value}")

    # 2. Train Black-Box Model
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 2: Training {args.model_type} model")
    logger.info(f"{'='*60}")

    model_config = config['black_box_models'][args.model_type][task_type]
    bb_model = BlackBoxModel(
        model_type=args.model_type,
        task=task_type,
        **model_config
    )

    bb_model.train(X_train, y_train)

    # Evaluate black-box model
    logger.info("\nEvaluating black-box model...")
    metrics = bb_model.evaluate(X_test, y_test, verbose=True)

    # Save black-box model
    model_path = f"results/models/{args.dataset}_{args.model_type}_model.pkl"
    bb_model.save_model(model_path)

    # 3. Compute Exact SHAP Values
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 3: Computing exact SHAP values")
    logger.info(f"{'='*60}")

    shap_config = config['shap_config']

    # Determine model type for SHAP
    if args.model_type in ['random_forest', 'xgboost', 'lightgbm']:
        shap_model_type = 'tree'
    else:
        shap_model_type = 'kernel'

    # Compute SHAP for training set (for training GAM)
    logger.info("\nComputing SHAP values for training set...")
    cache_path_train = f"results/models/{args.dataset}_{args.model_type}_shap_train.pkl"
    shap_train, shap_train_time = compute_exact_shap(
        model=bb_model.get_model(),
        X_data=X_train,
        model_type=shap_model_type,
        sample_size=shap_config['train_sample_size'],
        background_size=shap_config['background_size'],
        cache_path=cache_path_train if config['computation']['cache_shap'] else None
    )

    logger.info(f"Training SHAP shape: {shap_train.shape}")
    logger.info(f"Training SHAP computation time: {shap_train_time:.2f}s")

    # Compute SHAP for test set (for evaluation)
    logger.info("\nComputing SHAP values for test set...")
    cache_path_test = f"results/models/{args.dataset}_{args.model_type}_shap_test.pkl"
    shap_test, shap_test_time = compute_exact_shap(
        model=bb_model.get_model(),
        X_data=X_test,
        model_type=shap_model_type,
        sample_size=shap_config['test_sample_size'],
        background_size=shap_config['background_size'],
        cache_path=cache_path_test if config['computation']['cache_shap'] else None
    )

    logger.info(f"Test SHAP shape: {shap_test.shape}")
    logger.info(f"Test SHAP computation time: {shap_test_time:.2f}s")

    # 4. Train GAM Surrogate
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 4: Training GAM surrogate")
    logger.info(f"{'='*60}")

    gam_config = config['gam_config']
    surrogate = SHAPSurrogate(**gam_config)

    # Use subset of training data for GAM training
    n_train_samples = min(shap_train.shape[0], shap_config['train_sample_size'])
    X_train_gam = X_train[:n_train_samples]
    shap_train_gam = shap_train[:n_train_samples]

    surrogate.train(X_train_gam, shap_train_gam, feature_names=feature_names)

    # Save GAM surrogate
    surrogate_path = f"results/models/{args.dataset}_{args.model_type}_gam_surrogate.pkl"
    surrogate.save_model(surrogate_path)

    # 5. Predict SHAP with GAM and Evaluate
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 5: Predicting SHAP values with GAM")
    logger.info(f"{'='*60}")

    # Use subset of test data
    n_test_samples = min(shap_test.shape[0], shap_config['test_sample_size'])
    X_test_eval = X_test[:n_test_samples]
    shap_test_eval = shap_test[:n_test_samples]

    pred_shap, pred_time = surrogate.predict_shap(X_test_eval, return_time=True)

    logger.info(f"\nPrediction completed in {pred_time:.4f}s")
    logger.info(f"Speedup: {shap_test_time / pred_time:.2f}x faster")

    # 6. Comprehensive Evaluation
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 6: Evaluating results")
    logger.info(f"{'='*60}")

    evaluator = SHAPEvaluator(feature_names=feature_names)

    # Accuracy metrics
    accuracy_metrics = evaluator.compute_accuracy_metrics(shap_test_eval, pred_shap)
    logger.info("\nAccuracy Metrics:")
    for key, value in accuracy_metrics.items():
        if 'pvalue' not in key:
            logger.info(f"  {key}: {value:.6f}")

    # Speed metrics
    speed_metrics = evaluator.compute_speed_metrics(
        exact_time=shap_test_time,
        surrogate_time=pred_time,
        n_samples=n_test_samples
    )
    logger.info("\nSpeed Metrics:")
    for key, value in speed_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Feature ranking comparison
    ranking_results = evaluator.compare_feature_rankings(
        shap_test_eval, pred_shap, 
        top_k=config['evaluation']['top_k_features']
    )
    logger.info("\nFeature Ranking Metrics:")
    logger.info(f"  Top-{ranking_results['top_k']} overlap: {ranking_results['top_k_overlap']}/{ranking_results['top_k']}")
    logger.info(f"  Top-{ranking_results['top_k']} overlap ratio: {ranking_results['top_k_overlap_ratio']:.4f}")
    logger.info(f"  Ranking correlation: {ranking_results['ranking_correlation']:.4f}")

    # Per-feature metrics
    per_feature_df = evaluator.compute_per_feature_metrics(shap_test_eval, pred_shap)

    # 7. Generate Visualizations
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 7: Generating visualizations")
    logger.info(f"{'='*60}")

    fig_dir = f"results/figures/{args.dataset}_{args.model_type}"
    evaluator.generate_comparison_plots(
        shap_test_eval, pred_shap, 
        save_dir=fig_dir,
        dataset_name=f"{args.dataset}_{args.model_type}"
    )

    # 8. Save Results
    logger.info(f"\n{'='*60}")
    logger.info(f"Step 8: Saving results")
    logger.info(f"{'='*60}")

    # Save metrics to CSV
    results_df = pd.DataFrame([{
        'dataset': args.dataset,
        'model_type': args.model_type,
        **accuracy_metrics,
        **speed_metrics
    }])

    results_path = f"results/tables/{args.dataset}_{args.model_type}_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    # Save per-feature metrics
    per_feature_path = f"results/tables/{args.dataset}_{args.model_type}_per_feature.csv"
    per_feature_df.to_csv(per_feature_path, index=False)
    logger.info(f"Per-feature metrics saved to {per_feature_path}")

    # Save ranking comparison
    ranking_path = f"results/tables/{args.dataset}_{args.model_type}_rankings.csv"
    ranking_results['ranking_df'].to_csv(ranking_path, index=False)
    logger.info(f"Feature rankings saved to {ranking_path}")

    logger.info(f"\n{'='*60}")
    logger.info("Pipeline completed successfully!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstaSHAP Replication Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="california_housing",
        choices=["adult", "california_housing", "breast_cancer"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Black-box model type"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    main(args)
