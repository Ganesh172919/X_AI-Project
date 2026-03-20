#!/usr/bin/env python3
"""
Reproduce main results from InstaSHAP paper
Runs experiments across multiple datasets and models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging

from src.data_loader import DatasetLoader
from src.black_box_model import BlackBoxModel
from src.shap_computation import compute_exact_shap
from src.gam_surrogate import SHAPSurrogate
from src.evaluation import SHAPEvaluator
from src.utils import set_random_seed, setup_logging

logger = logging.getLogger(__name__)


def load_config():
    """Load configuration."""
    with open("config/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def run_single_experiment(dataset_name, model_type, config):
    """
    Run single experiment for one dataset + model combination.

    Returns:
        Dictionary of results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {dataset_name} + {model_type}")
    logger.info(f"{'='*80}")

    # Load data
    dataset_config = config['datasets'][dataset_name]
    loader = DatasetLoader(
        dataset_name=dataset_name,
        test_size=dataset_config['test_size'],
        random_state=config['random_seed']
    )
    X_train, X_test, y_train, y_test = loader.load_data()
    feature_names = loader.get_feature_names()
    task_type = loader.task_type

    # Train black-box model
    model_config = config['black_box_models'][model_type][task_type]
    bb_model = BlackBoxModel(model_type=model_type, task=task_type, **model_config)
    bb_model.train(X_train, y_train)

    # Evaluate black-box model
    bb_metrics = bb_model.evaluate(X_test, y_test, verbose=False)

    # Compute SHAP values
    shap_config = config['shap_config']
    shap_model_type = 'tree' if model_type in ['random_forest', 'xgboost', 'lightgbm'] else 'kernel'

    # Training SHAP
    shap_train, _ = compute_exact_shap(
        bb_model.get_model(), X_train,
        model_type=shap_model_type,
        sample_size=shap_config['train_sample_size'],
        background_size=shap_config['background_size']
    )

    # Test SHAP (for timing)
    n_test_samples = min(X_test.shape[0], shap_config['test_sample_size'])
    X_test_subset = X_test[:n_test_samples]

    # Time exact SHAP computation
    start = time.time()
    shap_test, _ = compute_exact_shap(
        bb_model.get_model(), X_test_subset,
        model_type=shap_model_type,
        sample_size=None,
        background_size=shap_config['background_size']
    )
    exact_shap_time = time.time() - start

    # Train GAM surrogate
    gam_config = config['gam_config']
    surrogate = SHAPSurrogate(**gam_config)

    n_train_samples = min(shap_train.shape[0], shap_config['train_sample_size'])
    surrogate.train(X_train[:n_train_samples], shap_train[:n_train_samples], feature_names=feature_names, verbose=False)

    # Predict with GAM
    pred_shap, gam_pred_time = surrogate.predict_shap(X_test_subset, return_time=True)

    # Evaluate
    evaluator = SHAPEvaluator(feature_names)
    accuracy_metrics = evaluator.compute_accuracy_metrics(shap_test, pred_shap)
    speed_metrics = evaluator.compute_speed_metrics(exact_shap_time, gam_pred_time, n_test_samples)
    ranking_results = evaluator.compare_feature_rankings(shap_test, pred_shap, top_k=10)

    # Compile results
    results = {
        'dataset': dataset_name,
        'model_type': model_type,
        'task_type': task_type,
        'n_features': X_train.shape[1],
        'n_train_samples': X_train.shape[0],
        'n_test_samples': X_test.shape[0],
        **bb_metrics,
        **accuracy_metrics,
        **speed_metrics,
        'top_k_overlap_ratio': ranking_results['top_k_overlap_ratio'],
        'ranking_correlation': ranking_results['ranking_correlation']
    }

    logger.info(f"Results: R²={accuracy_metrics['r2']:.4f}, Speedup={speed_metrics['speedup_factor']:.2f}x")

    return results


def generate_summary_table(results_df):
    """Generate summary table (Table 1 in paper)."""
    logger.info("\n" + "="*80)
    logger.info("TABLE 1: SHAP Prediction Accuracy")
    logger.info("="*80)

    summary_cols = ['dataset', 'model_type', 'mse', 'mae', 'r2', 'pearson_correlation']
    summary = results_df[summary_cols].copy()

    print("\n", summary.to_string(index=False))

    # Save to CSV
    summary.to_csv("results/tables/table1_accuracy.csv", index=False)
    logger.info("\nTable saved to: results/tables/table1_accuracy.csv")


def generate_speed_comparison(results_df):
    """Generate speed comparison figure (Figure 1 in paper)."""
    logger.info("\n" + "="*80)
    logger.info("FIGURE 1: Speed Comparison")
    logger.info("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Computation time comparison
    ax = axes[0]
    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width/2, results_df['exact_time_seconds'], width, label='Exact SHAP', color='coral', alpha=0.8)
    ax.bar(x + width/2, results_df['surrogate_time_seconds'], width, label='GAM Surrogate', color='steelblue', alpha=0.8)

    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('SHAP Computation Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['dataset']}\n{row['model_type']}" for _, row in results_df.iterrows()], 
                       rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    # Plot 2: Speedup factors
    ax = axes[1]
    bars = ax.bar(x, results_df['speedup_factor'], color='green', alpha=0.7)

    # Color bars by speedup magnitude
    for i, bar in enumerate(bars):
        speedup = results_df.iloc[i]['speedup_factor']
        if speedup < 10:
            bar.set_color('orange')
        elif speedup < 50:
            bar.set_color('yellowgreen')
        else:
            bar.set_color('green')

    ax.set_ylabel('Speedup Factor', fontsize=12)
    ax.set_title('Speedup: Exact SHAP / GAM Surrogate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['dataset']}\n{row['model_type']}" for _, row in results_df.iterrows()], 
                       rotation=45, ha='right', fontsize=9)
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No speedup')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/figures/figure1_speed_comparison.png", dpi=300, bbox_inches='tight')
    logger.info("Figure saved to: results/figures/figure1_speed_comparison.png")
    plt.close()


def generate_accuracy_summary(results_df):
    """Generate accuracy summary figure."""
    logger.info("\n" + "="*80)
    logger.info("FIGURE 2: Accuracy Summary")
    logger.info("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(results_df))
    labels = [f"{row['dataset']}\n{row['model_type']}" for _, row in results_df.iterrows()]

    # MSE
    axes[0, 0].bar(x, results_df['mse'], color='coral', alpha=0.7)
    axes[0, 0].set_title('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # MAE
    axes[0, 1].bar(x, results_df['mae'], color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # R²
    axes[1, 0].bar(x, results_df['r2'], color='green', alpha=0.7)
    axes[1, 0].set_title('R² Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='0.9 threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Correlation
    axes[1, 1].bar(x, results_df['pearson_correlation'], color='purple', alpha=0.7)
    axes[1, 1].set_title('Pearson Correlation', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Correlation')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_ylim([0.8, 1.0])
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("results/figures/figure2_accuracy_summary.png", dpi=300, bbox_inches='tight')
    logger.info("Figure saved to: results/figures/figure2_accuracy_summary.png")
    plt.close()


def main():
    """Main reproduction script."""
    setup_logging(log_level="INFO")
    config = load_config()
    set_random_seed(config['random_seed'])

    logger.info("="*80)
    logger.info("InstaSHAP Results Reproduction")
    logger.info("="*80)

    # Ensure output directories exist
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # Define experiments
    experiments = [
        ('california_housing', 'random_forest'),
        ('california_housing', 'xgboost'),
        ('breast_cancer', 'random_forest'),
        ('breast_cancer', 'lightgbm'),
    ]

    # Run all experiments
    all_results = []
    for dataset, model in experiments:
        try:
            results = run_single_experiment(dataset, model, config)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error in {dataset} + {model}: {e}")
            continue

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Save complete results
    results_df.to_csv("results/tables/complete_results.csv", index=False)
    logger.info("\nComplete results saved to: results/tables/complete_results.csv")

    # Generate summary visualizations
    generate_summary_table(results_df)
    generate_speed_comparison(results_df)
    generate_accuracy_summary(results_df)

    logger.info("\n" + "="*80)
    logger.info("Results reproduction completed!")
    logger.info("="*80)
    logger.info("\nKey Results:")
    logger.info(f"  Mean R²: {results_df['r2'].mean():.4f}")
    logger.info(f"  Mean Speedup: {results_df['speedup_factor'].mean():.2f}x")
    logger.info(f"  Mean Correlation: {results_df['pearson_correlation'].mean():.4f}")


if __name__ == "__main__":
    main()
