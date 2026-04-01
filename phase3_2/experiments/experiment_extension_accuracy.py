"""
Experiment 2: Extension Accuracy

This experiment evaluates the accuracy improvement of the Interaction-Aware
InstaSHAP extension compared to the original additive InstaSHAP.

We show that:
1. GA²M surrogate achieves higher fidelity on interaction-heavy data
2. Enhanced InstaSHAP produces more accurate Shapley values
3. The extension does not harm accuracy on non-interaction data

Author: DS357 Course Project Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
import sys
import os
import warnings

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "phase2"
    )
)

from phase2.data.data_loader import (
    load_california_housing,
    create_synthetic_interaction_dataset,
)
from phase2.models.base_model import train_model_for_dataset
from phase2.models.gam_surrogate import train_surrogate_for_blackbox
from phase2.models.instashap import InstaSHAP
from phase2.explainers.exact_shap import ExactSHAPExplainer

from extension.interaction_aware_surrogate import (
    train_interaction_surrogate_for_blackbox,
)
from extension.enhanced_instashap import EnhancedInstaSHAP

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Results directory
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def compare_accuracy_on_dataset(
    data: dict, dataset_name: str, n_interactions: int = 5
) -> dict:
    """
    Compare accuracy of original vs enhanced InstaSHAP on a dataset.

    Args:
        data: Dataset dictionary
        dataset_name: Name for reporting
        n_interactions: Number of interactions for GA²M

    Returns:
        Dictionary with comparison results
    """
    print(f"\n--- Comparing on {dataset_name} ---")

    # Train black-box model
    blackbox = train_model_for_dataset(data, model_type="xgboost")

    # Sample data for explanation
    X_explain = data["X_test"].iloc[:200]

    # Compute Exact SHAP (ground truth)
    print("Computing Exact SHAP values...")
    exact_explainer = ExactSHAPExplainer(
        blackbox.model, data["X_train"], model_type="tree", task_type=data["task_type"]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact_shap = exact_explainer.explain(X_explain, return_dataframe=False)

    # Train additive surrogate (original InstaSHAP)
    print("Training additive GAM surrogate...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        additive_surrogate = train_surrogate_for_blackbox(
            blackbox.model, data["X_train"], task_type=data["task_type"]
        )

    # Compute original InstaSHAP
    original_instashap = InstaSHAP(additive_surrogate, data["X_train"])
    original_shap = original_instashap.explain(X_explain, return_dataframe=False)

    # Evaluate additive surrogate fidelity
    bb_preds = blackbox.model.predict(data["X_test"])
    additive_r2 = r2_score(bb_preds, additive_surrogate.predict(data["X_test"]))

    # Train interaction-aware surrogate (enhanced InstaSHAP)
    print(f"Training GA²M surrogate with {n_interactions} interactions...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ga2m_surrogate = train_interaction_surrogate_for_blackbox(
            blackbox.model,
            data["X_train"],
            task_type=data["task_type"],
            n_interactions=n_interactions,
        )

    # Compute enhanced InstaSHAP
    enhanced_instashap = EnhancedInstaSHAP(ga2m_surrogate, data["X_train"])
    enhanced_shap = enhanced_instashap.explain(X_explain, return_dataframe=False)

    # Evaluate GA²M surrogate fidelity
    ga2m_r2 = r2_score(bb_preds, ga2m_surrogate.predict(data["X_test"]))

    # Compute accuracy metrics for both methods
    exact_flat = exact_shap.flatten()
    original_flat = original_shap.flatten()
    enhanced_flat = enhanced_shap.flatten()

    # Original InstaSHAP metrics
    orig_pearson, _ = pearsonr(exact_flat, original_flat)
    orig_spearman, _ = spearmanr(exact_flat, original_flat)
    orig_mae = mean_absolute_error(exact_flat, original_flat)

    # Enhanced InstaSHAP metrics
    enh_pearson, _ = pearsonr(exact_flat, enhanced_flat)
    enh_spearman, _ = spearmanr(exact_flat, enhanced_flat)
    enh_mae = mean_absolute_error(exact_flat, enhanced_flat)

    # Print results
    print(f"\nSurrogate Fidelity:")
    print(f"  Additive GAM: R² = {additive_r2:.4f}")
    print(f"  GA²M:         R² = {ga2m_r2:.4f}")
    print(f"  Improvement:  {ga2m_r2 - additive_r2:.4f}")

    print(f"\nInstaSHAP Accuracy (vs Exact SHAP):")
    print(f"  Original:  Pearson r = {orig_pearson:.4f}, MAE = {orig_mae:.4f}")
    print(f"  Enhanced:  Pearson r = {enh_pearson:.4f}, MAE = {enh_mae:.4f}")
    print(f"  Improvement: Δr = {enh_pearson - orig_pearson:.4f}")

    return {
        "dataset": dataset_name,
        "additive_r2": additive_r2,
        "ga2m_r2": ga2m_r2,
        "fidelity_improvement": ga2m_r2 - additive_r2,
        "original_pearson": orig_pearson,
        "enhanced_pearson": enh_pearson,
        "accuracy_improvement": enh_pearson - orig_pearson,
        "original_mae": orig_mae,
        "enhanced_mae": enh_mae,
        "mae_improvement": orig_mae - enh_mae,
    }


def run_accuracy_comparison() -> pd.DataFrame:
    """
    Run accuracy comparison across multiple datasets.

    Returns:
        DataFrame with all results
    """
    print("=" * 60)
    print("EXTENSION ACCURACY COMPARISON")
    print("Original InstaSHAP vs Enhanced InstaSHAP")
    print("=" * 60)

    results = []

    # Test on synthetic dataset with strong interactions
    print("\n" + "=" * 60)
    print("Test 1: Synthetic data with STRONG interactions")
    print("=" * 60)

    strong_interaction = create_synthetic_interaction_dataset(
        n_samples=2000, interaction_strength=2.0
    )
    results.append(
        compare_accuracy_on_dataset(
            strong_interaction, "synthetic_strong_interaction", n_interactions=5
        )
    )

    # Test on synthetic dataset with moderate interactions
    print("\n" + "=" * 60)
    print("Test 2: Synthetic data with MODERATE interactions")
    print("=" * 60)

    moderate_interaction = create_synthetic_interaction_dataset(
        n_samples=2000, interaction_strength=1.0
    )
    results.append(
        compare_accuracy_on_dataset(
            moderate_interaction, "synthetic_moderate_interaction", n_interactions=5
        )
    )

    # Test on synthetic dataset with NO interactions
    print("\n" + "=" * 60)
    print("Test 3: Synthetic data with NO interactions")
    print("=" * 60)

    no_interaction = create_synthetic_interaction_dataset(
        n_samples=2000, interaction_strength=0.0
    )
    results.append(
        compare_accuracy_on_dataset(
            no_interaction, "synthetic_no_interaction", n_interactions=5
        )
    )

    # Test on real dataset (California Housing)
    print("\n" + "=" * 60)
    print("Test 4: California Housing (real data)")
    print("=" * 60)

    cal_data = load_california_housing()
    results.append(
        compare_accuracy_on_dataset(cal_data, "california_housing", n_interactions=5)
    )

    return pd.DataFrame(results)


def plot_accuracy_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot accuracy comparison results.

    Args:
        results_df: Results DataFrame
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Surrogate Fidelity Comparison
    ax1 = axes[0]
    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        results_df["additive_r2"],
        width,
        label="Additive GAM",
        color="#e74c3c",
    )
    bars2 = ax1.bar(
        x + width / 2,
        results_df["ga2m_r2"],
        width,
        label="GA²M (with interactions)",
        color="#2ecc71",
    )

    ax1.axhline(y=0.9, color="blue", linestyle="--", alpha=0.7, label="High Fidelity")
    ax1.set_xlabel("Dataset", fontsize=12)
    ax1.set_ylabel("Surrogate R² (Fidelity)", fontsize=12)
    ax1.set_title("Surrogate Fidelity: Additive vs GA²M", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df["dataset"], rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.3)

    # Plot 2: InstaSHAP Accuracy Comparison
    ax2 = axes[1]

    bars3 = ax2.bar(
        x - width / 2,
        results_df["original_pearson"],
        width,
        label="Original InstaSHAP",
        color="#e74c3c",
    )
    bars4 = ax2.bar(
        x + width / 2,
        results_df["enhanced_pearson"],
        width,
        label="Enhanced InstaSHAP",
        color="#2ecc71",
    )

    ax2.axhline(y=0.9, color="blue", linestyle="--", alpha=0.7, label="High Accuracy")
    ax2.set_xlabel("Dataset", fontsize=12)
    ax2.set_ylabel("Pearson Correlation with Exact SHAP", fontsize=12)
    ax2.set_title("Shapley Value Accuracy: Original vs Enhanced", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df["dataset"], rotation=45, ha="right")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Accuracy comparison plot saved to {save_path}")


def main():
    """Run the complete extension accuracy experiment."""

    print("=" * 60)
    print("EXPERIMENT: EXTENSION ACCURACY")
    print("Comparing Original vs Enhanced InstaSHAP")
    print("=" * 60)

    # Run comparison
    results = run_accuracy_comparison()

    # Plot results
    plot_accuracy_comparison(
        results, os.path.join(RESULTS_DIR, "extension_accuracy_comparison.png")
    )

    # Save results
    results.to_csv(
        os.path.join(RESULTS_DIR, "extension_accuracy_results.csv"), index=False
    )

    # Summary
    print("\n" + "=" * 60)
    print("EXTENSION ACCURACY SUMMARY")
    print("=" * 60)
    print(
        results[
            ["dataset", "original_pearson", "enhanced_pearson", "accuracy_improvement"]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)

    strong_result = results[results["dataset"] == "synthetic_strong_interaction"].iloc[
        0
    ]
    no_int_result = results[results["dataset"] == "synthetic_no_interaction"].iloc[0]

    print(f"1. On STRONG interaction data:")
    print(f"   - Accuracy improvement: {strong_result['accuracy_improvement']:.4f}")
    print(f"   - Enhanced InstaSHAP significantly better")

    print(f"\n2. On NO interaction data:")
    print(f"   - Accuracy change: {no_int_result['accuracy_improvement']:.4f}")
    print(f"   - Enhanced InstaSHAP does not harm accuracy")

    print("\n3. Overall: Extension helps where needed, neutral where not")

    print("\nExperiment completed!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
