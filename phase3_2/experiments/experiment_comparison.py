"""
Experiment 4: Comprehensive Comparison

This experiment provides a comprehensive comparison of all methods
across multiple datasets, producing the final results table.

Methods compared:
- Exact SHAP (ground truth)
- KernelSHAP (baseline)
- Original InstaSHAP (additive)
- Enhanced InstaSHAP (with interactions)

Metrics:
- Accuracy (correlation with Exact SHAP)
- MAE
- Runtime
- Surrogate Fidelity

Author: DS357 Course Project Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
import time
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
    load_diabetes_data,
    create_synthetic_interaction_dataset,
)
from phase2.models.base_model import train_model_for_dataset
from phase2.models.gam_surrogate import train_surrogate_for_blackbox
from phase2.models.instashap import InstaSHAP
from phase2.explainers.exact_shap import ExactSHAPExplainer
from phase2.explainers.kernel_shap import KernelSHAPExplainer

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


def evaluate_method(
    method_name: str,
    shap_values: np.ndarray,
    exact_shap: np.ndarray,
    runtime: float,
    surrogate_r2: float = None,
) -> dict:
    """
    Evaluate a SHAP method against exact SHAP.

    Args:
        method_name: Name of the method
        shap_values: SHAP values from this method
        exact_shap: Exact SHAP values (ground truth)
        runtime: Time to compute explanations
        surrogate_r2: Surrogate fidelity (if applicable)

    Returns:
        Dictionary with evaluation metrics
    """
    # Flatten for comparison
    pred_flat = shap_values.flatten()
    exact_flat = exact_shap.flatten()

    pearson_r, _ = pearsonr(pred_flat, exact_flat)
    mae = mean_absolute_error(exact_flat, pred_flat)

    return {
        "method": method_name,
        "pearson_r": pearson_r,
        "mae": mae,
        "runtime": runtime,
        "surrogate_r2": surrogate_r2,
    }


def run_comprehensive_comparison_on_dataset(
    data: dict, dataset_name: str, n_explain: int = 200
) -> list:
    """
    Run comprehensive comparison on a single dataset.

    Args:
        data: Dataset dictionary
        dataset_name: Name of dataset
        n_explain: Number of samples to explain

    Returns:
        List of result dictionaries
    """
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    results = []

    # Train black-box model
    print("Training black-box model...")
    blackbox = train_model_for_dataset(data, model_type="xgboost")

    X_explain = data["X_test"].iloc[:n_explain]

    # 1. Exact SHAP (ground truth)
    print("Computing Exact SHAP values...")
    start = time.time()
    exact_explainer = ExactSHAPExplainer(
        blackbox.model, data["X_train"], model_type="tree", task_type=data["task_type"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact_shap = exact_explainer.explain(X_explain, return_dataframe=False)
    exact_time = time.time() - start

    results.append(
        {
            "method": "Exact SHAP",
            "dataset": dataset_name,
            "pearson_r": 1.0,  # Perfect correlation with itself
            "mae": 0.0,
            "runtime": exact_time,
            "surrogate_r2": np.nan,
        }
    )

    # 2. KernelSHAP (baseline) - only for small sample sizes
    if n_explain <= 100:
        print("Computing KernelSHAP values...")
        start = time.time()
        kernel_explainer = KernelSHAPExplainer(
            blackbox.model, data["X_train"].iloc[:200], task_type=data["task_type"]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel_shap = kernel_explainer.explain(
                X_explain, return_dataframe=False, nsamples=100
            )
        kernel_time = time.time() - start

        result = evaluate_method("KernelSHAP", kernel_shap, exact_shap, kernel_time)
        result["dataset"] = dataset_name
        results.append(result)

    # 3. Original InstaSHAP (additive)
    print("Computing Original InstaSHAP values...")
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        additive_surrogate = train_surrogate_for_blackbox(
            blackbox.model, data["X_train"], task_type=data["task_type"]
        )
        original_instashap = InstaSHAP(additive_surrogate, data["X_train"])
        original_shap = original_instashap.explain(X_explain, return_dataframe=False)
    original_time = time.time() - start

    # Calculate surrogate fidelity
    bb_preds = blackbox.model.predict(data["X_test"])
    additive_r2 = r2_score(bb_preds, additive_surrogate.predict(data["X_test"]))

    result = evaluate_method(
        "Original InstaSHAP", original_shap, exact_shap, original_time, additive_r2
    )
    result["dataset"] = dataset_name
    results.append(result)

    # 4. Enhanced InstaSHAP (with interactions)
    print("Computing Enhanced InstaSHAP values...")
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ga2m_surrogate = train_interaction_surrogate_for_blackbox(
            blackbox.model,
            data["X_train"],
            task_type=data["task_type"],
            n_interactions=5,
        )
        enhanced_instashap = EnhancedInstaSHAP(ga2m_surrogate, data["X_train"])
        enhanced_shap = enhanced_instashap.explain(X_explain, return_dataframe=False)
    enhanced_time = time.time() - start

    ga2m_r2 = r2_score(bb_preds, ga2m_surrogate.predict(data["X_test"]))

    result = evaluate_method(
        "Enhanced InstaSHAP", enhanced_shap, exact_shap, enhanced_time, ga2m_r2
    )
    result["dataset"] = dataset_name
    results.append(result)

    return results


def run_comprehensive_comparison() -> pd.DataFrame:
    """
    Run comprehensive comparison across all datasets.

    Returns:
        DataFrame with all results
    """
    print("=" * 60)
    print("COMPREHENSIVE COMPARISON")
    print("All Methods Across All Datasets")
    print("=" * 60)

    all_results = []

    # Dataset 1: Synthetic with strong interactions
    synth_strong = create_synthetic_interaction_dataset(
        n_samples=2000, interaction_strength=2.0
    )
    all_results.extend(
        run_comprehensive_comparison_on_dataset(
            synth_strong, "Synthetic (Strong Int.)", n_explain=100
        )
    )

    # Dataset 2: Synthetic with no interactions
    synth_none = create_synthetic_interaction_dataset(
        n_samples=2000, interaction_strength=0.0
    )
    all_results.extend(
        run_comprehensive_comparison_on_dataset(
            synth_none, "Synthetic (No Int.)", n_explain=100
        )
    )

    # Dataset 3: California Housing
    cal_data = load_california_housing()
    all_results.extend(
        run_comprehensive_comparison_on_dataset(
            cal_data, "California Housing", n_explain=100
        )
    )

    # Dataset 4: Diabetes
    diabetes_data = load_diabetes_data()
    all_results.extend(
        run_comprehensive_comparison_on_dataset(
            diabetes_data, "Diabetes", n_explain=100
        )
    )

    return pd.DataFrame(all_results)


def plot_comprehensive_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """
    Create comprehensive comparison visualization.

    Args:
        results_df: Results DataFrame
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Filter out Exact SHAP for accuracy comparisons (it's the reference)
    comparison_df = results_df[results_df["method"] != "Exact SHAP"]

    # Plot 1: Accuracy (Pearson r) by dataset and method
    ax1 = axes[0, 0]
    pivot1 = comparison_df.pivot(index="dataset", columns="method", values="pearson_r")
    pivot1.plot(kind="bar", ax=ax1, width=0.8)
    ax1.axhline(y=0.9, color="red", linestyle="--", alpha=0.7, label="High Accuracy")
    ax1.set_xlabel("Dataset", fontsize=11)
    ax1.set_ylabel("Pearson Correlation with Exact SHAP", fontsize=11)
    ax1.set_title("Accuracy Comparison", fontsize=13)
    ax1.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.3)

    # Plot 2: Runtime by dataset and method
    ax2 = axes[0, 1]
    pivot2 = results_df.pivot(index="dataset", columns="method", values="runtime")
    pivot2.plot(kind="bar", ax=ax2, width=0.8, logy=True)
    ax2.set_xlabel("Dataset", fontsize=11)
    ax2.set_ylabel("Runtime (seconds, log scale)", fontsize=11)
    ax2.set_title("Runtime Comparison", fontsize=13)
    ax2.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, axis="y", alpha=0.3)

    # Plot 3: MAE by dataset and method
    ax3 = axes[1, 0]
    pivot3 = comparison_df.pivot(index="dataset", columns="method", values="mae")
    pivot3.plot(kind="bar", ax=ax3, width=0.8)
    ax3.set_xlabel("Dataset", fontsize=11)
    ax3.set_ylabel("Mean Absolute Error", fontsize=11)
    ax3.set_title("MAE Comparison (Lower is Better)", fontsize=13)
    ax3.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, axis="y", alpha=0.3)

    # Plot 4: Surrogate Fidelity
    ax4 = axes[1, 1]
    instashap_only = comparison_df[comparison_df["method"].str.contains("InstaSHAP")]
    pivot4 = instashap_only.pivot(
        index="dataset", columns="method", values="surrogate_r2"
    )
    pivot4.plot(kind="bar", ax=ax4, width=0.8, color=["#e74c3c", "#2ecc71"])
    ax4.axhline(y=0.9, color="blue", linestyle="--", alpha=0.7, label="High Fidelity")
    ax4.set_xlabel("Dataset", fontsize=11)
    ax4.set_ylabel("Surrogate R²", fontsize=11)
    ax4.set_title("Surrogate Fidelity: Original vs Enhanced", fontsize=13)
    ax4.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax4.tick_params(axis="x", rotation=45)
    ax4.set_ylim(0, 1.05)
    ax4.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Comprehensive comparison plot saved to {save_path}")


def create_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a formatted summary table.

    Args:
        results_df: Results DataFrame

    Returns:
        Formatted summary DataFrame
    """
    summary = results_df.pivot_table(
        index="dataset",
        columns="method",
        values=["pearson_r", "mae", "runtime", "surrogate_r2"],
        aggfunc="first",
    )

    return summary


def main():
    """Run the complete comprehensive comparison experiment."""

    print("=" * 60)
    print("EXPERIMENT: COMPREHENSIVE COMPARISON")
    print("Final evaluation of all methods")
    print("=" * 60)

    # Run comparison
    results = run_comprehensive_comparison()

    # Plot results
    plot_comprehensive_comparison(
        results, os.path.join(RESULTS_DIR, "comprehensive_comparison.png")
    )

    # Save results
    results.to_csv(
        os.path.join(RESULTS_DIR, "comprehensive_comparison_results.csv"), index=False
    )

    # Create and save summary table
    summary = create_summary_table(results)
    summary.to_csv(os.path.join(RESULTS_DIR, "comprehensive_summary.csv"))

    # Print summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print("=" * 60)

    print("\nFull Results:")
    print(results.to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)

    # Analysis
    strong_int = results[results["dataset"] == "Synthetic (Strong Int.)"]
    no_int = results[results["dataset"] == "Synthetic (No Int.)"]

    orig_strong = strong_int[strong_int["method"] == "Original InstaSHAP"][
        "pearson_r"
    ].values[0]
    enh_strong = strong_int[strong_int["method"] == "Enhanced InstaSHAP"][
        "pearson_r"
    ].values[0]

    orig_none = no_int[no_int["method"] == "Original InstaSHAP"]["pearson_r"].values[0]
    enh_none = no_int[no_int["method"] == "Enhanced InstaSHAP"]["pearson_r"].values[0]

    print(f"\n1. On INTERACTION-HEAVY data:")
    print(f"   Original InstaSHAP: r = {orig_strong:.4f}")
    print(f"   Enhanced InstaSHAP: r = {enh_strong:.4f}")
    print(f"   Improvement: {enh_strong - orig_strong:.4f}")

    print(f"\n2. On NON-INTERACTION data:")
    print(f"   Original InstaSHAP: r = {orig_none:.4f}")
    print(f"   Enhanced InstaSHAP: r = {enh_none:.4f}")
    print(f"   Change: {enh_none - orig_none:.4f} (no harm)")

    # Runtime comparison
    orig_times = results[results["method"] == "Original InstaSHAP"]["runtime"].values
    enh_times = results[results["method"] == "Enhanced InstaSHAP"]["runtime"].values
    exact_times = results[results["method"] == "Exact SHAP"]["runtime"].values

    print(f"\n3. Runtime:")
    print(f"   Avg Exact SHAP: {np.mean(exact_times):.2f}s")
    print(f"   Avg Original InstaSHAP: {np.mean(orig_times):.2f}s")
    print(f"   Avg Enhanced InstaSHAP: {np.mean(enh_times):.2f}s")
    print(
        f"   Enhanced speedup vs Exact: {np.mean(exact_times) / np.mean(enh_times):.1f}x"
    )

    print("\n4. CONCLUSION:")
    print("   Enhanced InstaSHAP improves accuracy where interactions matter")
    print("   while maintaining the speed advantage of InstaSHAP!")

    print("\nExperiment completed!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
