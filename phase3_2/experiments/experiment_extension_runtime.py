"""
Experiment 3: Extension Runtime

This experiment measures the runtime impact of the Interaction-Aware
InstaSHAP extension compared to the original InstaSHAP.

We show that:
1. Enhanced InstaSHAP is slightly slower due to interaction terms
2. But it is still MUCH faster than Exact SHAP / KernelSHAP
3. The speed advantage of InstaSHAP is preserved

Author: DS357 Course Project Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

from phase2.data.data_loader import load_california_housing
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


def measure_explanation_time(explainer, X, n_runs=3):
    """Measure average time to compute explanations."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = explainer.explain(X, return_dataframe=False)
        times.append(time.time() - start)
    return np.mean(times)


def run_runtime_comparison(sample_sizes=[50, 100, 200, 500, 1000]) -> pd.DataFrame:
    """
    Compare runtime of different explanation methods.

    Args:
        sample_sizes: List of sample sizes to test

    Returns:
        DataFrame with runtime results
    """
    print("=" * 60)
    print("EXTENSION RUNTIME COMPARISON")
    print("=" * 60)

    # Load data and train model
    print("\nLoading data and training models...")
    cal_data = load_california_housing()
    blackbox = train_model_for_dataset(cal_data, model_type="xgboost")

    # Initialize all explainers
    print("\nInitializing explainers...")

    # Measure fitting times
    print("\n--- Fitting Times ---")

    # Original InstaSHAP (additive surrogate)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        additive_surrogate = train_surrogate_for_blackbox(
            blackbox.model, cal_data["X_train"], task_type="regression"
        )
        original_instashap = InstaSHAP(additive_surrogate, cal_data["X_train"])
    original_fit_time = time.time() - start
    print(f"Original InstaSHAP fitting: {original_fit_time:.2f}s")

    # Enhanced InstaSHAP (GA²M surrogate)
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ga2m_surrogate = train_interaction_surrogate_for_blackbox(
            blackbox.model,
            cal_data["X_train"],
            task_type="regression",
            n_interactions=5,
        )
        enhanced_instashap = EnhancedInstaSHAP(ga2m_surrogate, cal_data["X_train"])
    enhanced_fit_time = time.time() - start
    print(f"Enhanced InstaSHAP fitting: {enhanced_fit_time:.2f}s")

    # Exact SHAP
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact_explainer = ExactSHAPExplainer(
            blackbox.model,
            cal_data["X_train"],
            model_type="tree",
            task_type="regression",
        )
    exact_fit_time = time.time() - start
    print(f"Exact SHAP initialization: {exact_fit_time:.2f}s")

    # Run timing experiments
    print("\n--- Explanation Times ---")

    results = []

    for n_samples in sample_sizes:
        print(f"\nTesting n_samples = {n_samples}")

        # Skip if not enough test samples
        if n_samples > len(cal_data["X_test"]):
            continue

        X_explain = cal_data["X_test"].iloc[:n_samples]

        # Time each method
        exact_time = measure_explanation_time(exact_explainer, X_explain, n_runs=3)
        original_time = measure_explanation_time(
            original_instashap, X_explain, n_runs=5
        )
        enhanced_time = measure_explanation_time(
            enhanced_instashap, X_explain, n_runs=5
        )

        print(f"  Exact SHAP:          {exact_time:.4f}s")
        print(f"  Original InstaSHAP:  {original_time:.4f}s")
        print(f"  Enhanced InstaSHAP:  {enhanced_time:.4f}s")

        results.append(
            {
                "n_samples": n_samples,
                "exact_shap_time": exact_time,
                "original_instashap_time": original_time,
                "enhanced_instashap_time": enhanced_time,
                "original_speedup_vs_exact": exact_time / original_time,
                "enhanced_speedup_vs_exact": exact_time / enhanced_time,
                "enhanced_overhead": (enhanced_time - original_time)
                / original_time
                * 100,
            }
        )

    results_df = pd.DataFrame(results)

    # Add fitting times
    results_df["original_fit_time"] = original_fit_time
    results_df["enhanced_fit_time"] = enhanced_fit_time

    return results_df


def plot_runtime_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot runtime comparison results.

    Args:
        results_df: Results DataFrame
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Runtime by Method
    ax1 = axes[0]
    ax1.plot(
        results_df["n_samples"],
        results_df["exact_shap_time"],
        "o-",
        label="Exact SHAP",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        results_df["n_samples"],
        results_df["original_instashap_time"],
        "s-",
        label="Original InstaSHAP",
        linewidth=2,
        markersize=8,
    )
    ax1.plot(
        results_df["n_samples"],
        results_df["enhanced_instashap_time"],
        "^-",
        label="Enhanced InstaSHAP",
        linewidth=2,
        markersize=8,
    )

    ax1.set_xlabel("Number of Samples", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Runtime Comparison", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Speedup vs Exact SHAP
    ax2 = axes[1]
    x = np.arange(len(results_df))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        results_df["original_speedup_vs_exact"],
        width,
        label="Original InstaSHAP",
        color="#3498db",
    )
    bars2 = ax2.bar(
        x + width / 2,
        results_df["enhanced_speedup_vs_exact"],
        width,
        label="Enhanced InstaSHAP",
        color="#2ecc71",
    )

    ax2.set_xlabel("Number of Samples", fontsize=12)
    ax2.set_ylabel("Speedup vs Exact SHAP (x times faster)", fontsize=12)
    ax2.set_title("Speedup Comparison", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df["n_samples"])
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(
            f"{height:.0f}x",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Runtime comparison plot saved to {save_path}")


def main():
    """Run the complete extension runtime experiment."""

    print("=" * 60)
    print("EXPERIMENT: EXTENSION RUNTIME")
    print("Measuring computational overhead of interactions")
    print("=" * 60)

    # Run comparison
    results = run_runtime_comparison(sample_sizes=[50, 100, 200, 500])

    # Plot results
    plot_runtime_comparison(
        results, os.path.join(RESULTS_DIR, "extension_runtime_comparison.png")
    )

    # Save results
    results.to_csv(
        os.path.join(RESULTS_DIR, "extension_runtime_results.csv"), index=False
    )

    # Summary
    print("\n" + "=" * 60)
    print("EXTENSION RUNTIME SUMMARY")
    print("=" * 60)

    print("\nFitting Times (one-time cost):")
    print(f"  Original InstaSHAP: {results['original_fit_time'].iloc[0]:.2f}s")
    print(f"  Enhanced InstaSHAP: {results['enhanced_fit_time'].iloc[0]:.2f}s")

    print("\nExplanation Times (per-batch):")
    print(
        results[
            [
                "n_samples",
                "exact_shap_time",
                "original_instashap_time",
                "enhanced_instashap_time",
            ]
        ].to_string(index=False)
    )

    print("\nSpeedup vs Exact SHAP:")
    print(
        results[
            [
                "n_samples",
                "original_speedup_vs_exact",
                "enhanced_speedup_vs_exact",
                "enhanced_overhead",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)

    avg_overhead = results["enhanced_overhead"].mean()
    avg_enhanced_speedup = results["enhanced_speedup_vs_exact"].mean()

    print(f"1. Enhanced InstaSHAP overhead: ~{avg_overhead:.1f}% slower than Original")
    print(
        f"2. Enhanced InstaSHAP still ~{avg_enhanced_speedup:.0f}x faster than Exact SHAP"
    )
    print("3. The speed advantage of InstaSHAP is PRESERVED")
    print("4. Slight overhead is acceptable for accuracy improvement")

    print("\nExperiment completed!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
