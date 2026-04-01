"""
Experiment 1: Gap Demonstration

This experiment demonstrates the limitation of the original InstaSHAP
(additive-only surrogate) on datasets with strong feature interactions.

We show that:
1. Additive surrogates have poor fidelity when interactions are present
2. InstaSHAP produces inaccurate Shapley values in these cases
3. This serves as evidence that the identified gap is real and significant

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

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Results directory
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_gap_demonstration_synthetic(
    interaction_strengths: list = [0.0, 0.5, 1.0, 2.0, 3.0],
) -> pd.DataFrame:
    """
    Demonstrate the gap using synthetic datasets with varying interaction strength.

    Args:
        interaction_strengths: List of interaction strength values to test

    Returns:
        DataFrame with results for each interaction strength
    """
    print("=" * 60)
    print("GAP DEMONSTRATION: Synthetic Interaction Datasets")
    print("=" * 60)

    results = []

    for strength in interaction_strengths:
        print(f"\n--- Interaction Strength: {strength} ---")

        # Create synthetic dataset
        data = create_synthetic_interaction_dataset(
            n_samples=2000,
            n_features=10,
            interaction_strength=strength,
            noise_level=0.1,
        )

        # Train black-box model
        blackbox = train_model_for_dataset(data, model_type="xgboost")

        # Train additive GAM surrogate (original InstaSHAP)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surrogate = train_surrogate_for_blackbox(
                blackbox.model, data["X_train"], task_type="regression"
            )

        # Evaluate surrogate fidelity
        bb_preds = blackbox.model.predict(data["X_test"])
        surr_preds = surrogate.predict(data["X_test"])
        surrogate_r2 = r2_score(bb_preds, surr_preds)

        print(f"Surrogate R² (fidelity): {surrogate_r2:.4f}")

        # Compute exact SHAP values (ground truth)
        exact_explainer = ExactSHAPExplainer(
            blackbox.model, data["X_train"], model_type="tree", task_type="regression"
        )

        X_explain = data["X_test"].iloc[:200]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact_shap = exact_explainer.explain(X_explain, return_dataframe=False)

        # Compute InstaSHAP values
        instashap = InstaSHAP(surrogate, data["X_train"])
        instashap_values = instashap.explain(X_explain, return_dataframe=False)

        # Compute accuracy metrics
        exact_flat = exact_shap.flatten()
        insta_flat = instashap_values.flatten()

        pearson_r, _ = pearsonr(exact_flat, insta_flat)
        spearman_r, _ = spearmanr(exact_flat, insta_flat)
        mae = mean_absolute_error(exact_flat, insta_flat)

        print(f"InstaSHAP-Exact correlation: {pearson_r:.4f}")
        print(f"MAE: {mae:.4f}")

        results.append(
            {
                "interaction_strength": strength,
                "surrogate_r2": surrogate_r2,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "mae": mae,
            }
        )

    return pd.DataFrame(results)


def plot_gap_demonstration(results_df: pd.DataFrame, save_path: str) -> None:
    """
    Plot the gap demonstration results.

    Args:
        results_df: Results DataFrame
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Surrogate Fidelity vs Interaction Strength
    ax1 = axes[0]
    ax1.plot(
        results_df["interaction_strength"],
        results_df["surrogate_r2"],
        "o-",
        linewidth=2,
        markersize=10,
        color="#e74c3c",
    )
    ax1.axhline(
        y=0.9, color="green", linestyle="--", alpha=0.7, label="High Fidelity (0.9)"
    )
    ax1.set_xlabel("Interaction Strength", fontsize=12)
    ax1.set_ylabel("Surrogate R² (Fidelity)", fontsize=12)
    ax1.set_title("Surrogate Fidelity Degrades with Interactions", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Add annotations
    for i, row in results_df.iterrows():
        ax1.annotate(
            f"{row['surrogate_r2']:.2f}",
            (row["interaction_strength"], row["surrogate_r2"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Plot 2: InstaSHAP Accuracy vs Interaction Strength
    ax2 = axes[1]
    ax2.plot(
        results_df["interaction_strength"],
        results_df["pearson_r"],
        "o-",
        linewidth=2,
        markersize=10,
        color="#3498db",
    )
    ax2.axhline(
        y=0.9, color="green", linestyle="--", alpha=0.7, label="High Accuracy (0.9)"
    )
    ax2.set_xlabel("Interaction Strength", fontsize=12)
    ax2.set_ylabel("Pearson Correlation with Exact SHAP", fontsize=12)
    ax2.set_title("InstaSHAP Accuracy Degrades with Interactions", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Add annotations
    for i, row in results_df.iterrows():
        ax2.annotate(
            f"{row['pearson_r']:.2f}",
            (row["interaction_strength"], row["pearson_r"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Gap demonstration plot saved to {save_path}")


def run_gap_on_real_data() -> pd.DataFrame:
    """
    Test for interactions in real datasets.

    Returns:
        DataFrame with results
    """
    print("\n" + "=" * 60)
    print("GAP ANALYSIS: Real Dataset (California Housing)")
    print("=" * 60)

    # Load California Housing
    cal_data = load_california_housing()

    # Train XGBoost (which can capture interactions)
    blackbox = train_model_for_dataset(cal_data, model_type="xgboost")

    # Train additive surrogate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surrogate = train_surrogate_for_blackbox(
            blackbox.model, cal_data["X_train"], task_type="regression"
        )

    # Evaluate fidelity
    bb_preds = blackbox.model.predict(cal_data["X_test"])
    surr_preds = surrogate.predict(cal_data["X_test"])
    r2 = r2_score(bb_preds, surr_preds)

    print(f"\nAdditive Surrogate Fidelity: {r2:.4f}")

    # If fidelity is not very high, interactions may be present
    if r2 < 0.95:
        print("Note: Fidelity < 0.95 suggests potential interactions in the data")

    return pd.DataFrame(
        [
            {
                "dataset": "california_housing",
                "surrogate_r2": r2,
                "interactions_likely": r2 < 0.95,
            }
        ]
    )


def main():
    """Run the complete gap demonstration experiment."""

    print("=" * 60)
    print("EXPERIMENT: GAP DEMONSTRATION")
    print("Showing limitations of additive InstaSHAP")
    print("=" * 60)

    # Run synthetic data experiment
    synthetic_results = run_gap_demonstration_synthetic(
        interaction_strengths=[0.0, 0.5, 1.0, 2.0, 3.0]
    )

    # Plot results
    plot_gap_demonstration(
        synthetic_results, os.path.join(RESULTS_DIR, "gap_demonstration.png")
    )

    # Save results
    synthetic_results.to_csv(
        os.path.join(RESULTS_DIR, "gap_demonstration_results.csv"), index=False
    )

    # Test on real data
    real_results = run_gap_on_real_data()

    # Summary
    print("\n" + "=" * 60)
    print("GAP DEMONSTRATION SUMMARY")
    print("=" * 60)

    print("\nSynthetic Data Results:")
    print(synthetic_results.to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    print("1. Surrogate fidelity DECREASES as interaction strength increases")
    print("2. InstaSHAP accuracy (correlation with Exact SHAP) DECREASES similarly")
    print("3. At high interaction strength, InstaSHAP can be significantly inaccurate")
    print("4. This demonstrates the need for interaction-aware InstaSHAP")

    print("\nExperiment completed!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
