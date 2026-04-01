"""
experiment_gap_demonstration.py — Experiment 1: Demonstrate the Gap.

Create datasets with strong feature interactions and show that the
original additive InstaSHAP produces poor surrogate fidelity and
inaccurate Shapley values on such data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.data.data_loader import load_dataset
from phase2.models.base_model import train_blackbox_model
from phase2.models.gam_surrogate import fit_ebm_surrogate, surrogate_fidelity
from phase2.models.instashap import instashap_from_ebm
from phase2.explainers.exact_shap import compute_tree_shap

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def create_xor_dataset(n_samples=2000, noise=0.1):
    """Create a synthetic dataset with XOR-like interactions.

    y = x0 * x1 + x2 * x3 - x4 + noise
    Features x0*x1 and x2*x3 are strong interactions.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.randn(n_samples, 8)
    y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] + noise * rng.randn(n_samples)
    # Binarize for classification
    y_binary = (y > np.median(y)).astype(int)
    feature_names = [f"x{i}" for i in range(8)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_SEED, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": pd.DataFrame(X_train, columns=feature_names),
        "X_test": pd.DataFrame(X_test, columns=feature_names),
        "y_train": pd.Series(y_train),
        "y_test": pd.Series(y_test),
        "feature_names": feature_names,
        "task": "classification",
        "name": "XOR Synthetic",
    }


def create_multiplicative_dataset(n_samples=2000, noise=0.1):
    """Create a dataset with multiplicative interactions.

    y = 3*x0*x1 + 2*x2 - x3*x4*x5 + noise
    """
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.randn(n_samples, 8)
    y = (
        3 * X[:, 0] * X[:, 1]
        + 2 * X[:, 2]
        - X[:, 3] * X[:, 4] * X[:, 5]
        + noise * rng.randn(n_samples)
    )
    y_binary = (y > np.median(y)).astype(int)
    feature_names = [f"x{i}" for i in range(8)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_SEED, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": pd.DataFrame(X_train, columns=feature_names),
        "X_test": pd.DataFrame(X_test, columns=feature_names),
        "y_train": pd.Series(y_train),
        "y_test": pd.Series(y_test),
        "feature_names": feature_names,
        "task": "classification",
        "name": "Multiplicative Synthetic",
    }


def create_sklearn_interaction_dataset(n_samples=2000):
    """Use sklearn make_classification with redundant/clustered features
    to induce interactions."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=4,
        n_clusters_per_class=2,
        random_state=RANDOM_SEED,
    )
    feature_names = [f"f{i}" for i in range(12)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "X_train": pd.DataFrame(X_train, columns=feature_names),
        "X_test": pd.DataFrame(X_test, columns=feature_names),
        "y_train": pd.Series(y_train),
        "y_test": pd.Series(y_test),
        "feature_names": feature_names,
        "task": "classification",
        "name": "Sklearn Interaction Heavy",
    }


def evaluate_gap(dataset, dataset_name):
    """Evaluate additive InstaSHAP on the given dataset and report gaps."""
    print(f"\n{'=' * 60}")
    print(f"Gap Demonstration — {dataset_name}")
    print(f"{'=' * 60}")

    X_train = dataset["X_train"]
    X_test = dataset["X_test"]
    y_train = dataset["y_train"]
    feature_names = dataset["feature_names"]
    task = dataset["task"]

    # Train black-box
    print("Training black-box XGBoost ...")
    model = train_blackbox_model(X_train, y_train, task=task, model_type="xgboost")

    # Additive EBM surrogate
    print("Fitting additive EBM surrogate ...")
    ebm_additive = fit_ebm_surrogate(model, X_train, task=task, interactions=0)

    # GA²M surrogate
    print("Fitting GA²M surrogate (with interactions) ...")
    from phase3.extension.interaction_aware_surrogate import fit_interaction_ebm

    ebm_ga2m = fit_interaction_ebm(model, X_train, task=task, n_interactions=10)

    # Fidelity comparison
    fid_add = surrogate_fidelity(model, ebm_additive, X_test, task=task)
    fid_ga2m = surrogate_fidelity(model, ebm_ga2m, X_test, task=task)

    print(f"\n  Additive surrogate R²:  {fid_add['r2']:.4f}")
    print(f"  GA²M surrogate R²:      {fid_ga2m['r2']:.4f}")
    print(f"  Fidelity improvement:   {fid_ga2m['r2'] - fid_add['r2']:.4f}")

    # Exact SHAP
    print("\nComputing Exact SHAP ...")
    exact_result = compute_tree_shap(model, X_test, feature_names)
    shap_exact = exact_result["shap_values"]

    # InstaSHAP (additive)
    shap_additive, _ = instashap_from_ebm(
        ebm_additive, np.asarray(X_test), feature_names
    )
    shap_additive = np.asarray(shap_additive)

    # InstaSHAP (GA²M)
    from phase3.extension.enhanced_instashap import enhanced_instashap_from_ebm

    shap_ga2m, _, info = enhanced_instashap_from_ebm(
        ebm_ga2m, np.asarray(X_test), feature_names
    )
    shap_ga2m = np.asarray(shap_ga2m)

    # Accuracy comparison
    def _compare(shap_a, shap_b, label_a, label_b):
        flat_a = shap_a.flatten()
        flat_b = shap_b.flatten()
        pr, _ = pearsonr(flat_a, flat_b)
        mae = np.mean(np.abs(flat_a - flat_b))
        ss_res = np.sum((flat_a - flat_b) ** 2)
        ss_tot = np.sum((flat_a - np.mean(flat_a)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        print(f"\n  {label_a} vs {label_b}:")
        print(f"    Pearson r: {pr:.4f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        return {"pearson": pr, "mae": mae, "r2": r2}

    m_add = _compare(shap_exact, shap_additive, "Exact SHAP", "Additive InstaSHAP")
    m_ga2m = _compare(shap_exact, shap_ga2m, "Exact SHAP", "GA²M InstaSHAP")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    flat_exact = shap_exact.flatten()
    ax.scatter(
        flat_exact,
        shap_additive.flatten(),
        alpha=0.2,
        s=5,
        label=f"Additive (r={m_add['pearson']:.3f})",
    )
    ax.scatter(
        flat_exact,
        shap_ga2m.flatten(),
        alpha=0.2,
        s=5,
        label=f"GA²M (r={m_ga2m['pearson']:.3f})",
    )
    lims = [flat_exact.min(), flat_exact.max()]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Exact SHAP")
    ax.set_ylabel("InstaSHAP")
    ax.set_title(f"SHAP Value Comparison — {dataset_name}")
    ax.legend()

    ax = axes[1]
    methods = ["Additive\nInstaSHAP", "GA²M\nInstaSHAP"]
    r2_vals = [fid_add["r2"], fid_ga2m["r2"]]
    bars = ax.bar(methods, r2_vals, color=["#e74c3c", "#2ecc71"])
    ax.set_ylabel("Surrogate R² vs Black-Box")
    ax.set_title("Surrogate Fidelity")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, r2_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            fontsize=11,
        )

    plt.suptitle(f"Gap Demonstration — {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, f"gap_demo_{dataset_name.replace(' ', '_')}.png"),
        dpi=150,
    )
    plt.close()

    return {
        "dataset": dataset_name,
        "additive_r2": fid_add["r2"],
        "ga2m_r2": fid_ga2m["r2"],
        "additive_shap_pearson": m_add["pearson"],
        "ga2m_shap_pearson": m_ga2m["pearson"],
        "additive_shap_mae": m_add["mae"],
        "ga2m_shap_mae": m_ga2m["mae"],
    }


def run_gap_demonstration():
    """Run gap demonstration on interaction-heavy datasets."""
    datasets = {
        "XOR Synthetic": create_xor_dataset(),
        "Multiplicative Synthetic": create_multiplicative_dataset(),
        "Sklearn Interaction Heavy": create_sklearn_interaction_dataset(),
    }

    # Also include a low-interaction baseline
    try:
        from phase2.data.data_loader import load_dataset

        datasets["California Housing"] = load_dataset("california_housing")
    except Exception:
        pass

    results = []
    for name, ds in datasets.items():
        row = evaluate_gap(ds, name)
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "gap_demonstration.csv"), index=False)

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["additive_r2"], w, label="Additive Surrogate", color="#e74c3c")
    ax.bar(x + w / 2, df["ga2m_r2"], w, label="GA²M Surrogate", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=20, ha="right")
    ax.set_ylabel("Surrogate R²")
    ax.set_title("Surrogate Fidelity: Additive vs GA²M")
    ax.legend()
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(
        x - w / 2,
        df["additive_shap_pearson"],
        w,
        label="Additive InstaSHAP",
        color="#e74c3c",
    )
    ax.bar(
        x + w / 2, df["ga2m_shap_pearson"], w, label="GA²M InstaSHAP", color="#2ecc71"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=20, ha="right")
    ax.set_ylabel("Pearson r vs Exact SHAP")
    ax.set_title("SHAP Accuracy: Additive vs GA²M")
    ax.legend()
    ax.set_ylim(0, 1.05)

    plt.suptitle("Gap Demonstration Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gap_demo_summary.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Gap demonstration complete. Results saved to results/")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_gap_demonstration()
