"""
experiment_extension_accuracy.py — Experiment 2: Extension Accuracy.

Apply Interaction-Aware InstaSHAP (GA²M surrogate) to interaction-heavy
datasets and show improved surrogate fidelity and Shapley value accuracy.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.data.data_loader import load_dataset
from phase2.models.base_model import train_blackbox_model
from phase2.models.gam_surrogate import fit_ebm_surrogate, surrogate_fidelity
from phase2.models.instashap import instashap_from_ebm
from phase2.explainers.exact_shap import compute_tree_shap
from phase3.extension.interaction_aware_surrogate import fit_interaction_ebm
from phase3.extension.enhanced_instashap import enhanced_instashap_from_ebm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _make_xor_dataset(n_samples=2000):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.randn(n_samples, 8)
    y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] + 0.1 * rng.randn(n_samples)
    y_binary = (y > np.median(y)).astype(int)
    fnames = [f"x{i}" for i in range(8)]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_SEED, stratify=y_binary
    )
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns=fnames)
    X_test = pd.DataFrame(sc.transform(X_test), columns=fnames)
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": pd.Series(y_train),
        "y_test": pd.Series(y_test),
        "feature_names": fnames,
        "task": "classification",
        "name": "XOR Synthetic",
    }


def run_extension_accuracy():
    """Run accuracy comparison: Additive vs GA²M InstaSHAP."""
    datasets = {}

    # Synthetic interaction-heavy
    datasets["XOR Synthetic"] = _make_xor_dataset()

    # Real datasets
    for name in ["california_housing", "diabetes", "wine_binary"]:
        try:
            datasets[name] = load_dataset(name)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    all_results = []

    for ds_name, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Extension Accuracy — {ds_name}")
        print(f"{'=' * 60}")

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        feature_names = data["feature_names"]
        task = data["task"]

        # Train black-box
        model = train_blackbox_model(X_train, y_train, task=task, model_type="xgboost")

        # Exact SHAP
        print("Computing Exact SHAP ...")
        exact_result = compute_tree_shap(model, X_test, feature_names)
        shap_exact = exact_result["shap_values"]

        # Additive InstaSHAP
        print("Fitting additive surrogate ...")
        ebm_add = fit_ebm_surrogate(model, X_train, task=task, interactions=0)
        shap_add, _ = instashap_from_ebm(ebm_add, np.asarray(X_test), feature_names)
        shap_add = np.asarray(shap_add)
        fid_add = surrogate_fidelity(model, ebm_add, X_test, task=task)

        # GA²M InstaSHAP
        print("Fitting GA²M surrogate ...")
        ebm_ga2m = fit_interaction_ebm(model, X_train, task=task, n_interactions=10)
        shap_ga2m, _, info = enhanced_instashap_from_ebm(
            ebm_ga2m, np.asarray(X_test), feature_names
        )
        shap_ga2m = np.asarray(shap_ga2m)
        fid_ga2m = surrogate_fidelity(model, ebm_ga2m, X_test, task=task)

        # Metrics
        def _metrics(exact, approx, label):
            flat_e = exact.flatten()
            flat_a = approx.flatten()
            pr, _ = pearsonr(flat_e, flat_a)
            sr, _ = spearmanr(flat_e, flat_a)
            mae = np.mean(np.abs(flat_e - flat_a))
            ss_res = np.sum((flat_e - flat_a) ** 2)
            ss_tot = np.sum((flat_e - np.mean(flat_e)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            print(
                f"  {label}: Pearson={pr:.4f}, Spearman={sr:.4f}, MAE={mae:.6f}, R²={r2:.4f}"
            )
            return {
                "method": label,
                "pearson": pr,
                "spearman": sr,
                "mae": mae,
                "r2": r2,
            }

        m_add = _metrics(shap_exact, shap_add, "Additive InstaSHAP")
        m_ga2m = _metrics(shap_exact, shap_ga2m, "GA²M InstaSHAP")

        print(
            f"  Surrogate R²: additive={fid_add['r2']:.4f}, ga2m={fid_ga2m['r2']:.4f}"
        )

        for m in [m_add, m_ga2m]:
            m["dataset"] = ds_name
            m["surrogate_r2"] = (
                fid_add["r2"] if "Additive" in m["method"] else fid_ga2m["r2"]
            )
            all_results.append(m)

        # Per-feature scatter
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, shap_vals, title in [
            (axes[0], shap_add, "Additive InstaSHAP"),
            (axes[1], shap_ga2m, "GA²M InstaSHAP"),
        ]:
            flat_e = shap_exact.flatten()
            flat_a = shap_vals.flatten()
            pr, _ = pearsonr(flat_e, flat_a)
            ax.scatter(flat_e, flat_a, alpha=0.2, s=5)
            lims = [min(flat_e.min(), flat_a.min()), max(flat_e.max(), flat_a.max())]
            ax.plot(lims, lims, "r--", linewidth=1)
            ax.set_xlabel("Exact SHAP")
            ax.set_ylabel("InstaSHAP")
            ax.set_title(f"{title} (r={pr:.3f})")
        plt.suptitle(f"Extension Accuracy — {ds_name}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, f"ext_accuracy_{ds_name.replace(' ', '_')}.png"),
            dpi=150,
        )
        plt.close()

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "extension_accuracy.csv"), index=False)

    # Summary grouped bar
    datasets_unique = df["dataset"].unique()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = [
        ("pearson", "Pearson r"),
        ("mae", "MAE"),
        ("surrogate_r2", "Surrogate R²"),
    ]
    for ax, (col, ylabel) in zip(axes, metrics_to_plot):
        x = np.arange(len(datasets_unique))
        w = 0.35
        add_vals = [
            df[(df["dataset"] == d) & (df["method"] == "Additive InstaSHAP")][
                col
            ].values[0]
            for d in datasets_unique
        ]
        ga2m_vals = [
            df[(df["dataset"] == d) & (df["method"] == "GA²M InstaSHAP")][col].values[0]
            for d in datasets_unique
        ]
        ax.bar(x - w / 2, add_vals, w, label="Additive", color="#e74c3c")
        ax.bar(x + w / 2, ga2m_vals, w, label="GA²M", color="#2ecc71")
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_unique, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
    plt.suptitle("Extension Accuracy Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "extension_accuracy_summary.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Extension accuracy experiment complete.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_extension_accuracy()
