"""
experiment_comparison.py — Experiment 4: Comprehensive Comparison.

Run all methods (Exact SHAP, KernelSHAP, Original InstaSHAP,
Interaction-Aware InstaSHAP) across multiple datasets.
"""

import os
import sys
import time
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
from phase2.explainers.kernel_shap import compute_kernel_shap
from phase3.extension.interaction_aware_surrogate import fit_interaction_ebm
from phase3.extension.enhanced_instashap import enhanced_instashap_from_ebm
from phase3.extension.adaptive_surrogate import AdaptiveInstaSHAP

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def _make_xor_dataset(n_samples=2000):
    rng = np.random.RandomState(RANDOM_SEED)
    X = rng.randn(n_samples, 8)
    y = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] - X[:, 4] + 0.1 * rng.randn(n_samples)
    y_binary = (y > np.median(y)).astype(int)
    fnames = [f"x{i}" for i in range(8)]
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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


def run_comprehensive_comparison():
    """Run comprehensive comparison across all methods and datasets."""
    datasets = {"XOR Synthetic": _make_xor_dataset()}
    for name in ["california_housing", "diabetes", "wine_binary"]:
        try:
            datasets[name] = load_dataset(name)
        except Exception:
            pass

    all_results = []

    for ds_name, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Comprehensive Comparison — {ds_name}")
        print(f"{'=' * 60}")

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        feature_names = data["feature_names"]
        task = data["task"]

        model = train_blackbox_model(X_train, y_train, task=task, model_type="xgboost")

        # Subsample X_test for speed
        n_explain = min(500, len(X_test))
        X_explain = (
            X_test.iloc[:n_explain] if hasattr(X_test, "iloc") else X_test[:n_explain]
        )
        X_np = np.asarray(X_explain)

        # --- Exact SHAP (ground truth) ---
        print("  Exact SHAP ...")
        t0 = time.time()
        exact_res = compute_tree_shap(model, X_explain, feature_names)
        shap_exact = exact_res["shap_values"]
        t_exact = exact_res["runtime"]

        # --- Kernel SHAP (small subset only) ---
        shap_kernel = None
        t_kernel = None
        if n_explain <= 200:
            print("  Kernel SHAP ...")
            try:
                kernel_res = compute_kernel_shap(
                    model, X_explain, background_size=50, n_samples=200
                )
                shap_kernel = kernel_res["shap_values"]
                t_kernel = kernel_res["runtime"]
            except Exception as e:
                print(f"    Kernel SHAP failed: {e}")

        # --- Additive InstaSHAP ---
        print("  Additive InstaSHAP ...")
        t0 = time.time()
        ebm_add = fit_ebm_surrogate(model, X_train, task=task, interactions=0)
        shap_add, _ = instashap_from_ebm(ebm_add, X_np, feature_names)
        shap_add = np.asarray(shap_add)
        t_add = time.time() - t0
        fid_add = surrogate_fidelity(model, ebm_add, X_explain, task=task)

        # --- GA²M InstaSHAP ---
        print("  GA²M InstaSHAP ...")
        t0 = time.time()
        ebm_ga2m = fit_interaction_ebm(model, X_train, task=task, n_interactions=10)
        shap_ga2m, _, info = enhanced_instashap_from_ebm(ebm_ga2m, X_np, feature_names)
        shap_ga2m = np.asarray(shap_ga2m)
        t_ga2m = time.time() - t0
        fid_ga2m = surrogate_fidelity(model, ebm_ga2m, X_explain, task=task)

        # --- Adaptive InstaSHAP ---
        print("  Adaptive InstaSHAP ...")
        t0 = time.time()
        adaptive = AdaptiveInstaSHAP(model, task=task, feature_names=feature_names)
        adaptive.fit(X_train)
        shap_adapt, _, adapt_info = adaptive.explain(X_explain)
        shap_adapt = np.asarray(shap_adapt)
        t_adapt = time.time() - t0

        # Metrics vs Exact SHAP
        def _metrics(exact, approx, label):
            flat_e = exact.flatten()
            flat_a = approx.flatten()
            if np.std(flat_e) < 1e-12 or np.std(flat_a) < 1e-12:
                return {"method": label, "pearson": 0, "spearman": 0, "mae": 0, "r2": 0}
            pr, _ = pearsonr(flat_e, flat_a)
            sr, _ = spearmanr(flat_e, flat_a)
            mae = np.mean(np.abs(flat_e - flat_a))
            ss_res = np.sum((flat_e - flat_a) ** 2)
            ss_tot = np.sum((flat_e - np.mean(flat_e)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            return {
                "method": label,
                "pearson": pr,
                "spearman": sr,
                "mae": mae,
                "r2": r2,
            }

        comparisons = [
            ("Additive InstaSHAP", shap_add, t_add, fid_add["r2"]),
            ("GA²M InstaSHAP", shap_ga2m, t_ga2m, fid_ga2m["r2"]),
            ("Adaptive InstaSHAP", shap_adapt, t_adapt, None),
        ]

        if shap_kernel is not None:
            comparisons.append(("KernelSHAP", shap_kernel, t_kernel, None))

        for method_name, shap_vals, runtime, surr_r2 in comparisons:
            m = _metrics(shap_exact, shap_vals, method_name)
            m["dataset"] = ds_name
            m["runtime"] = runtime
            m["surrogate_r2"] = surr_r2
            m["adaptive_mode"] = (
                adapt_info.get("mode") if "Adaptive" in method_name else None
            )
            all_results.append(m)
            print(
                f"    {method_name}: r={m['pearson']:.4f}, MAE={m['mae']:.6f}, time={runtime:.3f}s"
            )

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_comparison.csv"), index=False)

    # Summary table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)
    pivot = df.pivot_table(
        index=["dataset", "method"],
        values=["pearson", "mae", "runtime", "surrogate_r2"],
        aggfunc="first",
    ).round(4)
    print(pivot.to_string())

    # Plot
    datasets_unique = df["dataset"].unique()
    methods_unique = df["method"].unique()
    colors = {
        "Additive InstaSHAP": "#e74c3c",
        "GA²M InstaSHAP": "#2ecc71",
        "Adaptive InstaSHAP": "#3498db",
        "KernelSHAP": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Pearson r
    ax = axes[0]
    x = np.arange(len(datasets_unique))
    w = 0.8 / len(methods_unique)
    for i, method in enumerate(methods_unique):
        vals = []
        for ds in datasets_unique:
            row = df[(df["dataset"] == ds) & (df["method"] == method)]
            vals.append(row["pearson"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * w, vals, w, label=method, color=colors.get(method, "#999"))
    ax.set_xticks(x + w * (len(methods_unique) - 1) / 2)
    ax.set_xticklabels(datasets_unique, rotation=20, ha="right")
    ax.set_ylabel("Pearson r vs Exact SHAP")
    ax.set_title("Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Runtime
    ax = axes[1]
    for i, method in enumerate(methods_unique):
        vals = []
        for ds in datasets_unique:
            row = df[(df["dataset"] == ds) & (df["method"] == method)]
            vals.append(row["runtime"].values[0] if len(row) > 0 else 0)
        ax.bar(x + i * w, vals, w, label=method, color=colors.get(method, "#999"))
    ax.set_xticks(x + w * (len(methods_unique) - 1) / 2)
    ax.set_xticklabels(datasets_unique, rotation=20, ha="right")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Speed")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    # Surrogate R²
    ax = axes[2]
    methods_with_fid = [
        m
        for m in methods_unique
        if any(df[(df["method"] == m) & (df["surrogate_r2"].notna())].shape[0] > 0)
    ]
    x2 = np.arange(len(datasets_unique))
    w2 = 0.8 / len(methods_with_fid)
    for i, method in enumerate(methods_with_fid):
        vals = []
        for ds in datasets_unique:
            row = df[(df["dataset"] == ds) & (df["method"] == method)]
            vals.append(
                row["surrogate_r2"].values[0]
                if len(row) > 0 and pd.notna(row["surrogate_r2"].values[0])
                else 0
            )
        ax.bar(x2 + i * w2, vals, w2, label=method, color=colors.get(method, "#999"))
    ax.set_xticks(x2 + w2 * (len(methods_with_fid) - 1) / 2)
    ax.set_xticklabels(datasets_unique, rotation=20, ha="right")
    ax.set_ylabel("Surrogate R²")
    ax.set_title("Surrogate Fidelity")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    plt.suptitle("Comprehensive Comparison: All Methods", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comprehensive_comparison.png"), dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("Comprehensive comparison complete. Results saved to results/")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_comprehensive_comparison()
