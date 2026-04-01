"""
experiment_extension_runtime.py — Experiment 3: Extension Runtime.

Measure computation time for the extended GA²M method vs baselines.
Show that it's still orders of magnitude faster than Exact/Kernel SHAP.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phase2.data.data_loader import load_dataset
from phase2.models.base_model import train_blackbox_model
from phase2.models.gam_surrogate import fit_ebm_surrogate
from phase2.models.instashap import instashap_from_ebm
from phase3.extension.interaction_aware_surrogate import fit_interaction_ebm
from phase3.extension.enhanced_instashap import enhanced_instashap_from_ebm

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SAMPLE_SIZES = [50, 100, 200, 500, 1000]


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
        "feature_names": fnames,
        "task": "classification",
        "name": "XOR Synthetic",
    }


def run_extension_runtime():
    """Run runtime comparison."""
    datasets = {
        "XOR Synthetic": _make_xor_dataset(),
    }
    try:
        datasets["California Housing"] = load_dataset("california_housing")
    except Exception:
        pass

    all_results = []

    for ds_name, data in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"Runtime Experiment — {ds_name}")
        print(f"{'=' * 60}")

        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        task = data["task"]

        model = train_blackbox_model(X_train, y_train, task=task, model_type="xgboost")

        for n in SAMPLE_SIZES:
            if n > len(X_test):
                continue
            X_sub = X_test.iloc[:n] if hasattr(X_test, "iloc") else X_test[:n]
            X_np = np.asarray(X_sub)

            print(f"\n  n={n}")

            # --- Additive InstaSHAP ---
            times = []
            for _ in range(3):
                t0 = time.time()
                ebm = fit_ebm_surrogate(model, X_train, task=task, interactions=0)
                _ = instashap_from_ebm(ebm, X_np)
                times.append(time.time() - t0)
            t_add = np.median(times)
            print(f"    Additive InstaSHAP: {t_add:.4f}s")

            # --- GA²M InstaSHAP ---
            times = []
            for _ in range(3):
                t0 = time.time()
                ebm = fit_interaction_ebm(model, X_train, task=task, n_interactions=10)
                _ = enhanced_instashap_from_ebm(ebm, X_np)
                times.append(time.time() - t0)
            t_ga2m = np.median(times)
            print(f"    GA²M InstaSHAP:     {t_ga2m:.4f}s")

            # --- Tree SHAP ---
            times = []
            for _ in range(3):
                t0 = time.time()
                try:
                    import shap

                    explainer = shap.TreeExplainer(model)
                    _ = explainer.shap_values(X_np)
                except Exception:
                    pass
                times.append(time.time() - t0)
            t_tree = np.median(times)
            print(f"    Tree SHAP:          {t_tree:.4f}s")

            # --- Kernel SHAP (only for small n) ---
            t_kernel = None
            if n <= 200:
                times = []
                for _ in range(3):
                    t0 = time.time()
                    try:
                        import shap

                        bg = X_np[: min(50, len(X_np))]
                        explainer = shap.KernelExplainer(model.predict, bg)
                        _ = explainer.shap_values(
                            X_np[: min(20, len(X_np))], nsamples=100
                        )
                    except Exception:
                        pass
                    times.append(time.time() - t0)
                t_kernel = np.median(times)
                print(f"    KernelSHAP:         {t_kernel:.4f}s")

            for method, t in [
                ("Additive InstaSHAP", t_add),
                ("GA²M InstaSHAP", t_ga2m),
                ("Tree SHAP", t_tree),
            ]:
                all_results.append(
                    {
                        "dataset": ds_name,
                        "n_samples": n,
                        "method": method,
                        "runtime_seconds": t,
                    }
                )
            if t_kernel is not None:
                all_results.append(
                    {
                        "dataset": ds_name,
                        "n_samples": n,
                        "method": "KernelSHAP",
                        "runtime_seconds": t_kernel,
                    }
                )

    # Save and plot
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULTS_DIR, "extension_runtime.csv"), index=False)

    for ds_name in df["dataset"].unique():
        sub = df[df["dataset"] == ds_name]
        fig, ax = plt.subplots(figsize=(10, 6))
        for method in sub["method"].unique():
            msub = sub[sub["method"] == method].sort_values("n_samples")
            ax.plot(
                msub["n_samples"], msub["runtime_seconds"], marker="o", label=method
            )
        ax.set_xlabel("Number of Explained Samples")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"Runtime Comparison — {ds_name}")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(RESULTS_DIR, f"ext_runtime_{ds_name.replace(' ', '_')}.png"),
            dpi=150,
        )
        plt.close()

    print("\n" + "=" * 60)
    print("Extension runtime experiment complete.")
    print("=" * 60)
    return df


if __name__ == "__main__":
    run_extension_runtime()
