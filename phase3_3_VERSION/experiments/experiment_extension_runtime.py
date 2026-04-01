"""Runtime comparison including the interaction-aware InstaSHAP extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from phase2.data.data_loader import load_dataset
from phase2.explainers.exact_shap import compute_exact_shap
from phase2.explainers.instashap_explainer import InstaSHAPExplainer
from phase2.explainers.kernel_shap import compute_kernel_shap
from phase2.models.base_model import predict_black_box, train_black_box_model
from phase2.utils import (
    PHASE2_ROOT,
    configure_plotting,
    save_dataframe,
    save_figure,
    seed_everything,
    select_background_frame,
    timed_call,
)
from phase3.extension.enhanced_instashap import compute_interaction_aware_instashap
from phase3.extension.interaction_aware_surrogate import train_interaction_aware_surrogate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="friedman1", choices=["friedman1"])
    parser.add_argument("--model-name", default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[50, 100, 250])
    parser.add_argument("--background-size", type=int, default=75)
    parser.add_argument("--kernel-nsamples", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PHASE2_ROOT.parent / "phase3" / "results" / "extension_runtime",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    """Benchmark Exact SHAP, KernelSHAP, original InstaSHAP, and the extension."""
    seed_everything()
    configure_plotting()
    output_dir = Path(args.output_dir)
    bundle = load_dataset(args.dataset)
    model_bundle = train_black_box_model(
        X_train=bundle.X_train,
        y_train=bundle.y_train,
        task=bundle.task,
        model_name=args.model_name,
        save_dir=output_dir / "artifacts",
    )
    background = select_background_frame(bundle.X_train, max_rows=args.background_size)

    additive_explainer = InstaSHAPExplainer(
        black_box_model=model_bundle.model,
        task=bundle.task,
        feature_names=bundle.feature_names,
    ).fit(bundle.X_train)
    interaction_bundle = train_interaction_aware_surrogate(
        X_train=bundle.X_train,
        black_box_predictions=predict_black_box(model_bundle.model, bundle.X_train, task=bundle.task),
        feature_names=bundle.feature_names,
        interaction_pairs=[("x_1", "x_2")],
        interaction_count=3,
        save_dir=output_dir / "artifacts" / "interaction_surrogate",
    )

    records: list[dict[str, float | int | str]] = []
    for n_samples in args.sample_sizes:
        X_explain = bundle.X_test.head(min(n_samples, len(bundle.X_test))).copy()
        _, elapsed = timed_call(
            compute_exact_shap,
            model=model_bundle.model,
            X_background=background,
            X_explain=X_explain,
            task=bundle.task,
            feature_names=bundle.feature_names,
        )
        records.append({"method": "exact_shap", "n_samples": len(X_explain), "runtime_seconds": elapsed})

        _, elapsed = timed_call(
            compute_kernel_shap,
            model=model_bundle.model,
            X_background=background,
            X_explain=X_explain,
            task=bundle.task,
            feature_names=bundle.feature_names,
            nsamples=args.kernel_nsamples,
        )
        records.append({"method": "kernelshap", "n_samples": len(X_explain), "runtime_seconds": elapsed})

        _, elapsed = timed_call(additive_explainer.explain, X_explain)
        records.append({"method": "original_instashap", "n_samples": len(X_explain), "runtime_seconds": elapsed})

        _, elapsed = timed_call(
            compute_interaction_aware_instashap,
            surrogate=interaction_bundle.surrogate,
            X=X_explain,
            reference_data=bundle.X_train,
            feature_names=bundle.feature_names,
        )
        records.append(
            {
                "method": "interaction_aware_instashap",
                "n_samples": len(X_explain),
                "runtime_seconds": elapsed,
            }
        )

    runtime_df = pd.DataFrame(records)
    save_dataframe(runtime_df, output_dir / "extension_runtime.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=runtime_df, x="n_samples", y="runtime_seconds", hue="method", marker="o")
    plt.title("Runtime Comparison with the Interaction-Aware Extension")
    plt.xlabel("Number of explained samples")
    plt.ylabel("Runtime (seconds)")
    save_figure(output_dir / "extension_runtime.png")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
