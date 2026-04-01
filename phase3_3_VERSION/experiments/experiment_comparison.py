"""Comprehensive multi-dataset comparison across all explanation methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from phase2.data.data_loader import available_datasets, load_dataset
from phase2.explainers.exact_shap import compute_exact_shap
from phase2.explainers.instashap_explainer import InstaSHAPExplainer
from phase2.explainers.kernel_shap import compute_kernel_shap
from phase2.models.base_model import predict_black_box, train_black_box_model
from phase2.models.gam_surrogate import evaluate_surrogate_fidelity
from phase2.utils import (
    PHASE2_ROOT,
    compute_alignment_metrics,
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
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["friedman1", "diabetes", "breast_cancer"],
        choices=sorted(available_datasets()),
    )
    parser.add_argument("--model-name", default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--explain-samples", type=int, default=96)
    parser.add_argument("--background-size", type=int, default=75)
    parser.add_argument("--kernel-nsamples", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PHASE2_ROOT.parent / "phase3" / "results" / "comparison",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    """Create the cross-dataset comparison table requested in Phase 3."""
    seed_everything()
    configure_plotting()
    output_dir = Path(args.output_dir)
    records: list[dict[str, float | str]] = []

    for dataset_name in args.datasets:
        bundle = load_dataset(dataset_name)
        model_bundle = train_black_box_model(
            X_train=bundle.X_train,
            y_train=bundle.y_train,
            task=bundle.task,
            model_name=args.model_name,
            save_dir=output_dir / "artifacts" / dataset_name,
        )
        background = select_background_frame(bundle.X_train, max_rows=args.background_size)
        X_explain = bundle.X_test.head(args.explain_samples).copy()
        exact_explanation, exact_runtime = timed_call(
            compute_exact_shap,
            model=model_bundle.model,
            X_background=background,
            X_explain=X_explain,
            task=bundle.task,
            feature_names=bundle.feature_names,
        )
        records.append(
            {
                "method": "exact_shap",
                "dataset": dataset_name,
                "accuracy_pearson": 1.0,
                "mae": 0.0,
                "runtime_seconds": exact_runtime,
                "surrogate_r2": float("nan"),
            }
        )

        kernel_explanation, kernel_runtime = timed_call(
            compute_kernel_shap,
            model=model_bundle.model,
            X_background=background,
            X_explain=X_explain,
            task=bundle.task,
            feature_names=bundle.feature_names,
            nsamples=args.kernel_nsamples,
        )
        kernel_alignment = compute_alignment_metrics(exact_explanation.values, kernel_explanation.values).as_dict()
        records.append(
            {
                "method": "kernelshap",
                "dataset": dataset_name,
                "accuracy_pearson": kernel_alignment["pearson"],
                "mae": kernel_alignment["mae"],
                "runtime_seconds": kernel_runtime,
                "surrogate_r2": float("nan"),
            }
        )

        additive_explainer = InstaSHAPExplainer(
            black_box_model=model_bundle.model,
            task=bundle.task,
            feature_names=bundle.feature_names,
        ).fit(bundle.X_train)
        additive_explanation, additive_runtime = timed_call(additive_explainer.explain, X_explain)
        additive_alignment = compute_alignment_metrics(exact_explanation.values, additive_explanation.values).as_dict()
        additive_fidelity = additive_explainer.fidelity(bundle.X_test)
        records.append(
            {
                "method": "original_instashap",
                "dataset": dataset_name,
                "accuracy_pearson": additive_alignment["pearson"],
                "mae": additive_alignment["mae"],
                "runtime_seconds": additive_runtime,
                "surrogate_r2": additive_fidelity["r2"],
            }
        )

        interaction_pairs = [("x_1", "x_2")] if dataset_name == "friedman1" else None
        interaction_bundle = train_interaction_aware_surrogate(
            X_train=bundle.X_train,
            black_box_predictions=predict_black_box(model_bundle.model, bundle.X_train, task=bundle.task),
            feature_names=bundle.feature_names,
            interaction_pairs=interaction_pairs,
            interaction_count=3,
            save_dir=output_dir / "artifacts" / f"{dataset_name}_interaction",
        )
        enhanced_explanation, enhanced_runtime = timed_call(
            compute_interaction_aware_instashap,
            surrogate=interaction_bundle.surrogate,
            X=X_explain,
            reference_data=bundle.X_train,
            feature_names=bundle.feature_names,
        )
        enhanced_alignment = compute_alignment_metrics(exact_explanation.values, enhanced_explanation.values).as_dict()
        enhanced_fidelity = evaluate_surrogate_fidelity(
            surrogate=interaction_bundle.surrogate,
            X_eval=bundle.X_test,
            black_box_predictions=predict_black_box(model_bundle.model, bundle.X_test, task=bundle.task),
        )
        records.append(
            {
                "method": "interaction_aware_instashap",
                "dataset": dataset_name,
                "accuracy_pearson": enhanced_alignment["pearson"],
                "mae": enhanced_alignment["mae"],
                "runtime_seconds": enhanced_runtime,
                "surrogate_r2": enhanced_fidelity["r2"],
            }
        )

    comparison_df = pd.DataFrame(records)
    save_dataframe(comparison_df, output_dir / "comparison_summary.csv", index=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x="dataset", y="accuracy_pearson", hue="method")
    plt.title("Cross-Dataset Accuracy Comparison")
    plt.xlabel("Dataset")
    plt.ylabel("Pearson correlation with Exact SHAP")
    save_figure(output_dir / "comparison_accuracy.png")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=comparison_df, x="dataset", y="runtime_seconds", hue="method")
    plt.title("Cross-Dataset Runtime Comparison")
    plt.xlabel("Dataset")
    plt.ylabel("Runtime (seconds)")
    save_figure(output_dir / "comparison_runtime.png")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
