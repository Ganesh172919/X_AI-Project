"""Accuracy comparison for the interaction-aware InstaSHAP extension."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from phase2.data.data_loader import load_dataset
from phase2.explainers.exact_shap import compute_exact_shap
from phase2.explainers.instashap_explainer import InstaSHAPExplainer
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
)
from phase3.extension.enhanced_instashap import compute_interaction_aware_instashap
from phase3.extension.interaction_aware_surrogate import train_interaction_aware_surrogate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="friedman1", choices=["friedman1"])
    parser.add_argument("--model-name", default="xgboost", choices=["xgboost", "random_forest"])
    parser.add_argument("--explain-samples", type=int, default=128)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PHASE2_ROOT.parent / "phase3" / "results" / "extension_accuracy",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    """Compare additive and interaction-aware InstaSHAP against Exact SHAP."""
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
    X_explain = bundle.X_test.head(args.explain_samples).copy()
    exact = compute_exact_shap(
        model=model_bundle.model,
        X_background=background,
        X_explain=X_explain,
        task=bundle.task,
        feature_names=bundle.feature_names,
    )

    additive_explainer = InstaSHAPExplainer(
        black_box_model=model_bundle.model,
        task=bundle.task,
        feature_names=bundle.feature_names,
    ).fit(bundle.X_train)
    additive_output = additive_explainer.explain(X_explain)

    train_predictions = predict_black_box(model_bundle.model, bundle.X_train, task=bundle.task)
    test_predictions = predict_black_box(model_bundle.model, bundle.X_test, task=bundle.task)
    interaction_bundle = train_interaction_aware_surrogate(
        X_train=bundle.X_train,
        black_box_predictions=train_predictions,
        feature_names=bundle.feature_names,
        interaction_pairs=[("x_1", "x_2")],
        interaction_count=3,
        save_dir=output_dir / "artifacts" / "interaction_surrogate",
    )
    enhanced_output = compute_interaction_aware_instashap(
        surrogate=interaction_bundle.surrogate,
        X=X_explain,
        reference_data=bundle.X_train,
        feature_names=bundle.feature_names,
    )

    additive_alignment = compute_alignment_metrics(exact.values, additive_output.values).as_dict()
    enhanced_alignment = compute_alignment_metrics(exact.values, enhanced_output.values).as_dict()
    additive_fidelity = additive_explainer.fidelity(bundle.X_test)
    enhanced_fidelity = evaluate_surrogate_fidelity(
        surrogate=interaction_bundle.surrogate,
        X_eval=bundle.X_test,
        black_box_predictions=test_predictions,
    )

    comparison_df = pd.DataFrame(
        [
            {
                "method": "original_instashap",
                "dataset": bundle.name,
                **additive_fidelity,
                **additive_alignment,
            },
            {
                "method": "interaction_aware_instashap",
                "dataset": bundle.name,
                **enhanced_fidelity,
                **enhanced_alignment,
            },
        ]
    )
    save_dataframe(comparison_df, output_dir / "extension_accuracy_summary.csv", index=False)
    save_dataframe(exact.values, output_dir / "exact_shap_values.csv", index=True)
    save_dataframe(additive_output.values, output_dir / "original_instashap_values.csv", index=True)
    save_dataframe(enhanced_output.values, output_dir / "interaction_aware_instashap_values.csv", index=True)

    metric_long_df = comparison_df.melt(
        id_vars=["method", "dataset"],
        value_vars=["pearson", "spearman", "mae", "r2"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metric_long_df, x="metric", y="value", hue="method")
    plt.title("Extension Accuracy Metrics on the Interaction-Heavy Dataset")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    save_figure(output_dir / "extension_accuracy_metrics.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=exact.values.to_numpy().ravel(),
        y=enhanced_output.values.to_numpy().ravel(),
        alpha=0.35,
        label="Interaction-aware InstaSHAP",
    )
    sns.scatterplot(
        x=exact.values.to_numpy().ravel(),
        y=additive_output.values.to_numpy().ravel(),
        alpha=0.20,
        label="Original InstaSHAP",
    )
    plt.title("Exact SHAP vs Original and Interaction-Aware InstaSHAP")
    plt.xlabel("Exact SHAP value")
    plt.ylabel("Approximate SHAP value")
    save_figure(output_dir / "extension_accuracy_scatter.png")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
