"""Show where additive-only InstaSHAP breaks on an interaction-heavy benchmark."""

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
from phase2.utils import (
    PHASE2_ROOT,
    compute_alignment_metrics,
    configure_plotting,
    save_dataframe,
    save_figure,
    seed_everything,
    select_background_frame,
)


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
        default=PHASE2_ROOT.parent / "phase3" / "results" / "gap_demonstration",
    )
    return parser.parse_args()


def run_experiment(args: argparse.Namespace) -> None:
    """Execute the gap-demonstration experiment."""
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

    additive_explainer = InstaSHAPExplainer(
        black_box_model=model_bundle.model,
        task=bundle.task,
        feature_names=bundle.feature_names,
    ).fit(bundle.X_train)
    background = select_background_frame(bundle.X_train, max_rows=args.background_size)
    X_explain = bundle.X_test.head(args.explain_samples).copy()
    exact = compute_exact_shap(
        model=model_bundle.model,
        X_background=background,
        X_explain=X_explain,
        task=bundle.task,
        feature_names=bundle.feature_names,
    )
    additive_values = additive_explainer.explain(X_explain)
    additive_fidelity = additive_explainer.fidelity(bundle.X_test)
    alignment = compute_alignment_metrics(exact.values, additive_values.values).as_dict()

    summary_df = pd.DataFrame(
        [
            {
                "dataset": bundle.name,
                "surrogate": "additive_gam",
                **additive_fidelity,
                **alignment,
            }
        ]
    )
    save_dataframe(summary_df, output_dir / "gap_summary.csv", index=False)
    save_dataframe(exact.values, output_dir / "exact_shap_values.csv", index=True)
    save_dataframe(additive_values.values, output_dir / "original_instashap_values.csv", index=True)

    black_box_pred = predict_black_box(model_bundle.model, bundle.X_test, task=bundle.task)
    surrogate_pred = additive_explainer.surrogate.predict(bundle.X_test)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=black_box_pred, y=surrogate_pred, alpha=0.35)
    plt.title("Gap Demonstration: Additive Surrogate Fidelity on Friedman1")
    plt.xlabel("Black-box prediction")
    plt.ylabel("Additive surrogate prediction")
    save_figure(output_dir / "gap_surrogate_fidelity_scatter.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=exact.values.to_numpy().ravel(),
        y=additive_values.values.to_numpy().ravel(),
        alpha=0.35,
    )
    plt.title("Gap Demonstration: Original InstaSHAP vs Exact SHAP")
    plt.xlabel("Exact SHAP value")
    plt.ylabel("Original InstaSHAP value")
    save_figure(output_dir / "gap_shap_alignment_scatter.png")


def main() -> None:
    """Script entry point."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
