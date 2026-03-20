#!/usr/bin/env python3
"""
Phase 3 extension experiment: interaction-aware InstaSHAP surrogate.

Research gap addressed:
The default InstaSHAP-style surrogate in this project is a pure additive GAM
(interactions=0), which can underfit SHAP patterns driven by feature
interactions in the black-box model.

This script runs a controlled before-vs-after comparison:
- Baseline: additive SHAP surrogate (interactions=0)
- Improved: interaction-aware surrogate (interactions>0)

Outputs are saved in Phase3/results.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.black_box_model import BlackBoxModel
from src.data_loader import DatasetLoader
from src.evaluation import SHAPEvaluator
from src.gam_surrogate import SHAPSurrogate
from src.shap_computation import compute_exact_shap
from src.utils import ensure_dir, load_config, set_random_seed, setup_logging


@dataclass
class VariantResult:
    variant: str
    interactions: int
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float
    pearson_correlation: float
    spearman_correlation: float
    top_k_overlap_ratio: float
    ranking_correlation: float
    exact_time_seconds: float
    surrogate_time_seconds: float
    speedup_factor: float


def run_variant(
    variant_name: str,
    interactions: int,
    X_train_shap,
    shap_train,
    X_test_eval,
    shap_test,
    feature_names,
    gam_config,
    top_k: int,
    exact_time_seconds: float,
) -> tuple[VariantResult, pd.DataFrame]:
    """Train and evaluate one surrogate variant."""
    variant_gam_config = dict(gam_config)
    variant_gam_config["interactions"] = interactions

    surrogate = SHAPSurrogate(**variant_gam_config)

    train_start = time.time()
    surrogate.train(X_train_shap, shap_train, feature_names=feature_names, verbose=False)
    train_seconds = time.time() - train_start

    pred_shap, surrogate_time = surrogate.predict_shap(X_test_eval, return_time=True)

    evaluator = SHAPEvaluator(feature_names)
    accuracy = evaluator.compute_accuracy_metrics(shap_test, pred_shap)
    speed = evaluator.compute_speed_metrics(
        exact_time=exact_time_seconds,
        surrogate_time=surrogate_time,
        n_samples=X_test_eval.shape[0],
    )
    ranking = evaluator.compare_feature_rankings(shap_test, pred_shap, top_k=top_k)
    per_feature = evaluator.compute_per_feature_metrics(shap_test, pred_shap)
    per_feature.insert(0, "variant", variant_name)

    result = VariantResult(
        variant=variant_name,
        interactions=interactions,
        mse=accuracy["mse"],
        mae=accuracy["mae"],
        rmse=accuracy["rmse"],
        r2=accuracy["r2"],
        mape=accuracy["mape"],
        pearson_correlation=accuracy["pearson_correlation"],
        spearman_correlation=accuracy["spearman_correlation"],
        top_k_overlap_ratio=ranking["top_k_overlap_ratio"],
        ranking_correlation=ranking["ranking_correlation"],
        exact_time_seconds=speed["exact_time_seconds"],
        surrogate_time_seconds=speed["surrogate_time_seconds"],
        speedup_factor=speed["speedup_factor"],
    )

    print(
        f"[{variant_name}] interactions={interactions} | "
        f"R2={result.r2:.4f}, MAE={result.mae:.5f}, "
        f"Pearson={result.pearson_correlation:.4f}, "
        f"Speedup={result.speedup_factor:.1f}x, TrainTime={train_seconds:.2f}s"
    )

    return result, per_feature


def plot_before_after(results_df: pd.DataFrame, output_png: Path) -> None:
    """Save a compact before-vs-after metric figure."""
    baseline = results_df.loc[results_df["variant"] == "baseline_additive"].iloc[0]
    improved = results_df.loc[results_df["variant"] == "improved_interaction_aware"].iloc[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(["Before", "After"], [baseline["r2"], improved["r2"]], color=["#9aa5b1", "#2f855a"])
    axes[0].set_title("R2 (Higher is better)")
    axes[0].set_ylim(min(-0.1, baseline["r2"], improved["r2"]) - 0.05, 1.0)

    axes[1].bar(["Before", "After"], [baseline["mae"], improved["mae"]], color=["#9aa5b1", "#2f855a"])
    axes[1].set_title("MAE (Lower is better)")

    axes[2].bar(
        ["Before", "After"],
        [baseline["speedup_factor"], improved["speedup_factor"]],
        color=["#9aa5b1", "#2f855a"],
    )
    axes[2].set_title("Speedup vs exact SHAP")

    fig.suptitle("Phase 3: Additive vs Interaction-Aware Surrogate", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 research-gap experiment")
    parser.add_argument("--dataset", default="california_housing", choices=["adult", "california_housing", "breast_cancer"])
    parser.add_argument("--model-type", default="random_forest", choices=["random_forest", "xgboost", "lightgbm"])
    parser.add_argument("--train-sample-size", type=int, default=800)
    parser.add_argument("--test-sample-size", type=int, default=400)
    parser.add_argument("--improved-interactions", type=int, default=8)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    setup_logging("WARNING")
    config = load_config(args.config)
    set_random_seed(config["random_seed"])

    output_root = Path("Phase3/results")
    ensure_dir(str(output_root / "tables"))
    ensure_dir(str(output_root / "figures"))

    dataset_cfg = config["datasets"][args.dataset]
    loader = DatasetLoader(
        dataset_name=args.dataset,
        test_size=dataset_cfg["test_size"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = loader.load_data()
    feature_names = loader.get_feature_names()
    task_type = loader.task_type

    model_cfg = config["black_box_models"][args.model_type][task_type]
    black_box = BlackBoxModel(model_type=args.model_type, task=task_type, **model_cfg)
    black_box.train(X_train, y_train)

    train_n = min(args.train_sample_size, X_train.shape[0])
    test_n = min(args.test_sample_size, X_test.shape[0])

    # Deterministic subset selection avoids train-SHAP alignment issues.
    X_train_shap = X_train[:train_n]
    X_test_eval = X_test[:test_n]

    shap_model_type = "tree" if args.model_type in {"random_forest", "xgboost", "lightgbm"} else "kernel"
    background_size = config["shap_config"]["background_size"]

    train_exact_start = time.time()
    shap_train, _ = compute_exact_shap(
        model=black_box.get_model(),
        X_data=X_train_shap,
        model_type=shap_model_type,
        sample_size=None,
        background_size=background_size,
        cache_path=None,
    )
    _ = time.time() - train_exact_start

    test_exact_start = time.time()
    shap_test, _ = compute_exact_shap(
        model=black_box.get_model(),
        X_data=X_test_eval,
        model_type=shap_model_type,
        sample_size=None,
        background_size=background_size,
        cache_path=None,
    )
    exact_time_seconds = time.time() - test_exact_start

    gam_config = dict(config["gam_config"])
    top_k = config["evaluation"]["top_k_features"]

    baseline_result, baseline_per_feature = run_variant(
        variant_name="baseline_additive",
        interactions=0,
        X_train_shap=X_train_shap,
        shap_train=shap_train,
        X_test_eval=X_test_eval,
        shap_test=shap_test,
        feature_names=feature_names,
        gam_config=gam_config,
        top_k=top_k,
        exact_time_seconds=exact_time_seconds,
    )

    improved_result, improved_per_feature = run_variant(
        variant_name="improved_interaction_aware",
        interactions=args.improved_interactions,
        X_train_shap=X_train_shap,
        shap_train=shap_train,
        X_test_eval=X_test_eval,
        shap_test=shap_test,
        feature_names=feature_names,
        gam_config=gam_config,
        top_k=top_k,
        exact_time_seconds=exact_time_seconds,
    )

    results_df = pd.DataFrame([asdict(baseline_result), asdict(improved_result)])
    results_df.insert(0, "dataset", args.dataset)
    results_df.insert(1, "model_type", args.model_type)
    results_df.insert(2, "train_samples", train_n)
    results_df.insert(3, "test_samples", test_n)

    per_feature_df = pd.concat([baseline_per_feature, improved_per_feature], ignore_index=True)

    summary_path = output_root / "tables" / "before_after_summary.csv"
    per_feature_path = output_root / "tables" / "before_after_per_feature.csv"
    figure_path = output_root / "figures" / "before_after_metrics.png"

    results_df.to_csv(summary_path, index=False)
    per_feature_df.to_csv(per_feature_path, index=False)
    plot_before_after(results_df, figure_path)

    baseline_r2 = float(results_df.loc[results_df["variant"] == "baseline_additive", "r2"].iloc[0])
    improved_r2 = float(results_df.loc[results_df["variant"] == "improved_interaction_aware", "r2"].iloc[0])
    delta_r2 = improved_r2 - baseline_r2

    baseline_mae = float(results_df.loc[results_df["variant"] == "baseline_additive", "mae"].iloc[0])
    improved_mae = float(results_df.loc[results_df["variant"] == "improved_interaction_aware", "mae"].iloc[0])
    mae_reduction_pct = ((baseline_mae - improved_mae) / baseline_mae * 100.0) if baseline_mae > 0 else 0.0

    print("\nSaved:")
    print(f"- {summary_path}")
    print(f"- {per_feature_path}")
    print(f"- {figure_path}")
    print(
        f"\nBefore vs After summary: Delta R2={delta_r2:.4f}, "
        f"MAE reduction={mae_reduction_pct:.2f}%"
    )


if __name__ == "__main__":
    main()
