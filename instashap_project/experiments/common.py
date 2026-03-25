"""Shared orchestration for the tabular InstaSHAP experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from instashap_project.data.loaders import DatasetBundle
from instashap_project.data.preprocessing import TabularPreprocessor, make_splits
from instashap_project.training.evaluate import evaluate_supervised_model, predict_targets
from instashap_project.training.train import (
    TrainingResult,
    train_blackbox_model,
    train_gam_model,
    train_instashap_model,
    train_masked_surrogate,
)
from instashap_project.utils.metrics import benchmark_callable, explanation_error
from instashap_project.utils.reproducibility import ensure_dir, resolve_device, write_json
from instashap_project.utils.visualization import (
    plot_explanation_alignment,
    plot_feature_importance,
    plot_interaction_heatmap,
    plot_metric_bars,
    plot_shape_function,
    plot_training_curves,
)
from instashap_project.xai.instashap_explainer import InstaSHAPExplainer
from instashap_project.xai.shap_wrapper import ShapBaselineExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class ExperimentResult:
    """Serialized summary of an experiment run."""

    dataset: str
    summary_path: Path
    metrics_table_path: Path
    paper_comparison_path: Path
    plots: list[str]


def _reshape_targets(target: pd.Series, task: str) -> np.ndarray:
    if task == "classification":
        return target.to_numpy(dtype=np.float32).reshape(-1, 1)
    return target.to_numpy(dtype=np.float32).reshape(-1, 1)


def _labels(target: pd.Series, task: str) -> np.ndarray:
    dtype = np.int64 if task == "classification" else np.float32
    return target.to_numpy(dtype=dtype)


def _select_model_flags(model_name: str) -> dict[str, bool]:
    normalized = model_name.lower()
    if normalized == "all":
        return {"blackbox": True, "gam": True, "shap": True, "instashap": True}
    if normalized == "gam":
        return {"blackbox": True, "gam": True, "shap": False, "instashap": False}
    if normalized == "shap":
        return {"blackbox": True, "gam": False, "shap": True, "instashap": False}
    if normalized == "instashap":
        return {"blackbox": True, "gam": False, "shap": True, "instashap": True}
    if normalized == "blackbox":
        return {"blackbox": True, "gam": False, "shap": False, "instashap": False}
    raise ValueError(f"Unsupported model selector: {model_name}")


def _select_output_per_sample(values: np.ndarray, output_indices: np.ndarray) -> np.ndarray:
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[2] == 1:
        return values[:, :, 0]
    selected = np.zeros((values.shape[0], values.shape[1]), dtype=np.float32)
    for sample_index, output_index in enumerate(output_indices):
        selected[sample_index, :] = values[sample_index, :, int(output_index)]
    return selected


def _primary_metric_name(task: str) -> str:
    return "nmse_pct" if task == "regression" else "accuracy"


def run_tabular_experiment(
    bundle: DatasetBundle,
    config: dict[str, Any],
    selected_model: str,
    focus_features: list[str],
    focus_interaction: tuple[str, str] | None = None,
) -> ExperimentResult:
    """Run one full dataset experiment and persist results."""

    dataset_name = bundle.metadata.name
    flags = _select_model_flags(selected_model)
    device = resolve_device(config["global"]["device"])
    seed = int(config["global"]["seed"])
    dataset_cfg = config["datasets"][dataset_name]
    training_cfg = config["training"]

    if config["global"].get("fast_dev_run", False):
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["max_rows"] = min(dataset_cfg.get("max_rows") or 4000, 4000)
        for key in ("blackbox", "gam", "surrogate", "instashap"):
            training_cfg[key] = dict(training_cfg[key])
            training_cfg[key]["epochs"] = min(int(training_cfg[key]["epochs"]), 4)

    bundle = bundle.sample(max_rows=dataset_cfg.get("max_rows"), seed=seed)
    splits = make_splits(
        bundle,
        test_size=float(dataset_cfg["test_size"]),
        val_size=float(dataset_cfg["val_size"]),
        seed=seed,
    )
    preprocessor = TabularPreprocessor(bundle.metadata).fit(splits.X_train)
    X_train = preprocessor.transform(splits.X_train)
    X_val = preprocessor.transform(splits.X_val)
    X_test = preprocessor.transform(splits.X_test)

    y_train_fit = _reshape_targets(splits.y_train, bundle.metadata.task)
    y_val_fit = _reshape_targets(splits.y_val, bundle.metadata.task)
    y_test_labels = _labels(splits.y_test, bundle.metadata.task)
    y_train_labels = _labels(splits.y_train, bundle.metadata.task)

    output_dim = 1 if bundle.metadata.task == "regression" else int(np.unique(y_train_labels).size)
    interactions = [tuple(pair) for pair in dataset_cfg.get("interaction_pairs", [])]
    results_dir = ensure_dir(PROJECT_ROOT / config["global"]["output_root"])
    tables_dir = ensure_dir(results_dir / "tables")
    plots_dir = ensure_dir(results_dir / "plots" / dataset_name)
    artifacts_dir = ensure_dir(results_dir / "artifacts" / dataset_name)

    metrics_rows: list[dict[str, float | str]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    saved_plots: list[str] = []

    blackbox_result: TrainingResult | None = None
    gam1_result: TrainingResult | None = None
    gam2_result: TrainingResult | None = None
    surrogate_result: TrainingResult | None = None
    instashap_result: TrainingResult | None = None

    if flags["blackbox"]:
        start = perf_counter()
        blackbox_result = train_blackbox_model(
            task=bundle.metadata.task,
            input_dim=preprocessor.input_dim,
            output_dim=output_dim,
            X_train=X_train,
            y_train=y_train_fit,
            X_val=X_val,
            y_val=y_val_fit,
            config=training_cfg["blackbox"],
            device=device,
            seed=seed,
            log_dir=artifacts_dir / "blackbox_logs",
        )
        training_seconds = perf_counter() - start
        histories["blackbox"] = blackbox_result.history
        metric_row = {"model": "blackbox", **evaluate_supervised_model(bundle.metadata.task, blackbox_result.model, X_test, y_test_labels, device)}
        inference_stats = benchmark_callable(lambda: predict_targets(bundle.metadata.task, blackbox_result.model, X_test[:512], device))
        metric_row["training_seconds"] = training_seconds
        metric_row["inference_seconds_mean"] = inference_stats["seconds_mean"]
        metrics_rows.append(metric_row)

    if flags["gam"]:
        start = perf_counter()
        gam1_result = train_gam_model(
            task=bundle.metadata.task,
            preprocessor=preprocessor,
            output_dim=output_dim,
            X_train=X_train,
            y_train=y_train_fit,
            X_val=X_val,
            y_val=y_val_fit,
            config=training_cfg["gam"],
            interactions=[],
            device=device,
            log_dir=artifacts_dir / "gam1_logs",
        )
        training_seconds = perf_counter() - start
        histories["gam1"] = gam1_result.history
        metric_row = {"model": "gam1", **evaluate_supervised_model(bundle.metadata.task, gam1_result.model, X_test, y_test_labels, device)}
        metric_row["training_seconds"] = training_seconds
        metric_row["inference_seconds_mean"] = benchmark_callable(lambda: predict_targets(bundle.metadata.task, gam1_result.model, X_test[:512], device))["seconds_mean"]
        metrics_rows.append(metric_row)

        if interactions:
            start = perf_counter()
            gam2_result = train_gam_model(
                task=bundle.metadata.task,
                preprocessor=preprocessor,
                output_dim=output_dim,
                X_train=X_train,
                y_train=y_train_fit,
                X_val=X_val,
                y_val=y_val_fit,
                config=training_cfg["gam"],
                interactions=interactions,
                device=device,
                log_dir=artifacts_dir / "gam2_logs",
            )
            training_seconds = perf_counter() - start
            histories["gam2"] = gam2_result.history
            metric_row = {"model": "gam2", **evaluate_supervised_model(bundle.metadata.task, gam2_result.model, X_test, y_test_labels, device)}
            metric_row["training_seconds"] = training_seconds
            metric_row["inference_seconds_mean"] = benchmark_callable(lambda: predict_targets(bundle.metadata.task, gam2_result.model, X_test[:512], device))["seconds_mean"]
            metrics_rows.append(metric_row)

    shap_selected: np.ndarray | None = None
    instashap_selected: np.ndarray | None = None
    explanation_rows: list[dict[str, float | str]] = []

    if flags["instashap"]:
        if blackbox_result is None:
            raise RuntimeError("InstaSHAP requires the black-box baseline to be trained first.")
        start = perf_counter()
        surrogate_result = train_masked_surrogate(
            blackbox_model=blackbox_result.model,
            preprocessor=preprocessor,
            X_train=X_train,
            X_val=X_val,
            config=training_cfg["surrogate"],
            device=device,
            seed=seed,
            log_dir=artifacts_dir / "surrogate_logs",
        )
        training_seconds = perf_counter() - start
        histories["surrogate"] = surrogate_result.history
        explanation_rows.append({"model": "surrogate", "training_seconds": training_seconds})

        start = perf_counter()
        instashap_result = train_instashap_model(
            preprocessor=preprocessor,
            surrogate_model=surrogate_result.model,
            X_train=X_train,
            X_val=X_val,
            config=training_cfg["instashap"],
            interactions=interactions,
            device=device,
            seed=seed,
            log_dir=artifacts_dir / "instashap_logs",
        )
        training_seconds = perf_counter() - start
        histories["instashap"] = instashap_result.history
        metric_row = {"model": "instashap", **evaluate_supervised_model(bundle.metadata.task, instashap_result.model, X_test, y_test_labels, device)}
        metric_row["training_seconds"] = training_seconds
        metric_row["inference_seconds_mean"] = benchmark_callable(lambda: predict_targets(bundle.metadata.task, instashap_result.model, X_test[:512], device))["seconds_mean"]
        metrics_rows.append(metric_row)

    shap_eval_size = int(dataset_cfg.get("shap_sample_size", config["global"]["shap_eval_samples"]))
    if flags["shap"] and blackbox_result is not None:
        background_size = min(int(config["global"]["shap_background_size"]), len(X_train))
        evaluation_size = min(shap_eval_size, len(X_test))
        background = X_train[:background_size]
        evaluation_inputs = X_test[:evaluation_size]
        shap_explainer = ShapBaselineExplainer(
            model=blackbox_result.model,
            preprocessor=preprocessor,
            device=str(device),
            max_evals=int(config["global"]["shap_max_evals"]),
        )
        start = perf_counter()
        shap_result = shap_explainer.explain(background, evaluation_inputs)
        shap_time = perf_counter() - start
        blackbox_eval_outputs = predict_targets(bundle.metadata.task, blackbox_result.model, evaluation_inputs, device)
        if bundle.metadata.task == "classification":
            output_indices = blackbox_eval_outputs["predictions"]
        else:
            output_indices = np.zeros(evaluation_size, dtype=int)
        shap_selected = _select_output_per_sample(shap_result.grouped_values, output_indices)
        explanation_rows.append({"model": "shap", "seconds_total": shap_time, "samples": evaluation_size})

        importance_plot = plots_dir / f"{dataset_name}_shap_importance.png"
        plot_feature_importance(shap_selected, bundle.metadata.feature_names, importance_plot, f"{dataset_name.title()} SHAP importance")
        saved_plots.append(str(importance_plot.relative_to(PROJECT_ROOT)))

        if instashap_result is not None:
            instashap_explainer = InstaSHAPExplainer(instashap_result.model, str(device))
            start = perf_counter()
            instashap_values = instashap_explainer.explain(evaluation_inputs).grouped_values
            instashap_time = perf_counter() - start
            instashap_selected = _select_output_per_sample(instashap_values, output_indices)
            explanation_rows.append({"model": "instashap", "seconds_total": instashap_time, "samples": evaluation_size})

            error_values = explanation_error(shap_selected, instashap_selected)
            explanation_rows.append({"model": "shap_vs_instashap", **error_values})
            comparison_plot = plots_dir / f"{dataset_name}_shap_vs_instashap_alignment.png"
            plot_explanation_alignment(
                shap_selected,
                instashap_selected,
                bundle.metadata.feature_names,
                comparison_plot,
                f"{dataset_name.title()} SHAP vs InstaSHAP",
            )
            saved_plots.append(str(comparison_plot.relative_to(PROJECT_ROOT)))

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty:
        primary_metric = _primary_metric_name(bundle.metadata.task)
        metrics_path = tables_dir / f"{dataset_name}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        metric_plot_path = plots_dir / f"{dataset_name}_{primary_metric}.png"
        plot_metric_bars(metrics_df, primary_metric, metric_plot_path, f"{dataset_name.title()} {primary_metric}")
        saved_plots.append(str(metric_plot_path.relative_to(PROJECT_ROOT)))
    else:
        metrics_path = tables_dir / f"{dataset_name}_metrics.csv"
        pd.DataFrame().to_csv(metrics_path, index=False)

    if histories:
        training_plot_path = plots_dir / f"{dataset_name}_training_curves.png"
        plot_training_curves(histories, training_plot_path, f"{dataset_name.title()} training curves")
        saved_plots.append(str(training_plot_path.relative_to(PROJECT_ROOT)))

    additive_model = gam2_result.model if gam2_result is not None else gam1_result.model if gam1_result is not None else instashap_result.model if instashap_result is not None else None
    if additive_model is not None:
        for feature_name in focus_features:
            shape_plot_path = plots_dir / f"{dataset_name}_shape_{feature_name}.png"
            plot_shape_function(
                additive_model,
                preprocessor,
                splits.X_train,
                feature_name,
                shape_plot_path,
                device,
            )
            saved_plots.append(str(shape_plot_path.relative_to(PROJECT_ROOT)))
        if focus_interaction is not None and hasattr(additive_model, "interactions") and focus_interaction in getattr(additive_model, "interactions", []):
            interaction_plot_path = plots_dir / f"{dataset_name}_interaction_{focus_interaction[0]}_{focus_interaction[1]}.png"
            plot_interaction_heatmap(
                additive_model,
                preprocessor,
                splits.X_train,
                focus_interaction[0],
                focus_interaction[1],
                interaction_plot_path,
                device,
            )
            saved_plots.append(str(interaction_plot_path.relative_to(PROJECT_ROOT)))

    paper_rows: list[dict[str, float | str]] = []
    if bundle.metadata.paper_metrics:
        metric_lookup = {row["model"]: row for row in metrics_rows}
        if bundle.metadata.task == "regression":
            paper_rows.extend(
                [
                    {"model": "blackbox", "reproduced": metric_lookup.get("blackbox", {}).get("nmse_pct"), "paper": bundle.metadata.paper_metrics.get("paper_blackbox_nmse_pct")},
                    {"model": "gam1", "reproduced": metric_lookup.get("gam1", {}).get("nmse_pct"), "paper": bundle.metadata.paper_metrics.get("paper_gam1_nmse_pct")},
                    {"model": "gam2", "reproduced": metric_lookup.get("gam2", {}).get("nmse_pct"), "paper": bundle.metadata.paper_metrics.get("paper_low_dim_gam_nmse_pct")},
                ]
            )
        elif dataset_name == "adult":
            paper_rows.extend(
                [
                    {"model": "gam1", "reproduced": metric_lookup.get("gam1", {}).get("accuracy"), "paper": bundle.metadata.paper_metrics.get("paper_vanilla_gam_accuracy")},
                    {"model": "instashap", "reproduced": metric_lookup.get("instashap", {}).get("accuracy"), "paper": bundle.metadata.paper_metrics.get("paper_instashap_gam_accuracy")},
                ]
            )
        else:
            paper_rows.extend(
                [
                    {"model": "blackbox", "reproduced": metric_lookup.get("blackbox", {}).get("accuracy"), "paper": bundle.metadata.paper_metrics.get("paper_blackbox_accuracy")},
                    {"model": "gam1", "reproduced": metric_lookup.get("gam1", {}).get("accuracy"), "paper": bundle.metadata.paper_metrics.get("paper_gam1_accuracy")},
                    {"model": "gam2", "reproduced": metric_lookup.get("gam2", {}).get("accuracy"), "paper": bundle.metadata.paper_metrics.get("paper_low_dim_gam_accuracy")},
                ]
            )
    paper_df = pd.DataFrame(paper_rows)
    paper_path = tables_dir / f"{dataset_name}_paper_comparison.csv"
    paper_df.to_csv(paper_path, index=False)

    explanation_df = pd.DataFrame(explanation_rows)
    explanation_path = tables_dir / f"{dataset_name}_explanation_comparison.csv"
    explanation_df.to_csv(explanation_path, index=False)

    summary_payload = {
        "dataset": dataset_name,
        "task": bundle.metadata.task,
        "device": str(device),
        "features": bundle.metadata.feature_names,
        "interaction_pairs": interactions,
        "metrics_table": str(metrics_path.relative_to(PROJECT_ROOT)),
        "paper_comparison_table": str(paper_path.relative_to(PROJECT_ROOT)),
        "explanation_table": str(explanation_path.relative_to(PROJECT_ROOT)),
        "plots": saved_plots,
        "paper_metadata": bundle.metadata.paper_metrics,
    }
    summary_path = artifacts_dir / f"{dataset_name}_summary.json"
    write_json(summary_path, summary_payload)
    return ExperimentResult(
        dataset=dataset_name,
        summary_path=summary_path,
        metrics_table_path=metrics_path,
        paper_comparison_path=paper_path,
        plots=saved_plots,
    )
