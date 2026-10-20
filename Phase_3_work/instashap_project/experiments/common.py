"""Phase 3 experiment orchestration for the Covertype extension study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from instashap_project.data.loaders import DatasetBundle
from instashap_project.data.preprocessing import TabularPreprocessor, make_splits
from instashap_project.masking import MaskingConfig, build_background_bank, build_masked_batch
from instashap_project.training.evaluate import evaluate_supervised_model, predict_targets
from instashap_project.training.train import (
    TrainingResult,
    mean_blackbox_outputs,
    mean_surrogate_outputs,
    sample_shapley_feature_masks,
    train_blackbox_model,
    train_gam_model,
    train_instashap_model,
    train_masked_surrogate,
)
from instashap_project.utils.logging_utils import format_log_event, get_logger
from instashap_project.utils.metrics import benchmark_callable, explanation_metrics
from instashap_project.utils.reproducibility import ensure_dir, resolve_device, set_global_seed, write_json
from instashap_project.utils.visualization import (
    plot_explanation_alignment,
    plot_feature_importance,
    plot_interaction_heatmap,
    plot_named_metric_bars,
    plot_shape_function,
    plot_training_curves,
)
from instashap_project.xai.instashap_explainer import InstaSHAPExplainer
from instashap_project.xai.shap_wrapper import ShapBaselineExplainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = get_logger(__name__)


@dataclass(slots=True)
class ExperimentResult:
    """Top-level experiment output paths."""

    dataset: str
    summary_path: Path
    tables: dict[str, str]
    plots: list[str]
    reports: dict[str, str]


def _reshape_targets(target: pd.Series, task: str) -> np.ndarray:
    dtype = np.int64 if task == "classification" else np.float32
    return target.to_numpy(dtype=dtype).reshape(-1, 1)


def _labels(target: pd.Series, task: str) -> np.ndarray:
    dtype = np.int64 if task == "classification" else np.float32
    return target.to_numpy(dtype=dtype)


def _numeric_summary(frame: pd.DataFrame, group_key: str) -> pd.DataFrame:
    numeric_columns = [column for column in frame.columns if column != group_key and pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric_columns:
        return pd.DataFrame()
    aggregated = frame.groupby(group_key)[numeric_columns].agg(["mean", "std"]).reset_index()
    aggregated.columns = [group_key] + [f"{name}_{agg}" for name, agg in aggregated.columns.tolist()[1:]]
    return aggregated.sort_values(group_key).reset_index(drop=True)


def _masking_configs(config: dict[str, Any]) -> dict[str, MaskingConfig]:
    masking_root = config["masking"]
    return {
        "zero_mask": MaskingConfig(
            strategy="zero_mask",
            background_bank_size=int(masking_root["background_bank_size"]),
            background_samples_train=1,
            background_samples_eval=1,
            seed=int(masking_root["seed"]),
        ),
        "empirical_background": MaskingConfig(
            strategy="empirical_background",
            background_bank_size=int(masking_root["background_bank_size"]),
            background_samples_train=int(masking_root["background_samples_train"]),
            background_samples_eval=int(masking_root["background_samples_eval"]),
            seed=int(masking_root["seed"]),
        ),
    }


def _selected_variants(variant: str) -> list[str]:
    if variant == "baseline":
        return ["zero_mask"]
    if variant == "improved":
        return ["empirical_background"]
    if variant == "compare":
        return ["zero_mask", "empirical_background"]
    raise ValueError(f"Unsupported variant: {variant}")


def _variant_label(strategy: str) -> str:
    return "instashap_zero" if strategy == "zero_mask" else "instashap_bg"


def _surrogate_label(strategy: str) -> str:
    return "surrogate_zero" if strategy == "zero_mask" else "surrogate_bg"


def _fidelity_label(strategy: str) -> str:
    return "zero_mask" if strategy == "zero_mask" else "empirical_background"


def evaluate_coalition_fidelity(
    *,
    blackbox_model: object,
    surrogate_model: object,
    preprocessor: TabularPreprocessor,
    transformed_inputs: np.ndarray,
    masking_config: MaskingConfig,
    background_bank: np.ndarray | None,
    device: Any,
    seed: int,
    num_masks: int,
    edge_mask_probability: float,
) -> dict[str, float]:
    """Evaluate how well a surrogate matches the coalition value function."""

    rng = np.random.default_rng(seed)
    feature_mask = sample_shapley_feature_masks(
        batch_size=len(transformed_inputs),
        num_features=preprocessor.num_original_features,
        rng=rng,
        edge_mask_probability=edge_mask_probability,
    )
    masked_inputs = build_masked_batch(
        preprocessor=preprocessor,
        transformed_inputs=transformed_inputs,
        feature_mask=feature_mask,
        strategy=masking_config.strategy,
        rng=rng,
        background_bank=background_bank,
        background_samples=num_masks,
    )
    blackbox_outputs = mean_blackbox_outputs(blackbox_model, masked_inputs, device=device)
    surrogate_outputs = mean_surrogate_outputs(
        surrogate_model,
        masked_inputs,
        feature_mask,
        transformed_inputs,
        device=device,
    )
    difference = surrogate_outputs - blackbox_outputs
    return {
        "mse": float(np.mean(np.square(difference))),
        "mae": float(np.mean(np.abs(difference))),
    }


def run_phase3_experiment(
    *,
    bundle: DatasetBundle,
    config: dict[str, Any],
    variant: str,
) -> ExperimentResult:
    """Run the full Phase 3 Covertype experiment and persist outputs."""

    dataset_name = bundle.metadata.name
    LOGGER.info(format_log_event("phase3.start", dataset=dataset_name, variant=variant))

    global_cfg = config["global"]
    dataset_cfg = dict(config["dataset"])
    training_cfg = {name: dict(values) for name, values in config["training"].items()}
    masking_cfgs = _masking_configs(config)
    selected_variants = _selected_variants(variant)
    device = resolve_device(global_cfg["device"])
    results_dir = ensure_dir(PROJECT_ROOT / global_cfg["output_root"])
    tables_dir = ensure_dir(results_dir / "tables")
    plots_dir = ensure_dir(results_dir / "plots" / dataset_name)
    artifacts_dir = ensure_dir(results_dir / "artifacts" / dataset_name)

    if global_cfg.get("fast_dev_run", False):
        dataset_cfg["max_rows"] = min(int(dataset_cfg.get("max_rows") or 12000), 8000)
        dataset_cfg["shap_sample_size"] = min(int(dataset_cfg["shap_sample_size"]), 12)
        dataset_cfg["coalition_eval_size"] = min(int(dataset_cfg["coalition_eval_size"]), 16)
        for name in training_cfg:
            training_cfg[name]["epochs"] = min(int(training_cfg[name]["epochs"]), 4)
        masking_cfgs["empirical_background"].background_samples_train = min(
            masking_cfgs["empirical_background"].background_samples_train,
            2,
        )
        masking_cfgs["empirical_background"].background_samples_eval = min(
            masking_cfgs["empirical_background"].background_samples_eval,
            3,
        )

    predictive_rows: list[dict[str, float | str | int]] = []
    runtime_rows: list[dict[str, float | str | int]] = []
    explanation_rows: list[dict[str, float | str | int]] = []
    coalition_rows: list[dict[str, float | str | int]] = []
    plots: list[str] = []

    representative_shapes_done = False
    representative_alignment_done = {"zero_mask": False, "empirical_background": False}

    for seed in list(global_cfg["seeds"]):
        set_global_seed(int(seed))
        LOGGER.info(format_log_event("seed.start", dataset=dataset_name, seed=seed))
        sampled_bundle = bundle.sample(max_rows=dataset_cfg.get("max_rows"), seed=int(seed))
        splits = make_splits(
            sampled_bundle,
            test_size=float(dataset_cfg["test_size"]),
            val_size=float(dataset_cfg["val_size"]),
            seed=int(seed),
        )
        preprocessor = TabularPreprocessor(sampled_bundle.metadata).fit(splits.X_train)
        X_train = preprocessor.transform(splits.X_train)
        X_val = preprocessor.transform(splits.X_val)
        X_test = preprocessor.transform(splits.X_test)
        y_train_fit = _reshape_targets(splits.y_train, sampled_bundle.metadata.task)
        y_val_fit = _reshape_targets(splits.y_val, sampled_bundle.metadata.task)
        y_test_labels = _labels(splits.y_test, sampled_bundle.metadata.task)
        y_train_labels = _labels(splits.y_train, sampled_bundle.metadata.task)
        output_dim = int(np.unique(y_train_labels).size)
        interactions = [tuple(pair) for pair in dataset_cfg["interaction_pairs"]]

        seed_artifacts = ensure_dir(artifacts_dir / f"seed_{seed}")
        histories: dict[str, list[dict[str, float]]] = {}

        background_bank = build_background_bank(
            X_train,
            max_rows=int(masking_cfgs["empirical_background"].background_bank_size),
            seed=int(seed),
        )

        start = perf_counter()
        blackbox_result = train_blackbox_model(
            task=sampled_bundle.metadata.task,
            input_dim=preprocessor.input_dim,
            output_dim=output_dim,
            X_train=X_train,
            y_train=y_train_fit,
            X_val=X_val,
            y_val=y_val_fit,
            config=training_cfg["blackbox"],
            device=device,
            seed=int(seed),
            log_dir=seed_artifacts / "blackbox_logs",
        )
        blackbox_training_seconds = perf_counter() - start
        histories["blackbox"] = blackbox_result.history
        blackbox_metrics = evaluate_supervised_model(sampled_bundle.metadata.task, blackbox_result.model, X_test, y_test_labels, device)
        predictive_rows.append({"seed": int(seed), "model": "blackbox", **blackbox_metrics})
        runtime_rows.append(
            {
                "seed": int(seed),
                "model": "blackbox",
                "stage": "predict",
                "training_seconds": blackbox_training_seconds,
                "inference_seconds_mean": benchmark_callable(
                    lambda: predict_targets(sampled_bundle.metadata.task, blackbox_result.model, X_test[:512], device)
                )["seconds_mean"],
                "explanation_seconds_total": np.nan,
                "samples": len(X_test),
            }
        )

        start = perf_counter()
        gam1_result = train_gam_model(
            task=sampled_bundle.metadata.task,
            preprocessor=preprocessor,
            output_dim=output_dim,
            X_train=X_train,
            y_train=y_train_fit,
            X_val=X_val,
            y_val=y_val_fit,
            config=training_cfg["gam"],
            interactions=[],
            device=device,
            log_dir=seed_artifacts / "gam1_logs",
        )
        gam1_training_seconds = perf_counter() - start
        histories["gam1"] = gam1_result.history
        gam1_metrics = evaluate_supervised_model(sampled_bundle.metadata.task, gam1_result.model, X_test, y_test_labels, device)
        predictive_rows.append({"seed": int(seed), "model": "gam1", **gam1_metrics})
        runtime_rows.append(
            {
                "seed": int(seed),
                "model": "gam1",
                "stage": "predict",
                "training_seconds": gam1_training_seconds,
                "inference_seconds_mean": benchmark_callable(
                    lambda: predict_targets(sampled_bundle.metadata.task, gam1_result.model, X_test[:512], device)
                )["seconds_mean"],
                "explanation_seconds_total": np.nan,
                "samples": len(X_test),
            }
        )

        start = perf_counter()
        gam2_result = train_gam_model(
            task=sampled_bundle.metadata.task,
            preprocessor=preprocessor,
            output_dim=output_dim,
            X_train=X_train,
            y_train=y_train_fit,
            X_val=X_val,
            y_val=y_val_fit,
            config=training_cfg["gam"],
            interactions=interactions,
            device=device,
            log_dir=seed_artifacts / "gam2_logs",
        )
        gam2_training_seconds = perf_counter() - start
        histories["gam2"] = gam2_result.history
        gam2_metrics = evaluate_supervised_model(sampled_bundle.metadata.task, gam2_result.model, X_test, y_test_labels, device)
        predictive_rows.append({"seed": int(seed), "model": "gam2", **gam2_metrics})
        runtime_rows.append(
            {
                "seed": int(seed),
                "model": "gam2",
                "stage": "predict",
                "training_seconds": gam2_training_seconds,
                "inference_seconds_mean": benchmark_callable(
                    lambda: predict_targets(sampled_bundle.metadata.task, gam2_result.model, X_test[:512], device)
                )["seconds_mean"],
                "explanation_seconds_total": np.nan,
                "samples": len(X_test),
            }
        )

        evaluation_size = min(int(dataset_cfg["shap_sample_size"]), len(X_test))
        background_size = min(int(global_cfg["shap_background_size"]), len(X_train))
        evaluation_inputs = X_test[:evaluation_size]
        shap_explainer = ShapBaselineExplainer(
            model=blackbox_result.model,
            preprocessor=preprocessor,
            device=str(device),
            max_evals=int(global_cfg["shap_max_evals"]),
        )
        start = perf_counter()
        shap_result = shap_explainer.explain(X_train[:background_size], evaluation_inputs)
        shap_seconds_total = perf_counter() - start
        blackbox_eval_outputs = predict_targets(sampled_bundle.metadata.task, blackbox_result.model, evaluation_inputs, device)
        output_indices = blackbox_eval_outputs["predictions"]
        shap_selected = np.zeros((evaluation_size, len(sampled_bundle.metadata.feature_names)), dtype=np.float32)
        for sample_index, output_index in enumerate(output_indices):
            shap_selected[sample_index, :] = shap_result.grouped_values[sample_index, :, int(output_index)]
        runtime_rows.append(
            {
                "seed": int(seed),
                "model": "permutation_shap",
                "stage": "explain",
                "training_seconds": 0.0,
                "inference_seconds_mean": np.nan,
                "explanation_seconds_total": shap_seconds_total,
                "samples": evaluation_size,
            }
        )

        shap_plot_path = plots_dir / f"covertype_seed_{seed}_shap_importance.png"
        plot_feature_importance(
            shap_selected,
            sampled_bundle.metadata.feature_names,
            shap_plot_path,
            f"Covertype Seed {seed} SHAP importance",
        )
        plots.append(str(shap_plot_path.relative_to(PROJECT_ROOT)))

        strategy_models: dict[str, TrainingResult] = {}
        for strategy in selected_variants:
            start = perf_counter()
            surrogate_result = train_masked_surrogate(
                blackbox_model=blackbox_result.model,
                preprocessor=preprocessor,
                X_train=X_train,
                X_val=X_val,
                config=training_cfg["surrogate"],
                masking_config=masking_cfgs[strategy],
                background_bank=background_bank if strategy == "empirical_background" else None,
                device=device,
                seed=int(seed),
                log_dir=seed_artifacts / f"{_surrogate_label(strategy)}_logs",
            )
            surrogate_training_seconds = perf_counter() - start
            histories[_surrogate_label(strategy)] = surrogate_result.history

            start = perf_counter()
            instashap_result = train_instashap_model(
                preprocessor=preprocessor,
                surrogate_model=surrogate_result.model,
                X_train=X_train,
                X_val=X_val,
                config=training_cfg["instashap"],
                interactions=interactions,
                masking_config=masking_cfgs[strategy],
                background_bank=background_bank if strategy == "empirical_background" else None,
                device=device,
                seed=int(seed),
                log_dir=seed_artifacts / f"{_variant_label(strategy)}_logs",
            )
            instashap_training_seconds = perf_counter() - start
            histories[_variant_label(strategy)] = instashap_result.history
            strategy_models[strategy] = instashap_result

            instashap_metrics = evaluate_supervised_model(
                sampled_bundle.metadata.task,
                instashap_result.model,
                X_test,
                y_test_labels,
                device,
            )
            predictive_rows.append({"seed": int(seed), "model": _variant_label(strategy), **instashap_metrics})
            runtime_rows.extend(
                [
                    {
                        "seed": int(seed),
                        "model": _surrogate_label(strategy),
                        "stage": "train",
                        "training_seconds": surrogate_training_seconds,
                        "inference_seconds_mean": np.nan,
                        "explanation_seconds_total": np.nan,
                        "samples": len(X_train),
                    },
                    {
                        "seed": int(seed),
                        "model": _variant_label(strategy),
                        "stage": "predict",
                        "training_seconds": instashap_training_seconds,
                        "inference_seconds_mean": benchmark_callable(
                            lambda: predict_targets(sampled_bundle.metadata.task, instashap_result.model, X_test[:512], device)
                        )["seconds_mean"],
                        "explanation_seconds_total": np.nan,
                        "samples": len(X_test),
                    },
                ]
            )

            instashap_explainer = InstaSHAPExplainer(instashap_result.model, str(device))
            start = perf_counter()
            instashap_values = instashap_explainer.explain(evaluation_inputs).grouped_values
            instashap_seconds_total = perf_counter() - start
            instashap_selected = np.zeros((evaluation_size, len(sampled_bundle.metadata.feature_names)), dtype=np.float32)
            for sample_index, output_index in enumerate(output_indices):
                instashap_selected[sample_index, :] = instashap_values[sample_index, :, int(output_index)]

            explanation_rows.append(
                {
                    "seed": int(seed),
                    "model": _variant_label(strategy),
                    **explanation_metrics(shap_selected, instashap_selected),
                }
            )
            runtime_rows.append(
                {
                    "seed": int(seed),
                    "model": _variant_label(strategy),
                    "stage": "explain",
                    "training_seconds": 0.0,
                    "inference_seconds_mean": np.nan,
                    "explanation_seconds_total": instashap_seconds_total,
                    "samples": evaluation_size,
                }
            )

            coalition_metrics = evaluate_coalition_fidelity(
                blackbox_model=blackbox_result.model,
                surrogate_model=surrogate_result.model,
                preprocessor=preprocessor,
                transformed_inputs=X_test[: min(int(dataset_cfg["coalition_eval_size"]), len(X_test))],
                masking_config=masking_cfgs[strategy],
                background_bank=background_bank if strategy == "empirical_background" else None,
                device=device,
                seed=int(seed),
                num_masks=int(masking_cfgs[strategy].background_samples_eval),
                edge_mask_probability=float(training_cfg["surrogate"]["edge_mask_probability"]),
            )
            coalition_rows.append(
                {
                    "seed": int(seed),
                    "model": _surrogate_label(strategy),
                    "masking_strategy": _fidelity_label(strategy),
                    **coalition_metrics,
                }
            )

            if not representative_alignment_done[strategy]:
                alignment_path = plots_dir / f"covertype_{_variant_label(strategy)}_alignment.png"
                plot_explanation_alignment(
                    shap_selected,
                    instashap_selected,
                    sampled_bundle.metadata.feature_names,
                    alignment_path,
                    f"Covertype SHAP vs {_variant_label(strategy)}",
                )
                plots.append(str(alignment_path.relative_to(PROJECT_ROOT)))
                representative_alignment_done[strategy] = True

            if strategy == "empirical_background" and not representative_shapes_done:
                for feature_name in ("elevation", "soil_climate_zone", "aspect"):
                    shape_path = plots_dir / f"covertype_shape_{feature_name}.png"
                    plot_shape_function(
                        instashap_result.model,
                        preprocessor,
                        splits.X_train,
                        feature_name,
                        shape_path,
                        device,
                    )
                    plots.append(str(shape_path.relative_to(PROJECT_ROOT)))
                interaction_path = plots_dir / "covertype_interaction_elevation_soil_climate_zone.png"
                plot_interaction_heatmap(
                    instashap_result.model,
                    preprocessor,
                    splits.X_train,
                    "elevation",
                    "soil_climate_zone",
                    interaction_path,
                    device,
                )
                plots.append(str(interaction_path.relative_to(PROJECT_ROOT)))
                representative_shapes_done = True

        training_curves_path = plots_dir / f"covertype_seed_{seed}_training_curves.png"
        plot_training_curves(histories, training_curves_path, f"Covertype training curves (seed {seed})")
        plots.append(str(training_curves_path.relative_to(PROJECT_ROOT)))

        seed_payload = {
            "seed": int(seed),
            "dataset": dataset_name,
            "variant": variant,
            "selected_variants": selected_variants,
            "device": str(device),
            "plots": [plot for plot in plots if f"seed_{seed}" in plot],
        }
        write_json(seed_artifacts / "seed_summary.json", seed_payload)
        LOGGER.info(format_log_event("seed.complete", dataset=dataset_name, seed=seed))

    predictive_frame = pd.DataFrame(predictive_rows)
    runtime_frame = pd.DataFrame(runtime_rows)
    explanation_frame = pd.DataFrame(explanation_rows)
    coalition_frame = pd.DataFrame(coalition_rows)

    predictive_summary = _numeric_summary(predictive_frame, "model")
    runtime_summary = _numeric_summary(runtime_frame, "model")
    explanation_summary = _numeric_summary(explanation_frame, "model")
    coalition_summary = _numeric_summary(coalition_frame, "model")

    predictive_path = tables_dir / "covertype_predictive_metrics.csv"
    predictive_summary_path = tables_dir / "covertype_predictive_summary.csv"
    runtime_path = tables_dir / "covertype_runtime_metrics.csv"
    runtime_summary_path = tables_dir / "covertype_runtime_summary.csv"
    explanation_path = tables_dir / "covertype_explanation_fidelity.csv"
    explanation_summary_path = tables_dir / "covertype_explanation_summary.csv"
    coalition_path = tables_dir / "covertype_coalition_fidelity.csv"
    coalition_summary_path = tables_dir / "covertype_coalition_summary.csv"

    predictive_frame.to_csv(predictive_path, index=False)
    predictive_summary.to_csv(predictive_summary_path, index=False)
    runtime_frame.to_csv(runtime_path, index=False)
    runtime_summary.to_csv(runtime_summary_path, index=False)
    explanation_frame.to_csv(explanation_path, index=False)
    explanation_summary.to_csv(explanation_summary_path, index=False)
    coalition_frame.to_csv(coalition_path, index=False)
    coalition_summary.to_csv(coalition_summary_path, index=False)

    accuracy_plot_path = plots_dir / "covertype_accuracy_comparison.png"
    if not predictive_summary.empty and "accuracy_mean" in predictive_summary.columns:
        plot_named_metric_bars(
            predictive_summary,
            x="model",
            y="accuracy_mean",
            output_path=accuracy_plot_path,
            title="Covertype predictive accuracy comparison",
        )
        plots.append(str(accuracy_plot_path.relative_to(PROJECT_ROOT)))

    explanation_plot_path = plots_dir / "covertype_explanation_mae_comparison.png"
    if not explanation_summary.empty and "mae_mean" in explanation_summary.columns:
        plot_named_metric_bars(
            explanation_summary,
            x="model",
            y="mae_mean",
            output_path=explanation_plot_path,
            title="Covertype explanation MAE comparison",
            color="#be185d",
        )
        plots.append(str(explanation_plot_path.relative_to(PROJECT_ROOT)))

    runtime_plot_path = plots_dir / "covertype_explanation_runtime_comparison.png"
    runtime_explain = runtime_frame[runtime_frame["stage"] == "explain"].copy()
    runtime_explain_summary = _numeric_summary(runtime_explain, "model")
    if not runtime_explain_summary.empty and "explanation_seconds_total_mean" in runtime_explain_summary.columns:
        plot_named_metric_bars(
            runtime_explain_summary,
            x="model",
            y="explanation_seconds_total_mean",
            output_path=runtime_plot_path,
            title="Covertype explanation runtime comparison",
            color="#1d4ed8",
        )
        plots.append(str(runtime_plot_path.relative_to(PROJECT_ROOT)))

    summary_payload = {
        "dataset": dataset_name,
        "variant": variant,
        "seeds": list(global_cfg["seeds"]),
        "tables": {
            "predictive_metrics": str(predictive_path.relative_to(PROJECT_ROOT)),
            "predictive_summary": str(predictive_summary_path.relative_to(PROJECT_ROOT)),
            "runtime_metrics": str(runtime_path.relative_to(PROJECT_ROOT)),
            "runtime_summary": str(runtime_summary_path.relative_to(PROJECT_ROOT)),
            "explanation_fidelity": str(explanation_path.relative_to(PROJECT_ROOT)),
            "explanation_summary": str(explanation_summary_path.relative_to(PROJECT_ROOT)),
            "coalition_fidelity": str(coalition_path.relative_to(PROJECT_ROOT)),
            "coalition_summary": str(coalition_summary_path.relative_to(PROJECT_ROOT)),
        },
        "plots": sorted(set(plots)),
    }
    summary_path = artifacts_dir / "covertype_phase3_summary.json"
    write_json(summary_path, summary_payload)
    LOGGER.info(format_log_event("phase3.complete", dataset=dataset_name, summary_path=summary_path))
    return ExperimentResult(
        dataset=dataset_name,
        summary_path=summary_path,
        tables=summary_payload["tables"],
        plots=sorted(set(plots)),
        reports={},
    )
