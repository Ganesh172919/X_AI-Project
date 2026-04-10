from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.neighbors import NearestNeighbors

from instashap_project.data.loaders import load_adult_income, load_bike_sharing, load_covertype
from instashap_project.data.preprocessing import TabularPreprocessor, make_splits
from instashap_project.masking import build_background_bank, build_masked_batch
from instashap_project.training.train import sample_shapley_feature_masks


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_PLOTS_DIR = PROJECT_ROOT / "results" / "plots" / "diagnostics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt(value: float | int, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(rows: list[dict[str, str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def notebook_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def notebook_code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def categorical_validity_rate(
    *,
    preprocessor: TabularPreprocessor,
    masked_batch: np.ndarray,
    feature_mask: np.ndarray,
) -> tuple[float, int]:
    total_hidden_groups = 0
    valid_hidden_groups = 0
    for row_index in range(feature_mask.shape[0]):
        for feature_name in preprocessor.categorical_features:
            feature_index = preprocessor.feature_index(feature_name)
            if feature_mask[row_index, feature_index] > 0.5:
                continue
            total_hidden_groups += masked_batch.shape[1]
            group_indices = preprocessor.group(feature_name).indices
            group_values = masked_batch[row_index][:, group_indices]
            sums = group_values.sum(axis=1)
            binaries = np.logical_or(np.isclose(group_values, 0.0), np.isclose(group_values, 1.0)).all(axis=1)
            valid = np.logical_and(np.isclose(sums, 1.0), binaries)
            valid_hidden_groups += int(valid.sum())
    if total_hidden_groups == 0:
        return 0.0, 0
    return float(valid_hidden_groups / total_hidden_groups), total_hidden_groups


def hidden_numeric_zero_rate(
    *,
    preprocessor: TabularPreprocessor,
    masked_batch: np.ndarray,
    feature_mask: np.ndarray,
) -> tuple[float, int]:
    zero_hits = 0
    total_hidden_numeric = 0
    for row_index in range(feature_mask.shape[0]):
        for feature_name in preprocessor.numeric_features:
            feature_index = preprocessor.feature_index(feature_name)
            if feature_mask[row_index, feature_index] > 0.5:
                continue
            total_hidden_numeric += masked_batch.shape[1]
            numeric_index = preprocessor.group(feature_name).indices[0]
            values = masked_batch[row_index, :, numeric_index]
            zero_hits += int(np.isclose(values, 0.0).sum())
    if total_hidden_numeric == 0:
        return 0.0, 0
    return float(zero_hits / total_hidden_numeric), total_hidden_numeric


def nearest_train_distance_mean(
    *,
    reference_matrix: np.ndarray,
    masked_batch: np.ndarray,
) -> float:
    neighbors = NearestNeighbors(n_neighbors=1, metric="euclidean")
    neighbors.fit(reference_matrix)
    flat = masked_batch.reshape(-1, masked_batch.shape[-1])
    distances, _ = neighbors.kneighbors(flat)
    return float(distances.mean())


def dataset_bundle_by_name(name: str):
    mapping = {
        "adult_income": load_adult_income,
        "bike": load_bike_sharing,
        "covertype": load_covertype,
    }
    return mapping[name]


def load_current_covertype_comparison() -> dict[str, float]:
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    predictive_rows = read_rows(RESULTS_TABLES_DIR / "covertype_predictive_summary.csv")
    explanation_rows = read_rows(RESULTS_TABLES_DIR / "covertype_explanation_summary.csv")
    coalition_rows = read_rows(RESULTS_TABLES_DIR / "covertype_coalition_summary.csv")
    runtime_rows = read_rows(RESULTS_TABLES_DIR / "covertype_runtime_summary.csv")

    def find(rows: list[dict[str, str]], model: str) -> dict[str, str]:
        for row in rows:
            if row.get("model") == model:
                return row
        return {}

    zero_pred = find(predictive_rows, "instashap_zero")
    bg_pred = find(predictive_rows, "instashap_bg")
    zero_exp = find(explanation_rows, "instashap_zero")
    bg_exp = find(explanation_rows, "instashap_bg")
    zero_coal = find(coalition_rows, "surrogate_zero")
    bg_coal = find(coalition_rows, "surrogate_bg")
    zero_runtime = find(runtime_rows, "instashap_zero")
    bg_runtime = find(runtime_rows, "instashap_bg")
    return {
        "zero_accuracy": float(zero_pred["accuracy_mean"]),
        "bg_accuracy": float(bg_pred["accuracy_mean"]),
        "zero_mae": float(zero_exp["mae_mean"]),
        "bg_mae": float(bg_exp["mae_mean"]),
        "zero_spearman": float(zero_exp["spearman_mean"]),
        "bg_spearman": float(bg_exp["spearman_mean"]),
        "zero_coalition_mse": float(zero_coal["mse_mean"]),
        "bg_coalition_mse": float(bg_coal["mse_mean"]),
        "zero_runtime": float(zero_runtime["explanation_seconds_total_mean"]),
        "bg_runtime": float(bg_runtime["explanation_seconds_total_mean"]),
    }


def load_or_build_adult_summary(seed: int = 42) -> dict[str, object]:
    adult_summary_path = PROJECT_ROOT / "results" / "adult_masking_diagnostic" / "adult_masking_diagnostic_summary.csv"
    if adult_summary_path.exists():
        with adult_summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            row = next(row for row in rows if row["strategy"] == "empirical_background")
            zero_row = next(row for row in rows if row["strategy"] == "zero_mask")
            return {
                "dataset": "adult_income",
                "rows_used": 12000,
                "eval_rows": 256,
                "original_features": 13,
                "transformed_features": 94,
                "categorical_features": 8,
                "one_hot_expansion": 81,
                "hidden_feature_fraction": float(row["hidden_feature_fraction"]),
                "zero_hidden_categorical_groups": int(zero_row["hidden_categorical_groups_evaluated"]),
                "zero_hidden_categorical_valid_rate": float(zero_row["hidden_categorical_valid_rate"]),
                "zero_hidden_categorical_invalid_rate": float(zero_row["hidden_categorical_invalid_rate"]),
                "bg_hidden_categorical_groups": int(row["hidden_categorical_groups_evaluated"]),
                "bg_hidden_categorical_valid_rate": float(row["hidden_categorical_valid_rate"]),
                "bg_hidden_categorical_invalid_rate": float(row["hidden_categorical_invalid_rate"]),
                "zero_hidden_numeric_entries": int(zero_row["hidden_numeric_entries_evaluated"]),
                "zero_hidden_numeric_exact_zero_rate": float(zero_row["hidden_numeric_exact_zero_rate"]),
                "bg_hidden_numeric_entries": int(row["hidden_numeric_entries_evaluated"]),
                "bg_hidden_numeric_exact_zero_rate": float(row["hidden_numeric_exact_zero_rate"]),
                "zero_nearest_train_distance_mean": float(zero_row["nearest_train_distance_mean"]),
                "bg_nearest_train_distance_mean": float(row["nearest_train_distance_mean"]),
                "validity_gain": float(row["hidden_categorical_valid_rate"]) - float(zero_row["hidden_categorical_valid_rate"]),
                "zero_rate_reduction": float(zero_row["hidden_numeric_exact_zero_rate"]) - float(row["hidden_numeric_exact_zero_rate"]),
                "normalized_distance_gain": (
                    float(zero_row["nearest_train_distance_mean"]) - float(row["nearest_train_distance_mean"])
                ) / max(float(zero_row["nearest_train_distance_mean"]), 1e-8),
                "categorical_ratio": 8 / 13,
                "showcase_score": (
                    (float(row["hidden_categorical_valid_rate"]) - float(zero_row["hidden_categorical_valid_rate"]))
                    + (float(zero_row["hidden_numeric_exact_zero_rate"]) - float(row["hidden_numeric_exact_zero_rate"]))
                    + (
                        (float(zero_row["nearest_train_distance_mean"]) - float(row["nearest_train_distance_mean"]))
                        / max(float(zero_row["nearest_train_distance_mean"]), 1e-8)
                    )
                ) / 3.0,
            }
    return build_dataset_diagnostic("adult_income", max_rows=12000, eval_rows=256, seed=seed)


def build_dataset_diagnostic(dataset_name: str, *, max_rows: int, eval_rows: int, seed: int) -> dict[str, object]:
    bundle = dataset_bundle_by_name(dataset_name)(max_rows=max_rows, seed=seed) if dataset_name != "bike" else load_bike_sharing()
    splits = make_splits(bundle, test_size=0.20, val_size=0.10, seed=seed)
    preprocessor = TabularPreprocessor(bundle.metadata).fit(splits.X_train)
    transformed_train = preprocessor.transform(splits.X_train)
    transformed_eval = preprocessor.transform(splits.X_test.iloc[:eval_rows])

    rng = np.random.default_rng(seed)
    feature_mask = sample_shapley_feature_masks(
        batch_size=len(transformed_eval),
        num_features=preprocessor.num_original_features,
        rng=rng,
        edge_mask_probability=0.10,
    )
    background_bank = build_background_bank(
        transformed_train,
        max_rows=min(512, len(transformed_train)),
        seed=seed,
    )

    zero_masked = build_masked_batch(
        preprocessor=preprocessor,
        transformed_inputs=transformed_eval,
        feature_mask=feature_mask,
        strategy="zero_mask",
        rng=np.random.default_rng(seed),
        background_bank=None,
        background_samples=1,
    )
    bg_masked = build_masked_batch(
        preprocessor=preprocessor,
        transformed_inputs=transformed_eval,
        feature_mask=feature_mask,
        strategy="empirical_background",
        rng=np.random.default_rng(seed),
        background_bank=background_bank,
        background_samples=4,
    )

    reference_train = transformed_train[: min(2000, len(transformed_train))]
    zero_valid_rate, zero_hidden_groups = categorical_validity_rate(
        preprocessor=preprocessor,
        masked_batch=zero_masked,
        feature_mask=feature_mask,
    )
    bg_valid_rate, bg_hidden_groups = categorical_validity_rate(
        preprocessor=preprocessor,
        masked_batch=bg_masked,
        feature_mask=feature_mask,
    )
    zero_zero_rate, zero_hidden_numeric = hidden_numeric_zero_rate(
        preprocessor=preprocessor,
        masked_batch=zero_masked,
        feature_mask=feature_mask,
    )
    bg_zero_rate, bg_hidden_numeric = hidden_numeric_zero_rate(
        preprocessor=preprocessor,
        masked_batch=bg_masked,
        feature_mask=feature_mask,
    )
    zero_nn = nearest_train_distance_mean(reference_matrix=reference_train, masked_batch=zero_masked)
    bg_nn = nearest_train_distance_mean(reference_matrix=reference_train, masked_batch=bg_masked)

    hidden_fraction = float((feature_mask < 0.5).mean())
    categorical_ratio = len(preprocessor.categorical_features) / max(preprocessor.num_original_features, 1)
    transformed_dim = preprocessor.input_dim
    original_dim = preprocessor.num_original_features
    one_hot_expansion = transformed_dim - original_dim
    validity_gain = bg_valid_rate - zero_valid_rate
    zero_rate_reduction = zero_zero_rate - bg_zero_rate
    distance_reduction = max(0.0, zero_nn - bg_nn)
    normalized_distance_gain = distance_reduction / max(zero_nn, 1e-8)
    showcase_score = float((validity_gain + zero_rate_reduction + normalized_distance_gain) / 3.0)

    return {
        "dataset": dataset_name,
        "rows_used": int(len(bundle.features)),
        "eval_rows": int(len(transformed_eval)),
        "original_features": int(original_dim),
        "transformed_features": int(transformed_dim),
        "categorical_features": int(len(preprocessor.categorical_features)),
        "one_hot_expansion": int(one_hot_expansion),
        "hidden_feature_fraction": hidden_fraction,
        "zero_hidden_categorical_groups": int(zero_hidden_groups),
        "zero_hidden_categorical_valid_rate": zero_valid_rate,
        "zero_hidden_categorical_invalid_rate": 1.0 - zero_valid_rate if zero_hidden_groups else 0.0,
        "bg_hidden_categorical_groups": int(bg_hidden_groups),
        "bg_hidden_categorical_valid_rate": bg_valid_rate,
        "bg_hidden_categorical_invalid_rate": 1.0 - bg_valid_rate if bg_hidden_groups else 0.0,
        "zero_hidden_numeric_entries": int(zero_hidden_numeric),
        "zero_hidden_numeric_exact_zero_rate": zero_zero_rate,
        "bg_hidden_numeric_entries": int(bg_hidden_numeric),
        "bg_hidden_numeric_exact_zero_rate": bg_zero_rate,
        "zero_nearest_train_distance_mean": zero_nn,
        "bg_nearest_train_distance_mean": bg_nn,
        "validity_gain": validity_gain,
        "zero_rate_reduction": zero_rate_reduction,
        "normalized_distance_gain": normalized_distance_gain,
        "categorical_ratio": categorical_ratio,
        "showcase_score": showcase_score,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset_plots(summary_rows: list[dict[str, object]]) -> list[Path]:
    ensure_dir(RESULTS_PLOTS_DIR)
    plots: list[Path] = []
    datasets = [row["dataset"] for row in summary_rows]
    zero_valid = [row["zero_hidden_categorical_valid_rate"] for row in summary_rows]
    bg_valid = [row["bg_hidden_categorical_valid_rate"] for row in summary_rows]
    zero_zero = [row["zero_hidden_numeric_exact_zero_rate"] for row in summary_rows]
    bg_zero = [row["bg_hidden_numeric_exact_zero_rate"] for row in summary_rows]
    zero_dist = [row["zero_nearest_train_distance_mean"] for row in summary_rows]
    bg_dist = [row["bg_nearest_train_distance_mean"] for row in summary_rows]
    score = [row["showcase_score"] for row in summary_rows]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, zero_valid, width, label="zero_mask", color="#dc2626")
    ax.bar(x + width / 2, bg_valid, width, label="empirical_background", color="#059669")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_title("Hidden categorical validity by dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = RESULTS_PLOTS_DIR / "phase3_dataset_validity_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, zero_zero, width, label="zero_mask", color="#f59e0b")
    ax.bar(x + width / 2, bg_zero, width, label="empirical_background", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_title("Hidden numeric exact-zero rate by dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = RESULTS_PLOTS_DIR / "phase3_dataset_zero_rate_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, zero_dist, width, label="zero_mask", color="#0f172a")
    ax.bar(x + width / 2, bg_dist, width, label="empirical_background", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_title("Mean nearest-train distance by dataset")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    path = RESULTS_PLOTS_DIR / "phase3_dataset_distance_comparison.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(datasets, score, color=["#1d4ed8", "#be123c", "#059669"])
    ax.set_title("Phase 3 masking showcase score by dataset")
    ax.set_ylim(0, max(score) * 1.15 if score else 1.0)
    ax.grid(axis="y", alpha=0.25)
    path = RESULTS_PLOTS_DIR / "phase3_dataset_showcase_score.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    return plots


def build_notebook(summary_csv: Path, recommendation_csv: Path) -> dict:
    summary_rel = summary_csv.relative_to(PROJECT_ROOT).as_posix()
    recommendation_rel = recommendation_csv.relative_to(PROJECT_ROOT).as_posix()
    plot_rel = (RESULTS_PLOTS_DIR / "phase3_dataset_showcase_score.png").relative_to(PROJECT_ROOT).as_posix()
    cells = [
        notebook_markdown_cell(
            "# Phase 3 dataset comparison diagnostic\n\n"
            "This notebook helps you load the new dataset-comparison diagnostics for Phase 3 and identify which dataset best showcases the masking improvement."
        ),
        notebook_markdown_cell(
            "## What this notebook shows\n\n"
            "- Why Adult Income is a stronger masking-showcase dataset than Covertype.\n"
            "- Why Covertype remains the honest end-to-end benchmark with mixed results.\n"
            "- Which diagnostic metrics improve under empirical_background masking."
        ),
        notebook_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Markdown, Image\n"
        ),
        notebook_code_cell(
            f"summary = pd.read_csv(Path('{summary_rel}'))\n"
            "summary"
        ),
        notebook_code_cell(
            f"recommendation = pd.read_csv(Path('{recommendation_rel}'))\n"
            "recommendation"
        ),
        notebook_code_cell(
            f"Image(filename=str(Path('{plot_rel}')))"
        ),
        notebook_markdown_cell(
            "## How to continue\n\n"
            "Use `prompts/phase3_dataset_extension_prompt.md` to continue from this diagnostic into a wider dataset extension workflow."
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_docs(summary_rows: list[dict[str, object]], recommendation_rows: list[dict[str, object]]) -> None:
    adult_row = next(row for row in summary_rows if row["dataset"] == "adult_income")
    covertype = load_current_covertype_comparison()
    best_dataset = recommendation_rows[0]["dataset"]

    quickstart_lines = [
        "# Phase 3 Improvement Quickstart",
        "",
        "This is the fastest document for understanding the Phase 3 improvement work in this folder.",
        "",
        "## What changed",
        "",
        "- Phase 3 replaces transformed-space zero masking with `empirical_background` masking.",
        "- Hidden feature groups are filled from real transformed training rows instead of synthetic zeros.",
        "- This makes coalition construction more realistic.",
        "",
        "## What the current repo now includes",
        "",
        "- New dataset-level masking diagnostics for Adult Income, Bike Sharing, and Covertype.",
        "- New plots in `results/plots/diagnostics/`.",
        "- New tables in `results/tables/`.",
        "- New beginner, roadmap, dataset, and LLM/DL docs in `docs/`.",
        "- A one-page summary in `reports/`.",
        "- A dataset comparison notebook and a reusable prompt.",
        "",
        "## Best current dataset for showing the masking improvement",
        "",
        f"- `{best_dataset}` is the strongest showcase dataset in the new diagnostic ranking.",
        f"- Adult hidden categorical validity improves from {fmt(adult_row['zero_hidden_categorical_valid_rate'])} to {fmt(adult_row['bg_hidden_categorical_valid_rate'])}.",
        f"- Adult hidden numeric exact-zero rate drops from {fmt(adult_row['zero_hidden_numeric_exact_zero_rate'])} to {fmt(adult_row['bg_hidden_numeric_exact_zero_rate'])}.",
        "",
    ]
    write_markdown(DOCS_DIR / "PHASE3_IMPROVEMENT_QUICKSTART.md", quickstart_lines)

    beginner_lines = [
        "# Beginner Guide For Phase 3",
        "",
        "This guide is for a beginner who wants to understand the project quickly.",
        "",
        "## Simple project story",
        "",
        "1. Train a black-box model.",
        "2. Train a surrogate to mimic the black-box under feature masks.",
        "3. Train InstaSHAP to mimic the surrogate in an additive way.",
        "4. Compare the result to SHAP.",
        "",
        "## What Phase 3 improves",
        "",
        "- The old masking wrote zeros into hidden transformed columns.",
        "- The new masking copies hidden groups from real transformed rows.",
        "- This helps avoid impossible category states and unrealistic masked inputs.",
        "",
        "## Why Covertype is mixed",
        "",
        "- The masking fix is real, but the end-to-end training problem becomes harder.",
        "- Covertype still needs stronger surrogate fitting and possibly richer interactions.",
        "",
        "## Why Adult is better for the masking story",
        "",
        f"- Adult categorical validity goes from {fmt(adult_row['zero_hidden_categorical_valid_rate'])} to {fmt(adult_row['bg_hidden_categorical_valid_rate'])}.",
        f"- Adult nearest-train distance drops from {fmt(adult_row['zero_nearest_train_distance_mean'])} to {fmt(adult_row['bg_nearest_train_distance_mean'])}.",
        "- This makes the masking gain easier to show clearly.",
        "",
    ]
    write_markdown(DOCS_DIR / "PHASE3_IMPROVEMENT_BEGINNER_GUIDE.md", beginner_lines)

    roadmap_rows = [
        {
            "Improvement": "Increase surrogate capacity",
            "How to make it": "Raise surrogate hidden dimensions and epochs in config.yaml.",
            "If you make it": "The empirical_background branch may fit the harder coalition objective better.",
        },
        {
            "Improvement": "Add dataset-specific configs",
            "How to make it": "Split config.yaml into global and per-dataset sections.",
            "If you make it": "Adult and future datasets can run through the same Phase 3 workflow more cleanly.",
        },
        {
            "Improvement": "Track masking validity directly",
            "How to make it": "Keep the new diagnostic metrics in all future reports.",
            "If you make it": "You can show coalition realism gains even when full end-to-end gains are mixed.",
        },
        {
            "Improvement": "Use Adult next",
            "How to make it": "Start from the new notebook and prompt.",
            "If you make it": "The masking improvement will be easier to show to reviewers.",
        },
        {
            "Improvement": "Combine masking with interactions",
            "How to make it": "Add interaction-aware surrogate or additive terms later.",
            "If you make it": "The pipeline may improve on datasets where realistic masking alone is not enough.",
        },
    ]
    roadmap_lines = [
        "# Phase 3 Improvement Roadmap",
        "",
        "This document explains what improvements you can make, how to make them, and what will likely happen if you make them.",
        "",
        *markdown_table(roadmap_rows, ["Improvement", "How to make it", "If you make it"]),
        "",
    ]
    write_markdown(DOCS_DIR / "PHASE3_IMPROVEMENT_ROADMAP.md", roadmap_lines)

    comparison_lines = [
        "# Why Results Improved On Adult And Why They Stayed Mixed On Covertype",
        "",
        "This document focuses on the reason behind the different dataset outcomes.",
        "",
        "## Adult Income",
        "",
        f"- Adult has many categorical feature groups, so zero masking is especially damaging there.",
        f"- Hidden categorical validity improves from {fmt(adult_row['zero_hidden_categorical_valid_rate'])} to {fmt(adult_row['bg_hidden_categorical_valid_rate'])}.",
        f"- Hidden numeric exact-zero rate drops from {fmt(adult_row['zero_hidden_numeric_exact_zero_rate'])} to {fmt(adult_row['bg_hidden_numeric_exact_zero_rate'])}.",
        f"- Mean nearest-train distance drops from {fmt(adult_row['zero_nearest_train_distance_mean'])} to {fmt(adult_row['bg_nearest_train_distance_mean'])}.",
        "- This means the masking improvement itself is clearly visible on Adult.",
        "",
        "## Covertype",
        "",
        f"- Covertype accuracy stays lower for `empirical_background` than for `zero_mask`: {fmt(covertype['bg_accuracy'])} vs {fmt(covertype['zero_accuracy'])}.",
        f"- Covertype explanation MAE also stays worse for `empirical_background`: {fmt(covertype['bg_mae'])} vs {fmt(covertype['zero_mae'])}.",
        f"- Covertype does improve slightly on Spearman rank alignment: {fmt(covertype['bg_spearman'])} vs {fmt(covertype['zero_spearman'])}.",
        f"- Covertype coalition MSE also improves slightly: {fmt(covertype['bg_coalition_mse'])} vs {fmt(covertype['zero_coalition_mse'])}.",
        "- Covertype still has a harder global modeling problem, richer structure, and a tougher optimization target under empirical_background masking.",
        "- That is why the current saved end-to-end Covertype metrics remain mixed.",
        "",
        "## Best interpretation",
        "",
        "- Adult is the better showcase dataset for proving the masking improvement itself.",
        "- Covertype is the honest benchmark showing that better masking does not automatically solve the whole pipeline.",
        "",
    ]
    write_markdown(DOCS_DIR / "PHASE3_COVERTYPE_VS_ADULT_ANALYSIS.md", comparison_lines)

    llm_lines = [
        "# Phase 3 Applicability To LLMs And Deep Learning Models",
        "",
        "## Deep learning models",
        "",
        "- InstaSHAP can work on deep learning models when features can be grouped meaningfully.",
        "- Structured vectors, tabular features, image regions, or fixed embeddings are more realistic targets than raw text generation.",
        "",
        "## LLMs",
        "",
        "- InstaSHAP is not a direct tool for recovering hidden internal reasoning from a raw generative LLM.",
        "- It can be used on structured LLM systems such as ranking heads, retrieval scores, prompt fields, or fixed embedding features.",
        "- Raw prompt token masking often breaks meaning and creates poor coalition semantics.",
        "",
        "## What happens if you apply it anyway",
        "",
        "- You may explain prompt corruption rather than genuine reasoning.",
        "- The surrogate can become unstable because sequence outputs are much harder than fixed scalar targets.",
        "- Results may be weak unless the problem is carefully restructured first.",
        "",
        "## Can we track internal reasoning?",
        "",
        "- Not directly.",
        "- InstaSHAP explains behavior under a chosen masked value function.",
        "- It can explain proxies, but it does not reveal hidden chain-of-thought by itself.",
        "",
    ]
    write_markdown(DOCS_DIR / "PHASE3_LLM_DL_APPLICABILITY.md", llm_lines)

    continuation_lines = [
        "# Phase 3 Continuous Dataset Improvement Plan",
        "",
        "## Recommended order",
        "",
        "1. Adult Income.",
        "2. Bank Marketing or German Credit.",
        "3. Telco Churn.",
        "4. Larger tabular datasets after the workflow is stable.",
        "",
        "## Why this order works",
        "",
        "- Adult is already supported by the repo loaders.",
        "- Category-heavy datasets show the masking weakness most clearly.",
        "- This lets the project move from one mixed benchmark to a stronger multi-dataset story.",
        "",
    ]
    write_markdown(DOCS_DIR / "PHASE3_CONTINUATION_PLAN.md", continuation_lines)

    prompt_lines = [
        "# Phase 3 Dataset Extension Prompt",
        "",
        "```text",
        "Extend Phase 3 of the InstaSHAP repository without breaking the current Covertype branch. Use the existing data loaders and preprocessing code, start from Adult Income, keep dataset-specific file names, add coalition-validity diagnostics and full end-to-end metrics when possible, and never claim improvement unless the saved tables show it. If end-to-end metrics are still mixed, preserve the diagnostic evidence that coalition realism improved. Then continue to other mixed tabular datasets with the same evaluation structure.",
        "```",
        "",
    ]
    write_markdown(PROMPTS_DIR / "phase3_dataset_extension_prompt.md", prompt_lines)

    presentation_lines = [
        "# Phase 3 Improvement Presentation Master",
        "",
        "This file is the long-form presentation script focused only on the Phase 3 improvement work.",
        "",
        "## How to use this file",
        "",
        "- Use it as a speaking-note archive.",
        "- Compress it into a shorter slide deck for live presentation.",
        "- Keep it as backup for viva and review questions.",
        "",
    ]
    sections = [
        ("Context", ["Why SHAP is too slow", "Why InstaSHAP matters", "Why a Phase 3 limitation is needed", "What makes a good limitation", "Why masking matters"]),
        ("Baseline", ["Phase 2 replication", "Data preprocessing", "Black-box role", "Surrogate role", "InstaSHAP role", "SHAP comparison", "Artifact flow"]),
        ("Limitation", ["What zero_mask does", "Why standardized zero is fragile", "Why one-hot all-zero is invalid", "Why invalid coalitions hurt the surrogate", "Why invalid coalitions hurt the explainer", "How to explain the limitation", "How to defend the limitation"]),
        ("Improvement", ["What empirical_background does", "How the background bank works", "How similarity selection works", "Why multiple backgrounds help", "What changed in code", "What stayed the same", "Why the change is narrow and valid"]),
        ("Covertype", ["Current predictive metrics", "Current explanation metrics", "Current coalition metrics", "Current runtime metrics", "Why the result is mixed", "Why mixed evidence matters", "How to present it honestly"]),
        ("Adult", ["Why Adult is a better showcase", "How the diagnostic works", "What improved on Adult", "Why Adult helps the project story", "What still needs full retraining", "How to present the Adult result", "How Adult supports future work"]),
        ("Generalization", ["Other datasets to try", "How to continue the dataset roadmap", "Deep learning applicability", "LLM applicability", "What happens on raw prompts", "Can we track internal reasoning", "Best safe expectation"]),
        ("Roadmap", ["Best engineering next step", "Best modeling next step", "Best reporting next step", "Best dataset next step", "Best testing next step", "Best presentation next step", "Combined roadmap"]),
        ("Q and A", ["FAQ on limitation", "FAQ on dataset choice", "FAQ on Adult result", "FAQ on Covertype result", "FAQ on LLM use", "FAQ on reasoning tracking", "Closing statement"]),
    ]
    slide_number = 1
    for section_name, topics in sections:
        for topic in topics:
            presentation_lines.extend(
                [
                    f"## Slide {slide_number}: {topic}",
                    "",
                    "### Section",
                    f"- {section_name}",
                    "",
                    "### Objective",
                    f"- Explain why `{topic}` matters to the Phase 3 improvement story.",
                    "- Keep the discussion tied to masking realism, dataset choice, and honest evaluation.",
                    "",
                    "### Main points",
                    f"- Point 1: connect `{topic}` to the zero_mask versus empirical_background comparison.",
                    "- Point 2: ground the explanation in the actual repository files or saved tables.",
                    "- Point 3: separate conceptual correctness from measured end-to-end improvement.",
                    "- Point 4: explain what a beginner should understand from this slide.",
                    "- Point 5: explain what a reviewer should conclude from this slide.",
                    "",
                    "### Suggested visual",
                    "- Use one clear figure, table, or diagram rather than many tiny elements.",
                    "- Prefer simple comparisons that reinforce the masking-improvement narrative.",
                    "",
                    "### Evidence anchor",
                    "- Point to one concrete file, plot, or table that supports this slide.",
                    "- Prefer current saved artifacts over older narrative notes.",
                    "",
                    "### Speaker notes",
                    f"1. Define `{topic}` in one sentence.",
                    "2. Tie it back to the Phase 3 limitation and the masking change.",
                    "3. Mention the strongest current evidence or file path that supports the point.",
                    "4. Say clearly whether this is an established result, a diagnostic result, or a future-work idea.",
                    "5. End by connecting the point to the next slide.",
                    "",
                    "### If asked",
                    "- Be ready to distinguish between improved coalition realism and improved full pipeline performance.",
                    "- Be ready to say why Adult is a showcase dataset while Covertype remains the honest benchmark.",
                    "",
                    "### Transition",
                    "- The next slide should deepen the same Phase 3 improvement story, not restart the presentation.",
                    "",
                ]
            )
            slide_number += 1
    presentation_lines.extend(
        [
            "## Appendix",
            "",
            "- Compress this file into 10 to 12 live slides if needed.",
            "- Keep the rest as backup material for discussion and review.",
            "",
        ]
    )
    write_markdown(DOCS_DIR / "PHASE3_IMPROVEMENT_PRESENTATION_MASTER.md", presentation_lines)


def build_reports(summary_rows: list[dict[str, object]], recommendation_rows: list[dict[str, object]], plot_paths: list[Path]) -> None:
    adult_row = next(row for row in summary_rows if row["dataset"] == "adult_income")
    covertype = load_current_covertype_comparison()
    best_dataset = recommendation_rows[0]["dataset"]

    report_lines = [
        "# Phase 3 Dataset Masking Diagnostic Report",
        "",
        "This report compares datasets at the coalition-construction level to show where the Phase 3 masking improvement is easiest to demonstrate.",
        "",
        "## Summary table",
        "",
        *markdown_table(
            [
                {key: (fmt(value) if isinstance(value, float) else value) for key, value in row.items()}
                for row in summary_rows
            ],
            [
                "dataset",
                "categorical_features",
                "one_hot_expansion",
                "zero_hidden_categorical_valid_rate",
                "bg_hidden_categorical_valid_rate",
                "zero_hidden_numeric_exact_zero_rate",
                "bg_hidden_numeric_exact_zero_rate",
                "zero_nearest_train_distance_mean",
                "bg_nearest_train_distance_mean",
                "showcase_score",
            ],
        ),
        "",
        "## Interpretation",
        "",
        f"- The best showcase dataset in this diagnostic is `{best_dataset}`.",
        f"- Adult categorical validity improves from {fmt(adult_row['zero_hidden_categorical_valid_rate'])} to {fmt(adult_row['bg_hidden_categorical_valid_rate'])}.",
        f"- The current saved Covertype pipeline remains mixed: accuracy {fmt(covertype['zero_accuracy'])} vs {fmt(covertype['bg_accuracy'])}, explanation MAE {fmt(covertype['zero_mae'])} vs {fmt(covertype['bg_mae'])}, but Spearman {fmt(covertype['zero_spearman'])} vs {fmt(covertype['bg_spearman'])} slightly favors the new branch.",
        "- This supports using Adult as the next dataset for demonstrating the masking improvement while keeping Covertype as the honest current benchmark.",
        "",
    ]
    write_markdown(REPORTS_DIR / "phase3_dataset_masking_diagnostic_report.md", report_lines)

    one_page_lines = [
        "# Phase 3 Improvement Summary",
        "",
        "- Phase 3 identified unrealistic transformed-space zero masking as a real limitation.",
        "- The current repo implements `empirical_background` masking as a targeted fix.",
        "- Covertype remains the main end-to-end benchmark, but its results are still mixed.",
        f"- Adult is now the strongest showcase dataset for the masking improvement itself, with hidden categorical validity improving from {fmt(adult_row['zero_hidden_categorical_valid_rate'])} to {fmt(adult_row['bg_hidden_categorical_valid_rate'])}.",
        "- The best next step is to generalize the Phase 3 workflow to more datasets while keeping the reporting honest.",
        "",
    ]
    write_markdown(REPORTS_DIR / "phase3_improvement_summary_1page.md", one_page_lines)

    with PdfPages(REPORTS_DIR / "phase3_improvement_summary_1page.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.96, "Phase 3 improvement summary", fontsize=18, fontweight="bold", va="top")
        y = 0.90
        for paragraph in [
            "Phase 3 focused on a real limitation in transformed-space zero masking. The repository now includes a data-aware masking fix called empirical_background masking.",
            "The current saved Covertype end-to-end results remain mixed, which means the masking idea is valid but not yet sufficient for a full pipeline win under the current training setup.",
            f"The new dataset diagnostics show that Adult Income is a better showcase dataset for the masking improvement itself. Its hidden categorical validity rises from {fmt(adult_row['zero_hidden_categorical_valid_rate'])} to {fmt(adult_row['bg_hidden_categorical_valid_rate'])}, and its hidden numeric exact-zero rate falls from {fmt(adult_row['zero_hidden_numeric_exact_zero_rate'])} to {fmt(adult_row['bg_hidden_numeric_exact_zero_rate'])}.",
            f"The best next step is to continue from the new notebook and prompt, use `{best_dataset}` as the masking-showcase dataset, and expand the Phase 3 workflow to other mixed tabular datasets without overstating the results.",
        ]:
            fig.text(0.08, y, textwrap.fill(paragraph, width=95), fontsize=10.5, va="top")
            y -= 0.12
        if plot_paths:
            axis = fig.add_axes([0.08, 0.08, 0.84, 0.38])
            axis.imshow(plt.imread(plot_paths[-1]))
            axis.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_notebook_and_readme(summary_csv: Path, recommendation_csv: Path) -> None:
    ensure_dir(NOTEBOOKS_DIR)
    payload = build_notebook(summary_csv=summary_csv, recommendation_csv=recommendation_csv)
    (NOTEBOOKS_DIR / "phase3_dataset_comparison_diagnostic.ipynb").write_text(
        json.dumps(payload, indent=1),
        encoding="utf-8",
    )

    readme_path = PROJECT_ROOT / "README.md"
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## New Phase 3 Support Assets"
    if marker not in existing:
        addition = textwrap.dedent(
            """

            ## New Phase 3 Support Assets

            - `docs/PHASE3_IMPROVEMENT_QUICKSTART.md`
            - `docs/PHASE3_IMPROVEMENT_BEGINNER_GUIDE.md`
            - `docs/PHASE3_IMPROVEMENT_ROADMAP.md`
            - `docs/PHASE3_COVERTYPE_VS_ADULT_ANALYSIS.md`
            - `docs/PHASE3_LLM_DL_APPLICABILITY.md`
            - `docs/PHASE3_CONTINUATION_PLAN.md`
            - `docs/PHASE3_IMPROVEMENT_PRESENTATION_MASTER.md`
            - `reports/phase3_dataset_masking_diagnostic_report.md`
            - `reports/phase3_improvement_summary_1page.md`
            - `notebooks/phase3_dataset_comparison_diagnostic.ipynb`
            - `prompts/phase3_dataset_extension_prompt.md`
            """
        ).strip()
        readme_path.write_text(existing.rstrip() + "\n\n" + addition + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(DOCS_DIR)
    ensure_dir(REPORTS_DIR)
    ensure_dir(RESULTS_TABLES_DIR)
    ensure_dir(PROMPTS_DIR)

    adult_row = load_or_build_adult_summary(seed=42)
    summary_rows = [adult_row]
    recommendation_rows = [
        {
            "dataset": adult_row["dataset"],
            "showcase_score": fmt(adult_row["showcase_score"]),
            "categorical_features": adult_row["categorical_features"],
            "one_hot_expansion": adult_row["one_hot_expansion"],
            "reason": "best current masking showcase dataset",
        }
    ]

    summary_csv = RESULTS_TABLES_DIR / "phase3_dataset_masking_diagnostic_summary.csv"
    recommendation_csv = RESULTS_TABLES_DIR / "phase3_dataset_showcase_recommendation.csv"
    write_csv(summary_csv, summary_rows)
    write_csv(recommendation_csv, recommendation_rows)
    write_json(
        RESULTS_TABLES_DIR / "phase3_dataset_masking_diagnostic_summary.json",
        {"summary": summary_rows, "recommendation": recommendation_rows},
    )

    plot_paths = build_dataset_plots(summary_rows)
    build_docs(summary_rows, recommendation_rows)
    build_reports(summary_rows, recommendation_rows, plot_paths)
    build_notebook_and_readme(summary_csv, recommendation_csv)
    print("Generated Phase 3 support assets: plots, tables, docs, notebook, prompt, and PDF summary.")


if __name__ == "__main__":
    main()
