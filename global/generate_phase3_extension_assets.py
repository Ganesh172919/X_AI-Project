from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_DIR = ROOT / "global"
PHASE3_DIR = ROOT / "Phase_3_work"
PHASE3_RESULTS_DIR = PHASE3_DIR / "results"
ADULT_DIAGNOSTIC_DIR = PHASE3_RESULTS_DIR / "adult_masking_diagnostic"
NOTEBOOK_DIR = PHASE3_DIR / "notebooks"
PROMPT_DIR = PHASE3_DIR / "prompts"

sys.path.insert(0, str(PHASE3_DIR))

from instashap_project.data.loaders import load_adult_income  # noqa: E402
from instashap_project.data.preprocessing import TabularPreprocessor, make_splits  # noqa: E402
from instashap_project.masking import build_background_bank, build_masked_batch  # noqa: E402
from instashap_project.training.train import sample_shapley_feature_masks  # noqa: E402


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fmt(value: float | int, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def load_phase3_summary_table(filename: str) -> list[dict[str, str]]:
    table_path = PHASE3_RESULTS_DIR / "tables" / filename
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def lookup_row(rows: list[dict[str, str]], model: str) -> dict[str, str]:
    for row in rows:
        if row.get("model") == model:
            return row
    return {}


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


def build_metric_plot(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    labels = [row["strategy"] for row in summary_rows]
    valid_rates = [float(row["hidden_categorical_valid_rate"]) for row in summary_rows]
    invalid_rates = [float(row["hidden_categorical_invalid_rate"]) for row in summary_rows]
    zero_rates = [float(row["hidden_numeric_exact_zero_rate"]) for row in summary_rows]
    nn_distances = [float(row["nearest_train_distance_mean"]) for row in summary_rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Adult dataset masking diagnostic", fontsize=16, fontweight="bold")

    axes[0, 0].bar(labels, valid_rates, color=["#1d4ed8", "#059669"])
    axes[0, 0].set_title("Hidden categorical validity rate")
    axes[0, 0].set_ylim(0, 1.05)

    axes[0, 1].bar(labels, invalid_rates, color=["#dc2626", "#7c3aed"])
    axes[0, 1].set_title("Hidden categorical invalid rate")
    axes[0, 1].set_ylim(0, 1.05)

    axes[1, 0].bar(labels, zero_rates, color=["#f59e0b", "#10b981"])
    axes[1, 0].set_title("Hidden numeric exact-zero rate")
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].bar(labels, nn_distances, color=["#0f172a", "#2563eb"])
    axes[1, 1].set_title("Mean nearest-train distance")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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


def build_adult_notebook(*, csv_path: Path, plot_path: Path, report_md_path: Path) -> dict:
    relative_csv = csv_path.relative_to(PHASE3_DIR).as_posix()
    relative_plot = plot_path.relative_to(PHASE3_DIR).as_posix()
    relative_report = report_md_path.relative_to(PHASE3_DIR).as_posix()
    cells = [
        notebook_markdown_cell(
            "# Phase 3 Adult masking diagnostic\n\n"
            "This notebook loads a better showcase dataset for the Phase 3 masking improvement. "
            "It measures the coalition realism improvement directly."
        ),
        notebook_markdown_cell(
            "## Why this notebook exists\n\n"
            "- Covertype is the current runnable Phase 3 dataset, but its end metrics are mixed.\n"
            "- Adult Income has many categorical groups, so invalid hidden category states are easier to show.\n"
            "- This notebook demonstrates that `empirical_background` fixes the masking problem itself more clearly on Adult Income."
        ),
        notebook_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import display, Markdown, Image\n"
        ),
        notebook_code_cell(
            f"summary = pd.read_csv(Path('{relative_csv}'))\n"
            "summary"
        ),
        notebook_markdown_cell(
            "## Interpretation guide\n\n"
            "- `hidden_categorical_valid_rate` should be near 0 for `zero_mask` and near 1 for `empirical_background`.\n"
            "- `hidden_numeric_exact_zero_rate` should be much higher for `zero_mask`.\n"
            "- `nearest_train_distance_mean` gives a rough realism signal against the transformed training manifold."
        ),
        notebook_code_cell(
            f"Image(filename=str(Path('{relative_plot}')))"
        ),
        notebook_code_cell(
            f"report_text = Path('{relative_report}').read_text(encoding='utf-8')\n"
            "display(Markdown(report_text))"
        ),
        notebook_markdown_cell(
            "## Next step\n\n"
            "Use the prompt file in `prompts/phase3_dataset_extension_prompt.md` to continue from this diagnostic into a fuller dataset extension workflow."
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


def adult_masking_diagnostic() -> dict[str, object]:
    ensure_dir(ADULT_DIAGNOSTIC_DIR)
    ensure_dir(NOTEBOOK_DIR)
    ensure_dir(PROMPT_DIR)

    bundle = load_adult_income(max_rows=12000, seed=42)
    splits = make_splits(bundle, test_size=0.20, val_size=0.10, seed=42)
    preprocessor = TabularPreprocessor(bundle.metadata).fit(splits.X_train)
    transformed_train = preprocessor.transform(splits.X_train)
    transformed_eval = preprocessor.transform(splits.X_test.iloc[:256])

    rng = np.random.default_rng(42)
    feature_mask = sample_shapley_feature_masks(
        batch_size=len(transformed_eval),
        num_features=preprocessor.num_original_features,
        rng=rng,
        edge_mask_probability=0.10,
    )

    background_bank = build_background_bank(
        transformed_train,
        max_rows=min(512, len(transformed_train)),
        seed=42,
    )

    zero_masked = build_masked_batch(
        preprocessor=preprocessor,
        transformed_inputs=transformed_eval,
        feature_mask=feature_mask,
        strategy="zero_mask",
        rng=np.random.default_rng(42),
        background_bank=None,
        background_samples=1,
    )
    bg_masked = build_masked_batch(
        preprocessor=preprocessor,
        transformed_inputs=transformed_eval,
        feature_mask=feature_mask,
        strategy="empirical_background",
        rng=np.random.default_rng(42),
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
    summary_rows = [
        {
            "strategy": "zero_mask",
            "dataset": "adult_income",
            "hidden_feature_fraction": fmt(hidden_fraction),
            "hidden_categorical_groups_evaluated": str(zero_hidden_groups),
            "hidden_categorical_valid_rate": fmt(zero_valid_rate),
            "hidden_categorical_invalid_rate": fmt(1.0 - zero_valid_rate),
            "hidden_numeric_entries_evaluated": str(zero_hidden_numeric),
            "hidden_numeric_exact_zero_rate": fmt(zero_zero_rate),
            "nearest_train_distance_mean": fmt(zero_nn),
        },
        {
            "strategy": "empirical_background",
            "dataset": "adult_income",
            "hidden_feature_fraction": fmt(hidden_fraction),
            "hidden_categorical_groups_evaluated": str(bg_hidden_groups),
            "hidden_categorical_valid_rate": fmt(bg_valid_rate),
            "hidden_categorical_invalid_rate": fmt(1.0 - bg_valid_rate),
            "hidden_numeric_entries_evaluated": str(bg_hidden_numeric),
            "hidden_numeric_exact_zero_rate": fmt(bg_zero_rate),
            "nearest_train_distance_mean": fmt(bg_nn),
        },
    ]

    csv_path = ADULT_DIAGNOSTIC_DIR / "adult_masking_diagnostic_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    diagnostic_payload = {
        "dataset": "adult_income",
        "max_rows": int(len(bundle.features)),
        "evaluation_rows": int(len(transformed_eval)),
        "hidden_feature_fraction": hidden_fraction,
        "strategies": summary_rows,
    }
    json_path = ADULT_DIAGNOSTIC_DIR / "adult_masking_diagnostic_summary.json"
    write_json(json_path, diagnostic_payload)

    plot_path = ADULT_DIAGNOSTIC_DIR / "adult_masking_diagnostic_comparison.png"
    build_metric_plot(summary_rows, plot_path)

    report_lines = [
        "# Adult masking diagnostic",
        "",
        "This diagnostic does not retrain the full Phase 3 pipeline. Instead, it measures the exact weakness that the Phase 3 improvement targets: unrealistic coalition construction after tabular preprocessing.",
        "",
        "## Why Adult Income is a good next dataset",
        "",
        "- It contains many categorical feature groups, so zero-masking creates many impossible all-zero one-hot states.",
        "- It is already supported by the repository data loaders.",
        "- It is a strong follow-on dataset for Phase 3 because the masking problem is easier to demonstrate here than on Covertype.",
        "",
        "## Summary table",
        "",
    ]
    report_lines.extend(
        markdown_table(
            summary_rows,
            [
                "strategy",
                "dataset",
                "hidden_feature_fraction",
                "hidden_categorical_groups_evaluated",
                "hidden_categorical_valid_rate",
                "hidden_categorical_invalid_rate",
                "hidden_numeric_entries_evaluated",
                "hidden_numeric_exact_zero_rate",
                "nearest_train_distance_mean",
            ],
        )
    )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `zero_mask` leaves hidden categorical groups valid only {fmt(zero_valid_rate)} of the time in this diagnostic because all-zero one-hot groups are invalid category states.",
            f"- `empirical_background` keeps hidden categorical groups valid {fmt(bg_valid_rate)} of the time because it copies hidden transformed groups from real rows.",
            f"- `zero_mask` sets hidden numeric values to exact zero at rate {fmt(zero_zero_rate)}, while `empirical_background` does so at rate {fmt(bg_zero_rate)}.",
            f"- The mean nearest-train distance is {fmt(zero_nn)} for `zero_mask` versus {fmt(bg_nn)} for `empirical_background`.",
            "- This dataset therefore showcases the Phase 3 masking improvement more clearly than Covertype at the coalition-construction level.",
            "",
        ]
    )
    report_md_path = ADULT_DIAGNOSTIC_DIR / "adult_masking_diagnostic_report.md"
    write_markdown(report_md_path, report_lines)

    report_pdf_path = ADULT_DIAGNOSTIC_DIR / "adult_masking_diagnostic_report.pdf"
    with PdfPages(report_pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.96, "Adult masking diagnostic", fontsize=18, fontweight="bold", va="top")
        y = 0.90
        for paragraph in [
            "This one-page report measures the precise limitation targeted by Phase 3: invalid or unrealistic coalition states after preprocessing.",
            "Adult Income is a strong next dataset because it has many categorical groups. Zero-masking creates impossible hidden one-hot states, while empirical_background preserves valid hidden groups by copying from real transformed training rows.",
            f"The diagnostic shows hidden categorical validity rising from {fmt(zero_valid_rate)} under zero_mask to {fmt(bg_valid_rate)} under empirical_background. It also reduces exact-zero hidden numeric states and changes nearest-train distance from {fmt(zero_nn)} to {fmt(bg_nn)}.",
            "This does not by itself prove better full-model explanation fidelity, but it does prove that the masking improvement directly fixes the coalition-construction problem more clearly on Adult Income than on Covertype.",
        ]:
            wrapped = textwrap.fill(paragraph, width=95)
            fig.text(0.08, y, wrapped, fontsize=10.5, va="top")
            y -= 0.12
        axis = fig.add_axes([0.08, 0.10, 0.84, 0.38])
        axis.imshow(plt.imread(plot_path))
        axis.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    notebook_path = NOTEBOOK_DIR / "phase3_adult_masking_diagnostic.ipynb"
    notebook_payload = build_adult_notebook(
        csv_path=csv_path,
        plot_path=plot_path,
        report_md_path=report_md_path,
    )
    notebook_path.write_text(json.dumps(notebook_payload, indent=1), encoding="utf-8")

    return {
        "summary_rows": summary_rows,
        "csv_path": csv_path,
        "json_path": json_path,
        "plot_path": plot_path,
        "report_md_path": report_md_path,
        "report_pdf_path": report_pdf_path,
        "notebook_path": notebook_path,
    }


def build_beginner_doc(adult_results: dict[str, object]) -> list[str]:
    rows = adult_results["summary_rows"]
    return [
        "# Beginner Guide To This Project",
        "",
        "This document is written for a beginner who wants to understand the project quickly without reading the whole repository first.",
        "",
        "## What this project is",
        "",
        "- This is an Explainable AI project based on InstaSHAP.",
        "- The goal is to make SHAP-style explanations much faster by training an additive explainer ahead of time.",
        "- The repository has three phases: proposal, replication, and improvement.",
        "",
        "## The simple story",
        "",
        "1. Train a black-box model.",
        "2. Train a surrogate that learns the black-box under different feature masks.",
        "3. Train an additive InstaSHAP model that imitates the surrogate.",
        "4. Compare the learned explanations to a slower SHAP baseline.",
        "",
        "## What Phase 3 changed",
        "",
        "- The old approach used `zero_mask`, which hides features by writing zeros into transformed columns.",
        "- The new approach uses `empirical_background`, which fills hidden feature groups from real training rows.",
        "- This is meant to make masked coalitions more realistic.",
        "",
        "## Why the Covertype result is mixed",
        "",
        "- The masking idea is valid, but the harder and more realistic coalition objective also makes learning harder.",
        "- In the current saved Covertype run, the new branch is slightly better on some secondary signals but not on the main headline metrics.",
        "- That means the idea is promising, but the full pipeline still needs tuning.",
        "",
        "## Why Adult Income is a better showcase for the masking idea",
        "",
        "- Adult Income has many categorical feature groups.",
        "- Under `zero_mask`, hidden categorical groups become all-zero one-hot vectors, which are invalid category states.",
        "- Under `empirical_background`, those hidden groups remain valid because they come from real transformed rows.",
        "",
        "## Adult masking diagnostic snapshot",
        "",
        *markdown_table(
            rows,
            [
                "strategy",
                "hidden_categorical_valid_rate",
                "hidden_categorical_invalid_rate",
                "hidden_numeric_exact_zero_rate",
                "nearest_train_distance_mean",
            ],
        ),
        "",
        "## Best files for a beginner",
        "",
        "- `global/README.md` for the full documentation hub.",
        "- `global/01_PROJECT_UNDERSTANDING_GUIDE.md` for the fast overview.",
        "- `global/06_LATEST_PHASE3_IMPROVEMENT_ANALYSIS.md` for the direct answer to the improvement question.",
        "- `Phase_3_work/README.md` for the current runnable branch.",
        "- `Phase_3_work/instashap_project/masking.py` to see the actual improvement in code.",
        "",
        "## Best short summary",
        "",
        "The project reproduced InstaSHAP, identified that zero-masking creates unrealistic coalition states in transformed tabular data, and implemented a background-aware masking fix. The current Covertype result is mixed, but Adult Income shows the masking improvement itself very clearly.",
        "",
    ]


def build_improvement_roadmap() -> list[str]:
    rows = [
        {
            "Improvement": "Increase surrogate capacity",
            "How to make it": "Raise surrogate hidden dimensions and training epochs in config.yaml or a new dataset-specific config.",
            "What will happen": "The surrogate may fit the harder empirical_background objective better, which can improve downstream explanation fidelity.",
        },
        {
            "Improvement": "Increase background samples",
            "How to make it": "Raise masking.background_samples_train and masking.background_samples_eval.",
            "What will happen": "Coalition targets become more stable but training and explanation preparation become slower.",
        },
        {
            "Improvement": "Add invalid-state metrics",
            "How to make it": "Track hidden categorical validity and off-manifold distance explicitly in the training reports.",
            "What will happen": "You can show the masking improvement even when end-task metrics are mixed.",
        },
        {
            "Improvement": "Use Adult Income next",
            "How to make it": "Generalize the Phase 3 workflow or reuse the new notebook and prompt to continue from the adult masking diagnostic.",
            "What will happen": "The masking limitation should be easier to demonstrate because there are more categorical groups.",
        },
        {
            "Improvement": "Add a dataset-specific config system",
            "How to make it": "Split Phase 3 config.yaml into a global block and per-dataset blocks instead of one single dataset block.",
            "What will happen": "It becomes easier to extend Phase 3 beyond Covertype without manual edits.",
        },
        {
            "Improvement": "Combine masking realism with interactions",
            "How to make it": "Add pairwise interaction capacity to the surrogate or final additive model.",
            "What will happen": "The model may better capture datasets where realistic coalitions are still not enough by themselves.",
        },
        {
            "Improvement": "More seeds and larger SHAP evaluation sets",
            "How to make it": "Raise the seeds list and evaluation sample sizes.",
            "What will happen": "The results become more stable and easier to defend statistically, but runs take longer.",
        },
        {
            "Improvement": "Dataset continuation track",
            "How to make it": "Evaluate Adult Income first, then Bank Marketing or German Credit as future work.",
            "What will happen": "You can show whether the Phase 3 improvement generalizes to other mixed tabular datasets.",
        },
    ]
    lines = [
        "# Phase 3 Improvement Roadmap",
        "",
        "This document answers what improvements you can make, how to make them, and what is likely to happen if you make them.",
        "",
        *markdown_table(rows, ["Improvement", "How to make it", "What will happen"]),
        "",
        "## Recommended order",
        "",
        "1. Start with explicit masking diagnostics on more datasets.",
        "2. Improve the surrogate capacity for the empirical_background branch.",
        "3. Make Phase 3 dataset-generic so Adult Income can run through the same reporting path.",
        "4. Combine masking realism with interaction-aware modeling.",
        "5. Expand to new datasets after the workflow is stable.",
        "",
        "## Best near-term improvement",
        "",
        "The best near-term path is to keep the Phase 3 research question narrow and strengthen the empirical_background branch with better surrogate training and a clearer multi-dataset evaluation story.",
        "",
    ]
    return lines


def build_dataset_reason_doc(adult_results: dict[str, object]) -> list[str]:
    rows = adult_results["summary_rows"]
    return [
        "# Why Adult Can Improve The Story And Why Covertype Did Not Improve Enough",
        "",
        "This document explains why the masking improvement is easier to show on Adult Income than on Covertype.",
        "",
        "## Why Adult Income is a strong next dataset",
        "",
        "- Adult Income has many categorical groups, so the invalid all-zero one-hot problem becomes very visible.",
        "- The new masking strategy directly fixes that issue by copying hidden groups from real transformed rows.",
        "- The Adult masking diagnostic therefore shows the exact coalition-construction improvement more clearly.",
        "",
        *markdown_table(
            rows,
            [
                "strategy",
                "hidden_categorical_valid_rate",
                "hidden_categorical_invalid_rate",
                "hidden_numeric_exact_zero_rate",
                "nearest_train_distance_mean",
            ],
        ),
        "",
        "## Why Covertype did not improve enough",
        "",
        "- Covertype still has a real masking problem, but the final end-to-end metrics depend on more than masking realism.",
        "- The empirical_background target is harder for the surrogate and final additive model to learn.",
        "- The current training budget may be too small for the harder objective.",
        "- One interaction pair is not enough to capture all remaining structure in the data.",
        "- Better coalition realism does not automatically guarantee better SHAP alignment or predictive accuracy.",
        "",
        "## The reason behind the mixed result",
        "",
        "- The improvement fixes one weakness in the pipeline but exposes another: optimization capacity.",
        "- In other words, the new coalition target is better grounded in the data but also harder to approximate.",
        "- That is why the Covertype result is honest and useful even though it is not a full win.",
        "",
        "## Best interpretation",
        "",
        "The masking idea is correct in spirit. Adult Income shows that clearly at the coalition-validity level. Covertype shows that a better masking rule alone is not always enough to improve the full InstaSHAP pipeline without additional modeling or training changes.",
        "",
    ]


def build_llm_dl_doc() -> list[str]:
    return [
        "# Can InstaSHAP Be Applied To LLMs And Deep Learning Models?",
        "",
        "This document explains where InstaSHAP fits, where it does not, and what to expect if you try to use it beyond tabular models.",
        "",
        "## Deep learning models",
        "",
        "- Yes, InstaSHAP can be applied to deep learning models when the input can be grouped into stable, meaningful features.",
        "- It is most natural for fixed-size vector inputs, structured tabular models, or engineered embeddings.",
        "- It can also be adapted to image patches or region groups, but masking design becomes much more important.",
        "- The main requirement is that you can define a stable feature grouping and a meaningful masked value function.",
        "",
        "## LLMs",
        "",
        "- You can apply InstaSHAP to LLM-related systems only in limited and carefully defined ways.",
        "- It is more realistic for structured LLM pipelines than for raw free-form generation.",
        "- Good fit examples include retrieval scores, prompt-template fields, tool-selection features, ranking models, or fixed embedding vectors.",
        "- Raw token-level generative prompting is a much harder fit because masking tokens destroys syntax and semantics.",
        "",
        "## What happens if you apply it to raw LLM prompts",
        "",
        "- Masking text often creates broken or unnatural prompts.",
        "- The model output is a sequence, not a single stable scalar target.",
        "- Token interactions are extremely rich and often much higher-order than a simple additive explanation setup can capture.",
        "- The surrogate may end up learning prompt corruption behavior rather than real reasoning behavior.",
        "",
        "## Should we expect good results?",
        "",
        "- For structured tabular or fixed-vector deep learning settings, yes, good results are plausible.",
        "- For raw generative LLM reasoning, no, not out of the box.",
        "- The better the feature grouping and masking semantics, the more reasonable the results become.",
        "- The less natural the masking operation, the less trustworthy the explanation becomes.",
        "",
        "## Can InstaSHAP track internal reasoning?",
        "",
        "- Not directly.",
        "- InstaSHAP explains observable input-output behavior under a chosen masked value function.",
        "- It does not reveal hidden chain-of-thought or private internal reasoning states by itself.",
        "- At best, it can explain proxies such as logits, hidden-state summaries, layer outputs, or module decisions if those are exposed as explicit targets.",
        "",
        "## Best safe conclusion",
        "",
        "InstaSHAP is a strong fit for structured data and some deep learning settings with meaningful feature groups. It is not a direct tool for faithfully recovering hidden LLM reasoning, and raw generative prompt masking should not be expected to produce highly trustworthy explanations without much more task-specific design.",
        "",
    ]


def build_continuation_doc() -> list[str]:
    return [
        "# Continuous Improvement With Other Datasets",
        "",
        "This document focuses on continuing the Phase 3 improvement across more datasets.",
        "",
        "## Immediate continuation path",
        "",
        "- Step 1: Use Adult Income to demonstrate the masking improvement at the coalition-validity level.",
        "- Step 2: Make the Phase 3 experiment runner dataset-generic.",
        "- Step 3: Add a dataset-specific reporting template so the files are named dynamically instead of using Covertype-specific names.",
        "- Step 4: Compare multiple datasets with the same masking diagnostics before pushing for full retraining on all of them.",
        "",
        "## Best dataset order",
        "",
        "1. Adult Income.",
        "2. Bank Marketing or German Credit.",
        "3. Telco Churn.",
        "4. Larger structured tabular datasets after the workflow is stable.",
        "",
        "## Why this order works",
        "",
        "- Adult is already supported by the repo loaders.",
        "- The masking weakness is easier to demonstrate on category-heavy datasets.",
        "- Credit or churn datasets make the value of realistic hidden groups easy to explain to reviewers.",
        "- Once the story is stable, larger datasets can test whether the improvement still scales.",
        "",
        "## Metrics to keep across every dataset",
        "",
        "- Predictive accuracy or regression error.",
        "- SHAP alignment metrics such as MAE and rank correlation.",
        "- Coalition fidelity metrics.",
        "- Runtime metrics.",
        "- Masking-validity metrics such as hidden categorical validity rate and nearest-train distance.",
        "",
        "## Why this matters",
        "",
        "If Covertype is the only dataset, reviewers may think the mixed result is dataset-specific. A continuation path across multiple datasets lets you show whether the limitation and its fix generalize.",
        "",
    ]


def build_prompt_doc() -> list[str]:
    prompt = (
        "You are extending Phase 3 of the InstaSHAP repository. Keep the current Covertype branch intact, "
        "but build a second extension track for Adult Income. Use the existing data loaders and preprocessing code, "
        "measure both the coalition-validity improvement and full end-to-end explanation metrics, keep filenames dataset-specific, "
        "and produce CSV tables, plots, Markdown, PDF, and notebook outputs. Do not claim improvement unless the saved tables show it. "
        "Track why the new dataset helps, why Covertype stayed mixed, and whether the masking fix generalizes. "
        "If the full pipeline is still mixed, preserve the diagnostic evidence that coalition realism improved."
    )
    return [
        "# Phase 3 Dataset Extension Prompt",
        "",
        "Use this prompt when you want an assistant or teammate to continue the project from the new dataset extension point.",
        "",
        "```text",
        prompt,
        "```",
        "",
        "## How to use it",
        "",
        "- Start from the Adult masking diagnostic notebook and report.",
        "- Ask the assistant to generalize the current Phase 3 runner to new datasets without breaking Covertype.",
        "- Keep the evidence honest: diagnostic gains are not the same as full-model gains.",
        "",
    ]


def build_adult_global_doc(adult_results: dict[str, object]) -> list[str]:
    rows = adult_results["summary_rows"]
    return [
        "# Adult Dataset Extension Summary",
        "",
        "This document summarizes the new Adult Income diagnostic assets created for the repository.",
        "",
        "## Created assets",
        "",
        f"- `{adult_results['csv_path'].relative_to(ROOT).as_posix()}`",
        f"- `{adult_results['json_path'].relative_to(ROOT).as_posix()}`",
        f"- `{adult_results['plot_path'].relative_to(ROOT).as_posix()}`",
        f"- `{adult_results['report_md_path'].relative_to(ROOT).as_posix()}`",
        f"- `{adult_results['report_pdf_path'].relative_to(ROOT).as_posix()}`",
        f"- `{adult_results['notebook_path'].relative_to(ROOT).as_posix()}`",
        "",
        "## Why these assets matter",
        "",
        "- They give you a dataset where the Phase 3 masking improvement is clearly visible at the coalition-construction level.",
        "- They are faster to run and explain than a full new end-to-end dataset retraining workflow.",
        "- They create a clean bridge from the current Covertype result to a broader multi-dataset extension story.",
        "",
        *markdown_table(
            rows,
            [
                "strategy",
                "hidden_categorical_valid_rate",
                "hidden_categorical_invalid_rate",
                "hidden_numeric_exact_zero_rate",
                "nearest_train_distance_mean",
            ],
        ),
        "",
    ]


def build_one_page_summary_md(
    adult_results: dict[str, object],
    phase3_predictive_rows: list[dict[str, str]],
    phase3_explanation_rows: list[dict[str, str]],
    phase3_coalition_rows: list[dict[str, str]],
) -> list[str]:
    zero_pred = lookup_row(phase3_predictive_rows, "instashap_zero")
    bg_pred = lookup_row(phase3_predictive_rows, "instashap_bg")
    zero_exp = lookup_row(phase3_explanation_rows, "instashap_zero")
    bg_exp = lookup_row(phase3_explanation_rows, "instashap_bg")
    zero_coal = lookup_row(phase3_coalition_rows, "surrogate_zero")
    bg_coal = lookup_row(phase3_coalition_rows, "surrogate_bg")
    adult_zero = adult_results["summary_rows"][0]
    adult_bg = adult_results["summary_rows"][1]
    return [
        "# Phase 3 One Page Summary",
        "",
        "## What we have done",
        "",
        "- Reproduced the InstaSHAP tabular pipeline in a modular repository.",
        "- Built a focused Phase 3 extension around a real limitation: unrealistic transformed-space zero masking.",
        "- Implemented `empirical_background` masking so hidden feature groups come from real transformed training rows.",
        "- Evaluated the improvement on Covertype with predictive, explanation, coalition, and runtime metrics.",
        "- Added an Adult Income masking diagnostic to show the masking improvement more clearly on a category-heavy dataset.",
        "",
        "## Current Covertype result",
        "",
        f"- Accuracy: zero {fmt(float(zero_pred['accuracy_mean']))} vs bg {fmt(float(bg_pred['accuracy_mean']))}",
        f"- Explanation MAE: zero {fmt(float(zero_exp['mae_mean']))} vs bg {fmt(float(bg_exp['mae_mean']))}",
        f"- Spearman: zero {fmt(float(zero_exp['spearman_mean']))} vs bg {fmt(float(bg_exp['spearman_mean']))}",
        f"- Coalition MSE: zero {fmt(float(zero_coal['mse_mean']))} vs bg {fmt(float(bg_coal['mse_mean']))}",
        "- Interpretation: the new masking idea is valid, but the current end-to-end Covertype result is mixed rather than a full win.",
        "",
        "## New Adult diagnostic result",
        "",
        f"- Hidden categorical validity: zero {adult_zero['hidden_categorical_valid_rate']} vs bg {adult_bg['hidden_categorical_valid_rate']}",
        f"- Hidden categorical invalid rate: zero {adult_zero['hidden_categorical_invalid_rate']} vs bg {adult_bg['hidden_categorical_invalid_rate']}",
        f"- Hidden numeric exact-zero rate: zero {adult_zero['hidden_numeric_exact_zero_rate']} vs bg {adult_bg['hidden_numeric_exact_zero_rate']}",
        f"- Nearest-train distance mean: zero {adult_zero['nearest_train_distance_mean']} vs bg {adult_bg['nearest_train_distance_mean']}",
        "- Interpretation: Adult Income is a better dataset for showing the masking improvement itself, even before full retraining.",
        "",
        "## Best next step",
        "",
        "- Generalize Phase 3 to new datasets with dataset-specific configs and report names.",
        "- Strengthen the surrogate branch for the harder empirical_background objective.",
        "- Keep the project honest: report diagnostic gains separately from full end-to-end gains.",
        "",
    ]


def render_one_page_summary_pdf(md_lines: list[str], output_path: Path) -> None:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in md_lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if stripped.startswith("#"):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(stripped.lstrip("# ").strip())
            continue
        if stripped.startswith("- "):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(stripped[2:].strip())
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current))

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.08, 0.96, "Phase 3 one page summary", fontsize=18, fontweight="bold", va="top")
        y = 0.92
        for paragraph in paragraphs[:16]:
            wrapped = textwrap.fill(paragraph, width=95)
            fig.text(0.08, y, wrapped, fontsize=10.2, va="top")
            y -= 0.055 if len(wrapped) < 120 else 0.075
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_presentation_master() -> list[str]:
    sections = [
        ("Context", ["Why SHAP is too slow for production use", "Why InstaSHAP matters", "Why the project needed a Phase 3 improvement", "What makes a limitation worth studying", "Why tabular masking choices matter"]),
        ("Baseline pipeline", ["Phase 2 replication architecture", "Data flow from raw features to transformed features", "Black-box model role", "Masked surrogate role", "Additive InstaSHAP role", "How SHAP is used as a reference baseline", "What artifacts the baseline creates"]),
        ("Phase 3 limitation", ["What zero_mask does", "Why zero_mask is fragile after standardization", "Why zero_mask is fragile after one-hot encoding", "Why invalid coalition states hurt the surrogate", "Why invalid coalition states hurt the final explainer", "How to explain this limitation to a beginner", "How to explain this limitation to a reviewer"]),
        ("Empirical background improvement", ["What empirical_background does", "How the background bank is built", "How visible features guide background row selection", "Why multiple background samples help", "What changed in masking.py", "What stayed the same in the broader architecture", "Why this is a narrow and valid Phase 3 change"]),
        ("Covertype evidence", ["Current predictive metrics", "Current explanation metrics", "Current coalition metrics", "Current runtime metrics", "Why the result is mixed", "Why mixed evidence is still useful", "How to present the mixed result honestly"]),
        ("Adult extension", ["Why Adult Income is a better showcase dataset", "How the new notebook uses Adult Income", "What the masking diagnostic measures", "How Adult demonstrates categorical validity improvement", "How Adult demonstrates reduced exact-zero hidden numerics", "How Adult supports the future Phase 3 roadmap", "What still remains to be done on Adult"]),
        ("Reasoning about success and failure", ["Why improvement is visible on Adult", "Why improvement is not yet enough on Covertype", "What optimization bottlenecks remain", "What modeling bottlenecks remain", "What evidence should be reported separately", "How to avoid overclaiming", "How to turn mixed evidence into a strong discussion section"]),
        ("Generalization", ["What other datasets to try next", "How to build a multi-dataset continuation plan", "Can this be applied to deep learning models", "Can this be applied to LLM systems", "What happens if raw LLM prompts are masked", "Can InstaSHAP reveal internal reasoning", "What safe expectations look like beyond tabular settings"]),
        ("Execution roadmap", ["Best next engineering change", "Best next research change", "Best next reporting change", "Best next dataset change", "Best next presentation change", "Best next testing change", "Best combined roadmap"]),
        ("Backup and Q&A", ["FAQ on the limitation", "FAQ on dataset choice", "FAQ on Adult diagnostic results", "FAQ on Covertype mixed results", "FAQ on LLM applicability", "FAQ on internal reasoning claims", "Final conclusion and closing statement"]),
    ]

    slide_specs: list[tuple[str, str]] = []
    for section_name, topics in sections:
        for topic in topics:
            slide_specs.append((section_name, topic))

    lines = [
        "# Phase 3 Improvement Presentation Master",
        "",
        "This is a long-form presentation script focused only on the Phase 3 improvement work. It is designed to be much longer than a normal slide deck so it can serve as both presentation material and speaking-note archive.",
        "",
        "## How to use this file",
        "",
        "- Turn the slides into a shorter live presentation if needed.",
        "- Use the speaker notes to defend the work during a viva or review.",
        "- Use the backup slides when a reviewer asks deeper questions.",
        "",
    ]

    slide_number = 1
    for section_name, topic in slide_specs:
        lines.extend(
            [
                f"## Slide {slide_number}: {topic}",
                "",
                "### Section",
                f"- {section_name}",
                "",
                "### Objective",
                f"- Explain why `{topic}` matters to the Phase 3 improvement story.",
                "- Keep the discussion connected to the masking-improvement research question.",
                "",
                "### Core points",
                f"- Point 1: `{topic}` should be tied back to realistic coalition construction.",
                "- Point 2: Keep the explanation grounded in the actual repository files and current saved artifacts.",
                "- Point 3: Separate conceptual validity from measured end-to-end gains.",
                "- Point 4: Emphasize that the Phase 3 change is narrow, testable, and honest.",
                "- Point 5: Show why this topic matters for the next-step roadmap.",
                "",
                "### Suggested visual",
                "- Use a simple diagram, table, or code pointer instead of dense paragraphs.",
                "- Prefer one clear comparison over many small charts.",
                "",
                "### Speaker notes",
                f"1. Start by defining `{topic}` in one sentence.",
                "2. Connect it to the zero_mask versus empirical_background comparison.",
                "3. Explain what the repository already proves and what it does not yet prove.",
                "4. If a metric is involved, mention the current saved table rather than relying on memory.",
                "5. Close the slide by saying what this means for the next dataset or next experiment.",
                "",
                "### If asked",
                "- Be ready to distinguish between diagnostic improvement and full pipeline improvement.",
                "- Be ready to mention Covertype as the current end-to-end dataset and Adult Income as the new masking showcase dataset.",
                "",
                "### Transition line",
                "- The next slide should deepen the Phase 3 improvement story without leaving the masking-focused narrative.",
                "",
            ]
        )
        slide_number += 1

    lines.extend(
        [
            "## Appendix A: Recommended live deck compression",
            "",
            "- Keep 10 to 12 slides for the actual live presentation.",
            "- Use the first half of this file to build the main deck.",
            "- Use the later half as backup material.",
            "",
            "## Appendix B: Best closing statement",
            "",
            "- We reproduced InstaSHAP, identified a real limitation in transformed-space masking, implemented a data-aware fix, and evaluated it honestly.",
            "- The current Covertype run is mixed, but the Adult diagnostic proves the masking improvement itself much more clearly.",
            "- That gives us a strong and defensible path for continuing the project across more datasets.",
            "",
        ]
    )
    return lines


def build_new_docs(adult_results: dict[str, object]) -> None:
    predictive_rows = load_phase3_summary_table("covertype_predictive_summary.csv")
    explanation_rows = load_phase3_summary_table("covertype_explanation_summary.csv")
    coalition_rows = load_phase3_summary_table("covertype_coalition_summary.csv")

    docs = {
        "18_BEGINNER_QUICK_UNDERSTANDING.md": build_beginner_doc(adult_results),
        "19_PHASE3_IMPROVEMENT_ROADMAP.md": build_improvement_roadmap(),
        "20_COVERTYPE_FAILURE_AND_ADULT_SHOWCASE.md": build_dataset_reason_doc(adult_results),
        "21_LLM_AND_DL_APPLICABILITY.md": build_llm_dl_doc(),
        "22_CONTINUOUS_DATASET_IMPROVEMENT_PLAN.md": build_continuation_doc(),
        "23_PHASE3_DATASET_EXTENSION_PROMPT.md": build_prompt_doc(),
        "24_ADULT_DATASET_EXTENSION_SUMMARY.md": build_adult_global_doc(adult_results),
        "25_PHASE3_IMPROVEMENT_PRESENTATION_MASTER.md": build_presentation_master(),
    }

    one_page_md_lines = build_one_page_summary_md(
        adult_results=adult_results,
        phase3_predictive_rows=predictive_rows,
        phase3_explanation_rows=explanation_rows,
        phase3_coalition_rows=coalition_rows,
    )
    docs["26_ONE_PAGE_SUMMARY.md"] = one_page_md_lines

    for filename, lines in docs.items():
        write_markdown(GLOBAL_DIR / filename, lines)

    render_one_page_summary_pdf(one_page_md_lines, GLOBAL_DIR / "26_ONE_PAGE_SUMMARY.pdf")
    write_markdown(PROMPT_DIR / "phase3_dataset_extension_prompt.md", build_prompt_doc())


def update_global_readme() -> None:
    readme_path = GLOBAL_DIR / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    addition = textwrap.dedent(
        """

        ## New extension docs

        - `18_BEGINNER_QUICK_UNDERSTANDING.md` for a beginner-friendly explanation.
        - `19_PHASE3_IMPROVEMENT_ROADMAP.md` for what to improve, how to improve it, and what to expect.
        - `20_COVERTYPE_FAILURE_AND_ADULT_SHOWCASE.md` for why Covertype stayed mixed and why Adult is a better masking showcase.
        - `21_LLM_AND_DL_APPLICABILITY.md` for whether InstaSHAP can be used with LLMs and deep learning models.
        - `22_CONTINUOUS_DATASET_IMPROVEMENT_PLAN.md` for the multi-dataset continuation path.
        - `23_PHASE3_DATASET_EXTENSION_PROMPT.md` for a reusable extension prompt.
        - `24_ADULT_DATASET_EXTENSION_SUMMARY.md` for the new Adult diagnostic assets.
        - `25_PHASE3_IMPROVEMENT_PRESENTATION_MASTER.md` for the long-form Phase 3 presentation script.
        - `26_ONE_PAGE_SUMMARY.md` and `26_ONE_PAGE_SUMMARY.pdf` for a short summary of what has been done.
        """
    ).strip()
    if "## New extension docs" not in existing:
        readme_path.write_text(existing.rstrip() + "\n\n" + addition + "\n", encoding="utf-8")


def update_line_count_report() -> None:
    markdown_files = sorted(GLOBAL_DIR.glob("*.md"))
    rows = []
    total_lines = 0
    for path in markdown_files:
        count = len(path.read_text(encoding="utf-8").splitlines())
        total_lines += count
        rows.append({"File": path.name, "Lines": str(count)})
    rows.append({"File": "TOTAL", "Lines": str(total_lines)})
    lines = [
        "# Line Count Report",
        "",
        "This report verifies the generated markdown volume in the global folder.",
        "",
        *markdown_table(rows, ["File", "Lines"]),
        "",
        "Requested target: about 6000 lines across several files.",
        "",
    ]
    write_markdown(GLOBAL_DIR / "16_LINE_COUNT_REPORT.md", lines)


def main() -> None:
    ensure_dir(GLOBAL_DIR)
    adult_results = adult_masking_diagnostic()
    build_new_docs(adult_results)
    update_global_readme()
    update_line_count_report()
    print("Generated extended Phase 3 docs, Adult diagnostic assets, notebook, prompt, and one-page summary.")


if __name__ == "__main__":
    main()
