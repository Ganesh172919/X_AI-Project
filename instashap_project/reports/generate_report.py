"""Generate a multi-page PDF report from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _collect_summaries(project_root: Path) -> list[dict]:
    summaries: list[dict] = []
    for summary_path in sorted((project_root / "results" / "artifacts").glob("*/*_summary.json")):
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
    return summaries


def _figure_with_text(title: str, lines: list[str], figsize: tuple[float, float] = (8.27, 11.69)) -> plt.Figure:
    fig = plt.figure(figsize=figsize)
    fig.text(0.08, 0.95, title, fontsize=18, fontweight="bold", va="top")
    y = 0.9
    for line in lines:
        fig.text(0.08, y, line, fontsize=10, va="top", family="monospace" if "|" in line else None)
        y -= 0.03
    return fig


def _table_to_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["No rows available."]
    rendered = df.round(4).to_string(index=False)
    return rendered.splitlines()


def _add_image_page(pdf: PdfPages, project_root: Path, title: str, relative_path: str) -> None:
    image_path = project_root / relative_path
    if not image_path.exists():
        return
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(title, fontsize=16, y=0.97)
    axis = fig.add_axes([0.06, 0.05, 0.88, 0.88])
    axis.imshow(plt.imread(image_path))
    axis.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def generate_full_report(project_root: Path | None = None) -> Path:
    """Build the reproducibility report PDF from saved experiment artifacts."""

    project_root = project_root or _project_root()
    summaries = _collect_summaries(project_root)
    output_path = project_root / "reports" / "instashap_reproducibility_report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        intro_lines = [
            "Objective: reproduce the key InstaSHAP tabular experiments from the ICLR 2025 paper.",
            "Datasets: Bike Sharing (synergy), Covertype (redundancy), Adult Income (supplementary).",
            "Methodology: black-box baseline, additive GAMs, masked surrogate, and InstaSHAP Eq. (20).",
            "Data source implementation: ucimlrepo fetch_ucirepo(id=275 / 31 / 2).",
            "Outputs: metrics tables, plots, SHAP vs InstaSHAP comparison, and paper-reference tables.",
        ]
        pdf.savefig(_figure_with_text("InstaSHAP Reproducibility Report", intro_lines))
        plt.close("all")

        for summary in summaries:
            dataset = summary["dataset"]
            metrics_df = pd.read_csv(project_root / summary["metrics_table"])
            paper_df = pd.read_csv(project_root / summary["paper_comparison_table"])
            explanation_df = pd.read_csv(project_root / summary["explanation_table"])
            overview_lines = [
                f"Dataset: {dataset}",
                f"Task: {summary['task']}",
                f"Features: {', '.join(summary['features'])}",
                f"Interaction pairs: {summary['interaction_pairs']}",
                "",
                "Implementation notes:",
                " - Black-box model: MLP by default, with optional RandomForest.",
                " - GAM-1/GAM-2: additive neural subnetworks over feature groups.",
                " - InstaSHAP: masked additive training against a learned masked surrogate.",
                " - SHAP: permutation explainer aggregated back to original feature groups.",
            ]
            fig = _figure_with_text(f"{dataset.title()} Overview", overview_lines)
            pdf.savefig(fig)
            plt.close(fig)

            metrics_fig = _figure_with_text(f"{dataset.title()} Metrics", _table_to_lines(metrics_df))
            pdf.savefig(metrics_fig)
            plt.close(metrics_fig)

            paper_fig = _figure_with_text(f"{dataset.title()} vs Paper", _table_to_lines(paper_df))
            pdf.savefig(paper_fig)
            plt.close(paper_fig)

            explanation_fig = _figure_with_text(f"{dataset.title()} Explanation Comparison", _table_to_lines(explanation_df))
            pdf.savefig(explanation_fig)
            plt.close(explanation_fig)

            for relative_plot in summary.get("plots", [])[:4]:
                _add_image_page(pdf, project_root, f"{dataset.title()} Plot", relative_plot)

        conclusion_lines = [
            "Observations:",
            " - The report juxtaposes reproduced metrics against the values quoted in the paper.",
            " - Remaining gaps usually come from limited epochs, smaller subsets, and missing original hyperparameters.",
            " - InstaSHAP artifacts include both predictive performance and explanation-fidelity measurements.",
            "",
            "Recommendation:",
            "Increase epochs and Covertype sample size for a closer numerical match to the paper before final benchmarking.",
        ]
        final_fig = _figure_with_text("Comparison and Observations", conclusion_lines)
        pdf.savefig(final_fig)
        plt.close(final_fig)

    return output_path


if __name__ == "__main__":
    report_path = generate_full_report()
    print(report_path)

