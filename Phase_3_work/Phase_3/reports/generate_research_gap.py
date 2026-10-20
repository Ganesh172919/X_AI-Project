"""1-page Research Gap PDF generator."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from utils.reproducibility import ensure_dir


def generate_research_gap_report(config: dict) -> None:
    """Generate a compact 1-page research gap PDF."""
    output_root = Path(config["global"].get("output_root", "results"))
    report_dir = Path("reports")
    ensure_dir(report_dir)

    summary_path = output_root / "artifacts" / "covertype" / "covertype_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    pdf_path = report_dir / "phase3_research_gap_1page.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        fig = plt.figure(figsize=(11, 8.5))

        # Title
        fig.text(0.5, 0.96, "Research Gap Analysis: Improving InstaSHAP via Three Targeted Innovations",
                 ha="center", fontsize=14, fontweight="bold")
        fig.text(0.5, 0.93, "Phase 3 — Covertype Dataset Study", ha="center", fontsize=11, color="#555")

        # Left column: Gaps
        ax_left = fig.add_axes([0.05, 0.45, 0.43, 0.45])
        ax_left.axis("off")
        gap_text = (
            "IDENTIFIED GAPS IN INSTASHAP (ICLR 2025)\n\n"
            "Gap 1 [CRITICAL]: Zero-Masking Off-Distribution\n"
            "  • x*mask creates invalid categorical states\n"
            "  • Zeroed one-hot groups = impossible inputs\n"
            "  • Surrogate learns wrong value function\n\n"
            "Gap 2 [HIGH]: Uniform Coalition Difficulty\n"
            "  • All mask sizes treated equally from epoch 1\n"
            "  • No progressive complexity → slow convergence\n\n"
            "Gap 3 [HIGH]: Single-Surrogate Fragility\n"
            "  • One surrogate cascades all errors\n"
            "  • No confidence signal on explanations\n"
        )
        ax_left.text(0, 1, gap_text, fontsize=8.5, fontfamily="monospace",
                     verticalalignment="top", transform=ax_left.transAxes)

        # Right column: Innovations
        ax_right = fig.add_axes([0.52, 0.45, 0.43, 0.45])
        ax_right.axis("off")
        innov_text = (
            "PROPOSED INNOVATIONS\n\n"
            "Innovation 1: Empirical-Background Masking\n"
            "  • Replace absent features with real training rows\n"
            "  • Preserves one-hot validity + marginal distribution\n"
            "  • Ref: Aas et al. 2019, ViaSHAP ICML 2025\n\n"
            "Innovation 2: Curriculum-Weighted Training\n"
            "  • 3-phase schedule: warm-up → standard → hard\n"
            "  • Temperature-controlled Shapley kernel\n"
            "  • Ref: Bengio 2009, ICLR 2026 Coalition Sampling\n\n"
            "Innovation 3: Multi-Surrogate Ensemble\n"
            "  • Train 3 surrogates, average predictions\n"
            "  • Variance → explanation confidence score\n"
            "  • Ref: Explanation Multiplicity 2026\n"
        )
        ax_right.text(0, 1, innov_text, fontsize=8.5, fontfamily="monospace",
                      verticalalignment="top", transform=ax_right.transAxes)

        # Bottom: Results table
        ax_table = fig.add_axes([0.05, 0.05, 0.90, 0.35])
        ax_table.axis("off")
        ax_table.set_title("Results Summary (mean ± std across seeds)", fontsize=11, fontweight="bold", pad=10)

        variants = ["instashap_zero", "instashap_bg", "instashap_curriculum", "instashap_full"]
        headers = ["Variant", "Accuracy", "Expl MSE", "Spearman ρ"]
        rows = []
        for vk in variants:
            acc = summary.get(f"{vk}_accuracy_mean", 0)
            acc_s = summary.get(f"{vk}_accuracy_std", 0)
            mse = summary.get(f"{vk}_explanation_mse_mean", 0)
            rho = summary.get(f"{vk}_spearman_rho_mean", 0)
            rows.append([
                vk.replace("instashap_", ""),
                f"{acc:.4f}±{acc_s:.4f}" if isinstance(acc, float) else "—",
                f"{mse:.6f}" if isinstance(mse, float) else "—",
                f"{rho:.4f}" if isinstance(rho, float) else "—",
            ])

        table = ax_table.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.3, 1.8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#45B7D1")
                cell.set_text_props(color="white", fontweight="bold")
            elif row == len(rows):
                cell.set_facecolor("#e8f5e9")
                cell.set_text_props(fontweight="bold")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Research gap PDF saved: {pdf_path}")

    # Markdown companion
    md_path = report_dir / "phase3_research_gap_1page.md"
    md_lines = [
        "# Research Gap: Improving InstaSHAP via Three Targeted Innovations",
        "",
        "## Problem: InstaSHAP (ICLR 2025) has three key limitations:",
        "1. **Zero-masking** creates off-distribution inputs (invalid one-hot states)",
        "2. **Uniform mask sampling** wastes capacity on hard coalitions early in training",
        "3. **Single surrogate** fragility with no uncertainty signal",
        "",
        "## Solution: Three layered innovations:",
        "1. **Empirical-Background Masking** — replace absent features with real training data",
        "2. **Curriculum-Weighted Training** — progressive warm-up → standard → hard schedule",
        "3. **Multi-Surrogate Ensemble** — average 3 surrogates + variance as confidence",
        "",
        "## Key References",
        "- Enouen & Liu (ICLR 2025), Lundberg & Lee (2017), Aas et al. (2019)",
        "- ViaSHAP (ICML 2025), Curriculum Learning (Bengio 2009)",
        "- Explanation Multiplicity (2026), SHAP-IQ (NeurIPS 2024)",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
