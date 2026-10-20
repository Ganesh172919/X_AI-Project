"""Extended visualization for Phase 3 — innovation-specific comparison plots."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns

PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96E6A1", "#DDA0DD", "#F4D03F"]
VARIANT_COLORS = {"instashap_zero": "#FF6B6B", "instashap_bg": "#4ECDC4",
                  "instashap_curriculum": "#45B7D1", "instashap_full": "#96E6A1"}
VARIANT_LABELS = {"instashap_zero": "Zero-Mask\n(Baseline)", "instashap_bg": "Background\n(Innov. 1)",
                  "instashap_curriculum": "Curriculum\n(Innov. 1+2)", "instashap_full": "Full\n(Innov. 1+2+3)"}


def savefig(fig: plt.Figure, filepath: str | Path, dpi: int = 150) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training_curves(
    histories: dict[str, list[dict[str, float]]],
    title: str,
    filepath: str | Path,
) -> None:
    """Training-loss curves side-by-side for train/val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, split in enumerate(("train_loss", "val_loss")):
        for name, hist in histories.items():
            vals = [h.get(split, np.nan) for h in hist]
            axes[i].plot(vals, label=name, linewidth=2)
        axes[i].set_title(f"{'Train' if i == 0 else 'Validation'} Loss")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, filepath)


def plot_innovation_comparison_bars(
    metrics: dict[str, dict[str, float]],
    metric_key: str,
    title: str,
    ylabel: str,
    filepath: str | Path,
    higher_is_better: bool = True,
) -> None:
    """Grouped bar chart comparing 4 variants for a given metric."""
    variants = list(metrics.keys())
    values = [metrics[v].get(metric_key, 0.0) for v in variants]
    stds = [metrics[v].get(f"{metric_key}_std", 0.0) for v in variants]
    colors = [VARIANT_COLORS.get(v, "#999") for v in variants]
    labels = [VARIANT_LABELS.get(v, v) for v in variants]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, yerr=stds, color=colors, edgecolor="white", linewidth=2, capsize=6, alpha=0.9)

    best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
    bars[best_idx].set_edgecolor("#2E7D32")
    bars[best_idx].set_linewidth(3)

    for bar, val, std in zip(bars, values, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, filepath)


def plot_explanation_scatter_multi(
    shap_values: np.ndarray,
    variant_values: dict[str, np.ndarray],
    feature_names: list[str],
    filepath: str | Path,
    max_features: int = 4,
) -> None:
    """Multi-panel scatter: SHAP vs each InstaSHAP variant for top features."""
    n_variants = len(variant_values)
    show_feats = min(max_features, len(feature_names))

    fig, axes = plt.subplots(n_variants, show_feats, figsize=(4 * show_feats, 4 * n_variants))
    if n_variants == 1:
        axes = axes[np.newaxis, :]

    for vi, (vname, vals) in enumerate(variant_values.items()):
        color = VARIANT_COLORS.get(vname, "#999")
        label = VARIANT_LABELS.get(vname, vname).replace("\n", " ")
        for fi in range(show_feats):
            ax = axes[vi, fi]
            sv = shap_values[:, fi].ravel()
            iv = vals[:, fi].ravel()
            ax.scatter(sv, iv, alpha=0.5, s=10, color=color)
            lo = min(sv.min(), iv.min())
            hi = max(sv.max(), iv.max())
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, alpha=0.5)
            if vi == 0:
                ax.set_title(feature_names[fi], fontsize=10)
            if fi == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xlabel("SHAP" if vi == n_variants - 1 else "", fontsize=8)
            ax.grid(True, alpha=0.2)

    fig.suptitle("SHAP vs InstaSHAP — Per Feature × Variant", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, filepath)


def plot_convergence_comparison(
    histories: dict[str, list[float]],
    filepath: str | Path,
) -> None:
    """Val-loss curves for different surrogate strategies with 95% threshold line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, losses in histories.items():
        color = VARIANT_COLORS.get(name, "#999")
        label = VARIANT_LABELS.get(name, name).replace("\n", " ")
        ax.plot(losses, label=label, linewidth=2, color=color)
        best = min(losses)
        threshold = best / 0.95
        ep = next((i for i, v in enumerate(losses) if v <= threshold), len(losses))
        ax.axvline(ep, linestyle=":", color=color, alpha=0.5)
        ax.text(ep + 0.3, threshold, f"95%@{ep}", fontsize=8, color=color)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Surrogate Convergence — Curriculum vs Standard", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, filepath)


def plot_stability_distribution(
    stability_scores: dict[str, np.ndarray],
    filepath: str | Path,
) -> None:
    """Histogram of per-sample stability scores for ensemble vs single."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, scores in stability_scores.items():
        color = VARIANT_COLORS.get(name, "#999")
        label = VARIANT_LABELS.get(name, name).replace("\n", " ")
        ax.hist(scores, bins=30, alpha=0.5, color=color, label=f"{label} (μ={scores.mean():.3f})", edgecolor="white")

    ax.set_xlabel("Stability Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Explanation Stability Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, filepath)


def plot_coalition_fidelity_by_size(
    fidelity_data: dict[str, dict[int, float]],
    filepath: str | Path,
) -> None:
    """Line plot: surrogate fidelity MSE vs coalition size for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, size_mse in fidelity_data.items():
        color = VARIANT_COLORS.get(name, "#999")
        label = VARIANT_LABELS.get(name, name).replace("\n", " ")
        sizes = sorted(size_mse.keys())
        values = [size_mse[s] for s in sizes]
        ax.plot(sizes, values, marker="o", linewidth=2, color=color, label=label)

    ax.set_xlabel("Coalition Size |S|", fontsize=12)
    ax.set_ylabel("Surrogate Fidelity MSE", fontsize=12)
    ax.set_title("Coalition Fidelity by Size — Masking Strategy Impact", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    savefig(fig, filepath)


def plot_metric_bar_chart(
    model_metrics: dict[str, float],
    metric_name: str,
    title: str,
    filepath: str | Path,
    paper_benchmarks: dict[str, float] | None = None,
) -> None:
    """Simple bar chart for model comparison with optional paper benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(model_metrics.keys())
    values = list(model_metrics.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=2, alpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    if paper_benchmarks:
        for name, pval in paper_benchmarks.items():
            if name in names:
                idx = names.index(name)
                ax.plot([idx - 0.3, idx + 0.3], [pval, pval], "--", color="black", linewidth=2)
                ax.text(idx + 0.35, pval, f"Paper: {pval:.3f}", fontsize=8, color="black")

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    savefig(fig, filepath)


def plot_shape_functions(
    preprocessor: object,
    gam_model: object,
    focus_features: list[str],
    X_train: np.ndarray,
    device: object,
    filepath: str | Path,
    title: str = "Learned Shape Functions",
) -> None:
    """Univariate shape function plots for GAM components."""
    import torch
    from data.preprocessing import TabularPreprocessor

    prep = preprocessor
    n = len(focus_features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, fn in zip(axes, focus_features):
        grp = prep.group(fn)
        if grp.kind == "numeric":
            col_vals = X_train[:, grp.start]
            sort_idx = np.argsort(col_vals)
            ref_input = torch.zeros(len(col_vals), prep.input_dim, device=device)
            ref_input[:, grp.start] = torch.from_numpy(col_vals[sort_idx]).float()
            with torch.no_grad():
                gam_model.eval()
                comp = gam_model.single_component(ref_input, (fn,))
            if comp.dim() > 1:
                comp = comp[:, 0]
            ax.plot(col_vals[sort_idx], comp.cpu().numpy(), linewidth=2, color="#45B7D1")
            ax.set_xlabel(fn)
        else:
            cats = grp.categories or []
            bars_vals = []
            for ci, cat in enumerate(cats):
                one_hot = torch.zeros(1, prep.input_dim, device=device)
                one_hot[0, grp.start + ci] = 1.0
                with torch.no_grad():
                    gam_model.eval()
                    comp = gam_model.single_component(one_hot, (fn,))
                bars_vals.append(float(comp[0, 0]))
            ax.bar(cats, bars_vals, color="#4ECDC4", edgecolor="white")
            ax.set_xlabel(fn)
            ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Component Output")
        ax.set_title(fn, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    savefig(fig, filepath)


def plot_summary_radar(
    variant_metrics: dict[str, dict[str, float]],
    metric_keys: list[str],
    filepath: str | Path,
) -> None:
    """Radar chart comparing all variants across multiple metrics."""
    from math import pi

    N = len(metric_keys)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for vname, mdict in variant_metrics.items():
        values = [mdict.get(k, 0.0) for k in metric_keys]
        values += values[:1]
        color = VARIANT_COLORS.get(vname, "#999")
        label = VARIANT_LABELS.get(vname, vname).replace("\n", " ")
        ax.plot(angles, values, linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_keys, fontsize=9)
    ax.set_title("Multi-Metric Comparison Radar", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    savefig(fig, filepath)
