"""Visualization helpers for additive-model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from instashap_project.data.preprocessing import TabularPreprocessor
from instashap_project.models.gam import GAMModel


sns.set_theme(style="whitegrid")


def _ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_training_curves(histories: dict[str, list[dict[str, float]]], output_path: str | Path, title: str) -> None:
    """Plot train/validation curves for one or more models."""

    output_path = _ensure_parent(output_path)
    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        if not history:
            continue
        frame = pd.DataFrame(history)
        plt.plot(frame["epoch"], frame["train_loss"], label=f"{label} train")
        plt.plot(frame["epoch"], frame["val_loss"], linestyle="--", label=f"{label} val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, metric: str, output_path: str | Path, title: str) -> None:
    """Plot a simple bar chart for one metric across models."""

    output_path = _ensure_parent(output_path)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=metrics_df, x="model", y=metric, hue="model", dodge=False, legend=False, palette="crest")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _output_vector(values: np.ndarray, output_index: int) -> np.ndarray:
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    return values[:, output_index]


def plot_shape_function(
    model: GAMModel,
    preprocessor: TabularPreprocessor,
    raw_frame: pd.DataFrame,
    feature_name: str,
    output_path: str | Path,
    device: str | torch.device,
    output_index: int = 0,
    num_points: int = 100,
) -> None:
    """Plot the learned univariate component for a single feature."""

    output_path = _ensure_parent(output_path)
    feature_group = preprocessor.group(feature_name)
    if feature_group.kind == "numeric":
        values = np.linspace(raw_frame[feature_name].min(), raw_frame[feature_name].max(), num_points)
    else:
        values = feature_group.categories or list(pd.Series(raw_frame[feature_name]).astype(str).unique())

    feature_frame = preprocessor.make_feature_frame(feature_name, values)
    transformed = preprocessor.transform(feature_frame)
    with torch.no_grad():
        tensor_inputs = torch.from_numpy(transformed.astype(np.float32)).to(torch.device(device))
        contributions = model.single_component(tensor_inputs, (feature_name,)).cpu().numpy()
    contributions = _output_vector(contributions, output_index)

    plt.figure(figsize=(8, 4))
    if feature_group.kind == "numeric":
        plt.plot(values, contributions, color="#0f766e", linewidth=2.0)
        plt.xlabel(feature_name.replace("_", " ").title())
    else:
        sns.barplot(x=list(map(str, values)), y=contributions, color="#0f766e")
        plt.xticks(rotation=40, ha="right")
        plt.xlabel(feature_name.replace("_", " ").title())
    plt.ylabel("Component value")
    plt.title(f"Shape function: {feature_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_interaction_heatmap(
    model: GAMModel,
    preprocessor: TabularPreprocessor,
    raw_frame: pd.DataFrame,
    first_feature: str,
    second_feature: str,
    output_path: str | Path,
    device: str | torch.device,
    output_index: int = 0,
    grid_points: int = 50,
) -> None:
    """Plot the pairwise interaction component as a heatmap."""

    output_path = _ensure_parent(output_path)
    first_group = preprocessor.group(first_feature)
    second_group = preprocessor.group(second_feature)

    first_values: list[Any] | np.ndarray
    second_values: list[Any] | np.ndarray
    if first_group.kind == "numeric":
        first_values = np.linspace(raw_frame[first_feature].min(), raw_frame[first_feature].max(), grid_points)
    else:
        first_values = first_group.categories or list(pd.Series(raw_frame[first_feature]).astype(str).unique())
    if second_group.kind == "numeric":
        second_values = np.linspace(raw_frame[second_feature].min(), raw_frame[second_feature].max(), grid_points)
    else:
        second_values = second_group.categories or list(pd.Series(raw_frame[second_feature]).astype(str).unique())

    grid_frame = preprocessor.make_interaction_frame(first_feature, first_values, second_feature, second_values)
    transformed = preprocessor.transform(grid_frame)
    with torch.no_grad():
        tensor_inputs = torch.from_numpy(transformed.astype(np.float32)).to(torch.device(device))
        values = model.single_component(tensor_inputs, (first_feature, second_feature)).cpu().numpy()
    values = _output_vector(values, output_index).reshape(len(first_values), len(second_values))

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        values,
        cmap="coolwarm",
        xticklabels=list(map(str, second_values)),
        yticklabels=list(map(str, first_values)),
    )
    plt.title(f"Interaction heatmap: {first_feature} x {second_feature}")
    plt.xlabel(second_feature.replace("_", " ").title())
    plt.ylabel(first_feature.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_importance(
    attributions: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot mean absolute attribution per feature."""

    output_path = _ensure_parent(output_path)
    if attributions.ndim == 3:
        summary = np.mean(np.abs(attributions), axis=(0, 2))
    else:
        summary = np.mean(np.abs(attributions), axis=0)
    frame = pd.DataFrame({"feature": feature_names, "importance": summary}).sort_values("importance", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=frame, x="importance", y="feature", color="#0f766e")
    plt.title(title)
    plt.xlabel("Mean absolute attribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_explanation_alignment(
    reference: np.ndarray,
    candidate: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot per-feature mean absolute difference between two explanation tensors."""

    output_path = _ensure_parent(output_path)
    difference = np.abs(reference - candidate)
    if difference.ndim == 3:
        summary = difference.mean(axis=(0, 2))
    else:
        summary = difference.mean(axis=0)
    frame = pd.DataFrame({"feature": feature_names, "mae": summary}).sort_values("mae", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=frame, x="mae", y="feature", color="#be185d")
    plt.title(title)
    plt.xlabel("Mean absolute error")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
