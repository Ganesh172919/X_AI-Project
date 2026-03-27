"""Bike Sharing experiment runner."""

from __future__ import annotations

from typing import Any

from instashap_project.data.loaders import load_bike_sharing
from instashap_project.experiments.common import ExperimentResult, run_tabular_experiment


def run(config: dict[str, Any], selected_model: str = "all") -> ExperimentResult:
    """Run the Bike Sharing synergy experiment."""

    bundle = load_bike_sharing()
    return run_tabular_experiment(
        bundle=bundle,
        config=config,
        selected_model=selected_model,
        focus_features=["hour", "workingday", "temp"],
        focus_interaction=("hour", "workingday"),
    )

