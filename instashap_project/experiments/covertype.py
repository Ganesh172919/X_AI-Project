"""Covertype experiment runner."""

from __future__ import annotations

from typing import Any

from instashap_project.data.loaders import load_covertype
from instashap_project.experiments.common import ExperimentResult, run_tabular_experiment


def run(config: dict[str, Any], selected_model: str = "all") -> ExperimentResult:
    """Run the Covertype redundancy experiment."""

    bundle = load_covertype(
        max_rows=config["datasets"]["covertype"].get("max_rows"),
        seed=int(config["global"]["seed"]),
    )
    return run_tabular_experiment(
        bundle=bundle,
        config=config,
        selected_model=selected_model,
        focus_features=["elevation", "soil_climate_zone", "aspect"],
        focus_interaction=("elevation", "soil_climate_zone"),
    )

