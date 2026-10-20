"""Adult Income experiment runner."""

from __future__ import annotations

from typing import Any

from instashap_project.data.loaders import load_adult_income
from instashap_project.experiments.common import ExperimentResult, run_tabular_experiment


def run(config: dict[str, Any], selected_model: str = "all") -> ExperimentResult:
    """Run the Adult Income supplementary experiment."""

    bundle = load_adult_income(
        max_rows=config["datasets"]["adult"].get("max_rows"),
        seed=int(config["global"]["seed"]),
    )
    return run_tabular_experiment(
        bundle=bundle,
        config=config,
        selected_model=selected_model,
        focus_features=["age", "capital_gain", "hours_per_week", "education"],
        focus_interaction=None,
    )
