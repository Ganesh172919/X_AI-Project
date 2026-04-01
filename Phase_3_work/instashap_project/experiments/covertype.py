"""Covertype runner for the Phase 3 background-aware InstaSHAP study."""

from __future__ import annotations

from typing import Any

from instashap_project.data.loaders import load_covertype
from instashap_project.experiments.common import ExperimentResult, run_phase3_experiment


def run(config: dict[str, Any], variant: str = "compare") -> ExperimentResult:
    """Run the standalone Phase 3 Covertype experiment."""

    bundle = load_covertype(
        max_rows=config["dataset"].get("max_rows"),
        seed=int(config["global"]["seeds"][0]),
    )
    return run_phase3_experiment(bundle=bundle, config=config, variant=variant)
