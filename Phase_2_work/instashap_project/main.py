"""Entry point for the InstaSHAP reproducibility project."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import yaml

from instashap_project.utils.logging_utils import configure_logging, format_log_event
from instashap_project.utils.reproducibility import set_global_seed


def load_config(config_path: str | Path) -> dict:
    """Load the YAML configuration."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="InstaSHAP reproducibility runner")
    parser.add_argument("--dataset", choices=["bike", "covertype", "adult", "all"], required=True)
    parser.add_argument("--model", choices=["all", "blackbox", "gam", "shap", "instashap"], default="all")
    parser.add_argument("--config", default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--fast-dev-run", action="store_true", help="Use smaller subsets and shorter training.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    """Run one or more experiments."""

    args = parse_args()
    logger = configure_logging(
        level=args.log_level,
        log_file=Path(__file__).resolve().parent / "results" / "run.log",
    )
    config = load_config(args.config)
    config["global"]["fast_dev_run"] = bool(args.fast_dev_run)
    set_global_seed(int(config["global"]["seed"]))

    from instashap_project.experiments import adult_income, bike_sharing, covertype

    runners = {
        "bike": bike_sharing.run,
        "covertype": covertype.run,
        "adult": adult_income.run,
    }

    logger.info(
        format_log_event(
            "run.start",
            dataset=args.dataset,
            model=args.model,
            fast_dev_run=args.fast_dev_run,
            config=Path(args.config),
        )
    )

    datasets = list(runners) if args.dataset == "all" else [args.dataset]
    results = []
    for dataset_name in datasets:
        result = runners[dataset_name](config=config, selected_model=args.model)
        results.append(result)
        logger.info(format_log_event("dataset.ready", dataset=dataset_name, summary_path=result.summary_path))

    logger.info(format_log_event("run.complete", datasets=datasets))


if __name__ == "__main__":
    main()
