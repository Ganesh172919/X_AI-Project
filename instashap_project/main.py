"""Entry point for the InstaSHAP reproducibility project."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import yaml

from instashap_project.experiments import adult_income, bike_sharing, covertype
from instashap_project.reports.generate_report import generate_full_report
from instashap_project.reports.summary_1page import generate_one_page_summary
from instashap_project.utils.reproducibility import set_global_seed


RUNNERS = {
    "bike": bike_sharing.run,
    "covertype": covertype.run,
    "adult": adult_income.run,
}


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
    parser.add_argument("--skip-report", action="store_true", help="Do not generate the full PDF report.")
    parser.add_argument("--skip-summary", action="store_true", help="Do not generate the one-page summary PDF.")
    return parser.parse_args()


def main() -> None:
    """Run one or more experiments."""

    args = parse_args()
    config = load_config(args.config)
    config["global"]["fast_dev_run"] = bool(args.fast_dev_run)
    set_global_seed(int(config["global"]["seed"]))

    datasets = list(RUNNERS) if args.dataset == "all" else [args.dataset]
    results = []
    for dataset_name in datasets:
        result = RUNNERS[dataset_name](config=config, selected_model=args.model)
        results.append(result)
        print(f"Completed {dataset_name}: {result.summary_path}")

    if not args.skip_report:
        report_path = generate_full_report()
        print(f"Full report: {report_path}")
    if not args.skip_summary:
        summary_path = generate_one_page_summary()
        print(f"One-page summary: {summary_path}")


if __name__ == "__main__":
    main()
