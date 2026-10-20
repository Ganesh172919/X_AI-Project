"""Entry point for the standalone Phase 3 Covertype extension project."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import warnings

import yaml

from instashap_project.reporting import generate_reports
from instashap_project.utils.logging_utils import configure_logging, format_log_event
from instashap_project.utils.reproducibility import set_global_seed, write_json


warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str | Path) -> dict:
    """Load the YAML configuration."""

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Phase 3 background-aware InstaSHAP runner")
    parser.add_argument("--dataset", choices=["covertype"], default="covertype")
    parser.add_argument("--variant", choices=["baseline", "improved", "compare"], default="compare")
    parser.add_argument("--config", default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--fast-dev-run", action="store_true", help="Use smaller subsets and shorter training.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate Markdown/PDF reports from saved artifacts.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def _default_summary_path() -> Path:
    return PROJECT_ROOT / "results" / "artifacts" / "covertype" / "covertype_phase3_summary.json"


def main() -> None:
    """Run the Phase 3 Covertype experiment or regenerate reports."""

    args = parse_args()
    logger = configure_logging(
        level=args.log_level,
        log_file=PROJECT_ROOT / "results" / "run.log",
    )
    config = load_config(args.config)
    config["global"]["fast_dev_run"] = bool(args.fast_dev_run)
    set_global_seed(int(config["global"]["seeds"][0]))

    logger.info(
        format_log_event(
            "phase3.cli.start",
            dataset=args.dataset,
            variant=args.variant,
            fast_dev_run=args.fast_dev_run,
            report_only=args.report_only,
        )
    )

    if args.report_only:
        summary_path = _default_summary_path()
        if not summary_path.exists():
            raise FileNotFoundError(f"Cannot regenerate reports because summary JSON is missing: {summary_path}")
        reports = generate_reports(project_root=PROJECT_ROOT, summary_path=summary_path)
        logger.info(format_log_event("phase3.report.ready", reports=reports))
        return

    from instashap_project.experiments import covertype

    result = covertype.run(config=config, variant=args.variant)
    reports = generate_reports(project_root=PROJECT_ROOT, summary_path=result.summary_path)

    summary_payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    summary_payload["reports"] = reports
    write_json(result.summary_path, summary_payload)

    logger.info(format_log_event("phase3.run.complete", summary_path=result.summary_path, reports=reports))


if __name__ == "__main__":
    main()
