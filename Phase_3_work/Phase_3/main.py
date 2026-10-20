"""Phase 3: InstaSHAP with Three Research Innovations — CLI Entry Point.

Usage:
    python main.py --variant compare                     # Full 3-seed study
    python main.py --variant compare --fast-dev-run      # Quick smoke test
    python main.py --variant baseline                    # Zero-mask only
    python main.py --variant improved                    # All 3 innovations
    python main.py --report-only                         # Regenerate PDFs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: InstaSHAP Innovation Study")
    parser.add_argument("--dataset", default="covertype", help="Dataset name (covertype only)")
    parser.add_argument("--variant", default="compare",
                        choices=["baseline", "improved", "compare"],
                        help="baseline=zero-mask | improved=all innovations | compare=full ablation")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick validation (4k rows, 4 epochs, 1 seed)")
    parser.add_argument("--report-only", action="store_true", help="Regenerate reports from existing results")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-report", action="store_true", help="Skip PDF report generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = get_logger("main", log_file="results/run.log")
    log.info(f"Phase 3 InstaSHAP — variant={args.variant}, fast_dev_run={args.fast_dev_run}")

    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.fast_dev_run:
        config["global"]["fast_dev_run"] = True

    if args.report_only:
        log.info("Regenerating reports from existing results...")
        try:
            from reports.generate_experiment_report import generate_experiment_report
            generate_experiment_report(config)
            log.info("Experiment report generated.")
        except Exception as e:
            log.warning(f"Report generation failed: {e}")
        try:
            from reports.generate_research_gap import generate_research_gap_report
            generate_research_gap_report(config)
            log.info("Research gap report generated.")
        except Exception as e:
            log.warning(f"Research gap report failed: {e}")
        return

    # Run experiment
    from experiments.covertype_comparison import run_comparison
    results = run_comparison(config, variant=args.variant, fast_dev_run=args.fast_dev_run)

    # Generate reports
    if not args.skip_report:
        log.info("Generating reports...")
        try:
            from reports.generate_experiment_report import generate_experiment_report
            generate_experiment_report(config)
        except Exception as e:
            log.warning(f"Experiment report failed: {e}")
        try:
            from reports.generate_research_gap import generate_research_gap_report
            generate_research_gap_report(config)
        except Exception as e:
            log.warning(f"Research gap report failed: {e}")

    log.info("Phase 3 complete. Results in: results/")


if __name__ == "__main__":
    main()
