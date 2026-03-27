"""Structured logging helpers for clean CLI output."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import warnings


def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> logging.Logger:
    """Configure root logging and suppress noisy third-party warnings."""

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    warnings.filterwarnings(
        "ignore",
        message=r".*Passing `palette` without assigning `hue` is deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*np\.object.*",
        category=FutureWarning,
    )

    for logger_name in ("tensorflow", "absl", "matplotlib", "PIL", "numexpr"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    return logging.getLogger("instashap")


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""

    return logging.getLogger(name)


def format_log_event(event: str, **fields: object) -> str:
    """Format a structured log line using stable key ordering."""

    if not fields:
        return event
    serialized = " ".join(
        f"{key}={json.dumps(_normalize_value(value), ensure_ascii=True)}"
        for key, value in sorted(fields.items())
    )
    return f"{event} | {serialized}"


def _normalize_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return value
