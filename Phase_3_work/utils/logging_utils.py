"""Structured logging utilities."""

from __future__ import annotations
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)-5s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(fmt)
    logger.addHandler(stream_h)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_h.setFormatter(fmt)
        logger.addHandler(file_h)

    logger.propagate = False
    return logger
