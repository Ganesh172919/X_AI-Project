"""
Utility functions for InstaSHAP replication.

Provides helpers for configuration management, reproducibility, logging,
and object serialization used throughout the pipeline.

Functions
---------
load_config
    Load a YAML configuration file into a dictionary.
set_random_seed
    Set random seeds across Python, NumPy, and optionally PyTorch.
setup_logging
    Configure Python logging with console and optional file handlers.
save_object / load_object
    Serialize and deserialize Python objects via joblib.
ensure_dir
    Create a directory (and parents) if it does not exist.
format_time
    Convert seconds to a human-readable string (ms, s, min, hr).
print_dict
    Pretty-print a nested dictionary.

Example
-------
>>> from src.utils import load_config, set_random_seed, setup_logging
>>> config = load_config("config/config.yaml")
>>> set_random_seed(config["random_seed"])
>>> logger = setup_logging("INFO")
"""

import os
import yaml
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
import joblib


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Set seeds for other libraries if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), format=log_format, handlers=handlers
    )

    return logging.getLogger(__name__)


def save_object(obj: Any, filepath: str) -> None:
    """
    Save Python object using joblib.

    Args:
        obj: Object to save
        filepath: Path to save file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_object(filepath: str) -> Any:
    """
    Load Python object using joblib.

    Args:
        filepath: Path to saved file

    Returns:
        Loaded object
    """
    return joblib.load(filepath)


def ensure_dir(directory: str) -> None:
    """
    Ensure directory exists, create if not.

    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} min"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hr"


def print_dict(d: Dict[str, Any], indent: int = 0) -> None:
    """
    Pretty print dictionary.

    Args:
        d: Dictionary to print
        indent: Indentation level
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
