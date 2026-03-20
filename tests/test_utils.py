"""
Unit tests for utils module.
"""

import pytest
import numpy as np
import os
import yaml
import logging
from pathlib import Path
from src.utils import (
    load_config,
    set_random_seed,
    setup_logging,
    save_object,
    load_object,
    ensure_dir,
    format_time,
    print_dict,
)


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config."""
        config_data = {"key": "value", "nested": {"a": 1}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        result = load_config(str(config_file))
        assert result == config_data

    def test_load_missing_file_raises(self, tmp_path):
        """Test that loading a missing file raises an error."""
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


class TestSetRandomSeed:
    """Tests for set_random_seed."""

    def test_reproducibility(self):
        """Test that setting the seed produces reproducible results."""
        set_random_seed(42)
        a = np.random.randn(10)

        set_random_seed(42)
        b = np.random.randn(10)

        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        set_random_seed(42)
        a = np.random.randn(10)

        set_random_seed(123)
        b = np.random.randn(10)

        assert not np.array_equal(a, b)


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging("INFO")
        assert isinstance(logger, logging.Logger)

    def test_log_file_creation(self, tmp_path):
        """Test that log file is created when specified."""
        log_file = str(tmp_path / "test.log")
        setup_logging("INFO", log_file=log_file)

        logger = logging.getLogger("test_utils")
        logger.info("Test message")

        # Force flush
        for handler in logging.getLogger().handlers:
            handler.flush()

        assert os.path.exists(log_file)


class TestSerialization:
    """Tests for save_object and load_object."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading a Python object."""
        obj = {"key": [1, 2, 3], "nested": {"a": 1.5}}
        filepath = str(tmp_path / "obj.pkl")

        save_object(obj, filepath)
        loaded = load_object(filepath)

        assert loaded == obj

    def test_save_creates_directories(self, tmp_path):
        """Test that save_object creates parent directories."""
        filepath = str(tmp_path / "sub" / "dir" / "obj.pkl")
        save_object({"test": True}, filepath)

        assert os.path.exists(filepath)


class TestEnsureDir:
    """Tests for ensure_dir."""

    def test_creates_directory(self, tmp_path):
        """Test that ensure_dir creates a directory."""
        dir_path = str(tmp_path / "new_dir")
        ensure_dir(dir_path)
        assert os.path.isdir(dir_path)

    def test_creates_nested_directories(self, tmp_path):
        """Test that ensure_dir creates nested directories."""
        dir_path = str(tmp_path / "a" / "b" / "c")
        ensure_dir(dir_path)
        assert os.path.isdir(dir_path)

    def test_existing_directory(self, tmp_path):
        """Test that ensure_dir handles existing directories."""
        ensure_dir(str(tmp_path))
        ensure_dir(str(tmp_path))  # Should not raise
        assert os.path.isdir(str(tmp_path))


class TestFormatTime:
    """Tests for format_time."""

    def test_milliseconds(self):
        """Test formatting sub-second times."""
        assert "ms" in format_time(0.5)

    def test_seconds(self):
        """Test formatting seconds."""
        assert "s" in format_time(30)
        assert "ms" not in format_time(30)

    def test_minutes(self):
        """Test formatting minutes."""
        assert "min" in format_time(120)

    def test_hours(self):
        """Test formatting hours."""
        assert "hr" in format_time(7200)


class TestPrintDict:
    """Tests for print_dict."""

    def test_flat_dict(self, capsys):
        """Test printing a flat dictionary."""
        print_dict({"a": 1, "b": 2})
        captured = capsys.readouterr()
        assert "a: 1" in captured.out
        assert "b: 2" in captured.out

    def test_nested_dict(self, capsys):
        """Test printing a nested dictionary."""
        print_dict({"outer": {"inner": 42}})
        captured = capsys.readouterr()
        assert "outer:" in captured.out
        assert "inner: 42" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
