"""InstaSHAP reproducibility package."""

from __future__ import annotations

import os

# Set TensorFlow's C++ log level before SHAP or other optional dependencies import it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
