# File Reference: Package `__init__.py` Files

## Purpose

The package `__init__.py` files serve mostly as package markers and import anchors.

## Notable Case

The top-level `__init__.py` also sets `TF_CPP_MIN_LOG_LEVEL` early so optional SHAP/TensorFlow internals remain quieter during runtime.

## Why This File Exists

It is included here because these files are small but still contribute to project behavior and package layout.
