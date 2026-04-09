# Phase 2 Replication Reference

Phase 2 is the baseline system that the extension depends on conceptually and structurally.

## Where It Lives

- Main package: Phase_2_work/instashap_project/.
- CLI entrypoint: Phase_2_work/instashap_project/main.py.
- Configuration: Phase_2_work/instashap_project/config.yaml.
- Main outputs: results/, reports/, notebooks/.

## Datasets

- Bike Sharing for regression and interaction analysis.
- Covertype for classification and elevation/soil interpretation.
- Adult Income for additive stability context.
- Each dataset is wired through a thin experiment runner on top of experiments/common.py.

## Important Modules

- data/loaders.py defines the benchmark views and metadata.
- data/preprocessing.py builds grouped transformed features.
- models/blackbox_model.py defines predictive and surrogate backbones.
- models/gam.py defines the additive neural model.
- training/train.py implements the learning loops.
- xai/shap_wrapper.py defines the reference SHAP computation path.

## Why Phase 2 Matters

- It proves the team could implement the paper pipeline before attempting improvements.
- It provides the baseline mental model for black-box -> surrogate -> InstaSHAP training.
- It creates the comparison context for any later extension claim.
- It already contains substantial reporting and visualization infrastructure.

## Phase 2 Limitations

- Masking remained relatively simple and synthetic.
- The broader replication scope made it harder to isolate one failure mode deeply.
- That gap naturally leads into the focused Phase 3 branch.
