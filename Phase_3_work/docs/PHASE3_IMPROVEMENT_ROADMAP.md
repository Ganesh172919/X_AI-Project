# Phase 3 Improvement Roadmap

This document explains what improvements you can make, how to make them, and what will likely happen if you make them.

| Improvement | How to make it | If you make it |
| --- | --- | --- |
| Increase surrogate capacity | Raise surrogate hidden dimensions and epochs in config.yaml. | The empirical_background branch may fit the harder coalition objective better. |
| Add dataset-specific configs | Split config.yaml into global and per-dataset sections. | Adult and future datasets can run through the same Phase 3 workflow more cleanly. |
| Track masking validity directly | Keep the new diagnostic metrics in all future reports. | You can show coalition realism gains even when full end-to-end gains are mixed. |
| Use Adult next | Start from the new notebook and prompt. | The masking improvement will be easier to show to reviewers. |
| Combine masking with interactions | Add interaction-aware surrogate or additive terms later. | The pipeline may improve on datasets where realistic masking alone is not enough. |
