# Phase 3 Improvement Roadmap

This document answers what improvements you can make, how to make them, and what is likely to happen if you make them.

| Improvement | How to make it | What will happen |
| --- | --- | --- |
| Increase surrogate capacity | Raise surrogate hidden dimensions and training epochs in config.yaml or a new dataset-specific config. | The surrogate may fit the harder empirical_background objective better, which can improve downstream explanation fidelity. |
| Increase background samples | Raise masking.background_samples_train and masking.background_samples_eval. | Coalition targets become more stable but training and explanation preparation become slower. |
| Add invalid-state metrics | Track hidden categorical validity and off-manifold distance explicitly in the training reports. | You can show the masking improvement even when end-task metrics are mixed. |
| Use Adult Income next | Generalize the Phase 3 workflow or reuse the new notebook and prompt to continue from the adult masking diagnostic. | The masking limitation should be easier to demonstrate because there are more categorical groups. |
| Add a dataset-specific config system | Split Phase 3 config.yaml into a global block and per-dataset blocks instead of one single dataset block. | It becomes easier to extend Phase 3 beyond Covertype without manual edits. |
| Combine masking realism with interactions | Add pairwise interaction capacity to the surrogate or final additive model. | The model may better capture datasets where realistic coalitions are still not enough by themselves. |
| More seeds and larger SHAP evaluation sets | Raise the seeds list and evaluation sample sizes. | The results become more stable and easier to defend statistically, but runs take longer. |
| Dataset continuation track | Evaluate Adult Income first, then Bank Marketing or German Credit as future work. | You can show whether the Phase 3 improvement generalizes to other mixed tabular datasets. |

## Recommended order

1. Start with explicit masking diagnostics on more datasets.
2. Improve the surrogate capacity for the empirical_background branch.
3. Make Phase 3 dataset-generic so Adult Income can run through the same reporting path.
4. Combine masking realism with interaction-aware modeling.
5. Expand to new datasets after the workflow is stable.

## Best near-term improvement

The best near-term path is to keep the Phase 3 research question narrow and strengthen the empirical_background branch with better surrogate training and a clearer multi-dataset evaluation story.
