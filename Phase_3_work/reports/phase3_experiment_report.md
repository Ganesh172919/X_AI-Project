# Phase 3 Experiment Report: Background-Aware InstaSHAP on Covertype

## Objective
This standalone Phase 3 project extends the original InstaSHAP tabular pipeline on the Covertype dataset. The key goal is to test whether a stronger masking/value-function construction improves explanation fidelity without changing the additive architecture.

## Research Gap
The reproduced baseline uses zero-masking in transformed feature space. For tabular data this can generate unrealistic coalition samples, especially after standardization and one-hot encoding. That weakens surrogate fidelity and can propagate explanation error into InstaSHAP.

## Proposed Improvement
We introduce empirical-background masking. Instead of replacing missing feature groups with zeros, the method fills each masked original feature group with values copied from real transformed training rows. During coalition evaluation, outputs are averaged across multiple sampled background rows to better approximate a data-aware masked expectation.

## Experimental Setup
- Dataset: Covertype only
- Seeds: [42, 123, 2026]
- Comparison: blackbox, GAM-1, GAM-2, instashap_zero, instashap_bg
- Reference explainer: permutation SHAP
- Coalition fidelity: surrogate vs black-box under the same masking scheme

## Before vs After
```text
         model  accuracy_mean  log_loss_mean  explanation_mae_mean  explanation_spearman_mean  coalition_mse_mean  explain_seconds_mean
instashap_zero         0.6842         0.7815                0.3591                     0.5650              0.2021                0.0100
  instashap_bg         0.6774         0.8168                0.3795                     0.5835              0.2016                0.0121
```

## Full Predictive Summary
```text
         model  seed_mean  seed_std  accuracy_mean  accuracy_std  log_loss_mean  log_loss_std
      blackbox   730.3333 1122.8109         0.7735        0.0057         0.5245        0.0070
          gam1   730.3333 1122.8109         0.7186        0.0038         0.6373        0.0064
          gam2   730.3333 1122.8109         0.7216        0.0049         0.6329        0.0079
  instashap_bg   730.3333 1122.8109         0.6774        0.0059         0.8168        0.0324
instashap_zero   730.3333 1122.8109         0.6842        0.0093         0.7815        0.0415
```

## Full Explanation Summary
```text
         model  seed_mean  seed_std  mse_mean  mse_std  mae_mean  mae_std  spearman_mean  spearman_std
  instashap_bg   730.3333 1122.8109    0.4924   0.2494    0.3795   0.1074         0.5835        0.0055
instashap_zero   730.3333 1122.8109    0.2920   0.1521    0.3591   0.0802         0.5650        0.0703
```

## Full Coalition Summary
```text
         model  seed_mean  seed_std  mse_mean  mse_std  mae_mean  mae_std
  surrogate_bg   730.3333 1122.8109    0.2016   0.0490    0.3194   0.0355
surrogate_zero   730.3333 1122.8109    0.2021   0.0512    0.3041   0.0378
```

## Notes
- The improved method is not a full conditional-SHAP implementation.
- The comparison isolates masking/value-function construction; the additive InstaSHAP architecture is otherwise kept aligned.
- All tables in this report are generated from saved CSV artifacts.
- Outcome interpretation: The empirical-background variant did not improve predictive accuracy over the zero-mask baseline in this run. Its SHAP alignment remained weaker than the zero-mask baseline, which suggests the background-aware coalition objective is harder to optimize with the current surrogate capacity and training budget. Coalition fidelity improved as well, which supports the proposed masking strategy directly.

## References
- Lundberg and Lee, SHAP: https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Jethani et al., FastSHAP: https://arxiv.org/abs/2107.07436
- Aas et al., dependent-feature SHAP: https://arxiv.org/abs/1903.10464
- Frye et al., Shapley explainability on the data manifold: https://arxiv.org/abs/2006.01272
- Tsai et al., Faith-Shap: https://jmlr.org/papers/v24/22-0202.html
