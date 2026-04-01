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
instashap_zero         0.6098         0.9312                0.2805                     0.4951              0.2863                0.0035
  instashap_bg         0.6135         0.9356                0.3109                     0.4518              0.3793                0.0050
```

## Full Predictive Summary
```text
         model  seed_mean  seed_std  accuracy_mean  accuracy_std  log_loss_mean  log_loss_std
      blackbox   730.3333 1122.8109         0.6942        0.0198         0.7463        0.0333
          gam1   730.3333 1122.8109         0.6977        0.0170         0.7467        0.0241
          gam2   730.3333 1122.8109         0.7056        0.0158         0.7123        0.0257
  instashap_bg   730.3333 1122.8109         0.6135        0.0178         0.9356        0.0088
instashap_zero   730.3333 1122.8109         0.6098        0.0391         0.9312        0.0175
```

## Full Explanation Summary
```text
         model  seed_mean  seed_std  mse_mean  mse_std  mae_mean  mae_std  spearman_mean  spearman_std
  instashap_bg   730.3333 1122.8109    0.1647   0.0253    0.3109   0.0033         0.4518        0.0755
instashap_zero   730.3333 1122.8109    0.1501   0.0403    0.2805   0.0074         0.4951        0.0427
```

## Full Coalition Summary
```text
         model  seed_mean  seed_std  mse_mean  mse_std  mae_mean  mae_std
  surrogate_bg   730.3333 1122.8109    0.3793   0.0862    0.4715   0.0526
surrogate_zero   730.3333 1122.8109    0.2863   0.0567    0.4047   0.0469
```

## Notes
- The improved method is not a full conditional-SHAP implementation.
- The comparison isolates masking/value-function construction; the additive InstaSHAP architecture is otherwise kept aligned.
- All tables in this report are generated from saved CSV artifacts.
- Outcome interpretation: The empirical-background variant improved predictive accuracy over the zero-mask baseline. Its SHAP alignment remained weaker than the zero-mask baseline, which suggests the background-aware coalition objective is harder to optimize with the current surrogate capacity and training budget. Coalition fidelity also remained weaker in this run, so the most responsible interpretation is that the idea is promising but not yet a definitive win under the current experimental budget.

## References
- Lundberg and Lee, SHAP: https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Jethani et al., FastSHAP: https://arxiv.org/abs/2107.07436
- Aas et al., dependent-feature SHAP: https://arxiv.org/abs/1903.10464
- Frye et al., Shapley explainability on the data manifold: https://arxiv.org/abs/2006.01272
- Tsai et al., Faith-Shap: https://jmlr.org/papers/v24/22-0202.html
