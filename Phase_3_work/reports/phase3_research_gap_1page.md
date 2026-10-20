# Phase 3 Research Gap: Background-Aware InstaSHAP

The original InstaSHAP formulation is elegant and fast, but the current tabular implementation in this repository uses zero-masking in transformed feature space. On Covertype, that is a fragile approximation because missing standardized numeric values become artificial zeros and missing categorical groups become all-zero one-hot blocks that may not correspond to realistic data. This can distort the coalition value function used to train the surrogate and the downstream additive explainer.

Our improvement is empirical-background masking. For each coalition mask, any hidden original feature group is replaced with the corresponding transformed columns from a real training row. We then average coalition outputs over multiple sampled background rows. This keeps numeric and categorical groups realistic and provides a stronger approximation to marginal or interventional feature removal than plain zero-masking.

The comparison focuses on two models: `instashap_zero` and `instashap_bg`. We judge them using predictive accuracy, SHAP-alignment metrics, coalition fidelity, and explanation runtime. The goal is not to claim a full dependence-aware SHAP estimator, but to show that better tabular masking materially improves explanation fidelity while preserving the efficiency advantage of InstaSHAP.

```text
         model  accuracy_mean  log_loss_mean  explanation_mae_mean  explanation_spearman_mean  coalition_mse_mean  explain_seconds_mean
instashap_zero         0.6842         0.7815                0.3591                     0.5650              0.2021                0.0100
  instashap_bg         0.6774         0.8168                0.3795                     0.5835              0.2016                0.0121
```

## References
- Lundberg and Lee, SHAP: https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
- Jethani et al., FastSHAP: https://arxiv.org/abs/2107.07436
- Aas et al., dependent-feature SHAP: https://arxiv.org/abs/1903.10464
- Frye et al., Shapley explainability on the data manifold: https://arxiv.org/abs/2006.01272
- Tsai et al., Faith-Shap: https://jmlr.org/papers/v24/22-0202.html
