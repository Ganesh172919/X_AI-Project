# Phase 3: Research Gap Identification and Extension

## Title
Interaction-Aware Surrogate for Faster and More Faithful InstaSHAP Approximation

## 1. Research Gap
The replicated InstaSHAP pipeline trains one surrogate GAM per SHAP dimension with pure additivity (`interactions=0`). This is a practical and interpretable design, but it has a known limitation: additive models cannot fully represent SHAP response surfaces created by strong feature interactions in the original black-box model.

In tree ensembles (Random Forest, XGBoost, LightGBM), interactions are often a major source of predictive behavior. If the surrogate ignores them, SHAP approximation may lose fidelity, especially in local regions with interaction-driven attribution changes.

In short, the gap is:
- Existing surrogate in baseline form is interaction-blind.
- Original InstaSHAP framing prioritizes speed and additive interpretability, but this can underfit interaction-heavy attribution structure.

## 2. Proposed Improvement
I implemented an interaction-aware surrogate extension in `Phase3/run_phase3_experiment.py` by enabling pairwise interaction terms in Explainable Boosting Machines (EBM):
- Before (baseline): `interactions = 0`
- After (improved): `interactions = 4`

The experiment keeps all other settings fixed and compares both surrogates on the same dataset/model and same SHAP train/test subsets. This isolates the effect of adding interaction capacity.

## 3. Experimental Setup
- Dataset: California Housing
- Black-box model: Random Forest
- SHAP explainer: TreeSHAP
- SHAP train subset: 120 samples
- SHAP test subset: 80 samples
- Metrics: MSE, MAE, RMSE, R2, Pearson/Spearman correlation, speedup

Artifacts generated:
- `Phase3/results/tables/before_after_summary.csv`
- `Phase3/results/tables/before_after_per_feature.csv`
- `Phase3/results/figures/before_after_metrics.png`

## 4. Before vs After Results
From `before_after_summary.csv`:

| Variant | Interactions | MSE | MAE | R2 | Pearson | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline additive | 0 | 0.009545 | 0.057249 | 0.8746 | 0.9354 | 281.03x |
| Improved interaction-aware | 4 | 0.007540 | 0.051870 | 0.9010 | 0.9493 | 29.91x |

Observed gains:
- R2 improvement: +0.0263 (from 0.8746 to 0.9010)
- MAE reduction: about 9.4%
- MSE reduction: about 21.0%
- Correlation improvement: +0.0139 (Pearson)

Interpretation:
- Adding controlled interactions improved SHAP fidelity.
- Trade-off: the interaction-aware model is slower at inference than the additive model, so speedup vs exact SHAP decreases.
- Despite this, both variants remain much faster than exact SHAP, while the interaction-aware model gives better approximation quality.

## 5. Why This Is Supported by Literature
This improvement is grounded in established GAM/GA2M literature:
- Lou et al. (2013) showed that adding pairwise interactions to additive models (GA2M) can improve predictive fidelity while preserving interpretability.
- Caruana et al. (2015) demonstrated interpretable boosted additive models with interaction terms for practical tabular tasks.

Hence, an interaction-aware surrogate is a principled extension of an additive InstaSHAP-style surrogate for interaction-rich black-box models.

## 6. Conclusion
The identified limitation (interaction blindness in additive surrogate approximation) is real and practically relevant. The implemented extension (interaction-aware EBM surrogate) improves SHAP approximation quality in a controlled before-vs-after experiment, satisfying the research extension objective with measurable empirical gains.

## References
1. Nori et al., InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly, ICLR 2025.
2. Lou et al., Accurate Intelligible Models with Pairwise Interactions, KDD 2013.
3. Caruana et al., Intelligible Models for Classification and Regression, KDD 2015.
