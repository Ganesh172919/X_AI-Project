# Research Gap: Additive Surrogates Miss Interaction Structure

## Limitation

The original InstaSHAP formulation assumes that a purely additive surrogate is sufficient to approximate the black-box model. In an additive model, the prediction is decomposed into independent per-feature terms. That design gives InstaSHAP its speed, but it also creates a hard representational limit: the surrogate cannot model pairwise or higher-order interactions such as `x_i * x_j`, XOR-like logic, or multiplicative nonlinear effects.

## Why This Matters

This limitation matters both practically and theoretically.

- Practically, many real tabular problems contain interaction effects. If the surrogate cannot represent them, its predictions drift away from the black-box model and its explanations become less faithful.
- Theoretically, Shapley values are defined on the true model behavior. If the surrogate is misspecified, the closed-form attribution is analytically correct for the surrogate but not necessarily accurate for the black-box function being explained.

## Evidence in This Project

This repository operationalizes the gap on the sklearn `friedman1` benchmark, which contains a known interaction term involving `x_1` and `x_2`.

- `phase3/experiments/experiment_gap_demonstration.py` fits the original additive-only InstaSHAP pipeline on `friedman1`.
- `phase3/results/gap_demonstration/gap_summary.csv` records the additive surrogate's fidelity and attribution agreement with Exact SHAP.
- `phase3/results/gap_demonstration/gap_surrogate_fidelity_scatter.png` visualizes the surrogate mismatch.
- `phase3/results/gap_demonstration/gap_shap_alignment_scatter.png` visualizes the attribution mismatch.

The expected failure mode is:

- lower surrogate `R²` than on the more additive Phase 2 datasets
- weaker Pearson and Spearman alignment with Exact SHAP
- larger MAE in explanation space

Those failure patterns provide direct empirical evidence that the additive-only assumption can break when interactions drive the black-box predictions.

## Proposed Extension

The extension implemented in this phase replaces the purely additive surrogate with a **GA²M-style surrogate** that includes pairwise interaction terms. Once those terms are present, the explanation rule is extended so that each centered interaction contribution is split fairly across the participating features. For pairwise terms, this is a 50/50 split, consistent with Shapley-style interaction allocation.

## Supporting Literature

The literature directly supports this gap and the proposed remedy.

- Lundberg and Lee formalized SHAP and later work on SHAP interaction values established a principled way to separate main effects from interaction effects.
- Lou et al. showed that GA²M models can remain interpretable while adding pairwise interactions.
- Bordt and von Luxburg analyzed the relationship between additive models and Shapley values, reinforcing why model class assumptions matter.
- The Faith-Shap work in JMLR further studies explanation faithfulness in settings where interactions and attribution quality matter.

See [supporting_references.md](../references/supporting_references.md) for the formatted bibliography.
