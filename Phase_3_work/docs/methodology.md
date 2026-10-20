# Methodology

## Research Question

The guiding question of this project is:

> If we replace zero-masked coalitions with more realistic background-filled coalitions, do we obtain a better training signal for surrogate-based InstaSHAP on Covertype?

The study is intentionally narrow so that the comparison is attributable to the masking strategy rather than to unrelated architectural changes.

## Dataset Choice

The project uses **Covertype** as the single primary dataset because:

- it is already supported by the previous project
- it contains both numeric and grouped categorical structure
- it includes a meaningful paper-aligned interaction pair: `elevation × soil_climate_zone`
- it gives a stronger motivation for investigating dependence-aware coalition construction than a simpler additive-only benchmark

## Baseline Pipeline

The baseline branch reproduces the original tabular idea as closely as possible inside the Phase 3 standalone repository:

1. Load Covertype with the same grouped soil-climate representation.
2. Split into train, validation, and test sets using the configured seed.
3. Fit the tabular preprocessor.
4. Train a black-box model on the transformed data.
5. Train GAM-1 and GAM-2 reference models on the supervised task.
6. Train a masked surrogate using zero-masked transformed inputs.
7. Train an additive InstaSHAP model against surrogate coalition outputs.
8. Compare explanations against permutation SHAP.

This branch produces the model name `instashap_zero`.

## Improved Pipeline

The improved branch keeps the same high-level modeling structure but changes coalition construction:

1. Sample a binary coalition mask in original feature space.
2. Expand that mask to transformed columns.
3. Keep visible columns from the original sample.
4. Replace hidden columns with values from real transformed training rows.
5. Choose those background rows using similarity on the visible transformed columns.
6. Evaluate multiple such background completions.
7. Average the coalition outputs.
8. Train the surrogate and the additive model using those averaged coalition targets.

This branch produces the model name `instashap_bg`.

## Why This Is A Reasonable Improvement

The proposal is motivated by a simple modeling concern: the coalition value function should be built from plausible hidden-feature completions. Pure zero-masking does not guarantee this in transformed tabular space. Empirical-background masking is still approximate, but it is more data-aware because hidden groups are filled using real training rows rather than synthetic zero patterns.

## Controlled Variables

To make the baseline vs improved comparison fair, the project holds these parts fixed:

- same dataset
- same split policy
- same seed list
- same black-box family
- same additive architecture family
- same interaction pair
- same report generation path

Only the coalition construction and the resulting surrogate training target differ.

## Metrics

The project records four metric families.

### Predictive Metrics

- accuracy
- log-loss

These show whether the additive explainer remains a usable predictor.

### Explanation Fidelity Metrics

- mean squared error against permutation SHAP
- mean absolute error against permutation SHAP
- Spearman correlation against permutation SHAP

These measure how close the one-pass explainer is to the SHAP baseline.

### Coalition Fidelity Metrics

- mean squared error between surrogate coalition outputs and black-box coalition outputs
- mean absolute error between surrogate coalition outputs and black-box coalition outputs

These measure whether the surrogate is learning the intended coalition function under the corresponding masking scheme.

### Runtime Metrics

- training time
- prediction latency
- explanation latency

These preserve the original motivation for InstaSHAP: fast inference-time explanations.

## Fast-Dev vs Full Run

The repository supports two execution modes.

### Fast-Dev Run

This is a smoke test used for implementation validation:

- smaller dataset cap
- shorter training schedules
- fewer coalition/background samples

### Full Run

This is the run intended for final submission-quality artifacts:

- full configured seeds
- larger training budget
- larger coalition evaluation budget

## Interpreting Results Responsibly

The generated reports are intentionally conservative.

- If the improved branch helps only in prediction but not in SHAP alignment, the report says so.
- If coalition fidelity remains weaker, the report says so.
- If the evidence is mixed, the conclusion is framed as a promising but incomplete improvement rather than a definitive breakthrough.

This is important because the assignment asks for research quality, and research quality depends on honest interpretation as much as on implementation quality.
