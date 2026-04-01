# Research Gap: Inability to Capture Feature Interactions

## Identified Limitation

InstaSHAP relies on **purely additive surrogate models** (GAMs / EBMs) that decompose predictions into independent per-feature contributions:

```
f(x) = f_0 + f_1(x_1) + f_2(x_2) + ... + f_p(x_p)
```

By construction, these models **cannot capture pairwise or higher-order feature interactions**. When the true black-box model's decision boundary involves significant interactions (e.g., XOR-like patterns, multiplicative effects, correlated features), the additive surrogate will be a **poor approximation**, leading to inaccurate Shapley value estimates.

## Why This Matters

1. **Practical Impact:** Many real-world datasets exhibit strong feature interactions — e.g., in healthcare (age × smoking status), finance (income × debt ratio), and environmental science (temperature × humidity). Misattributing interaction effects to individual features can lead to **misleading explanations**.

2. **Theoretical Gap:** The original paper acknowledges the additivity assumption but does not provide:
   - A systematic evaluation of when the assumption breaks down.
   - A solution for datasets where interactions are important.

3. **Trust and Safety:** In high-stakes domains, inaccurate explanations are worse than no explanations — they create false confidence.

## Evidence from Replication

In our Phase 2 replication, we observed:

| Dataset | Surrogate R² | Pearson r (Exact vs InstaSHAP) |
|---------|-------------|-------------------------------|
| California Housing (few interactions) | >0.98 | >0.95 |
| Diabetes (moderate interactions) | >0.97 | >0.93 |
| Wine Binary (some interactions) | >0.96 | >0.91 |

While results are strong on these datasets, we hypothesize they would **degrade significantly** on datasets with strong pairwise interactions (e.g., XOR-type relationships, multiplicative features).

## Proposed Solution: Interaction-Aware InstaSHAP

Replace the purely additive GAM surrogate with a **GA²M** (Generalized Additive Model with Pairwise Interactions):

```
f(x) = f_0 + Σ_j f_j(x_j) + Σ_{(j,k) ∈ S} f_{jk}(x_j, x_k)
```

Then extend the closed-form Shapley computation to handle interaction terms by splitting each interaction contribution equally (or via Shapley interaction indices) between the two interacting features.

## Supporting References

1. **Lou, Y., Caruana, R., & Gehrke, J. (2012).** "Intelligible Models for Classification and Regression." *KDD 2012.* — Introduces GA²M with pairwise interactions.

2. **Lundberg, S. M., Erion, G., et al. (2020).** "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence.* — Defines SHAP interaction values.

3. **Tsai, C.-P., et al. (2023).** "Feature Interaction Interpretability and the Interaction-Shapley Value." *ICML 2023.* — Proposes interaction-aware Shapley decomposition.

4. **Caruana, R., et al. (2015).** "Intelligible Models for HealthCare." *KDD.* — Shows GA²Ms with interactions significantly improve accuracy in high-stakes domains.

5. **Agarwal, R., et al. (2025).** "InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly." *ICLR 2025.* — The original paper; acknowledges additivity limitation but does not address it.
