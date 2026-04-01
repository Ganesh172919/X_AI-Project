# Supporting References

## Papers Cited

1. **Agarwal, R., et al. (2025).** InstaSHAP: Interpretable Additive Models Explain Shapley Values Instantly. *International Conference on Learning Representations (ICLR) 2025.*
   [[OpenReview]](https://openreview.net/forum?id=ky7vVlBQBY)
   - The original paper being replicated and extended.

2. **Lou, Y., Caruana, R., & Gehrke, J. (2012).** Intelligible Models for Classification and Regression. *Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD),* 150–158.
   - Introduces Generalized Additive Models plus pairwise interactions (GA²M). Demonstrates that adding interaction terms significantly improves model accuracy while maintaining interpretability.

3. **Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S.-I. (2020).** From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence,* 2(1), 56–67.
   - Defines SHAP interaction values and proves their connection to Shapley interaction indices. Provides the theoretical basis for splitting interaction contributions between features.

4. **Tsai, C.-P., et al. (2023).** Feature Interaction Interpretability and the Interaction-Shapley Value. *International Conference on Machine Learning (ICML).*
   - Proposes a principled approach to interaction-aware feature attribution using Shapley interaction indices. Relevant to how we split pairwise interaction terms.

5. **Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., & Elhadad, N. (2015).** Intempible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission. *Proceedings of the 21st ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD),* 1721–1730.
   - Demonstrates the practical importance of interpretable models (including GA²Ms) in high-stakes domains. Shows that interaction terms capture clinically meaningful relationships that additive-only models miss.

6. **Friedman, J. H. (2001).** Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics,* 29(5), 1189–1232.
   - Foundation for Generalized Additive Models and the boosting framework used by EBMs.

7. **Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2021).** InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv preprint arXiv:1909.09223.*
   - Documentation and implementation reference for the Explainable Boosting Machine (EBM) used as the surrogate model.

---

## Connection to Our Extension

| Reference | How It Supports Our Work |
|-----------|------------------------|
| Lou et al. (2012) | Provides the GA²M model architecture we adopt as the interaction-aware surrogate |
| Lundberg et al. (2020) | Theoretical basis for splitting interaction contributions (Shapley interaction indices) |
| Tsai et al. (2023) | Methodology for interaction-aware feature attribution |
| Caruana et al. (2015) | Practical motivation for capturing interactions in high-stakes applications |
| Friedman (2001) | Mathematical foundation for additive and interaction models |
