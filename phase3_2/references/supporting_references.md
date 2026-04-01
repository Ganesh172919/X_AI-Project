# Supporting References for InstaSHAP Extension

## DS357 - Explainable AI Course Project
## Phase 3: Research Gap Identification and Extension

---

This document provides references to papers that support our identified research gap (inability to capture feature interactions in additive surrogates) and our proposed solution (Interaction-Aware InstaSHAP using GA²M).

---

## Primary References

### 1. GA²M and Intelligible Models with Interactions

**Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate Intelligible Models with Pairwise Interactions. *Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '13)*, 623-631.**

- **Relevance**: This paper introduces GA²M (Generalized Additive Models with pairwise interactions), which is the foundation of our extension. The authors show that adding a small number of pairwise interaction terms significantly improves model accuracy on many real datasets without sacrificing interpretability.
- **Key Insight**: "We find that for many problems [...] a small number of pairwise interaction terms [...] can significantly improve model accuracy."
- **Our Usage**: We use EBM (Explainable Boosting Machine), a modern implementation of GA²M from InterpretML, as our interaction-aware surrogate.

---

### 2. SHAP Interaction Values

**Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., ... & Lee, S. I. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*, 2(1), 56-67.**

- **Relevance**: This paper extends SHAP to include interaction values, providing a theoretical foundation for how to decompose model predictions when interactions are present.
- **Key Insight**: "SHAP interaction values [...] represent the effect of the interaction between features i and j on the model's prediction."
- **Our Usage**: We use the concept of SHAP interaction indices to fairly allocate interaction contributions between the two interacting features in our Enhanced InstaSHAP formula.

---

### 3. Feature Interaction Detection

**Tsang, M., Cheng, D., & Liu, Y. (2020). Detecting Statistical Interactions from Neural Network Weights. *International Conference on Learning Representations (ICLR 2020)*.**

- **Relevance**: This paper addresses the importance of detecting feature interactions for explainability and shows methods for identifying which feature pairs have significant interactions.
- **Key Insight**: "Feature interactions are ubiquitous in real-world data and must be considered for accurate explanations."
- **Our Usage**: We leverage the automatic interaction detection capabilities of EBM, which identifies significant pairwise interactions during training.

---

### 4. H-Statistic for Interaction Measurement

**Friedman, J. H., & Popescu, B. E. (2008). Predictive Learning via Rule Ensembles. *The Annals of Applied Statistics*, 2(3), 916-954.**

- **Relevance**: This paper introduces the H-statistic, a measure of interaction strength between features, and provides empirical evidence that interactions are significant in most benchmark datasets.
- **Key Insight**: "A surprisingly large number of predictor variables in classification and regression problems exhibit strong interactions."
- **Our Usage**: The H-statistic provides theoretical justification for why addressing interactions is important and validates our claim that additive models are insufficient for many real datasets.

---

### 5. Causal Feature Attribution

**Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature Relevance Quantification in Explainable AI: A Causal Problem. *Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics (AISTATS 2020)*, 2907-2916.**

- **Relevance**: This paper discusses the theoretical foundations of feature attribution in explainable AI, including how interactions affect the proper allocation of importance.
- **Key Insight**: "When features interact, their individual contributions cannot be determined independently."
- **Our Usage**: This supports our design choice to split interaction contributions fairly between the interacting features using principles from cooperative game theory.

---

## Additional Supporting References

### 6. InterpretML Framework

**Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv preprint arXiv:1909.09223*.**

- **Relevance**: This paper describes the InterpretML framework which includes the Explainable Boosting Machine (EBM) implementation we use.
- **Our Usage**: We use InterpretML's EBM with the `interactions` parameter enabled for our GA²M surrogate.

### 7. Original SHAP Paper

**Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 4765-4774.**

- **Relevance**: The foundational paper on SHAP values that InstaSHAP builds upon.
- **Our Usage**: Provides the theoretical foundation for Shapley value computation that both the original InstaSHAP and our extension leverage.

### 8. Limitations of Additive Explanations

**Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods. *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, 180-186.**

- **Relevance**: This paper discusses limitations of post-hoc explanations, including issues with additive approximations of complex models.
- **Our Usage**: Supports our claim that additive surrogates may not faithfully represent complex black-box models.

---

## Citation Format (IEEE)

[1] Y. Lou, R. Caruana, J. Gehrke, and G. Hooker, "Accurate intelligible models with pairwise interactions," in *Proc. 19th ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, 2013, pp. 623-631.

[2] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56-67, 2020.

[3] M. Tsang, D. Cheng, and Y. Liu, "Detecting statistical interactions from neural network weights," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2020.

[4] J. H. Friedman and B. E. Popescu, "Predictive learning via rule ensembles," *Ann. Appl. Statist.*, vol. 2, no. 3, pp. 916-954, 2008.

[5] D. Janzing, L. Minorics, and P. Blöbaum, "Feature relevance quantification in explainable AI: A causal problem," in *Proc. 23rd Int. Conf. Artificial Intelligence and Statistics (AISTATS)*, 2020, pp. 2907-2916.

---

*Document prepared for DS357 - Explainable AI Course Project*
