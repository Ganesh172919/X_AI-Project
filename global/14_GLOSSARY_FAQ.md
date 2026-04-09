# Glossary And FAQ

This appendix provides short definitions and ready answers for discussion, viva, or presentation use.

## Glossary

- SHAP: Feature attribution framework grounded in Shapley values.
- InstaSHAP: Amortized additive explainer that produces SHAP-style attributions in one forward pass after training.
- Black-box model: The predictive model we want to explain.
- Surrogate: A model trained to imitate the black-box under feature coalitions.
- Coalition: A subset of visible features in the Shapley setup.
- zero_mask: Baseline Phase 3 masking strategy that hides groups by zeroing transformed columns.
- empirical_background: Improved Phase 3 masking strategy that fills hidden groups from real transformed training rows.
- Coalition fidelity: Agreement between surrogate outputs and black-box outputs under the same masked inputs.
- Explanation fidelity: Agreement between InstaSHAP outputs and the SHAP reference explainer.
- Feature group: All transformed columns belonging to one original feature.
- Background bank: Sampled training rows used to complete hidden feature groups realistically.

## FAQ

- Q: What is the latest Phase 3 improvement?
- A: Empirical background masking in Phase_3_work/instashap_project/masking.py.

- Q: Is the current Phase 3 branch the same as the root architecture note?
- A: No. The root note describes an interaction-aware idea, while the runnable branch focuses on masking realism.

- Q: Did the latest branch improve InstaSHAP overall?
- A: Not overall in the current saved artifacts. The result is mixed.

- Q: Is it still a good limitation?
- A: Yes, because it is specific, realistic, and directly implemented in code.

- Q: Which dataset should we use to show the limitation?
- A: Covertype is the strongest current choice in this repository.

- Q: What should be the next improvement?
- A: Stronger surrogate training, direct invalid-state metrics, and eventually combining masking realism with interaction-aware modeling.
