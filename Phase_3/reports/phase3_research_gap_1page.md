# Research Gap: Improving InstaSHAP via Three Targeted Innovations

## Problem: InstaSHAP (ICLR 2025) has three key limitations:
1. **Zero-masking** creates off-distribution inputs (invalid one-hot states)
2. **Uniform mask sampling** wastes capacity on hard coalitions early in training
3. **Single surrogate** fragility with no uncertainty signal

## Solution: Three layered innovations:
1. **Empirical-Background Masking** — replace absent features with real training data
2. **Curriculum-Weighted Training** — progressive warm-up → standard → hard schedule
3. **Multi-Surrogate Ensemble** — average 3 surrogates + variance as confidence

## Key References
- Enouen & Liu (ICLR 2025), Lundberg & Lee (2017), Aas et al. (2019)
- ViaSHAP (ICML 2025), Curriculum Learning (Bengio 2009)
- Explanation Multiplicity (2026), SHAP-IQ (NeurIPS 2024)