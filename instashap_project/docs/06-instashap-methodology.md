# InstaSHAP Methodology

## The Core Equation

The project follows the paper's Equation (20) conceptually:

```text
E_x E_S [ f(x; S) - sum_T 1(T subseteq S) * phi_T(x_T) ]^2
```

There are two important ingredients:

1. Masks `S` are sampled from the **Shapley kernel**
2. Each additive component contributes only if all of its features are present in the mask

## What This Means Intuitively

Instead of computing SHAP values after a model is trained, InstaSHAP trains an additive model that already knows how masked subsets behave. Because of that:

- Univariate terms act like per-feature attribution building blocks
- Pairwise interaction terms capture low-order interaction structure
- Final per-feature attributions can be read off by distributing pairwise terms across features

## Masked Surrogate

The project trains a **masked surrogate** before training InstaSHAP. This surrogate approximates:

```text
f(x; S)
```

where:

- `x` is the transformed input
- `S` is the binary mask over original features

The surrogate gives a trainable target for the additive InstaSHAP model.

## Why Not Train Directly Against SHAP Values?

Because the paper's point is not just to imitate already-computed SHAP scores. The method instead learns from masked function behavior, then recovers SHAP-like attributions from the additive decomposition.

## How Attributions Are Recovered

For a model with:

- univariate components `phi_i`
- pairwise components `phi_ij`

the project returns feature attributions by:

- assigning each `phi_i` fully to feature `i`
- splitting `phi_ij` equally between feature `i` and feature `j`

This keeps the attribution extraction simple and fast at inference time.
