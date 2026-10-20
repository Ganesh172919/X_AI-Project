# Beginner Guide For Phase 3

This guide is for a beginner who wants to understand the project quickly.

## Simple project story

1. Train a black-box model.
2. Train a surrogate to mimic the black-box under feature masks.
3. Train InstaSHAP to mimic the surrogate in an additive way.
4. Compare the result to SHAP.

## What Phase 3 improves

- The old masking wrote zeros into hidden transformed columns.
- The new masking copies hidden groups from real transformed rows.
- This helps avoid impossible category states and unrealistic masked inputs.

## Why Covertype is mixed

- The masking fix is real, but the end-to-end training problem becomes harder.
- Covertype still needs stronger surrogate fitting and possibly richer interactions.

## Why Adult is better for the masking story

- Adult categorical validity goes from 0.0000 to 1.0000.
- Adult nearest-train distance drops from 2.1893 to 1.3329.
- This makes the masking gain easier to show clearly.
