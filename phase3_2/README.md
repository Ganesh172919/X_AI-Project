# Phase 3: InstaSHAP Extension - Interaction-Aware InstaSHAP

This folder is a cleaner write-up of the interaction-aware Phase 3 direction. Think of it as the more presentation-ready version of `phase3/`.

## Core question

```text
What happens when the black-box model depends on interactions,
but the InstaSHAP surrogate is only additive?
```

## Short answer

The additive surrogate can become misspecified, so its explanations may be analytically correct for the surrogate while still being wrong for the real black-box behavior.

## Proposed fix

Interaction-Aware InstaSHAP:
- add pairwise interaction terms
- preserve interpretability
- split interaction contributions fairly across the participating features
- use adaptive selection so extra complexity is only added when needed

## How InstaSHAP works in this branch

```text
1. create coalition-based training targets from the black-box
2. fit a surrogate that can include pairwise terms
3. center each learned term
4. allocate single-feature terms directly
5. split each pairwise term across the two features
6. return fast one-pass explanations from the learned surrogate structure
```

This branch is focused on making the explanation model class match the black-box behavior better when interactions matter.

## Beginner example

```text
house_price depends on:
  size
  neighborhood

But the effect of size is different in different neighborhoods.

Additive model:
  effect(size) + effect(neighborhood)

Interaction-aware model:
  effect(size)
  + effect(neighborhood)
  + effect(size, neighborhood)
```

## What this folder is best at

- explaining the assignment logic
- describing the research gap clearly
- showing why this is a valid Phase 3 extension

## What this folder is not best at

It is not the strongest evidence folder for final results. For saved outputs and the most complete interaction-aware run, open `../phase3_3_VERSION/README.md`.

## Key files

- `gap_analysis/research_gap.md`
- `extension/interaction_aware_surrogate.py`
- `extension/enhanced_instashap.py`
- `extension/adaptive_surrogate.py`
- `experiments/experiment_comparison.py`

## Important limitation

This track fixes the interaction problem, but not the masked-data realism problem. If the coalition construction itself is unrealistic, explanation quality can still degrade.

## What improves and what still fails

What improves:
- the model can express pairwise synergy instead of pretending everything is additive
- the assignment story is cleaner and easier to justify

What still fails:
- masking/data-manifold issues remain
- explanation quality can still lag even when surrogate fidelity improves
- sequence-generation and raw LLM use remain out of scope for the current design

## LLM note

This design is still mainly for structured inputs with stable feature groups. It is not a direct drop-in solution for explaining raw LLM token generation.
