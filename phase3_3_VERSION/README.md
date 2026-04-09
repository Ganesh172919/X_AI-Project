# Phase 3: Research Gap and Extension

This is the most complete interaction-aware InstaSHAP folder in the repository. If you want one folder that shows the interaction-gap story, start here.

## The research gap

Original InstaSHAP is fast because it relies on additive surrogates. That works well when the black-box behavior is close to additive, but it breaks down when important effects only appear through feature interactions.

Real example:

```text
y = x1 * x2

x1 alone is not enough
x2 alone is not enough
the pair is what matters
```

An additive surrogate cannot represent that exactly.

## The extension in this folder

This branch adds:
- an interaction-aware surrogate
- an interaction-aware InstaSHAP allocation rule
- an adaptive strategy that upgrades only when needed

## How InstaSHAP works in this branch

```text
black-box model
    ->
compute coalition-based explanation targets
    ->
fit additive surrogate
    ->
upgrade to interaction-aware surrogate if fidelity is too low
    ->
center surrogate terms over reference data
    ->
map main effects directly to features
    ->
split pairwise terms across the involved features
    ->
compare against exact SHAP
```

This is the clearest folder for understanding the interaction-aware Phase 3 idea end to end.

## What actually improved here

Saved `friedman1` results show:

| Method | Surrogate R2 | Pearson | MAE | Runtime |
| --- | --- | --- | --- | --- |
| Original InstaSHAP | 0.9195 | 0.9285 | 0.2918 | 0.0066s |
| Interaction-aware InstaSHAP | 0.9473 | 0.9225 | 0.3144 | 0.0131s |

## Honest interpretation

This folder teaches an important research lesson:

```text
better surrogate fidelity does not automatically guarantee better final attributions
```

What improved:
- the surrogate matched the black-box better

What did not clearly improve in the saved run:
- Pearson alignment
- MAE to exact SHAP

So this folder successfully demonstrates:
- why the additive gap is real
- why interaction-aware surrogates are a valid fix

But it does not yet prove that the whole pipeline is better in every downstream metric.

## What still fails

- pairwise terms help representation, but not every downstream attribution metric improves automatically
- masking/data-realism issues are still not addressed here
- higher-order interactions are still outside the current surrogate design
- raw LLM generation remains a poor fit because token masking and sequence outputs break the core assumptions

## Flow

```text
black-box model
    ->
fit additive surrogate
    ->
measure fidelity
    ->
fit interaction-aware surrogate
    ->
compute exact SHAP
    ->
compare original InstaSHAP vs interaction-aware InstaSHAP
```

## When this approach is best

Use this branch when your main failure source is:
- multiplicative interactions
- XOR-like logic
- pairwise dependency patterns
- low-order but important feature synergy

## When this approach is not enough

It does not fix unrealistic masking or data-manifold problems by itself. For that, study `../Phase_3_work/README.md`.

## Fine-tuned model note

This idea can work for a fine-tuned structured model with stable feature groups and a fixed prediction target.

It is not a good direct fit for raw LLM generation because:
- token masking changes semantics
- outputs are sequence-level
- the surrogate family is too simple for the full language process

## Best next read

- `gap_analysis/research_gap.md`
- `../phase3-architecture.md`
- `../Phase_3_work/docs/phase3_beginner_guide.md`
