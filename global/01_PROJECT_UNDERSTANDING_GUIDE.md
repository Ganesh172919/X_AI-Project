# Project Understanding Guide

This document is the fastest complete orientation for a new reader. It explains what the project is, how the phases fit together, and what the current repository state really means.

## Project In One Paragraph

- The repository studies InstaSHAP, a method for producing SHAP-style feature attributions in one forward pass after training an additive explanation model.
- Phase 1 proposed the project and the paper choice.
- Phase 2 built the replication baseline across Bike Sharing, Covertype, and Adult Income.
- Phase 3 narrowed the research gap to a masking limitation on Covertype and implemented empirical_background masking.
- A separate presentation package converts the technical work into a web presentation and PowerPoint deck.

## Core Problem

- SHAP is powerful but expensive because it depends on many coalition evaluations.
- InstaSHAP tries to amortize that cost into training so inference-time explanations are fast.
- The repository therefore balances predictive performance, explanation fidelity, and runtime.
- The main engineering question is how to preserve explanation quality while reducing explanation latency.
- The main research question in Phase 3 is whether more realistic coalition construction improves the pipeline.

## What The Project Built

- A replication package in Phase_2_work/instashap_project with modular data, model, training, evaluation, and XAI code.
- A runnable extension package in Phase_3_work with explicit comparison between zero_mask and empirical_background masking.
- Saved results in CSV, JSON, PNG, Markdown, and PDF formats.
- An HTML presentation and a PowerPoint generator in instashap_presentation/.
- A new global folder that normalizes the full project story and current evidence.

## What A Reviewer Needs To Know

- The latest runnable Phase 3 improvement is empirical_background masking in Phase_3_work/instashap_project/masking.py.
- The root phase3-architecture.md is a conceptual interaction-aware note, not the same thing as the current runnable Phase_3_work branch.
- The root README still describes an earlier EBM-flavored plan, so it should not be treated as the final truth source for the current runnable system.
- The current saved Phase 3 artifact snapshot favors instashap_zero on accuracy (0.6843 vs 0.6774) and explanation MAE (0.3591 vs 0.3795).
- The current saved Phase 3 artifact snapshot slightly favors instashap_bg on Spearman rank alignment (0.5835 vs 0.5650) and coalition MSE (0.2016 vs 0.2021).
- The limitation is still a good research choice because it is specific, code-level, measurable, and easy to explain to reviewers.

## Best Files To Open First

- README.md for the original course framing.
- Phase_2_work/instashap_project/README.md for the replication story.
- Phase_3_work/README.md for the current runnable extension story.
- Phase_3_work/main.py for the actual Phase 3 entrypoint.
- Phase_3_work/instashap_project/masking.py for the latest improvement itself.
- Phase_3_work/results/tables/*.csv for any final metric claim.

## What To Say In One Minute

- We chose InstaSHAP because SHAP is expensive and practical XAI needs faster explanations.
- We first reproduced the tabular pipeline in a modular codebase.
- We then identified that zero-masking in transformed tabular space can produce unrealistic coalition states.
- We implemented empirical_background masking so hidden groups come from real transformed training rows.
- The current evidence is mixed, which gives the project a strong and honest next-step research direction.
