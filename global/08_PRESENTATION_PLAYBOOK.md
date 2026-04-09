# Presentation Playbook

This document turns the repository into a practical presentation script.

## Suggested Story

- Start with why SHAP is useful but slow.
- Introduce InstaSHAP as an amortized explanation idea.
- Show that Phase 2 built the replication baseline.
- Introduce the Phase 3 limitation in simple terms.
- Explain empirical_background masking with a concrete example.
- Show the current results honestly.
- End with the future work direction rather than overclaiming success.

## Safe Claims

- We reproduced the pipeline in modular code.
- We implemented a targeted Phase 3 fix for a real limitation.
- The current results are mixed but informative.

## Unsafe Claims

- The latest branch definitively improves InstaSHAP overall.
- The current Phase 3 branch is the same as every older note in the repo.
- Every narrative file in the repo matches the current tables.

## Likely Questions

- What exactly is the limitation?
- Why is Covertype the right dataset?
- What improved and what did not?
- Why should we still trust the project if the gain is mixed?
- What would you do next?

## Best Answers

- The limitation is unrealistic coalition construction under transformed-space zero masking.
- Covertype is the best current dataset because the grouped categorical structure makes the issue easy to show.
- Some secondary signals improved, but overall end-metrics did not all improve.
- The project is stronger because it reports the result honestly and leaves a clear next-step path.
- The next step is stronger surrogate training and broader dataset validation.
