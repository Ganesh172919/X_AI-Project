# Phase 3 Applicability To LLMs And Deep Learning Models

## Deep learning models

- InstaSHAP can work on deep learning models when features can be grouped meaningfully.
- Structured vectors, tabular features, image regions, or fixed embeddings are more realistic targets than raw text generation.

## LLMs

- InstaSHAP is not a direct tool for recovering hidden internal reasoning from a raw generative LLM.
- It can be used on structured LLM systems such as ranking heads, retrieval scores, prompt fields, or fixed embedding features.
- Raw prompt token masking often breaks meaning and creates poor coalition semantics.

## What happens if you apply it anyway

- You may explain prompt corruption rather than genuine reasoning.
- The surrogate can become unstable because sequence outputs are much harder than fixed scalar targets.
- Results may be weak unless the problem is carefully restructured first.

## Can we track internal reasoning?

- Not directly.
- InstaSHAP explains behavior under a chosen masked value function.
- It can explain proxies, but it does not reveal hidden chain-of-thought by itself.
