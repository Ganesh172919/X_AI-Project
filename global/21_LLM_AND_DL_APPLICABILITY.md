# Can InstaSHAP Be Applied To LLMs And Deep Learning Models?

This document explains where InstaSHAP fits, where it does not, and what to expect if you try to use it beyond tabular models.

## Deep learning models

- Yes, InstaSHAP can be applied to deep learning models when the input can be grouped into stable, meaningful features.
- It is most natural for fixed-size vector inputs, structured tabular models, or engineered embeddings.
- It can also be adapted to image patches or region groups, but masking design becomes much more important.
- The main requirement is that you can define a stable feature grouping and a meaningful masked value function.

## LLMs

- You can apply InstaSHAP to LLM-related systems only in limited and carefully defined ways.
- It is more realistic for structured LLM pipelines than for raw free-form generation.
- Good fit examples include retrieval scores, prompt-template fields, tool-selection features, ranking models, or fixed embedding vectors.
- Raw token-level generative prompting is a much harder fit because masking tokens destroys syntax and semantics.

## What happens if you apply it to raw LLM prompts

- Masking text often creates broken or unnatural prompts.
- The model output is a sequence, not a single stable scalar target.
- Token interactions are extremely rich and often much higher-order than a simple additive explanation setup can capture.
- The surrogate may end up learning prompt corruption behavior rather than real reasoning behavior.

## Should we expect good results?

- For structured tabular or fixed-vector deep learning settings, yes, good results are plausible.
- For raw generative LLM reasoning, no, not out of the box.
- The better the feature grouping and masking semantics, the more reasonable the results become.
- The less natural the masking operation, the less trustworthy the explanation becomes.

## Can InstaSHAP track internal reasoning?

- Not directly.
- InstaSHAP explains observable input-output behavior under a chosen masked value function.
- It does not reveal hidden chain-of-thought or private internal reasoning states by itself.
- At best, it can explain proxies such as logits, hidden-state summaries, layer outputs, or module decisions if those are exposed as explicit targets.

## Best safe conclusion

InstaSHAP is a strong fit for structured data and some deep learning settings with meaningful feature groups. It is not a direct tool for faithfully recovering hidden LLM reasoning, and raw generative prompt masking should not be expected to produce highly trustworthy explanations without much more task-specific design.
