# Phase 3 Dataset Extension Prompt

Use this prompt when you want an assistant or teammate to continue the project from the new dataset extension point.

```text
You are extending Phase 3 of the InstaSHAP repository. Keep the current Covertype branch intact, but build a second extension track for Adult Income. Use the existing data loaders and preprocessing code, measure both the coalition-validity improvement and full end-to-end explanation metrics, keep filenames dataset-specific, and produce CSV tables, plots, Markdown, PDF, and notebook outputs. Do not claim improvement unless the saved tables show it. Track why the new dataset helps, why Covertype stayed mixed, and whether the masking fix generalizes. If the full pipeline is still mixed, preserve the diagnostic evidence that coalition realism improved.
```

## How to use it

- Start from the Adult masking diagnostic notebook and report.
- Ask the assistant to generalize the current Phase 3 runner to new datasets without breaking Covertype.
- Keep the evidence honest: diagnostic gains are not the same as full-model gains.
