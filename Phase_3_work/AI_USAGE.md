# AI Usage Declaration

## Tool Used

- OpenAI Codex / GPT-based coding assistant

## How AI Was Used

- repo inspection and planning
- code scaffolding from the Phase 2 baseline
- implementation of the Phase 3 masking, training, evaluation, reporting, and documentation flow
- generation of Markdown and PDF report content templates
- assistance with unit tests and reproducibility tooling

## Prompt Summary

The core prompt asked for a standalone Phase 3 project in `Phase_3_work` that:

- identifies a limitation in the original InstaSHAP paper and implementation
- proposes one literature-supported improvement
- uses the same dataset family and Covertype as the primary benchmark
- generates code, experiment outputs, updated documentation, and PDFs

## Verification and Modification

AI-generated suggestions were not accepted blindly. The final project was:

- grounded in the existing Phase 2 codebase
- adjusted to a proper standalone package layout
- validated with Python compile checks
- executed with `--fast-dev-run`
- verified by inspecting generated CSV, Markdown, PDF, and plot artifacts

## Important Note

The generated report text is intentionally conservative. It describes the implemented empirical-background masking idea and the observed results without overstating the conclusions.
