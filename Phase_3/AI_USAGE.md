# AI Usage Declaration — Phase 3

## Tool Used
- **AI Assistant:** Google Gemini (Antigravity agent)
- **Purpose:** Code generation, research synthesis, and experiment design assistance

## How AI Was Used

### 1. Literature Review & Gap Identification
- **Prompt category:** "Search for latest SHAP/XAI improvements 2025-2026"
- **Verification:** All cited papers were cross-referenced with OpenReview, arXiv, and conference proceedings
- **Manual contribution:** Selection of 3 most impactful gaps from 5 identified candidates

### 2. Code Generation
- **Prompt category:** "Build standalone Phase 3 project reproducing Phase 2 pipeline with 3 innovations"
- **Generated modules:** data/, masking/, models/, training/, experiments/, xai/, utils/, reports/
- **Verification:** All training loops, masking strategies, and metrics were reviewed for correctness
- **Manual contribution:** Hyperparameter tuning, architecture decisions, experiment design

### 3. Report Writing
- **Prompt category:** "Generate experiment report PDF from results"
- **Verification:** All numbers in reports match CSV outputs and JSON artifacts

## What Was Verified Manually
- [x] Masking logic preserves one-hot group validity
- [x] Curriculum schedule transitions at correct epoch fractions
- [x] Ensemble averaging reduces to single surrogate when M=1
- [x] Shapley kernel weights sum to 1.0
- [x] Background bank sampling uses non-replacement
- [x] All training loops implement early stopping correctly
- [x] Report PDF numbers match JSON artifacts
- [x] CLI --fast-dev-run executes end-to-end without errors

## Academic Integrity Statement
This work was developed with AI assistance for code generation and research synthesis. The research gaps, innovation design, and experimental methodology represent original analytical contribution built upon the foundations of the InstaSHAP paper (Enouen & Liu, ICLR 2025). All results are reproducible via the provided codebase with the documented random seeds (42, 123, 7).
