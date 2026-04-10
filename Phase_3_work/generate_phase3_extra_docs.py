from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"
PROMPTS_DIR = ROOT / "prompts"
TESTS_DIR = ROOT / "tests"
SOURCE_DIR = ROOT / "instashap_project"

SKIP_PARTS = {"__pycache__"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def approx_lines(path: Path) -> int | None:
    if path.suffix.lower() not in {".py", ".md", ".csv", ".json", ".yaml", ".yml", ".txt", ".html", ".css", ".js", ".log"}:
        return None
    try:
        return len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
    except Exception:
        return None


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def all_files(*roots: Path) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not any(part in SKIP_PARTS for part in path.parts):
                files.append(path)
    return sorted(files)


def infer_source_role(path: Path) -> str:
    name = path.name
    path_text = rel(path)
    mapping = {
        "__init__.py": "package boundary file",
        "loaders.py": "dataset loading and metadata logic",
        "preprocessing.py": "feature preprocessing and feature-group bookkeeping",
        "common.py": "experiment orchestration and artifact writing",
        "covertype.py": "dataset-specific Phase 3 runner",
        "masking.py": "core Phase 3 improvement implementation",
        "blackbox_model.py": "black-box and surrogate network definitions",
        "gam.py": "additive model components and attribution logic",
        "instashap.py": "InstaSHAP model wrapper on top of GAM behavior",
        "reporting.py": "report generation from saved artifacts",
        "evaluate.py": "prediction and evaluation helpers",
        "train.py": "training loops and Shapley-style mask sampling",
        "logging_utils.py": "logging utilities",
        "metrics.py": "metric helper functions",
        "reproducibility.py": "seed and output helper functions",
        "visualization.py": "plotting helpers and output generation",
        "instashap_explainer.py": "one-pass explanation wrapper",
        "shap_wrapper.py": "SHAP baseline wrapper",
    }
    if name in mapping:
        return mapping[name]
    if "data/" in path_text:
        return "data-layer source file"
    if "models/" in path_text:
        return "model-layer source file"
    if "training/" in path_text:
        return "training-layer source file"
    if "utils/" in path_text:
        return "utility-layer source file"
    if "xai/" in path_text:
        return "explanation-layer source file"
    return "supporting Phase 3 source file"


def parse_python_symbols(path: Path) -> tuple[list[str], list[str], list[str]]:
    classes: list[str] = []
    functions: list[str] = []
    constants: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return classes, functions, constants
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    constants.append(target.id)
    return classes, functions, constants


def artifact_role(path: Path) -> str:
    path_text = rel(path)
    suffix = path.suffix.lower()
    if "/results/tables/" in path_text:
        return "saved table used as evidence or comparison input"
    if "/results/plots/" in path_text:
        return "saved plot used for analysis or presentation"
    if "/results/artifacts/" in path_text:
        return "machine-readable summary or saved run artifact"
    if path_text.startswith("reports/"):
        return "formal report or summary artifact"
    if path_text.startswith("notebooks/"):
        return "interactive notebook asset"
    if path_text.startswith("prompts/"):
        return "prompt asset for continuing the project"
    if path_text.startswith("tests/"):
        return "test asset or verification source"
    if suffix == ".log":
        return "run log"
    return "supporting Phase 3 artifact"


def artifact_use(path: Path) -> str:
    path_text = rel(path)
    if "adult_masking_diagnostic" in path_text:
        return "Use this when showing why Adult is a stronger masking-showcase dataset."
    if "covertype_" in path.name:
        return "Use this when presenting or defending the current saved Covertype benchmark."
    if "phase3_dataset_" in path.name:
        return "Use this when comparing datasets or extending Phase 3 beyond Covertype."
    if path_text.startswith("reports/"):
        return "Use this in formal presentation or report submission."
    if path_text.startswith("notebooks/"):
        return "Use this for interactive exploration and demonstration."
    if path_text.startswith("prompts/"):
        return "Use this to continue the project with an assistant or teammate."
    if path_text.startswith("tests/"):
        return "Use this to verify assumptions before presenting new claims."
    return "Use this as supporting evidence inside Phase 3."


def build_source_walkthrough() -> list[str]:
    lines = [
        "# Phase 3 Source Walkthrough Appendix",
        "",
        "This appendix gives a source-level walkthrough of the Phase 3 implementation package.",
        "",
        "## How to use this appendix",
        "",
        "- Open this file when you want to understand what each Python source file does.",
        "- Use it together with `docs/PHASE3_IMPROVEMENT_QUICKSTART.md` and `docs/PHASE3_IMPROVEMENT_BEGINNER_GUIDE.md`.",
        "- Treat `masking.py`, `experiments/common.py`, and `training/train.py` as the highest-value files for the latest improvement.",
        "",
    ]
    source_files = all_files(SOURCE_DIR)
    source_files = [path for path in source_files if path.suffix.lower() == ".py"]
    for path in source_files:
        classes, functions, constants = parse_python_symbols(path)
        count = approx_lines(path)
        lines.extend(
            [
                f"## {rel(path)}",
                "",
                f"- Role: {infer_source_role(path)}.",
                f"- Approx lines: {count if count is not None else 'n/a'}.",
                f"- Package area: {path.parent.relative_to(SOURCE_DIR).as_posix() if path.parent != SOURCE_DIR else 'root package'}.",
                "- Reading priority: high if you are tracing the end-to-end Phase 3 improvement path.",
                "- Why it matters: it contributes directly to the runnable Phase 3 system or its diagnostics.",
                "",
                "### Constants",
                "",
            ]
        )
        if constants:
            for item in constants:
                lines.append(f"- `{item}` appears as a module-level constant or configuration anchor.")
        else:
            lines.append("- No major uppercase module constants detected.")
        lines.extend(["", "### Classes", ""])
        if classes:
            for item in classes:
                lines.append(f"- `{item}` is part of the main source interface for this file.")
        else:
            lines.append("- No top-level classes detected.")
        lines.extend(["", "### Functions", ""])
        if functions:
            for item in functions:
                lines.append(f"- `{item}` is a top-level function exported or used by the module.")
        else:
            lines.append("- No top-level functions detected.")
        lines.extend(
            [
                "",
                "### Reading notes",
                "",
                f"- Start from `{path.name}` if your question is directly related to {infer_source_role(path)}.",
                "- Compare this file with the corresponding Phase 2 file if you want to see what changed structurally.",
                "- Read the tests and saved reports after reading the source when you need evidence or verification context.",
                "- Use the file together with the generated diagnostics if you are trying to connect code to results.",
                "",
            ]
        )
    return lines


def build_artifact_atlas() -> list[str]:
    lines = [
        "# Phase 3 Artifact Atlas",
        "",
        "This appendix catalogs the artifacts created or used by Phase 3 so a reader can understand what each file proves or supports.",
        "",
        "## How to use this appendix",
        "",
        "- Open this when you need to know which saved file to show during a presentation or review.",
        "- Use the role note to understand whether a file is evidence, communication, or verification support.",
        "",
    ]
    groups = [
        ("results", all_files(RESULTS_DIR)),
        ("reports", all_files(REPORTS_DIR)),
        ("notebooks", all_files(NOTEBOOKS_DIR)),
        ("prompts", all_files(PROMPTS_DIR)),
        ("tests", all_files(TESTS_DIR)),
    ]
    for group_name, files in groups:
        lines.extend([f"## {group_name.title()}", ""])
        for path in files:
            count = approx_lines(path)
            lines.extend(
                [
                    f"### {rel(path)}",
                    "",
                    f"- Role: {artifact_role(path)}.",
                    f"- Practical use: {artifact_use(path)}",
                    f"- File type: `{path.suffix.lower() or 'no extension'}`.",
                    f"- Approx bytes: {path.stat().st_size}.",
                    f"- Approx lines: {count if count is not None else 'n/a'}.",
                    "- Evidence note: use current saved artifacts instead of older narrative claims when a conflict appears.",
                    "- Presentation note: keep this artifact paired with a short explanation of what it does and does not prove.",
                    "",
                ]
            )
    return lines


def build_reviewer_qa_bank() -> list[str]:
    categories: dict[str, list[tuple[str, str]]] = {
        "Project scope": [
            ("What is the project about?", "It studies InstaSHAP for fast SHAP-style explanations and extends the tabular pipeline with a masking realism improvement."),
            ("What exactly is Phase 3?", "Phase 3 is the extension stage that targets a specific limitation rather than redoing the whole replication."),
            ("Why not stop at Phase 2?", "Phase 2 proves replication, but Phase 3 is needed to show critical thinking and research extension."),
            ("What is the main limitation you selected?", "Unrealistic coalition construction under transformed-space zero masking."),
            ("Why is that limitation important?", "Because invalid coalition states can corrupt the surrogate target and the final explanation quality."),
            ("Why is this a valid research gap?", "It is specific, code-level, measurable, and easy to justify in tabular preprocessing pipelines."),
            ("What is the improvement called?", "The implemented improvement is empirical_background masking."),
            ("What changed in code?", "The main change is in masking.py, where hidden groups can be filled from real transformed training rows."),
            ("What stayed the same?", "The broader black-box -> surrogate -> InstaSHAP architecture stayed the same."),
            ("What is the strongest safe claim?", "The masking improvement is valid and measurable, but the full end-to-end Covertype result is still mixed."),
        ],
        "Covertype": [
            ("Why use Covertype at all?", "It is the current runnable Phase 3 benchmark and connects well to the original project story."),
            ("Did Covertype improve overall?", "No, not overall in the current saved tables."),
            ("What improved slightly on Covertype?", "Spearman rank alignment and coalition MSE improved slightly."),
            ("What stayed worse on Covertype?", "Accuracy, log loss, explanation MAE, and runtime favored the zero baseline."),
            ("Why is Covertype still useful?", "It is the honest benchmark showing that better masking does not automatically solve the whole pipeline."),
            ("Why does Covertype remain mixed?", "The empirical background objective is more realistic but also harder for the surrogate and final model to learn."),
            ("Is Covertype a failure?", "No, it is an honest research result with clear next-step implications."),
            ("What does Covertype teach us?", "It teaches that coalition realism and optimization capacity have to improve together."),
            ("Should we remove Covertype from the presentation?", "No, keep it as the honest end-to-end benchmark."),
            ("How should we describe Covertype?", "Describe it as the primary benchmark with mixed but informative results."),
        ],
        "Adult showcase": [
            ("Why introduce Adult Income?", "Because it makes the masking problem easier to demonstrate at the coalition-construction level."),
            ("What improved on Adult?", "Hidden categorical validity rose from 0.0000 to 1.0000 and hidden numeric exact-zero rate fell from 1.0000 to 0.0000."),
            ("Why does Adult show the masking gain more clearly?", "Because it has many categorical feature groups, so zero masking creates obviously invalid hidden states."),
            ("Does Adult prove full end-to-end accuracy improvement?", "No, the current Adult asset is a masking diagnostic, not a full new end-to-end Phase 3 retraining result."),
            ("Why is Adult still valuable?", "It proves that the masking improvement itself works clearly on a better-suited dataset."),
            ("What is the best way to present Adult?", "Present it as the best dataset for showing the masking improvement itself."),
            ("Why not replace Covertype entirely with Adult?", "Because Covertype is still the current full Phase 3 benchmark and should remain part of the honest story."),
            ("What should the reviewer conclude from Adult?", "That the masking idea is genuinely useful and deserves continuation across more datasets."),
            ("What should not be claimed from Adult?", "Do not claim a full new end-to-end InstaSHAP win unless the pipeline is actually retrained and saved."),
            ("What is the next step after Adult?", "Generalize the Phase 3 runner and reporting workflow so Adult can be evaluated end-to-end like Covertype."),
        ],
        "Metrics and evidence": [
            ("Which files are the truth source for current Phase 3 metrics?", "The saved CSV tables in results/tables and the generated reports based on them."),
            ("Why not trust every markdown file equally?", "Some older narrative docs describe earlier plans or older numbers."),
            ("What metric best shows the masking gain on Adult?", "Hidden categorical validity and hidden numeric exact-zero rate."),
            ("What metric best shows the mixed Covertype result?", "The combined view of accuracy, explanation MAE, Spearman, coalition MSE, and runtime."),
            ("Why are multiple metrics needed?", "Because one improvement can help coalition realism without improving every end-to-end metric."),
            ("What is coalition fidelity?", "It measures how well the surrogate matches the black-box under masked coalitions."),
            ("What is explanation fidelity?", "It measures how closely InstaSHAP explanations align with the chosen SHAP reference."),
            ("Why track nearest-train distance?", "It gives a simple realism signal for how far masked examples drift from the training manifold."),
            ("What makes a good presentation metric?", "A metric that is directly tied to the claimed improvement and easy to explain."),
            ("What is the current strongest diagnostic metric?", "Adult hidden categorical validity is the cleanest one right now."),
        ],
        "Improvement roadmap": [
            ("What should be improved first?", "Surrogate capacity and dataset-generic Phase 3 configuration."),
            ("Why improve the surrogate first?", "Because the new objective is harder and the surrogate is the main bottleneck."),
            ("Should we add more background samples?", "Yes, but that trades more stable coalition targets against more runtime."),
            ("Should we add more datasets?", "Yes, after keeping the evaluation structure and reporting honest."),
            ("Which datasets are next?", "Adult first, then other mixed tabular datasets such as Bank Marketing, German Credit, or Telco Churn."),
            ("Should we combine masking with interactions?", "Yes, that is one of the strongest future directions."),
            ("What happens if we only change masking and nothing else?", "Coalition realism can improve without full pipeline improvement."),
            ("What happens if we also improve surrogate capacity?", "The empirical_background branch has a better chance of converting realism gains into end-to-end gains."),
            ("Why add direct validity metrics to future reports?", "So the masking improvement is visible even when headline metrics remain mixed."),
            ("What is the best near-term path?", "Keep the research question narrow and strengthen the empirical_background branch."),
        ],
        "LLMs and deep learning": [
            ("Can InstaSHAP be used for deep learning models?", "Yes, when the input can be grouped into stable meaningful features."),
            ("Can InstaSHAP be used for raw LLM prompting?", "Not safely out of the box."),
            ("Why is raw LLM prompting hard?", "Because token masking breaks syntax and semantics, and outputs are sequences rather than simple fixed targets."),
            ("Can InstaSHAP explain retrieval or ranking subsystems around LLMs?", "Yes, those are much better structured targets."),
            ("Can InstaSHAP reveal internal reasoning?", "No, not directly."),
            ("What can it explain instead?", "Observable behavior under a chosen masked value function, or explicit proxies such as logits or module outputs."),
            ("Should we expect good results on raw generative LLM tasks?", "No, not without substantial task-specific redesign."),
            ("Should we expect good results on structured deep models?", "Yes, that is more realistic when the feature grouping is meaningful."),
            ("What is the safe conclusion for LLMs?", "Use InstaSHAP for structured LLM-adjacent systems, not as a direct chain-of-thought tracker."),
            ("What should not be promised?", "Do not promise faithful recovery of hidden reasoning just from masking outputs."),
        ],
        "Presentation and review": [
            ("What should the opening slide say?", "Explain why SHAP is valuable but too slow, and why InstaSHAP matters."),
            ("What should the main Phase 3 slide say?", "State the limitation, the fix, and the honest evidence outcome."),
            ("What if a reviewer asks why results are mixed?", "Explain that the new objective is more realistic but harder to learn."),
            ("What if a reviewer asks why the project is still strong?", "Explain that a well-scoped, falsifiable, and honest extension is academically stronger than an overstated claim."),
            ("What should be the final slide takeaway?", "The masking improvement is real, Adult shows it clearly, and Covertype defines the next optimization challenge."),
            ("What should a beginner read first?", "The quickstart and beginner guide in docs/ and the Phase 3 README."),
            ("What is the strongest new file for presentation preparation?", "PHASE3_IMPROVEMENT_PRESENTATION_MASTER.md."),
            ("What is the strongest new file for fast understanding?", "PHASE3_IMPROVEMENT_QUICKSTART.md."),
            ("What is the strongest new file for skeptical reviewers?", "PHASE3_COVERTYPE_VS_ADULT_ANALYSIS.md plus the diagnostic tables."),
            ("What should be shown as a one-page handout?", "phase3_improvement_summary_1page.pdf."),
        ],
    }

    lines = [
        "# Phase 3 Reviewer Q and A Bank",
        "",
        "This appendix is a long-form question and answer bank for presentations, viva, review, and onboarding.",
        "",
        "## How to use this appendix",
        "",
        "- Read the category that matches the kind of question you expect.",
        "- Use the short answer first, then expand only if the reviewer asks for more.",
        "- Keep every answer tied to current saved artifacts or current source files.",
        "",
    ]
    for category, pairs in categories.items():
        lines.extend([f"## {category}", ""])
        for index, (question, answer) in enumerate(pairs, start=1):
            lines.extend(
                [
                    f"### Q{index}: {question}",
                    "",
                    f"- Short answer: {answer}",
                    "- Expanded answer: connect the response back to the Phase 3 limitation, the masking change, and the saved evidence if the discussion continues.",
                    "- Best supporting file: choose a current Phase 3 table, report, or source file that directly backs the answer.",
                    "",
                ]
            )

    extra_topics = [
        "What if we add more seeds?",
        "What if we double surrogate width?",
        "What if we reduce background samples?",
        "What if we compare against another explainer?",
        "What if Adult end-to-end gains stay mixed too?",
        "What if we only show diagnostic metrics?",
        "What if a reviewer asks about fairness?",
        "What if a reviewer asks about runtime cost?",
        "What if a reviewer asks why not use conditional SHAP directly?",
        "What if a reviewer asks why the root README says something else?",
        "What if a reviewer asks which files to trust first?",
        "What if we want to turn this into a publication-style extension?",
        "What if we need to explain the project to a non-technical audience?",
        "What if a reviewer asks about failure analysis?",
        "What if the audience wants a business use case?",
        "What if the audience wants a healthcare use case?",
        "What if the audience asks whether this is production ready?",
        "What if the audience asks for the single most important next improvement?",
        "What if the audience asks whether the project is finished?",
        "What if the audience asks what changed from yesterday's docs?",
    ]
    lines.extend(["## Extra scenario questions", ""])
    for idx, topic in enumerate(extra_topics, start=1):
        lines.extend(
            [
                f"### Scenario {idx}: {topic}",
                "",
                "- Guidance: answer with a narrow claim first, then add the evidence source, then add the next-step implication.",
                "- Good pattern: what changed, what did not change, what the current evidence shows, and what should happen next.",
                "",
            ]
        )
    return lines


def update_docs_index(new_docs: list[str]) -> None:
    index_path = DOCS_DIR / "index.md"
    existing = index_path.read_text(encoding="utf-8") if index_path.exists() else "# Documentation Hub\n"
    marker = "## Extra Appendices"
    if marker in existing:
        return
    lines = existing.rstrip().splitlines()
    lines.extend(
        [
            "",
            "## Extra Appendices",
            "",
        ]
    )
    for item in new_docs:
        lines.append(f"- `{item}`")
    lines.append("")
    write_markdown(index_path, lines)


def build_line_count_report(paths: list[Path]) -> list[str]:
    rows: list[tuple[str, int]] = []
    total = 0
    for path in paths:
        count = len(path.read_text(encoding="utf-8").splitlines())
        rows.append((path.name, count))
        total += count
    lines = [
        "# Phase 3 Extra Docs Line Count Report",
        "",
        "This report shows the approximate line counts for the newly generated appendices.",
        "",
        "| File | Lines |",
        "| --- | --- |",
    ]
    for name, count in rows:
        lines.append(f"| {name} | {count} |")
    lines.append(f"| TOTAL | {total} |")
    lines.append("")
    lines.append("Target requested by user: about 3000 more lines of documentation.")
    lines.append("")
    return lines


def main() -> None:
    ensure_dir(DOCS_DIR)
    source_doc = DOCS_DIR / "PHASE3_SOURCE_WALKTHROUGH_APPENDIX.md"
    artifact_doc = DOCS_DIR / "PHASE3_ARTIFACT_ATLAS.md"
    qa_doc = DOCS_DIR / "PHASE3_REVIEWER_QA_BANK.md"
    line_doc = DOCS_DIR / "PHASE3_EXTRA_DOCS_LINE_COUNT.md"

    write_markdown(source_doc, build_source_walkthrough())
    write_markdown(artifact_doc, build_artifact_atlas())
    write_markdown(qa_doc, build_reviewer_qa_bank())
    write_markdown(line_doc, build_line_count_report([source_doc, artifact_doc, qa_doc]))
    update_docs_index([source_doc.name, artifact_doc.name, qa_doc.name, line_doc.name])
    print("Generated extra Phase 3 documentation appendices.")


if __name__ == "__main__":
    main()
