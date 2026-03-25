# System Flow Diagrams

## Purpose

This document gives a visual, systems-level understanding of how the project executes.

## End-to-End CLI Flow

```mermaid
flowchart TD
    A[User runs main.py] --> B[Load config.yaml]
    B --> C[Configure logging]
    C --> D[Set random seed]
    D --> E[Select dataset runner]
    E --> F[Load dataset]
    F --> G[Split train/val/test]
    G --> H[Fit preprocessor]
    H --> I[Train black-box]
    I --> J[Train GAM models]
    I --> K[Train masked surrogate]
    K --> L[Train InstaSHAP]
    I --> M[Run SHAP baseline]
    L --> N[Run InstaSHAP explainer]
    J --> O[Generate additive plots]
    M --> P[Compare explanations]
    N --> P
    P --> Q[Write tables and JSON summaries]
    Q --> R[Generate PDF reports]
```

## Module Dependency View

```mermaid
flowchart LR
    main --> experiments
    experiments --> data
    experiments --> training
    experiments --> xai
    experiments --> utils
    training --> models
    training --> utils
    xai --> training
    xai --> data
    reports --> results
```

## Runtime Artifact Flow

```mermaid
flowchart TD
    A[Training histories] --> B[results/artifacts/... logs]
    C[Metrics] --> D[results/tables]
    E[Plots] --> F[results/plots]
    D --> G[dataset summary JSON]
    F --> G
    G --> H[full report PDF]
    G --> I[1-page summary PDF]
```

## Why These Diagrams Matter

These diagrams help answer three important questions quickly:

- Where does a new feature belong?
- Which modules are on the critical path?
- Which artifacts depend on which earlier steps?

