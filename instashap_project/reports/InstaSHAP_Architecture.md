# InstaSHAP — Architecture, Problem Statement, Trade-offs & Use Cases

---

## 1. The Problem InstaSHAP Solves

### 1.1 The Core Challenge: Explainability Is Too Slow

```mermaid
graph LR
    subgraph "THE PROBLEM"
        A["🧠 Black-Box ML Model<br/>(Accurate but Opaque)"] -->|"WHY did you<br/>make this decision?"| B["❓ Need Explanation"]
        B -->|"Traditional SHAP"| C["⏱️ ~1000 model evaluations<br/>PER SAMPLE"]
        C --> D["🚫 Too Slow for<br/>Real-Time Use"]
    end

    style A fill:#ff6b6b,color:#fff
    style D fill:#ff6b6b,color:#fff
```

**In simple terms:** Modern ML models (deep neural networks, random forests) are powerful predictors, but they are "black boxes" — no one knows *why* they make specific decisions. **Shapley values** are the gold standard for explaining predictions, but computing them requires evaluating the model on **all possible feature subsets** — an exponentially expensive process.

| Scenario | Traditional SHAP | InstaSHAP |
|----------|-----------------|-----------|
| Explain 1 prediction | ~2-10 seconds | ~1-5 milliseconds |
| Explain 1000 predictions | ~30-60 minutes | < 5 seconds |
| Real-time dashboard | ❌ Impossible | ✅ Feasible |
| Regulatory audit (millions of records) | ❌ Days/weeks | ✅ Minutes |

### 1.2 Why This Matters

```mermaid
mindmap
  root((Why Explainability<br/>Matters))
    🏥 Healthcare
      Drug interaction predictions
      Diagnosis support
      Patients deserve to know WHY
    🏦 Finance
      Loan approval/denial reasons
      Fraud detection explanations
      Regulatory compliance (GDPR, ECOA)
    ⚖️ Legal & Ethics
      Bias detection in hiring AI
      Fair sentencing algorithms
      Transparency requirements
    🏭 Industry
      Quality control decisions
      Predictive maintenance triggers
      Supply chain optimization reasoning
```

### 1.3 The Problem in One Sentence

> **Traditional SHAP gives perfect explanations but is too slow for production; InstaSHAP gives near-perfect explanations instantly by training an additive model that naturally recovers Shapley values in a single forward pass.**

---

## 2. System Architecture (Visual)

### 2.1 High-Level Pipeline

```mermaid
flowchart TB
    subgraph INPUT["📥 DATA INPUT"]
        DS[("UCI Dataset<br/>Bike / Covertype / Adult")]
        PP["TabularPreprocessor<br/>• StandardScaler (numeric)<br/>• OneHotEncoder (categorical)<br/>• Feature group mapping"]
        DS --> PP
    end

    subgraph SPLIT["✂️ DATA SPLITTING"]
        TR["Train Set (70%)"]
        VL["Validation Set (10%)"]
        TS["Test Set (20%)"]
        PP --> TR & VL & TS
    end

    subgraph STAGE1["⬛ STAGE 1: BLACK-BOX"]
        BB["TabularMLP<br/>[256 → 128 → output]<br/>Standard supervised training"]
        TR -->|"Train"| BB
        VL -->|"Early stopping"| BB
    end

    subgraph STAGE2["🟧 STAGE 2: MASKED SURROGATE"]
        MS["MaskedSurrogateMLP<br/>[x·mask ∥ mask] → [256 → 128 → output]<br/>Learns f(x; S) for any subset S"]
        BB -->|"Frozen predictions<br/>as targets"| MS
        MASK1["Shapley Kernel<br/>Mask Sampler"]
        MASK1 -->|"Random subsets S"| MS
    end

    subgraph STAGE3["🟩 STAGE 3: INSTASHAP"]
        IS["InstaSHAPModel<br/>Additive GAM [96 → 64 per feature]<br/>Gated by feature mask"]
        MS -->|"Frozen surrogate<br/>outputs as targets"| IS
        MASK2["Same Shapley Kernel<br/>Mask Sampler"]
        MASK2 -->|"Random subsets S"| IS
    end

    subgraph STAGE4["📊 STAGE 4: EXPLAIN & COMPARE"]
        SHAP["Permutation SHAP<br/>(~1000 evals/sample)"]
        ISHA["InstaSHAP .explain()<br/>(1 forward pass)"]
        CMP["Compare Attributions<br/>MSE / MAE"]
        BB -->|"Model to explain"| SHAP
        IS -->|"Trained model"| ISHA
        SHAP --> CMP
        ISHA --> CMP
    end

    subgraph OUTPUT["📁 OUTPUTS"]
        CSV["CSV Metrics Tables"]
        PLT["Plots & Heatmaps"]
        PDF["PDF Reports"]
        CMP --> CSV & PLT & PDF
    end

    style INPUT fill:#e3f2fd,stroke:#1565c0
    style STAGE1 fill:#fce4ec,stroke:#c62828
    style STAGE2 fill:#fff3e0,stroke:#e65100
    style STAGE3 fill:#e8f5e9,stroke:#2e7d32
    style STAGE4 fill:#f3e5f5,stroke:#6a1b9a
    style OUTPUT fill:#fff9c4,stroke:#f9a825
```

### 2.2 Model Architecture Details

```mermaid
flowchart LR
    subgraph BB["BLACK-BOX MLP"]
        direction TB
        B1["Input<br/>(D features)"] --> B2["Linear(256) + ReLU + Dropout(0.1)"]
        B2 --> B3["Linear(128) + ReLU + Dropout(0.1)"]
        B3 --> B4["Linear(output_dim)"]
        B4 --> B5["Prediction ŷ"]
    end

    subgraph SR["MASKED SURROGATE"]
        direction TB
        S1["Input = x·mask ∥ mask<br/>(D + n_features)"] --> S2["Linear(256) + ReLU + Dropout(0.1)"]
        S2 --> S3["Linear(128) + ReLU + Dropout(0.1)"]
        S3 --> S4["Linear(output_dim)"]
        S4 --> S5["Approx f(x; S)"]
    end

    subgraph GAM["GAM / INSTASHAP ADDITIVE MODEL"]
        direction TB
        G1["Feature 1 → MLP₁(96→64)"] --> GSUM["Σ + bias"]
        G2["Feature 2 → MLP₂(96→64)"] --> GSUM
        G3["Feature 3 → MLP₃(96→64)"] --> GSUM
        GN["Feature n → MLPₙ(96→64)"] --> GSUM
        GI["Pair (i,j) → MLP_pair(96→64)"] --> GSUM
        GSUM --> GOUT["Prediction / SHAP values"]
    end

    style BB fill:#fce4ec,stroke:#c62828
    style SR fill:#fff3e0,stroke:#e65100
    style GAM fill:#e8f5e9,stroke:#2e7d32
```

### 2.3 The Masking Mechanism (Core Innovation)

```mermaid
sequenceDiagram
    participant Sampler as Shapley Mask Sampler
    participant Input as Input x = [x₁, x₂, x₃, x₄]
    participant Mask as Mask S = [1, 0, 1, 0]
    participant Model as InstaSHAP Model

    Sampler->>Mask: Draw subset from Shapley kernel
    Note over Mask: S = {feature 1, feature 3}

    Mask->>Input: Element-wise multiply
    Note over Input: x·S = [x₁, 0, x₃, 0]

    Input->>Model: Feed masked input
    Note over Model: g₁(x₁) × 1 = active<br/>g₂(x₂) × 0 = gated off<br/>g₃(x₃) × 1 = active<br/>g₄(x₄) × 0 = gated off

    Model-->>Model: Output = bias + g₁(x₁) + g₃(x₃)
    Note over Model: Train to match surrogate(x·S, S)
```

**After training:** Each `gᵢ(xᵢ)` equals the Shapley value `φᵢ(x)` — no separate SHAP computation needed!

### 2.4 Training Flow (What Trains Against What)

```mermaid
graph TD
    LABELS["Ground Truth Labels<br/>(y_true)"]
    BB_MODEL["🟥 Black-Box MLP"]
    GAM1["🟦 GAM-1<br/>(no interactions)"]
    GAM2["🟦 GAM-2<br/>(with interactions)"]
    SURR["🟧 Masked Surrogate"]
    ISHA["🟩 InstaSHAP"]

    LABELS -->|"MSE / CrossEntropy"| BB_MODEL
    LABELS -->|"MSE / CrossEntropy"| GAM1
    LABELS -->|"MSE / CrossEntropy"| GAM2
    BB_MODEL -->|"Raw outputs as targets<br/>(frozen)"| SURR
    SURR -->|"Masked outputs as targets<br/>(frozen)"| ISHA

    style LABELS fill:#fff9c4,stroke:#f9a825
    style BB_MODEL fill:#fce4ec,stroke:#c62828
    style GAM1 fill:#e3f2fd,stroke:#1565c0
    style GAM2 fill:#e3f2fd,stroke:#1565c0
    style SURR fill:#fff3e0,stroke:#e65100
    style ISHA fill:#e8f5e9,stroke:#2e7d32
```

> **Key insight:** InstaSHAP never sees the ground truth labels directly. It learns to replicate the surrogate's behavior under masking, which in turn replicates the black-box's behavior.

---

## 3. Data Flow Architecture

```mermaid
flowchart LR
    subgraph RAW["Raw Data"]
        UCI["UCI Repository<br/>(ucimlrepo API)"]
    end

    subgraph LOAD["Loaders (data/loaders.py)"]
        BIKE["load_bike_sharing()<br/>13 features, regression"]
        COV["load_covertype()<br/>11 features, 7-class"]
        ADT["load_adult_income()<br/>13 features, binary"]
    end

    subgraph PREP["Preprocessing (data/preprocessing.py)"]
        TAB["TabularPreprocessor"]
        FG["FeatureGroup Mapping<br/>original → transformed indices"]
        SPL["make_splits()<br/>70/10/20 stratified"]
    end

    subgraph TENSOR["Tensors"]
        XT["X_train (float32)"]
        XV["X_val (float32)"]
        XE["X_test (float32)"]
    end

    UCI --> BIKE & COV & ADT
    BIKE & COV & ADT --> TAB
    TAB --> FG
    TAB --> SPL
    SPL --> XT & XV & XE

    style RAW fill:#e3f2fd
    style PREP fill:#fff3e0
    style TENSOR fill:#e8f5e9
```

---

## 4. Trade-off Analysis

### 4.1 Speed vs. Fidelity

```mermaid
quadrantChart
    title Speed vs Explanation Fidelity
    x-axis "Slow" --> "Fast"
    y-axis "Low Fidelity" --> "High Fidelity"

    "Exact SHAP": [0.05, 0.95]
    "Permutation SHAP": [0.15, 0.85]
    "KernelSHAP": [0.30, 0.75]
    "InstaSHAP": [0.90, 0.80]
    "LIME": [0.60, 0.50]
    "Gradient-based": [0.85, 0.40]
```

| Method | Speed | Fidelity | Consistency | Amortized Cost |
|--------|-------|----------|-------------|----------------|
| **Exact SHAP** | 🔴 Exponential | 🟢 Perfect | 🟢 Always consistent | None — recomputed every time |
| **Permutation SHAP** | 🔴 ~1000 evals/sample | 🟢 High (stochastic) | 🟡 Approximate | None — recomputed every time |
| **InstaSHAP** | 🟢 1 forward pass | 🟡 Near-perfect | 🟢 Deterministic | 🔴 Upfront training cost |
| **LIME** | 🟡 ~100 evals/sample | 🔴 Low (local linear) | 🔴 Inconsistent across runs | None |
| **Gradient-based** | 🟢 1 backward pass | 🔴 Not Shapley-faithful | 🟡 Model-dependent | None |

### 4.2 Detailed Trade-offs

#### ✅ Advantages of InstaSHAP

| Advantage | Explanation |
|-----------|-------------|
| **Instant explanations** | Single forward pass — enables real-time dashboards, streaming predictions |
| **Deterministic** | Same input always gets the same explanation (no stochastic sampling) |
| **Shapley-faithful** | Satisfies efficiency, symmetry, linearity axioms (unlike LIME, gradients) |
| **Interpretable model** | Each component `gᵢ` is a meaningful shape function you can inspect |
| **Batch-efficient** | Explain thousands of samples in one GPU batch operation |

#### ⚠️ Disadvantages & Costs

| Disadvantage | Explanation |
|--------------|-------------|
| **Upfront training cost** | Must train 3 models (black-box → surrogate → InstaSHAP) before getting any explanations |
| **Surrogate approximation error** | The surrogate may not perfectly replicate `f(x; S)`, introducing cumulative error |
| **Model-specific** | Trained for ONE specific black-box model — if the model changes, retrain everything |
| **Additive constraint** | The InstaSHAP model is additive (even with GAM-2 interactions), limiting representational capacity vs. the black-box |
| **Feature interactions** | Only pairwise interactions are captured; higher-order interactions are approximated |
| **Data-dependent** | Explanation quality depends on training data distribution — may not generalize to out-of-distribution inputs |

### 4.3 When InstaSHAP Wins vs. When It Doesn't

```mermaid
flowchart TD
    Q1{"Need real-time<br/>explanations?"}
    Q2{"Model changes<br/>frequently?"}
    Q3{"Need exact Shapley<br/>values?"}
    Q4{"High-volume<br/>explanations?"}
    Q5{"Small dataset<br/>(< 1000 samples)?"}

    Q1 -->|Yes| WIN1["✅ Use InstaSHAP"]
    Q1 -->|No| Q2

    Q2 -->|Yes| LOSE1["❌ Use Permutation SHAP<br/>(avoid retraining)"]
    Q2 -->|No| Q3

    Q3 -->|Yes| LOSE2["❌ Use Exact/Permutation SHAP"]
    Q3 -->|No| Q4

    Q4 -->|Yes| WIN2["✅ Use InstaSHAP"]
    Q4 -->|No| Q5

    Q5 -->|Yes| LOSE3["❌ Not enough data<br/>to train InstaSHAP well"]
    Q5 -->|No| WIN3["✅ InstaSHAP is a good fit"]

    style WIN1 fill:#e8f5e9,stroke:#2e7d32
    style WIN2 fill:#e8f5e9,stroke:#2e7d32
    style WIN3 fill:#e8f5e9,stroke:#2e7d32
    style LOSE1 fill:#fce4ec,stroke:#c62828
    style LOSE2 fill:#fce4ec,stroke:#c62828
    style LOSE3 fill:#fce4ec,stroke:#c62828
```

---

## 5. Use Cases

### 5.1 Real-World Application Scenarios

```mermaid
graph TB
    subgraph UC1["🏦 FINANCIAL SERVICES"]
        F1["Loan Decision Engine"] --> F2["Real-time denial reasons<br/>'Income: -35%, Debt ratio: -28%'"]
        F3["Fraud Detection Pipeline"] --> F4["Instant flagging explanations<br/>for analyst review"]
        F5["Credit Scoring API"] --> F6["Serve SHAP values alongside<br/>every credit score"]
    end

    subgraph UC2["🏥 HEALTHCARE"]
        H1["Diagnostic Support Tool"] --> H2["Show doctors which symptoms<br/>drove the diagnosis"]
        H3["Drug Interaction Predictor"] --> H4["Explain which drug combinations<br/>are risky and why"]
        H5["Patient Risk Scoring"] --> H6["Real-time ICU monitoring<br/>with explanation updates"]
    end

    subgraph UC3["🏭 INDUSTRY & IoT"]
        I1["Predictive Maintenance"] --> I2["Explain which sensor readings<br/>triggered failure prediction"]
        I3["Quality Control"] --> I4["Instant defect attribution<br/>on production line"]
        I5["Energy Forecasting"] --> I6["Explain demand predictions<br/>for grid operators"]
    end

    subgraph UC4["📊 ML OPERATIONS"]
        M1["Model Monitoring"] --> M2["Drift detection via<br/>shifting explanation patterns"]
        M3["A/B Testing"] --> M4["Compare feature importance<br/>across model versions"]
        M5["Debugging"] --> M6["Identify why model fails<br/>on specific data segments"]
    end

    style UC1 fill:#e3f2fd,stroke:#1565c0
    style UC2 fill:#fce4ec,stroke:#c62828
    style UC3 fill:#fff3e0,stroke:#e65100
    style UC4 fill:#e8f5e9,stroke:#2e7d32
```

### 5.2 Detailed Use Case Table

| Use Case | Domain | Why InstaSHAP? | Traditional SHAP Problem |
|----------|--------|----------------|--------------------------|
| **Real-time loan decisions** | Finance | Regulators require explanations for every decision; can't wait 10s per application | Permutation SHAP too slow for 10K+ daily applications |
| **Clinical decision support** | Healthcare | Doctors need instant explanations during patient consultations | Can't wait minutes while patient is in consultation |
| **Fraud alert triage** | Finance | Analysts need to see *why* each transaction was flagged, instantly | Thousands of alerts/hour require sub-second explanations |
| **Model monitoring dashboard** | MLOps | Aggregate explanation statistics over millions of predictions | Computing SHAP for millions of records is infeasible |
| **Edge device deployment** | IoT / Mobile | No cloud roundtrip available; explanations must be local | SHAP requires background data and many evaluations |
| **Interactive data exploration** | Research | Users explore "what-if" scenarios and need instant feedback | Each scenario change triggers a new expensive SHAP run |
| **Regulatory compliance** | Finance / Healthcare | GDPR "right to explanation" requires explanations for ALL decisions | Can't retroactively explain millions of historical decisions with SHAP |
| **Autonomous systems** | Robotics / AV | Real-time decision justification for safety-critical systems | Latency constraints prohibit iterative SHAP |

### 5.3 This Project's Specific Use Cases (Datasets)

| Dataset | Real-World Scenario | Explanation Value |
|---------|---------------------|-------------------|
| **Bike Sharing** | City transport planning — predict hourly demand | Understand that `hour × workingday` interaction drives commute vs. leisure patterns. Planners can optimize bike station placement. |
| **Covertype** | Forest service resource allocation — classify vegetation | Understand that `elevation` is the dominant predictor. Rangers can focus surveys on elevation bands rather than costly soil analysis. |
| **Adult Income** | Social policy — predict income brackets | Identify which demographic factors most influence income predictions. Flag potential bias in `race` or `sex` attributions. |

---

## 6. Comparison with Alternative Approaches

```mermaid
graph LR
    subgraph METHODS["Explainability Methods"]
        SHAP["🔵 Permutation SHAP<br/>Gold standard, slow"]
        LIME["🟡 LIME<br/>Fast-ish, inconsistent"]
        GRAD["🟠 Gradient × Input<br/>Fast, not Shapley"]
        ISHA["🟢 InstaSHAP<br/>Fast AND Shapley-faithful"]
        IG["🟤 Integrated Gradients<br/>Fast, path-dependent"]
    end

    SHAP -->|"InstaSHAP matches<br/>these values"| ISHA
    LIME -.->|"Not Shapley-<br/>consistent"| ISHA
    GRAD -.->|"Different theory<br/>entirely"| ISHA

    style ISHA fill:#e8f5e9,stroke:#2e7d32,stroke-width:3px
```

| Criterion | Permutation SHAP | LIME | Gradient-Based | InstaSHAP |
|-----------|-----------------|------|----------------|-----------|
| **Theoretical basis** | Shapley values | Local linear model | Backpropagation | Shapley values (via additive model) |
| **Speed** | O(2ⁿ) approximate | O(100) perturbations | O(1) backward pass | O(1) forward pass |
| **Determinism** | ❌ Stochastic | ❌ Stochastic | ✅ Deterministic | ✅ Deterministic |
| **Model-agnostic** | ✅ Any model | ✅ Any model | ❌ Differentiable only | ❌ Needs training pipeline |
| **Efficiency axiom** | ✅ Attributions sum to prediction | ❌ No guarantee | ❌ No guarantee | ✅ By construction |
| **Setup cost** | None | None | None | High (3 models to train) |
| **Works for new data** | ✅ | ✅ | ✅ | ✅ (within distribution) |

---

## 7. Performance Architecture

### 7.1 Computational Cost Breakdown

```mermaid
pie title Training Time Distribution (Typical Full Run)
    "Black-Box MLP Training" : 15
    "Masked Surrogate Training" : 30
    "InstaSHAP Training" : 25
    "GAM-1/GAM-2 Training" : 15
    "Permutation SHAP (Comparison)" : 10
    "Visualization & Reports" : 5
```

### 7.2 Inference Speed Comparison

```mermaid
xychart-beta
    title "Explanation Latency (log scale, lower is better)"
    x-axis ["Exact SHAP", "Permutation SHAP", "KernelSHAP", "LIME", "InstaSHAP"]
    y-axis "Milliseconds per sample" 0 --> 10000
    bar [10000, 5000, 1000, 200, 2]
```

---

## 8. Key Architectural Decisions & Rationale

| Decision | Rationale | Alternative Considered |
|----------|-----------|----------------------|
| **Additive architecture (GAM)** | Shapley values emerge naturally from additive components under the masked objective | Full MLP — but wouldn't decompose into per-feature attributions |
| **Two-stage distillation (BB → Surrogate → InstaSHAP)** | Surrogate amortizes the cost of masked evaluations; InstaSHAP then learns from surrogate | Direct InstaSHAP training against black-box — too expensive (exponential masked evals) |
| **Shapley kernel sampling** | Matches the theoretical weighting of the Shapley formula; proven to converge | Uniform mask sampling — biased, doesn't match Shapley weights |
| **Edge mask probability (10%)** | Adding all-zero and all-one masks stabilizes training at the boundaries | No edge masks — training can be unstable for extreme subsets |
| **AdamW optimizer** | Better generalization through decoupled weight decay | Adam, SGD — AdamW is standard for modern neural networks |
| **Early stopping on validation loss** | Prevents overfitting; restores best checkpoint | Fixed epochs — risk of over/under-training |
| **Feature group bookkeeping** | One-hot columns must be aggregated back to original features for meaningful explanations | Treating each one-hot column as a separate feature — unintelligible explanations |

---

## 9. Summary: The InstaSHAP Value Proposition

```mermaid
graph LR
    PROBLEM["🔴 PROBLEM<br/>Shapley values are<br/>too slow for production"]
    SOLUTION["🟢 SOLUTION<br/>Train an additive model<br/>that outputs SHAP values<br/>in one forward pass"]
    RESULT["🏆 RESULT<br/>Near-perfect SHAP fidelity<br/>at 1000x speed"]

    PROBLEM --> SOLUTION --> RESULT

    style PROBLEM fill:#fce4ec,stroke:#c62828
    style SOLUTION fill:#fff3e0,stroke:#e65100
    style RESULT fill:#e8f5e9,stroke:#2e7d32
```

> **Bottom line:** InstaSHAP trades upfront training cost for near-instant, Shapley-faithful explanations at inference time — making real-time explainable AI practical for the first time.

---

*This architecture document complements the [main research report](InstaSHAP_Research_Project_Report.md) with visual diagrams, trade-off analysis, and practical use-case guidance.*
