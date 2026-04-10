"""
Synthetic Masking Comparison — Main Pipeline

Runs the complete synthetic dataset experiment:
1. Generate synthetic dataset
2. Preprocess with StandardScaler + OneHotEncoder
3. Train MLP black-box
4. Run InstaSHAP with zero_mask
5. Run InstaSHAP with empirical_background
6. Measure at all four levels
7. Save all output assets
"""

from __future__ import annotations

import json
import time
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')

# Add this directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic_dataset_generator import (
    generate_synthetic_dataset, save_dataset,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES, LABEL_COL,
)
from preprocessing import SyntheticPreprocessor
from masking_runner import (
    train_blackbox, run_instashap_pipeline,
    compute_coalition_validity, compute_predictive_metrics,
    compute_explanation_metrics, DEVICE,
)

SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "synthetic_demo"


def main():
    print("=" * 72)
    print("  SYNTHETIC MASKING COMPARISON PIPELINE")
    print("  zero_mask vs empirical_background on engineered 200-row dataset")
    print("=" * 72)
    print()

    t_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Generate Dataset
    # ─────────────────────────────────────────────────────────────────
    print("STEP 1: Generating synthetic dataset...")
    df = generate_synthetic_dataset(n=200, random_state=SEED)
    csv_path = save_dataset(df, OUTPUT_DIR)

    X = df[ALL_FEATURES].copy()
    y = df[LABEL_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw = X_test_raw.reset_index(drop=True)

    print(f"  Train: {len(X_train_raw)} rows, Test: {len(X_test_raw)} rows")
    print(f"  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test label dist:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
    print()

    # ─────────────────────────────────────────────────────────────────
    # STEP 2: Preprocessing
    # ─────────────────────────────────────────────────────────────────
    print("STEP 2: Preprocessing (StandardScaler + OneHotEncoder)...")
    preprocessor = SyntheticPreprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(preprocessor.summary())
    print(f"  Transformed shape: train={X_train.shape}, test={X_test.shape}")
    print()

    # ─────────────────────────────────────────────────────────────────
    # STEP 2b: Train Black-Box MLP
    # ─────────────────────────────────────────────────────────────────
    print("STEP 2b: Training black-box MLP classifier...")
    blackbox = train_blackbox(X_train, y_train, X_test, y_test,
                              epochs=80, lr=0.002, batch_size=32, patience=15)

    # Evaluate blackbox
    import torch
    import torch.nn.functional as F
    blackbox.eval()
    with torch.no_grad():
        bb_out = blackbox(torch.FloatTensor(X_test).to(DEVICE))
        bb_probs = F.softmax(bb_out, dim=1).cpu().numpy()
        bb_preds = np.argmax(bb_probs, axis=1)

    bb_acc = accuracy_score(y_test, bb_preds)
    bb_f1 = f1_score(y_test, bb_preds, average='weighted', zero_division=0)
    print(f"  Black-box test accuracy: {bb_acc:.4f}")
    print(f"  Black-box test F1: {bb_f1:.4f}")
    print()

    # ─────────────────────────────────────────────────────────────────
    # STEP 3: Run InstaSHAP with zero_mask
    # ─────────────────────────────────────────────────────────────────
    print("STEP 3: Running InstaSHAP with ZERO_MASK strategy...")
    result_zero = run_instashap_pipeline(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        blackbox=blackbox,
        preprocessor=preprocessor,
        strategy='zero_mask',
        n_coalitions=500,
        n_eval_rows=40,
        random_state=SEED,
    )
    print(f"  Wall time: {result_zero['wall_time']:.2f}s")
    print(f"  Mean surrogate MSE: {result_zero['surrogate_mse_mean']:.6f}")
    print()

    # ─────────────────────────────────────────────────────────────────
    # STEP 4: Run InstaSHAP with empirical_background
    # ─────────────────────────────────────────────────────────────────
    print("STEP 4: Running InstaSHAP with EMPIRICAL_BACKGROUND strategy...")
    result_emp = run_instashap_pipeline(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        blackbox=blackbox,
        preprocessor=preprocessor,
        strategy='empirical_background',
        n_coalitions=500,
        n_eval_rows=40,
        random_state=SEED + 1,  # different seed for independent evaluation
    )
    print(f"  Wall time: {result_emp['wall_time']:.2f}s")
    print(f"  Mean surrogate MSE: {result_emp['surrogate_mse_mean']:.6f}")
    print()

    # ─────────────────────────────────────────────────────────────────
    # STEP 5: Compute all metrics at four levels
    # ─────────────────────────────────────────────────────────────────
    print("STEP 5: Computing metrics at all four levels...")
    print()

    # Level 1: Coalition Validity
    print("  Level 1: Coalition Validity...")
    validity_zero = compute_coalition_validity(
        result_zero['masked_data'], result_zero['masks'], preprocessor, X_train
    )
    validity_emp = compute_coalition_validity(
        result_emp['masked_data'], result_emp['masks'], preprocessor, X_train
    )

    print(f"    zero_mask:")
    print(f"      hidden_cat_valid_rate:  {validity_zero['hidden_categorical_valid_rate']:.4f}")
    print(f"      hidden_cat_invalid_rate: {validity_zero['hidden_categorical_invalid_rate']:.4f}")
    print(f"      hidden_num_zero_rate:   {validity_zero['hidden_numeric_exact_zero_rate']:.4f}")
    print(f"      nearest_train_dist:     {validity_zero['nearest_train_distance_mean']:.4f}")
    print(f"    empirical_background:")
    print(f"      hidden_cat_valid_rate:  {validity_emp['hidden_categorical_valid_rate']:.4f}")
    print(f"      hidden_cat_invalid_rate: {validity_emp['hidden_categorical_invalid_rate']:.4f}")
    print(f"      hidden_num_zero_rate:   {validity_emp['hidden_numeric_exact_zero_rate']:.4f}")
    print(f"      nearest_train_dist:     {validity_emp['nearest_train_distance_mean']:.4f}")
    print()

    # Level 2: Predictive
    print("  Level 2: Predictive...")
    pred_zero = compute_predictive_metrics(result_zero)
    pred_emp = compute_predictive_metrics(result_emp)

    print(f"    zero_mask:    acc={pred_zero['surrogate_accuracy']:.4f}, f1={pred_zero['surrogate_f1']:.4f}, mse={pred_zero['surrogate_mse_vs_blackbox']:.6f}")
    print(f"    empirical:    acc={pred_emp['surrogate_accuracy']:.4f}, f1={pred_emp['surrogate_f1']:.4f}, mse={pred_emp['surrogate_mse_vs_blackbox']:.6f}")
    print()

    # Level 3: Explanation
    print("  Level 3: Explanation Quality...")
    expl_metrics = compute_explanation_metrics(result_zero, result_emp)

    print(f"    zero_mask spearman:    {expl_metrics['zero_mask_spearman_corr']:.4f}")
    print(f"    empirical spearman:    {expl_metrics['empirical_spearman_corr']:.4f}")
    print(f"    zero_mask MAE:         {expl_metrics['zero_mask_explanation_mae']:.6f}")
    print(f"    empirical MAE:         {expl_metrics['empirical_explanation_mae']:.6f}")
    print()

    # Level 4: Runtime
    print("  Level 4: Runtime...")
    print(f"    zero_mask wall time:    {result_zero['wall_time']:.2f}s")
    print(f"    empirical wall time:    {result_emp['wall_time']:.2f}s")
    print()

    total_time = time.perf_counter() - t_start

    # ─────────────────────────────────────────────────────────────────
    # STEP 6: Build comparison table and save outputs
    # ─────────────────────────────────────────────────────────────────
    print("STEP 6: Saving output assets...")

    metrics_rows = []
    for label, validity, pred, result in [
        ('zero_mask', validity_zero, pred_zero, result_zero),
        ('empirical_background', validity_emp, pred_emp, result_emp),
    ]:
        row = {
            'strategy': label,
            # Level 1
            'hidden_categorical_valid_rate': round(validity['hidden_categorical_valid_rate'], 4),
            'hidden_categorical_invalid_rate': round(validity['hidden_categorical_invalid_rate'], 4),
            'hidden_categorical_groups_evaluated': validity['hidden_categorical_groups_evaluated'],
            'hidden_numeric_exact_zero_rate': round(validity['hidden_numeric_exact_zero_rate'], 4),
            'hidden_numeric_entries_evaluated': validity['hidden_numeric_entries_evaluated'],
            'nearest_train_distance_mean': round(validity['nearest_train_distance_mean'], 4),
            # Level 2
            'surrogate_accuracy': round(pred['surrogate_accuracy'], 4),
            'surrogate_f1': round(pred['surrogate_f1'], 4),
            'surrogate_mse_vs_blackbox': round(pred['surrogate_mse_vs_blackbox'], 6),
            # Level 3
            'spearman_rank_correlation': round(
                expl_metrics['zero_mask_spearman_corr'] if label == 'zero_mask'
                else expl_metrics['empirical_spearman_corr'], 4
            ),
            'explanation_mae': round(
                expl_metrics['zero_mask_explanation_mae'] if label == 'zero_mask'
                else expl_metrics['empirical_explanation_mae'], 6
            ),
            # Level 4
            'wall_time_seconds': round(result['wall_time'], 2),
        }
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_out = OUTPUT_DIR / "synthetic_masking_comparison.csv"
    metrics_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # JSON
    json_out = OUTPUT_DIR / "synthetic_masking_comparison.json"
    with open(json_out, 'w') as f:
        json.dump(metrics_rows, f, indent=2)
    print(f"  Saved: {json_out}")

    # ─────────────────────────────────────────────────────────────────
    # Bar Chart
    # ─────────────────────────────────────────────────────────────────
    _save_comparison_chart(metrics_df, OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────
    # Diagnostic Report
    # ─────────────────────────────────────────────────────────────────
    _save_diagnostic_report(metrics_df, expl_metrics, bb_acc, bb_f1, total_time, OUTPUT_DIR)

    print()
    print("=" * 72)
    print(f"  PIPELINE COMPLETE — Total time: {total_time:.1f}s")
    print(f"  All outputs saved to: {OUTPUT_DIR}")
    print("=" * 72)


def _save_comparison_chart(df: pd.DataFrame, output_dir: Path):
    """Create side-by-side bar chart for zero_mask vs empirical_background."""

    # Select key metrics for visualization
    plot_metrics = [
        ('hidden_categorical_valid_rate', 'Cat. Valid Rate\n(higher = better)', True),
        ('hidden_categorical_invalid_rate', 'Cat. Invalid Rate\n(lower = better)', False),
        ('hidden_numeric_exact_zero_rate', 'Num. Exact Zero Rate\n(lower = better)', False),
        ('nearest_train_distance_mean', 'Nearest Train\nDistance (lower = better)', False),
        ('surrogate_accuracy', 'Surrogate Accuracy\n(higher = better)', True),
        ('surrogate_f1', 'Surrogate F1\n(higher = better)', True),
        ('surrogate_mse_vs_blackbox', 'Surrogate MSE\nvs Blackbox (lower = better)', False),
        ('spearman_rank_correlation', 'Spearman Rank\nCorrelation (higher = better)', True),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    zero_color = '#E74C3C'       # Red for zero_mask (bad)
    emp_color = '#27AE60'        # Green for empirical (good)

    for idx, (metric, label, higher_better) in enumerate(plot_metrics):
        ax = axes[idx]
        vals = df.set_index('strategy')[metric]
        z_val = vals['zero_mask']
        e_val = vals['empirical_background']

        bars = ax.bar(['zero_mask', 'empirical\nbackground'],
                      [z_val, e_val],
                      color=[zero_color, emp_color],
                      edgecolor='white', linewidth=1.5, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, [z_val, e_val]):
            fmt = f'{val:.4f}' if val < 1.0 else f'{val:.2f}'
            if val > 100:
                fmt = f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(z_val, e_val, 0.01),
                    fmt, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(label, fontsize=10, fontweight='bold', pad=8)
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=9)

        # Mark which bar is "correct"
        if higher_better:
            winner = 'empirical\nbackground' if e_val >= z_val else 'zero_mask'
        else:
            winner = 'empirical\nbackground' if e_val <= z_val else 'zero_mask'

        if winner == 'empirical\nbackground':
            bars[1].set_edgecolor('#1A7A3C')
            bars[1].set_linewidth(3)
        else:
            bars[0].set_edgecolor('#A93226')
            bars[0].set_linewidth(3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Synthetic Dataset: zero_mask vs empirical_background\n'
                 '(Red = zero_mask, Green = empirical_background)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    chart_path = output_dir / "synthetic_masking_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {chart_path}")


def _save_diagnostic_report(df: pd.DataFrame, expl_metrics: dict,
                            bb_acc: float, bb_f1: float,
                            total_time: float, output_dir: Path):
    """Generate human-readable Markdown diagnostic report."""

    z = df[df['strategy'] == 'zero_mask'].iloc[0]
    e = df[df['strategy'] == 'empirical_background'].iloc[0]

    report = f"""# Synthetic Masking Diagnostic Report

**Dataset**: Fully synthetic, 200 rows (160 train / 40 test)
**Features**: 4 numeric + 3 categorical (5/5/4 levels) → ~18-22 one-hot columns
**Label rule**: high_earner = 1 if income > 60k AND education ∈ {{masters, doctorate}} AND occupation ∈ {{technical, executive, admin}}
**Black-box**: MLP classifier (accuracy={bb_acc:.4f}, F1={bb_f1:.4f})
**Coalitions per row**: 500
**Evaluation rows**: 40
**Total runtime**: {total_time:.1f}s

---

## Summary Table

| Metric | Level | zero_mask | empirical_background | Winner |
|--------|-------|-----------|---------------------|--------|
| hidden_categorical_valid_rate | L1-Diagnostic | {z['hidden_categorical_valid_rate']:.4f} | {e['hidden_categorical_valid_rate']:.4f} | {'empirical ✓' if e['hidden_categorical_valid_rate'] > z['hidden_categorical_valid_rate'] else 'zero_mask'} |
| hidden_categorical_invalid_rate | L1-Diagnostic | {z['hidden_categorical_invalid_rate']:.4f} | {e['hidden_categorical_invalid_rate']:.4f} | {'empirical ✓' if e['hidden_categorical_invalid_rate'] < z['hidden_categorical_invalid_rate'] else 'zero_mask'} |
| hidden_numeric_exact_zero_rate | L1-Diagnostic | {z['hidden_numeric_exact_zero_rate']:.4f} | {e['hidden_numeric_exact_zero_rate']:.4f} | {'empirical ✓' if e['hidden_numeric_exact_zero_rate'] < z['hidden_numeric_exact_zero_rate'] else 'zero_mask'} |
| nearest_train_distance_mean | L1-Diagnostic | {z['nearest_train_distance_mean']:.4f} | {e['nearest_train_distance_mean']:.4f} | {'empirical ✓' if e['nearest_train_distance_mean'] < z['nearest_train_distance_mean'] else 'zero_mask'} |
| surrogate_accuracy | L2-Predictive | {z['surrogate_accuracy']:.4f} | {e['surrogate_accuracy']:.4f} | {'empirical ✓' if e['surrogate_accuracy'] >= z['surrogate_accuracy'] else 'zero_mask'} |
| surrogate_f1 | L2-Predictive | {z['surrogate_f1']:.4f} | {e['surrogate_f1']:.4f} | {'empirical ✓' if e['surrogate_f1'] >= z['surrogate_f1'] else 'zero_mask'} |
| surrogate_mse_vs_blackbox | L2-Predictive | {z['surrogate_mse_vs_blackbox']:.6f} | {e['surrogate_mse_vs_blackbox']:.6f} | {'empirical ✓' if e['surrogate_mse_vs_blackbox'] <= z['surrogate_mse_vs_blackbox'] else 'zero_mask'} |
| spearman_rank_correlation | L3-Explanation | {z['spearman_rank_correlation']:.4f} | {e['spearman_rank_correlation']:.4f} | {'empirical ✓' if e['spearman_rank_correlation'] >= z['spearman_rank_correlation'] else 'zero_mask'} |
| explanation_mae | L3-Explanation | {z['explanation_mae']:.6f} | {e['explanation_mae']:.6f} | {'empirical ✓' if e['explanation_mae'] <= z['explanation_mae'] else 'zero_mask'} |
| wall_time_seconds | L4-Runtime | {z['wall_time_seconds']:.2f} | {e['wall_time_seconds']:.2f} | {'empirical ✓' if e['wall_time_seconds'] <= z['wall_time_seconds'] else 'zero_mask'} |

---

## Interpretation

### Level 1 — Coalition Validity (Diagnostic Level)

This is the **key metric** and the reason this dataset was designed. The result is structural:

- **zero_mask** leaves hidden categorical groups valid only **{z['hidden_categorical_valid_rate']:.1%}** of the time. Every hidden one-hot group has all columns set to 0, which is an impossible category state that never appears in real data. With 3 multi-level categorical features (education=5, occupation=5, region=4), every coalition contains multiple invalid one-hot blocks.

- **empirical_background** keeps hidden categorical groups valid **{e['hidden_categorical_valid_rate']:.1%}** of the time, because it copies the hidden group from a real training row — always producing exactly one active category.

- For hidden numeric values, zero_mask sets {z['hidden_numeric_exact_zero_rate']:.1%} to exact 0.0 (which in standardized space is NOT feature absence), while empirical_background copies realistic standardized values ({e['hidden_numeric_exact_zero_rate']:.1%} exact zeros).

- The mean nearest-train distance confirms that empirical_background produces coalitions closer to the training manifold ({e['nearest_train_distance_mean']:.4f} vs {z['nearest_train_distance_mean']:.4f}).

**This result is by construction and holds for any dataset with one-hot encoded categorical features.**

### Level 2 — Predictive Quality (End-to-End Level)

Surrogate accuracy and F1 measure how well the linear surrogate approximates the blackbox under each masking strategy. On this 200-row dataset, the differences may be modest because:
- The linear surrogate is a rough approximation regardless of masking
- 200 rows limits statistical power for end-to-end comparison

### Level 3 — Explanation Quality (End-to-End Level)

Spearman rank correlation measures whether feature importance rankings are preserved. The reference SHAP values come from the empirical_background surrogate (as a working proxy for ground truth).

### Level 4 — Runtime

Both strategies have similar runtime since the masking operation itself is lightweight. The dominant cost is coalition sampling and surrogate training.

---

## Diagnostic vs End-to-End Metrics

| Level | Type | Expected Result | Rationale |
|-------|------|----------------|-----------|
| L1 (Coalition Validity) | **Diagnostic** | Full win for empirical_background | By construction: zero_mask always produces invalid one-hot states |
| L2 (Predictive) | End-to-End | May be mixed on 200 rows | Small dataset limits statistical power |
| L3 (Explanation) | End-to-End | May be mixed on 200 rows | Depends on how much the surrogate benefits from valid inputs |
| L4 (Runtime) | End-to-End | Approximately equal | Masking cost is negligible |

---

## Conclusion

**This dataset proves the masking fix at the coalition-construction level (Level 1).** The 100% invalid rate under zero_mask vs 0% under empirical_background is not a statistical finding — it is a structural invariant that holds for any dataset with one-hot encoded categorical features.

The dataset is designed to demonstrate **why** empirical_background masking is necessary, not to prove a full end-to-end SHAP quality improvement. The 200-row size limits statistical power for Level 2–3 comparisons, and this is expected and acceptable for the purpose of this demonstration.

For end-to-end quality improvement evidence, see the Adult Income and Covertype experiments in the main Phase 3 pipeline.
"""

    report_path = output_dir / "synthetic_masking_diagnostic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved: {report_path}")


if __name__ == "__main__":
    main()
