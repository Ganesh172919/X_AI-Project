"""
Step 1: Synthetic Dataset Generator for InstaSHAP Masking Comparison.

Generates a 200-row binary classification dataset with:
- 4 numeric features: income, experience, hours_per_week, age
- 3 categorical features: education (5 levels), occupation (5 levels), region (4 levels)
- 1 binary label: high_earner (0 or 1) based on genuine feature interactions

After one-hot encoding, the feature matrix has ~20 columns. This guarantees
that zero_mask produces all-zero one-hot groups (always invalid), making
the masking difference structurally unavoidable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


EDUCATION_LEVELS = ["high_school", "associates", "bachelors", "masters", "doctorate"]
OCCUPATION_LEVELS = ["admin", "service", "sales", "technical", "executive"]
REGION_LEVELS = ["northeast", "southeast", "midwest", "west"]

NUMERIC_FEATURES = ["income", "experience", "hours_per_week", "age"]
CATEGORICAL_FEATURES = ["education", "occupation", "region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL_COL = "high_earner"


def generate_synthetic_dataset(n: int = 200, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic binary classification dataset with genuine feature interactions.

    Label rule:
        high_earner = 1 if ALL of:
            - income > 60000
            - education in {masters, doctorate}
            - occupation in {technical, executive, admin}
        ELSE high_earner = 0

    This ensures categorical features carry real signal, so SHAP values are meaningful.
    Distributions are designed to produce ~35-45% positive class.

    Parameters
    ----------
    n : int
        Number of rows.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with numeric, categorical, and label columns.
    """
    rng = np.random.RandomState(random_state)

    # ── Numeric features ──
    # Income is correlated with education and occupation (added later via noise)
    base_income = rng.lognormal(mean=10.8, sigma=0.5, size=n).clip(20000, 200000)
    experience = rng.normal(loc=12, scale=6, size=n).clip(0, 40).round(1)
    hours_per_week = rng.normal(loc=40, scale=8, size=n).clip(10, 80).round(0)
    age = rng.normal(loc=38, scale=10, size=n).clip(18, 70).round(0)

    # ── Categorical features with skewed distributions ──
    # Education: skew toward bachelor's to create meaningful variance
    edu_probs = [0.15, 0.15, 0.30, 0.25, 0.15]
    education = rng.choice(EDUCATION_LEVELS, size=n, p=edu_probs)

    # Occupation: slightly skew toward service/sales to make top-3 meaningful
    occ_probs = [0.20, 0.20, 0.15, 0.25, 0.20]
    occupation = rng.choice(OCCUPATION_LEVELS, size=n, p=occ_probs)

    # Region: relatively balanced
    reg_probs = [0.25, 0.25, 0.25, 0.25]
    region = rng.choice(REGION_LEVELS, size=n, p=reg_probs)

    # ── Inject correlation: education/occupation affect income ──
    edu_bonus = np.array([
        {"high_school": -15000, "associates": -5000, "bachelors": 0,
         "masters": 12000, "doctorate": 25000}[e] for e in education
    ], dtype=float)
    occ_bonus = np.array([
        {"admin": 5000, "service": -10000, "sales": -5000,
         "technical": 10000, "executive": 20000}[o] for o in occupation
    ], dtype=float)
    income = (base_income + edu_bonus + occ_bonus).clip(15000, 250000).round(0)

    # ── Experience correlates with age ──
    experience = (experience + (age - 30) * 0.3 + rng.normal(0, 2, size=n)).clip(0, 45).round(1)

    # ── Build DataFrame ──
    df = pd.DataFrame({
        "income": income,
        "experience": experience,
        "hours_per_week": hours_per_week,
        "age": age,
        "education": education,
        "occupation": occupation,
        "region": region,
    })

    # ── Label rule: genuine multi-feature interaction ──
    top_education = {"masters", "doctorate"}
    top_occupation = {"technical", "executive", "admin"}

    high_earner = (
        (df["income"] > 60000) &
        (df["education"].isin(top_education)) &
        (df["occupation"].isin(top_occupation))
    ).astype(int)

    df[LABEL_COL] = high_earner

    return df


def save_dataset(df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save the synthetic dataset to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_dataset.csv"
    df.to_csv(path, index=False)
    print(f"Saved synthetic dataset: {path} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"  Label distribution: {df[LABEL_COL].value_counts().to_dict()}")
    print(f"  Positive rate: {df[LABEL_COL].mean():.2%}")
    return path


if __name__ == "__main__":
    df = generate_synthetic_dataset(n=200, random_state=42)
    save_dataset(df, Path(__file__).resolve().parent / "results" / "synthetic_demo")
    print("\nDataset preview:")
    print(df.head(10).to_string())
    print(f"\nFeature types:")
    for col in ALL_FEATURES:
        if col in NUMERIC_FEATURES:
            print(f"  {col}: numeric ({df[col].min():.0f} - {df[col].max():.0f})")
        else:
            print(f"  {col}: categorical ({df[col].nunique()} levels: {df[col].unique().tolist()})")
