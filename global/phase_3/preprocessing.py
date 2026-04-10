"""
Step 2: Preprocessing Pipeline with Feature Group Tracking.

Applies StandardScaler to numeric columns and OneHotEncoder to categorical columns.
Tracks which transformed columns map to each original feature group — this map
is required by both masking strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass
class FeatureGroup:
    """Mapping from one original feature to its transformed column range."""
    name: str
    kind: str          # 'numeric' or 'categorical'
    start: int
    end: int
    categories: list = field(default_factory=list)

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.end))

    @property
    def width(self) -> int:
        return self.end - self.start


class SyntheticPreprocessor:
    """
    Preprocessing pipeline with original-feature group bookkeeping.

    After fit_transform, provides:
        - feature_groups: dict mapping feature name -> FeatureGroup
        - feature_order: list of original feature names in processing order
        - transformed_feature_names: list of column names after transformation
        - input_dim: number of transformed columns
    """

    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.feature_order = self.numeric_features + self.categorical_features
        self.feature_groups: dict[str, FeatureGroup] = {}
        self.transformed_feature_names: list[str] = []
        self._pipeline: Optional[ColumnTransformer] = None

    def _build_pipeline(self) -> ColumnTransformer:
        numeric_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ])
        return ColumnTransformer([
            ('num', numeric_pipe, self.numeric_features),
            ('cat', categorical_pipe, self.categorical_features),
        ], sparse_threshold=0.0)

    def fit(self, frame: pd.DataFrame) -> 'SyntheticPreprocessor':
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(frame)
        self._build_feature_groups()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        result = self._pipeline.transform(frame)
        if hasattr(result, 'toarray'):
            result = result.toarray()
        return np.asarray(result, dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _build_feature_groups(self) -> None:
        self.feature_groups = {}
        self.transformed_feature_names = []
        cursor = 0

        for feat in self.numeric_features:
            self.feature_groups[feat] = FeatureGroup(
                name=feat, kind='numeric', start=cursor, end=cursor + 1
            )
            self.transformed_feature_names.append(feat)
            cursor += 1

        encoder = self._pipeline.named_transformers_['cat'].named_steps['onehot']
        for feat, cats in zip(self.categorical_features, encoder.categories_):
            cat_list = [str(c) for c in cats]
            width = len(cat_list)
            self.feature_groups[feat] = FeatureGroup(
                name=feat, kind='categorical', start=cursor, end=cursor + width,
                categories=cat_list
            )
            self.transformed_feature_names.extend([f'{feat}={c}' for c in cat_list])
            cursor += width

    @property
    def input_dim(self) -> int:
        return len(self.transformed_feature_names)

    @property
    def num_original_features(self) -> int:
        return len(self.feature_order)

    def feature_index(self, name: str) -> int:
        return self.feature_order.index(name)

    def group(self, name: str) -> FeatureGroup:
        return self.feature_groups[name]

    def expand_feature_mask(self, feature_mask: np.ndarray) -> np.ndarray:
        """
        Expand [batch, num_original_features] mask to [batch, input_dim] mask.
        Each original feature's mask value is repeated across its column group.
        """
        parts = []
        for feat in self.feature_order:
            grp = self.feature_groups[feat]
            idx = self.feature_index(feat)
            parts.append(np.repeat(feature_mask[:, [idx]], grp.width, axis=1))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def categorical_group_names(self) -> list[str]:
        """Return names of categorical features."""
        return [f for f in self.feature_order
                if self.feature_groups[f].kind == 'categorical']

    def numeric_group_names(self) -> list[str]:
        """Return names of numeric features."""
        return [f for f in self.feature_order
                if self.feature_groups[f].kind == 'numeric']

    def summary(self) -> str:
        """Print a summary of feature groups."""
        lines = [f"SyntheticPreprocessor: {self.num_original_features} features -> {self.input_dim} columns"]
        for name, grp in self.feature_groups.items():
            if grp.kind == 'categorical':
                lines.append(f"  {name:>20s}: cols [{grp.start}:{grp.end}] (width={grp.width}, kind={grp.kind}, cats={grp.categories})")
            else:
                lines.append(f"  {name:>20s}: cols [{grp.start}:{grp.end}] (width={grp.width}, kind={grp.kind})")
        return "\n".join(lines)
