"""Preprocessing utilities that preserve original feature groups."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data.loaders import DatasetBundle, DatasetMetadata


@dataclass(slots=True)
class FeatureGroup:
    """Mapping from one original feature to its transformed columns."""
    name: str
    kind: str
    start: int
    end: int
    categories: list[str] | None = None

    @property
    def indices(self) -> list[int]:
        return list(range(self.start, self.end))

    @property
    def width(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class SplitBundle:
    """Train/validation/test split container."""
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


class TabularPreprocessor:
    """Column transformer with original-feature group bookkeeping."""

    def __init__(self, metadata: DatasetMetadata) -> None:
        self.metadata = metadata
        self.numeric_features = list(metadata.numeric_features)
        self.categorical_features = list(metadata.categorical_features)
        self.feature_order = metadata.feature_names
        self.pipeline = self._build_pipeline()
        self.feature_groups: dict[str, FeatureGroup] = {}
        self.transformed_feature_names: list[str] = []
        self._template_values: dict[str, Any] = {}

    def _build_pipeline(self) -> ColumnTransformer:
        encoder_kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
        if "sparse_output" in signature(OneHotEncoder).parameters:
            encoder_kwargs["sparse_output"] = False
        else:
            encoder_kwargs["sparse"] = False

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(**encoder_kwargs)),
        ])
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_features),
                ("cat", categorical_pipe, self.categorical_features),
            ],
            sparse_threshold=0.0,
        )

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.pipeline.fit(frame)
        self._capture_template_values(frame)
        self._build_feature_groups()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self.pipeline.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed, dtype=np.float32)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def _capture_template_values(self, frame: pd.DataFrame) -> None:
        for feature in self.numeric_features:
            self._template_values[feature] = float(frame[feature].median())
        for feature in self.categorical_features:
            non_null = frame[feature].dropna()
            self._template_values[feature] = non_null.mode().iloc[0] if not non_null.empty else ""

    def _build_feature_groups(self) -> None:
        self.feature_groups = {}
        self.transformed_feature_names = []
        cursor = 0
        for feature in self.numeric_features:
            self.feature_groups[feature] = FeatureGroup(name=feature, kind="numeric", start=cursor, end=cursor + 1)
            self.transformed_feature_names.append(feature)
            cursor += 1
        encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        for feature, categories in zip(self.categorical_features, encoder.categories_):
            categories_list = [str(v) for v in categories]
            width = len(categories_list)
            self.feature_groups[feature] = FeatureGroup(
                name=feature, kind="categorical", start=cursor, end=cursor + width, categories=categories_list,
            )
            self.transformed_feature_names.extend([f"{feature}={v}" for v in categories_list])
            cursor += width

    @property
    def input_dim(self) -> int:
        return len(self.transformed_feature_names)

    @property
    def num_original_features(self) -> int:
        return len(self.feature_order)

    def feature_index(self, feature_name: str) -> int:
        return self.feature_order.index(feature_name)

    def group(self, feature_name: str) -> FeatureGroup:
        return self.feature_groups[feature_name]

    def slices_for(self, feature_names: tuple[str, ...] | list[str]) -> list[int]:
        indices: list[int] = []
        for fn in feature_names:
            indices.extend(self.feature_groups[fn].indices)
        return indices

    def expand_feature_mask(self, feature_mask: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []
        for fn in self.feature_order:
            g = self.feature_groups[fn]
            parts.append(np.repeat(feature_mask[:, [self.feature_index(fn)]], g.width, axis=1))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def make_reference_frame(self, num_rows: int) -> pd.DataFrame:
        rows = [{f: self._template_values[f] for f in self.feature_order} for _ in range(num_rows)]
        return pd.DataFrame(rows)

    def make_feature_frame(self, feature_name: str, values: list[Any] | np.ndarray) -> pd.DataFrame:
        frame = self.make_reference_frame(len(values))
        frame[feature_name] = values
        if feature_name in self.categorical_features:
            frame[feature_name] = pd.Categorical(values, categories=self.group(feature_name).categories, ordered=False)
        return frame

    def make_interaction_frame(self, f1: str, v1, f2: str, v2) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for a in v1:
            for b in v2:
                row = {f: self._template_values[f] for f in self.feature_order}
                row[f1] = a
                row[f2] = b
                rows.append(row)
        frame = pd.DataFrame(rows)
        if f1 in self.categorical_features:
            frame[f1] = pd.Categorical(frame[f1], categories=self.group(f1).categories)
        if f2 in self.categorical_features:
            frame[f2] = pd.Categorical(frame[f2], categories=self.group(f2).categories)
        return frame

    def build_background_bank(self, X_transformed: np.ndarray, bank_size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a fixed bank of transformed training rows for background masking."""
        n = min(bank_size, len(X_transformed))
        indices = rng.choice(len(X_transformed), size=n, replace=False)
        return X_transformed[indices].copy()


def make_splits(bundle: DatasetBundle, test_size: float, val_size: float, seed: int) -> SplitBundle:
    """Create train/validation/test splits with stratification."""
    stratify = bundle.target if bundle.metadata.task == "classification" else None
    X_tv, X_test, y_tv, y_test = train_test_split(
        bundle.features, bundle.target, test_size=test_size, random_state=seed, stratify=stratify,
    )
    relative_val = val_size / (1.0 - test_size)
    strat_tv = y_tv if bundle.metadata.task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=relative_val, random_state=seed, stratify=strat_tv,
    )
    return SplitBundle(
        X_train=X_train.reset_index(drop=True), X_val=X_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True), y_train=y_train.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True), y_test=y_test.reset_index(drop=True),
    )
