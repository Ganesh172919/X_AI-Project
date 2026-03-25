"""Dataset loaders aligned with the InstaSHAP paper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

ADULT_UCI_ID = 2
COVERTYPE_UCI_ID = 31
BIKE_UCI_ID = 275

SOIL_ELU_CODES = [
    2702,
    2703,
    2704,
    2705,
    2706,
    2717,
    3501,
    3502,
    4201,
    4703,
    4704,
    4744,
    4758,
    5101,
    5151,
    6101,
    6102,
    6731,
    7101,
    7102,
    7103,
    7201,
    7202,
    7700,
    7701,
    7702,
    7709,
    7710,
    7745,
    7746,
    7755,
    7756,
    7757,
    7790,
    8703,
    8707,
    8708,
    8771,
    8772,
    8776,
]

SOIL_CLIMATE_LABELS = {
    1: "lower montane",
    2: "lower montane",
    3: "lower montane",
    4: "upper montane",
    5: "upper montane",
    6: "subalpine",
    7: "subalpine",
    8: "alpine",
}
SOIL_CLIMATE_ORDER = ["lower montane", "upper montane", "subalpine", "alpine"]


@dataclass(slots=True)
class DatasetMetadata:
    """Metadata that travels with every dataset bundle."""

    name: str
    task: str
    target_name: str
    numeric_features: list[str]
    categorical_features: list[str]
    interaction_pairs: list[tuple[str, str]] = field(default_factory=list)
    paper_metrics: dict[str, float] = field(default_factory=dict)
    description: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


@dataclass(slots=True)
class DatasetBundle:
    """Loaded dataset plus metadata."""

    features: pd.DataFrame
    target: pd.Series
    metadata: DatasetMetadata

    def sample(self, max_rows: int | None, seed: int) -> "DatasetBundle":
        if max_rows is None or len(self.features) <= max_rows:
            return self

        stratify = self.target if self.metadata.task == "classification" else None
        sampled_idx, _ = train_test_split(
            np.arange(len(self.features)),
            train_size=max_rows,
            stratify=stratify,
            random_state=seed,
        )
        sampled_features = self.features.iloc[sampled_idx].reset_index(drop=True)
        sampled_target = self.target.iloc[sampled_idx].reset_index(drop=True)
        return DatasetBundle(sampled_features, sampled_target, self.metadata)

def load_bike_sharing() -> DatasetBundle:
    """Load the Bike Sharing dataset via ucimlrepo and apply paper-aligned preprocessing."""

    dataset = fetch_ucirepo(id=BIKE_UCI_ID)
    df = dataset.data.features.copy()
    target_df = dataset.data.targets.copy()

    date_series = pd.to_datetime(df["dteday"], errors="coerce")
    features = pd.DataFrame(
        {
            "day_of_month": date_series.dt.day.astype("Int64"),
            "temp": df["temp"].astype(float),
            "atemp": df["atemp"].astype(float),
            "hum": df["hum"].astype(float),
            "windspeed": df["windspeed"].astype(float),
            "season": df["season"].astype(str),
            "year": df["yr"].map({0: "2011", 1: "2012"}).astype(str),
            "month": df["mnth"].astype(int).astype(str),
            "hour": df["hr"].astype(int).astype(str),
            "holiday": df["holiday"].map({0: "no", 1: "yes"}).astype(str),
            "weekday": df["weekday"].astype(int).astype(str),
            "workingday": df["workingday"].map({0: "no", 1: "yes"}).astype(str),
            "weather_situation": df["weathersit"].astype(int).astype(str),
        }
    )
    target = target_df.iloc[:, 0].astype(float).rename("count")

    for col in ["season", "year", "month", "hour", "holiday", "weekday", "workingday", "weather_situation"]:
        features[col] = features[col].astype("category")

    metadata = DatasetMetadata(
        name="bike",
        task="regression",
        target_name="count",
        numeric_features=["day_of_month", "temp", "atemp", "hum", "windspeed"],
        categorical_features=[
            "season",
            "year",
            "month",
            "hour",
            "holiday",
            "weekday",
            "workingday",
            "weather_situation",
        ],
        interaction_pairs=[("hour", "workingday")],
        paper_metrics={
            "paper_blackbox_nmse_pct": 6.59,
            "paper_gam1_nmse_pct": 17.4,
            "paper_low_dim_gam_nmse_pct": 6.23,
        },
        description="Hourly bike demand regression with a known hour x workday synergistic interaction.",
        notes={
            "paper_total_features": 13,
            "uci_id": BIKE_UCI_ID,
            "repository_url": dataset.metadata.get("repository_url"),
            "raw_feature_count": int(dataset.data.features.shape[1]),
            "interaction_comment": "The paper highlights a strong interaction between hour and workingday.",
        },
    )
    return DatasetBundle(features.reset_index(drop=True), target.reset_index(drop=True), metadata)


def _soil_type_to_climate_label(soil_type_code: int) -> str:
    elu_code = SOIL_ELU_CODES[soil_type_code - 1]
    climate_digit = int(str(elu_code)[0])
    return SOIL_CLIMATE_LABELS[climate_digit]


def load_covertype(max_rows: int | None = None, seed: int = 42) -> DatasetBundle:
    """Load Covertype via ucimlrepo with a compact 10 numerical + 1 grouped soil categorical view."""

    dataset = fetch_ucirepo(id=COVERTYPE_UCI_ID)
    data = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].astype(int) - 1

    soil_cols = [col for col in data.columns if col.startswith("Soil_Type")]
    soil_type_code = data[soil_cols].to_numpy(dtype=np.int64).argmax(axis=1) + 1
    soil_climate_zone = pd.Series(soil_type_code).map(_soil_type_to_climate_label)

    features = pd.DataFrame(
        {
            "elevation": data["Elevation"].astype(float),
            "aspect": data["Aspect"].astype(float),
            "slope": data["Slope"].astype(float),
            "horizontal_distance_to_hydrology": data["Horizontal_Distance_To_Hydrology"].astype(float),
            "vertical_distance_to_hydrology": data["Vertical_Distance_To_Hydrology"].astype(float),
            "horizontal_distance_to_roadways": data["Horizontal_Distance_To_Roadways"].astype(float),
            "hillshade_9am": data["Hillshade_9am"].astype(float),
            "hillshade_noon": data["Hillshade_Noon"].astype(float),
            "hillshade_3pm": data["Hillshade_3pm"].astype(float),
            "horizontal_distance_to_fire_points": data["Horizontal_Distance_To_Fire_Points"].astype(float),
            "soil_climate_zone": pd.Categorical(
                soil_climate_zone,
                categories=SOIL_CLIMATE_ORDER,
                ordered=True,
            ),
        }
    )

    metadata = DatasetMetadata(
        name="covertype",
        task="classification",
        target_name="cover_type",
        numeric_features=[
            "elevation",
            "aspect",
            "slope",
            "horizontal_distance_to_hydrology",
            "vertical_distance_to_hydrology",
            "horizontal_distance_to_roadways",
            "hillshade_9am",
            "hillshade_noon",
            "hillshade_3pm",
            "horizontal_distance_to_fire_points",
        ],
        categorical_features=["soil_climate_zone"],
        interaction_pairs=[("elevation", "soil_climate_zone")],
        paper_metrics={
            "paper_blackbox_accuracy": 0.804,
            "paper_gam1_accuracy": 0.724,
            "paper_low_dim_gam_accuracy": 0.822,
        },
        description="Forest cover classification emphasizing the redundant elevation x soil interaction.",
        notes={
            "paper_total_features": 11,
            "uci_id": COVERTYPE_UCI_ID,
            "repository_url": dataset.metadata.get("repository_url"),
            "raw_feature_count": int(dataset.data.features.shape[1]),
            "soil_grouping": "Grouped from the first ELU climate digit in the UCI covtype.info metadata.",
        },
    )
    bundle = DatasetBundle(features.reset_index(drop=True), target.reset_index(drop=True), metadata)
    return bundle.sample(max_rows=max_rows, seed=seed)


def load_adult_income(max_rows: int | None = None, seed: int = 42) -> DatasetBundle:
    """Load the Adult dataset via ucimlrepo."""

    dataset = fetch_ucirepo(id=ADULT_UCI_ID)
    df = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    df["income"] = targets.iloc[:, 0]
    df = df.rename(
        columns={
            "marital-status": "marital_status",
            "capital-gain": "capital_gain",
            "capital-loss": "capital_loss",
            "hours-per-week": "hours_per_week",
            "native-country": "native_country",
        }
    )
    df = df.drop(columns=["education-num"])
    target = (df.pop("income") == ">50K").astype(int).rename("income_above_50k")

    features = df.copy()
    numeric_features = ["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]
    categorical_features = [col for col in features.columns if col not in numeric_features]

    for col in categorical_features:
        features[col] = features[col].astype("category")

    metadata = DatasetMetadata(
        name="adult",
        task="classification",
        target_name="income_above_50k",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        interaction_pairs=[],
        paper_metrics={
            "paper_vanilla_gam_accuracy": 0.842,
            "paper_instashap_gam_accuracy": 0.843,
        },
        description="Supplementary Adult Income classification benchmark used for 1D additive stability.",
        notes={
            "paper_total_features": 13,
            "uci_id": ADULT_UCI_ID,
            "repository_url": dataset.metadata.get("repository_url"),
            "raw_feature_count": int(dataset.data.features.shape[1]),
            "dropped_feature": "education-num",
        },
    )
    bundle = DatasetBundle(features.reset_index(drop=True), target.reset_index(drop=True), metadata)
    return bundle.sample(max_rows=max_rows, seed=seed)


def load_dataset(name: str, *, max_rows: int | None = None, seed: int = 42) -> DatasetBundle:
    """Dispatch dataset loading by canonical name."""

    normalized = name.strip().lower()
    if normalized in {"bike", "bike_sharing", "bikeshare"}:
        return load_bike_sharing()
    if normalized in {"covertype", "covtype", "treecover"}:
        return load_covertype(max_rows=max_rows, seed=seed)
    if normalized in {"adult", "adult_income", "income"}:
        return load_adult_income(max_rows=max_rows, seed=seed)
    raise ValueError(f"Unsupported dataset: {name}")
