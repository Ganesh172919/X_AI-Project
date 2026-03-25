# File Reference: `data/loaders.py`

## Purpose

This file defines dataset loading and dataset metadata.

## Main Objects

- `DatasetMetadata`
- `DatasetBundle`
- `load_bike_sharing()`
- `load_covertype()`
- `load_adult_income()`
- `load_dataset()`

## What It Does

- Fetches UCI datasets with `fetch_ucirepo`
- Converts raw data into project-specific feature sets
- Attaches metadata needed by downstream modules
- Defines paper comparison values and interaction pairs

## Special Logic

- Bike: converts date fields and categorical fields
- Covertype: compresses the raw 54-feature representation into 10 numeric + 1 grouped soil climate feature
- Adult: drops `education-num` for a 13-feature representation

## Extension Points

- Add new datasets
- Change feature selection
- Add metadata for new experiments

