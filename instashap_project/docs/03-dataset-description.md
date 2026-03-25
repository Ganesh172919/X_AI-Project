# Dataset Description

## Data Sources

All datasets are loaded with `ucimlrepo`:

- Bike Sharing: `fetch_ucirepo(id=275)`
- Covertype: `fetch_ucirepo(id=31)`
- Adult Income: `fetch_ucirepo(id=2)`

## Bike Sharing

- Task: regression
- Raw source: hourly bike demand from UCI Bike Sharing
- Raw feature count: 13
- Target: `cnt`

### Project-specific representation

The project uses:

- Numeric features:
  - `day_of_month`
  - `temp`
  - `atemp`
  - `hum`
  - `windspeed`
- Categorical features:
  - `season`
  - `year`
  - `month`
  - `hour`
  - `holiday`
  - `weekday`
  - `workingday`
  - `weather_situation`

### Important interaction

- `hour x workingday`

## Covertype

- Task: multiclass classification
- Raw source: UCI Forest Covertype
- Raw feature count: 54
- Target: `Cover_Type`

### Project-specific representation

To align with the paper's compact tabular discussion, the project compresses the raw 54-column representation into:

- 10 numeric terrain features
- 1 grouped categorical feature: `soil_climate_zone`

The grouped soil feature is derived from the UCI soil ELU codes and mapped into:

- `lower montane`
- `upper montane`
- `subalpine`
- `alpine`

### Important interaction

- `elevation x soil_climate_zone`

## Adult Income

- Task: binary classification
- Raw source: UCI Adult / Census Income
- Raw feature count: 14
- Target: `income`

### Project-specific representation

The project drops `education-num` to use a 13-feature representation closer to the paper's supplementary discussion.

- Numeric features:
  - `age`
  - `fnlwgt`
  - `capital_gain`
  - `capital_loss`
  - `hours_per_week`
- Categorical features:
  - `workclass`
  - `education`
  - `marital_status`
  - `occupation`
  - `relationship`
  - `race`
  - `sex`
  - `native_country`

## Preprocessing Summary

The preprocessing pipeline is shared across datasets:

- Numeric imputation with median
- Numeric standardization with `StandardScaler`
- Categorical imputation with most frequent value
- Categorical one-hot encoding with `OneHotEncoder(handle_unknown="ignore")`

## Splits

The project uses:

- Test split: `20%`
- Validation split: `10%` of the total dataset
- Train split: remaining `70%`

For classification tasks, splits are **stratified**. For regression tasks, they are not stratified.
