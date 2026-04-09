# Dataset Strategy

This document answers which dataset should be used to show the limitation clearly and how to choose future datasets.

## Best Current Choice

- Covertype is the strongest in-repo dataset for the current Phase 3 limitation.
- It combines scaled numerics with a grouped soil_climate_zone categorical feature.
- It gives an easy story for why unrealistic masked states matter.
- It already has runnable code, saved metrics, plots, and reports in Phase 3.

## Ranking Current Datasets

- Rank 1: Covertype for masking realism and grouped categorical validity.
- Rank 2: Adult Income for future mixed-categorical validation.
- Rank 3: Bike Sharing, which is better for interaction stories than this exact masking story.

## What Makes A Good Dataset

- Mixed numeric and categorical structure after preprocessing.
- Dependence between visible and hidden features.
- Enough rows for surrogate training.
- A story that reviewers can understand quickly.
- A place where invalid one-hot or unrealistic standardized states are easy to demonstrate.

## Good Future Datasets

- Adult Income for a next in-repo extension.
- Bank Marketing or German Credit for categorical-heavy tabular evaluation.
- Telco Churn for business-friendly categorical explanations.
- Census or ACS income variants for larger structured tabular evaluation.

## Best Viva Answer

- Use Covertype right now because it directly matches the current runnable Phase 3 branch.
- Use Adult Income or another mixed-feature credit/churn dataset next if the goal is to strengthen the invalid-category argument further.
