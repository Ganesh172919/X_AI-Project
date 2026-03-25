# Results and Artifacts

## Artifact Types

### Tables

- `*_metrics.csv`
- `*_paper_comparison.csv`
- `*_explanation_comparison.csv`

### Plots

- training curves
- metric bar charts
- feature importance summaries
- shape functions
- interaction heatmaps
- SHAP vs InstaSHAP alignment plots

### Summaries

- per-dataset JSON summary files
- full PDF report
- one-page PDF summary

## Artifact Locations

- `results/tables/`
- `results/plots/`
- `results/artifacts/`
- `reports/`

## Reading the JSON Summary

Each dataset summary includes:

- dataset name
- task type
- feature list
- interaction pairs
- saved table paths
- saved plot paths
- paper metadata

## Recommended Review Order

1. Check `*_metrics.csv`
2. Check `*_paper_comparison.csv`
3. Open shape function and interaction plots
4. Inspect `*_explanation_comparison.csv`
5. Read the generated PDF report

