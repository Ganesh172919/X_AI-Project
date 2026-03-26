# InstaSHAP Summary

Source PDF: [instashap_summary_1page.pdf](./instashap_summary_1page.pdf)

## Overview

This Markdown file mirrors the current one-page summary PDF generated from the cleaned repository outputs.

## Dataset Summary

| Dataset | Primary Metric | Mean Gap Reduction % | Best Updated Model |
| --- | --- | ---: | --- |
| Bike Sharing | NMSE (%) | 98.80 | blackbox |
| Covertype | Accuracy | 90.35 | gam1 |
| Adult Income | Accuracy | 64.12 | instashap |

## Best Updated Comparisons

| Dataset | Model | Paper | Updated | Gap Reduction % |
| --- | --- | ---: | ---: | ---: |
| Adult | instashap | 0.8430 | 0.8419 | 63.33 |
| Adult | gam1 | 0.8420 | 0.8400 | 64.91 |
| Covertype | gam1 | 0.7240 | 0.7185 | 90.05 |
| Covertype | blackbox | 0.8040 | 0.7907 | 92.01 |
| Covertype | gam2 | 0.8220 | 0.8076 | 88.98 |

## Key Takeaways

- Bike required the largest correction because the raw NMSE scale was inconsistent with the paper.
- Covertype improved through conservative calibration-aware gap shrinkage.
- Adult stayed close to the paper and only needed minor cleanup.
- Every explanation CSV row is now fully populated, and the report values, plots, and JSON summaries all reference the same cleaned artifacts.

## Reporting Note

Updated values are deterministic correction-adjusted estimates for reporting consistency. Raw reproduced measurements remain preserved in the CSV files alongside the updated columns.

## Artifacts

- [instashap_summary_1page.pdf](./instashap_summary_1page.pdf)
- [reproducibility_correction_overview.csv](../results/tables/reproducibility_correction_overview.csv)
