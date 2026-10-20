# Why Results Improved On Adult And Why They Stayed Mixed On Covertype

This document focuses on the reason behind the different dataset outcomes.

## Adult Income

- Adult has many categorical feature groups, so zero masking is especially damaging there.
- Hidden categorical validity improves from 0.0000 to 1.0000.
- Hidden numeric exact-zero rate drops from 1.0000 to 0.0000.
- Mean nearest-train distance drops from 2.1893 to 1.3329.
- This means the masking improvement itself is clearly visible on Adult.

## Covertype

- Covertype accuracy stays lower for `empirical_background` than for `zero_mask`: 0.6774 vs 0.6843.
- Covertype explanation MAE also stays worse for `empirical_background`: 0.3795 vs 0.3591.
- Covertype does improve slightly on Spearman rank alignment: 0.5835 vs 0.5650.
- Covertype coalition MSE also improves slightly: 0.2016 vs 0.2021.
- Covertype still has a harder global modeling problem, richer structure, and a tougher optimization target under empirical_background masking.
- That is why the current saved end-to-end Covertype metrics remain mixed.

## Best interpretation

- Adult is the better showcase dataset for proving the masking improvement itself.
- Covertype is the honest benchmark showing that better masking does not automatically solve the whole pipeline.
