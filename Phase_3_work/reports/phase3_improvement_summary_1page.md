# Phase 3 Improvement Summary

- Phase 3 identified unrealistic transformed-space zero masking as a real limitation.
- The current repo implements `empirical_background` masking as a targeted fix.
- Covertype remains the main end-to-end benchmark, but its results are still mixed.
- Adult is now the strongest showcase dataset for the masking improvement itself, with hidden categorical validity improving from 0.0000 to 1.0000.
- The best next step is to generalize the Phase 3 workflow to more datasets while keeping the reporting honest.
