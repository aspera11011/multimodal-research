# SGNet RGB-D-D reliability-gate results

This directory contains aggregate, reviewable results only. Checkpoints, complete logs, datasets, and server-specific paths are intentionally excluded.

## Result files

- `summary.json`: original 200-sample spatial-gate pilot and control comparisons.
- `confirmatory_multiseed.json`: three-seed full-NYU evaluation on clean, translation, scale, and synthetic texture conditions.
- `adaptive_frequency_multiseed.json`: fixed-threshold adaptive full/high-frequency gate evaluation across three seeds.
- `rgbdd_real_protocol_summary.json`: RGB-D-D real-input 4× baseline, direct-transfer failure, and 200-sample real-domain calibration failure.

## Interpretation boundary

The adaptive frequency gate is retained for the cross-dataset synthetic 16× protocol. Every seed improves every mean metric in the seven selected clean, translation, scale, and texture conditions; 102 of 105 paired confidence intervals are strictly below zero.

The same claim does not extend to real sensor input. Direct transfer to `SGNet_Real_R` and a 200-sample real-domain calibration both perform significantly worse than the real-input baseline on all five reported metrics. These experiments are retained as No-Go evidence.
