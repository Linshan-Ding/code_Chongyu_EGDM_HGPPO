# Pilot Results

This directory contains the compact, reviewable output of the EGDM-HGPPO pilot experiment.

## Contents

`_reports/run01/` is the report bundle generated from the `pilot-joint` run.

- `summary.csv`: method-level aggregate metrics by split and scale.
- `main-results.csv`: per-instance results for the main comparison.
- `ablation-results.csv`: full model and ablation results.
- `comparisons.csv`: pairwise wins, losses, win rates, and mean improvements.
- `all-results.csv`: combined result table used to build the summaries.
- `training-tail.csv`: late-training diagnostic statistics.
- `report.md`: short human-readable report.
- `report.json`: machine-readable report with protocol and acceptance fields.
- `claims.json`: machine-readable claim and evidence status.
- `gate-behavior.csv`, `effects-load.csv`, `effects-migration.csv`: supporting diagnostic tables.
- `fallback-status.csv`, `replay-checks.csv`, `job-status.csv`: execution and reproducibility checks.
- `reference-status.json`, `appendix-strong-reference.csv`, `generalization-L.csv`: reference and supplementary checks.

The raw training checkpoints, event traces, per-epoch artifacts, and large figures are intentionally omitted from this compact GitHub bundle. They remain available in the local experiment directory and can be shared separately when needed.

## Interpretation

This is a pilot/pre-experiment result bundle for advisor review. `mean_twt` is the mean total weighted tardiness; lower values are better. The tables should be interpreted together with the protocol and acceptance fields in `report.json`.
