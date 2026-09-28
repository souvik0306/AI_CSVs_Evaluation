# CSV velocity evaluation

This workspace evaluates velocity estimates that already exist as CSV files. It
uses the same processing stages as `AI_Velocity_Evaluation`, but skips ROS bag
extraction when estimate and ground-truth CSVs are already available.

## Layout

- `data/`: ground-truth and source CSV datasets.
- `results/`: model predictions, evaluation outputs, plots, and summaries.
- `scripts/pipeline/`: numbered processing and plotting stages.
- `scripts/evaluations/`: repeatable dataset-level evaluation drivers.
- `scripts/utilities/`: notebooks and one-off inspection tools.
- `archive/legacy_results/`: older runs retained for reference.
- `archive/compressed/`: original ZIP bundles retained without cluttering the root.
- `archive/generated/`: generated cache files that were present before cleanup.

## Typical usage

Run commands from the workspace root:

```bash
python3 scripts/evaluations/run_28jul_rmse_drift_pipeline.py --help
python3 scripts/evaluations/run_31jul_rmse_drift_pipeline.py --help
```

Both evaluation drivers resolve their defaults relative to this project, so
they do not depend on the current working directory. Individual pipeline stages
accept explicit CSV paths and can be run independently.

Historical artifacts were moved, not deleted. Old dated outputs are under
`archive/legacy_results`, and the top-level ZIP files are under
`archive/compressed`.
