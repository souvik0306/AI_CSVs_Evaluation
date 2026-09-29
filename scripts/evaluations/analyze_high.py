#!/usr/bin/env python3
"""Evaluate the high-dynamic velocity CSVs over configured 20-second windows."""

from pathlib import Path

from evaluation_common import parse_evaluation_args, run_dataset_evaluation


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    ROOT
    / "data"
    / "Finetuned_Air-io_Results_CSV_10"
    / "T-Lab_10th_September_dataset"
    / "High-Dynamic"
)
GT_ROOT = ROOT / "data" / "T-Lab_10th_September_dataset" / "High-Dynamic"
RUN_NAME = "eval_outputs_10sep_finetuned_20s"

WINDOWS = {
    "AI": {
        "flight_1": ("ai_12_23_36", 41.14, 61.14),
        "flight_2": ("ai_12_27_17", 40.24, 60.24),
        "flight_3": ("ai_12_30_45", 37.50, 57.50),
        "flight_4": ("ai_12_34_16", 38.23, 58.23),
    },
    "RAW": {
        "flight_1": ("raw_13_45_38", 36.19, 56.19),
        "flight_2": ("raw_13_47_24", 37.10, 57.10),
        "flight_3": ("raw_13_49_14", 37.42, 57.42),
        "flight_4": ("raw_13_51_01", 31.98, 51.98),
    },
}


def main() -> None:
    args = parse_evaluation_args(
        description="Calculate RMSE and drift for the high-dynamic flights.",
        results_root=RESULTS_ROOT,
        gt_root=GT_ROOT,
        run_name=RUN_NAME,
        modes=list(WINDOWS),
        default_mode="ALL",
    )
    run_dataset_evaluation(args, WINDOWS, dataset_tag="high")


if __name__ == "__main__":
    main()
