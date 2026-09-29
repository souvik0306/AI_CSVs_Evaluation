#!/usr/bin/env python3
"""Evaluate the medium-dynamic velocity CSVs over configured 20-second windows."""

from pathlib import Path

from evaluation_common import parse_evaluation_args, run_dataset_evaluation


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    ROOT
    / "data"
    / "Finetuned_Air-io_Results_CSV_10"
    / "T-Lab_10th_September_dataset"
    / "Medium-Dynamic"
)
GT_ROOT = ROOT / "data" / "T-Lab_10th_September_dataset" / "Medium-Dynamic"
RUN_NAME = "eval_outputs_10sep_finetuned_20s"

WINDOWS = {
    "AI": {
        "flight_1": ("ai_12_09_03", 34.43, 54.43),
        "flight_2": ("ai_12_11_43", 41.62, 61.62),
        "flight_3": ("ai_12_14_18", 35.62, 55.62),
    },
    "RAW": {
        "flight_1": ("raw_13_33_34", 38.48, 58.48),
        "flight_2": ("raw_13_36_16", 34.54, 54.54),
        "flight_3": ("raw_13_38_39", 36.91, 56.91),
    },
}


def main() -> None:
    args = parse_evaluation_args(
        description="Calculate RMSE and drift for the medium-dynamic flights.",
        results_root=RESULTS_ROOT,
        gt_root=GT_ROOT,
        run_name=RUN_NAME,
        modes=list(WINDOWS),
        default_mode="ALL",
    )
    run_dataset_evaluation(args, WINDOWS, dataset_tag="medium")


if __name__ == "__main__":
    main()
