#!/usr/bin/env python3
"""Evaluate the low-dynamic velocity CSVs over configured 20-second windows."""

from pathlib import Path

from evaluation_common import parse_evaluation_args, run_dataset_evaluation


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    ROOT
    / "data"
    / "Finetuned_Air-io_Results_CSV_10"
    / "T-Lab_10th_September_dataset"
    / "Low-Dynamic"
)
GT_ROOT = ROOT / "data" / "T-Lab_10th_September_dataset" / "Low-Dynamic"
RUN_NAME = "eval_outputs_10sep_finetuned_20s"

WINDOWS = {
    "AI": {
        "flight_1": ("ai_flight_1", 44.24, 64.24),
        "flight_2": ("ai_flight_2", 32.99, 52.99),
        "flight_3": ("ai_flight_3", 35.95, 55.95),
        "flight_4": ("ai_flight_4", 41.47, 61.47),
    },
    "RAW": {
        "flight_1": ("raw_13_16_49", 48.51, 68.51),
        "flight_2": ("raw_13_19_00", 33.29, 53.29),
        "flight_3": ("raw_13_20_51", 34.77, 54.77),
        "flight_4": ("raw_13_22_50", 39.63, 59.63),
        "flight_5": ("raw_13_25_41", 35.16, 55.16),
    },
}


def main() -> None:
    args = parse_evaluation_args(
        description="Calculate RMSE and drift for the low-dynamic flights.",
        results_root=RESULTS_ROOT,
        gt_root=GT_ROOT,
        run_name=RUN_NAME,
        modes=list(WINDOWS),
        default_mode="ALL",
    )
    run_dataset_evaluation(args, WINDOWS, dataset_tag="low")


if __name__ == "__main__":
    main()
