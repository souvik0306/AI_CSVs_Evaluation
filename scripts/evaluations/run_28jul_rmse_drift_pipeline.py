#!/usr/bin/env python3
"""Evaluate pre-cleaned 28 July velocity CSVs."""

from pathlib import Path

from evaluation_common import parse_evaluation_args, run_dataset_evaluation


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    ROOT
    / "results"
    / "Finetuned_Air-io_Results_Plots_28_weighted_gain_2"
    / "results_csv"
    / "T-Lab_28th_July_dataset"
)
GT_ROOT = ROOT / "data" / "T-Lab_28th_July_dataset"
RUN_NAME = "eval_outputs_28jul_direct"

AI_WINDOWS = {
    "flight_1": ("ai_10_20_35", 66.20, 75.20),
    "flight_2": ("ai_10_31_13", 59.65, 68.65),
    "flight_3": ("ai_10_34_48", 48.81, 57.81),
    "flight_4": ("ai_10_37_53", 45.39, 54.39),
    "flight_5": ("ai_10_40_49", 47.90, 56.90),
    "flight_6": ("ai_10_43_37", 41.65, 50.65),
    "flight_7": ("ai_10_46_31", 42.18, 51.18),
}

RAW_WINDOWS = {
    "flight_1": ("raw_10_18_50", 36.89, 45.89),
    "flight_2": ("raw_10_21_43", 38.09, 47.09),
    "flight_3": ("raw_10_24_22", 38.54, 47.54),
    "flight_4": ("raw_10_29_23", 42.35, 51.35),
    "flight_5": ("raw_10_32_09", 41.89, 50.89),
    "flight_6": ("raw_10_35_00", 37.46, 46.46),
    "flight_7": ("raw_10_37_51", 48.27, 57.27),
}

WINDOWS_BY_MODE = {"UN": AI_WINDOWS, "RAW": RAW_WINDOWS}


def main() -> None:
    args = parse_evaluation_args(
        description="Calculate RMSE and drift for the 28 July flights.",
        results_root=RESULTS_ROOT,
        gt_root=GT_ROOT,
        run_name=RUN_NAME,
        modes=list(WINDOWS_BY_MODE),
    )
    run_dataset_evaluation(args, WINDOWS_BY_MODE, dataset_tag="28jul")


if __name__ == "__main__":
    main()
