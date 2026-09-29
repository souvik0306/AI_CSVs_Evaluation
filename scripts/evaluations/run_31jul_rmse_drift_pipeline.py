#!/usr/bin/env python3
"""Evaluate pre-cleaned 31 July velocity CSVs."""

from pathlib import Path

from evaluation_common import parse_evaluation_args, run_dataset_evaluation


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    ROOT
    / "results"
    / "Finetuned_Air-io_Results_Plots_28_weighted_gain_2"
    / "results_csv"
    / "T-Lab_31st_July_dataset"
)
GT_ROOT = ROOT / "data" / "T-Lab_31st_July_dataset"
RUN_NAME = "eval_outputs_31jul_direct"

AI_WINDOWS = {
    "flight_1": ("ai_flight_1", 36.67, 45.67),
    "flight_2": ("ai_flight_2", 37.12, 46.12),
    "flight_3": ("ai_flight_3", 37.55, 46.55),
    "flight_4": ("ai_flight_4", 34.63, 43.63),
    "flight_5": ("ai_flight_5", 33.57, 42.57),
    "flight_6": ("ai_flight_6", 32.35, 41.35),
    "flight_7": ("ai_flight_7", 33.98, 42.98),
    "flight_8": ("ai_flight_8", 36.93, 45.93),
}

WINDOWS_BY_MODE = {"UN": AI_WINDOWS}


def main() -> None:
    args = parse_evaluation_args(
        description="Calculate RMSE and drift for the 31 July UN flights.",
        results_root=RESULTS_ROOT,
        gt_root=GT_ROOT,
        run_name=RUN_NAME,
        modes=list(WINDOWS_BY_MODE),
    )
    run_dataset_evaluation(args, WINDOWS_BY_MODE, dataset_tag="31jul")


if __name__ == "__main__":
    main()
