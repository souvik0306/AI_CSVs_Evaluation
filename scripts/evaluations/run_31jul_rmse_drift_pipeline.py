#!/usr/bin/env python3

import csv
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "scripts" / "pipeline"
RESULTS_ROOT = (
    ROOT
    / "results"
    / "Finetuned_Air-io_Results_CSV_28_gain_0"
    / "T-Lab_31st_July_dataset"
)
GT_ROOT = ROOT / "data" / "T-Lab_31st_July_dataset"
MODE = "UN"
RUN_NAME = "eval_outputs_31jul_rerun"
CLEAN_BOUND_ARGS = ["--bound", "vel_*:-8:8"]


AI_WINDOWS: Dict[str, Tuple[str, float, float]] = {
    "flight_1": ("ai_flight_1", 36.67, 45.67),
    "flight_2": ("ai_flight_2", 37.12, 46.12),
    "flight_3": ("ai_flight_3", 37.55, 46.55),
    "flight_4": ("ai_flight_4", 34.63, 43.63),
    "flight_5": ("ai_flight_5", 33.57, 42.57),
    "flight_6": ("ai_flight_6", 32.35, 41.35),
    "flight_7": ("ai_flight_7", 33.98, 42.98),
    "flight_8": ("ai_flight_8", 36.93, 45.93),
}


SUMMARY_FIELDS = [
    "mode",
    "flight",
    "label",
    "window_start_s",
    "window_end_s",
    "rows",
    "actual_duration_s",
    "rmse_x",
    "rmse_y",
    "rmse_z",
    "rmse_xy",
    "rmse_x_bias_corrected",
    "rmse_y_bias_corrected",
    "rmse_z_bias_corrected",
    "rmse_xy_bias_corrected",
    "drift_rate_mps2",
    "drift_intercept_mps",
    "out_dir",
]


def run(cmd: List[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def rmse(values: pd.Series) -> float:
    vals = values.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(vals ** 2)))


def rmse_xy(err_x: pd.Series, err_y: pd.Series) -> float:
    vals_x = err_x.to_numpy(dtype=float)
    vals_y = err_y.to_numpy(dtype=float)
    mask = np.isfinite(vals_x) & np.isfinite(vals_y)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(vals_x[mask] ** 2 + vals_y[mask] ** 2)))


def build_summary_row(
    flight: str,
    label: str,
    window_start: float,
    window_end: float,
    out_dir: Path,
) -> Dict[str, object]:
    bias_csv = out_dir / f"{flight}_velocity_clean_window_bias_errors.csv"
    if not bias_csv.exists():
        raise FileNotFoundError(bias_csv)

    df = pd.read_csv(bias_csv)
    rows = len(df)
    duration = float(df["time"].iloc[-1] - df["time"].iloc[0]) if rows else float("nan")
    t = df["time"].to_numpy(dtype=float)
    t_rel = t - t[0] if rows else np.array([])
    xy_err = np.sqrt(
        df["err_x"].to_numpy(dtype=float) ** 2
        + df["err_y"].to_numpy(dtype=float) ** 2
    )
    if rows >= 2:
        drift_rate, drift_intercept = np.polyfit(t_rel, xy_err, 1)
    else:
        drift_rate, drift_intercept = float("nan"), float("nan")

    return {
        "mode": MODE,
        "flight": flight,
        "label": label,
        "window_start_s": window_start,
        "window_end_s": window_end,
        "rows": rows,
        "actual_duration_s": duration,
        "rmse_x": rmse(df["err_x"]),
        "rmse_y": rmse(df["err_y"]),
        "rmse_z": rmse(df["err_z"]),
        "rmse_xy": rmse_xy(df["err_x"], df["err_y"]),
        "rmse_x_bias_corrected": rmse(df["err_x_bias"]),
        "rmse_y_bias_corrected": rmse(df["err_y_bias"]),
        "rmse_z_bias_corrected": rmse(df["err_z_bias"]),
        "rmse_xy_bias_corrected": rmse_xy(df["err_x_bias"], df["err_y_bias"]),
        "drift_rate_mps2": float(drift_rate),
        "drift_intercept_mps": float(drift_intercept),
        "out_dir": str(out_dir),
    }


def write_summary(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RMSE and drift evaluation for the 31 July UN flights."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory containing the UN folder with flight_*_velocity.csv files.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=GT_ROOT,
        help="Directory containing the UN/flight_*/data.csv ground-truth files.",
    )
    parser.add_argument(
        "--run-name",
        default=RUN_NAME,
        help="Output folder name to create inside the UN results folder.",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=None,
        help="Override each window end as window_start + this duration in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    gt_root = args.gt_root.resolve()

    source_dir = results_root / MODE
    gt_dir = gt_root / MODE
    output_root = source_dir / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)

    windows = AI_WINDOWS
    if args.window_duration is not None:
        windows = {
            flight: (label, window_start, window_start + args.window_duration)
            for flight, (label, window_start, _window_end) in AI_WINDOWS.items()
        }

    rows: List[Dict[str, object]] = []
    for flight, (label, window_start, window_end) in windows.items():
        out_dir = output_root / flight
        out_dir.mkdir(parents=True, exist_ok=True)

        est_csv = source_dir / f"{flight}_velocity.csv"
        gt_csv = gt_dir / flight / "data.csv"
        if not est_csv.exists():
            raise FileNotFoundError(est_csv)
        if not gt_csv.exists():
            raise FileNotFoundError(gt_csv)

        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "3_vel_csv_dataset_cleaner.py"),
                "--est_csv",
                str(est_csv),
                "--gt_csv",
                str(gt_csv),
                "--out_dir",
                str(out_dir),
                *CLEAN_BOUND_ARGS,
            ],
            cwd=out_dir,
        )

        clean_est = out_dir / f"{flight}_velocity_clean.csv"
        clean_gt = out_dir / "data_clean.csv"
        aligned_gt = out_dir / "data_clean_aligned.csv"

        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "4_align_vel_csv.py"),
                "--est_csv",
                str(clean_est),
                "--gt_csv",
                str(clean_gt),
                "--out",
                str(aligned_gt),
                "--gt_latency",
                "0.025",
            ],
            cwd=out_dir,
        )
        flight_time_zero = float(pd.read_csv(aligned_gt, usecols=["time"]).iloc[0, 0])

        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "5_manual_relative_window_clipping.py"),
                "--est_csv",
                str(clean_est),
                "--gt_csv",
                str(aligned_gt),
                "--window_start",
                str(window_start),
                "--window_end",
                str(window_end),
                "--output_suffix",
                "_window",
            ],
            cwd=out_dir,
        )

        est_window = out_dir / f"{flight}_velocity_clean_window.csv"
        gt_window = out_dir / "data_clean_aligned_window.csv"
        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "6_bias_zeroing_and_rmse.py"),
                "--est_csv",
                str(est_window),
                "--gt_csv",
                str(gt_window),
            ],
            cwd=out_dir,
        )

        bias_csv = out_dir / f"{flight}_velocity_clean_window_bias_errors.csv"
        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "7_plot_absolute_velocity_error.py"),
                "--bias_errors_csv",
                str(bias_csv),
                "--est_csv",
                str(est_window),
                "--gt_csv",
                str(gt_window),
                "--flight_time_zero",
                str(flight_time_zero),
                "--out_dir",
                str(out_dir / "plots_drift_rate"),
            ],
            cwd=out_dir,
        )

        rows.append(build_summary_row(flight, label, window_start, window_end, out_dir))

    summary_path = output_root / "un_31jul_bias_rmse_drift_summary.csv"
    write_summary(summary_path, rows)
    print(f"Saved UN summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
