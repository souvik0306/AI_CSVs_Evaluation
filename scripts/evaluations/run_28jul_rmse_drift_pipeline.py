#!/usr/bin/env python3

import csv
import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "scripts" / "pipeline"
RESULTS_ROOT = (
    ROOT
    / "results"
    / "Finetuned_Air-io_Results_CSV_28_gain_0"
    / "T-Lab_28th_July_dataset"
)
GT_ROOT = ROOT / "data" / "T-Lab_28th_July_dataset"
RUN_NAME = "eval_outputs_28jul_rerun"
CLEAN_BOUND_ARGS = ["--bound", "vel_*:-4:4"]


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
    print("Running:", " ".join(cmd))
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
    mode: str,
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
        "mode": mode,
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


def with_duration(windows: Dict[str, tuple], duration: Optional[float]) -> Dict[str, tuple]:
    if duration is None:
        return windows
    return {
        flight: (label, window_start, window_start + duration)
        for flight, (label, window_start, _window_end) in windows.items()
    }


def evaluate_group(
    mode: str,
    windows: Dict[str, tuple],
    results_root: Path,
    gt_root: Path,
    run_name: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    source_dir = results_root / mode
    gt_dir = gt_root / mode
    output_root = source_dir / run_name
    output_root.mkdir(parents=True, exist_ok=True)

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

        rows.append(
            build_summary_row(mode, flight, label, window_start, window_end, out_dir)
        )

    summary_path = output_root / f"{mode.lower()}_28jul_bias_rmse_drift_summary.csv"
    write_summary(summary_path, rows)
    print(f"Saved {mode} summary: {summary_path}")
    return rows


def write_summary(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_overlays(
    ai_rows: List[Dict[str, object]],
    raw_rows: List[Dict[str, object]],
    results_root: Path,
) -> None:
    raw_by_flight = {row["flight"]: row for row in raw_rows}
    overlay_dir = results_root / "overlay_drift_rate_28jul_rerun"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for ai_row in ai_rows:
        flight = str(ai_row["flight"])
        raw_row = raw_by_flight.get(flight)
        if raw_row is None:
            continue
        ai_csv = Path(str(ai_row["out_dir"])) / f"{flight}_velocity_clean_window_bias_errors.csv"
        raw_csv = Path(str(raw_row["out_dir"])) / f"{flight}_velocity_clean_window_bias_errors.csv"
        out_png = overlay_dir / f"{flight}_ai_vs_raw_drift_rate.png"
        run(
            [
                sys.executable,
                str(PIPELINE_DIR / "8_overlay_drift_rate_ai_vs_raw.py"),
                "--ai_csv",
                str(ai_csv),
                "--raw_csv",
                str(raw_csv),
                "--out_png",
                str(out_png),
            ],
            cwd=ROOT,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RMSE and drift evaluation for the 28 July flights."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Directory containing the UN/RAW result folders with flight_*_velocity.csv files.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=GT_ROOT,
        help="Directory containing the UN/RAW/flight_*/data.csv ground-truth files.",
    )
    parser.add_argument(
        "--run-name",
        default=RUN_NAME,
        help="Output folder name to create inside each evaluated mode folder.",
    )
    parser.add_argument(
        "--mode",
        choices=["UN", "RAW", "ALL"],
        default="UN",
        help="Which 28 July mode to evaluate (default: UN, matching the bundled finetuned results).",
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
    modes = ["UN", "RAW"] if args.mode == "ALL" else [args.mode]

    all_rows: List[Dict[str, object]] = []
    ai_rows: List[Dict[str, object]] = []
    raw_rows: List[Dict[str, object]] = []

    if "UN" in modes:
        ai_rows = evaluate_group(
            "UN",
            with_duration(AI_WINDOWS, args.window_duration),
            results_root,
            gt_root,
            args.run_name,
        )
        all_rows.extend(ai_rows)

    if "RAW" in modes:
        raw_rows = evaluate_group(
            "RAW",
            with_duration(RAW_WINDOWS, args.window_duration),
            results_root,
            gt_root,
            args.run_name,
        )
        all_rows.extend(raw_rows)

    if args.mode == "ALL":
        combined_path = results_root / args.run_name / "combined_28jul_bias_rmse_drift_summary.csv"
        write_summary(combined_path, all_rows)
        print(f"Saved combined summary: {combined_path}")

        write_overlays(ai_rows, raw_rows, results_root)


if __name__ == "__main__":
    main()
