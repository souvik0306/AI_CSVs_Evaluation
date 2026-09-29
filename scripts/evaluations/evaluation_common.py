#!/usr/bin/env python3
"""Shared evaluation for pre-cleaned, timestamp-aligned velocity CSVs."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_VELOCITY_COLUMNS = ["time", "vel_x", "vel_y", "vel_z"]

Window = Tuple[str, float, float]
Windows = Mapping[str, Window]

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


def parse_evaluation_args(
    description: str,
    results_root: Path,
    gt_root: Path,
    run_name: str,
    modes: Sequence[str],
    default_mode: Optional[str] = None,
) -> argparse.Namespace:
    """Create the common CLI used by dataset-specific evaluation scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=results_root,
        help="Directory containing mode folders with flight velocity CSVs.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=gt_root,
        help="Directory containing mode/flight_*/data.csv GT files.",
    )
    parser.add_argument(
        "--run-name",
        default=run_name,
        help="Output folder name to create inside each evaluated mode folder.",
    )
    parser.add_argument(
        "--window-duration",
        type=float,
        default=None,
        help="Override each configured window end with start + duration.",
    )
    if len(modes) > 1:
        parser.add_argument(
            "--mode",
            choices=[*modes, "ALL"],
            default=default_mode or modes[0],
            help="Mode to evaluate, or ALL to evaluate every configured mode.",
        )
    else:
        parser.set_defaults(mode=modes[0])
    args = parser.parse_args()
    if args.mode not in {*modes, "ALL"}:
        parser.error(f"default mode {args.mode!r} is not configured")
    return args


def with_duration(windows: Windows, duration: Optional[float]) -> Dict[str, Window]:
    if duration is None:
        return dict(windows)
    if duration <= 0:
        raise ValueError("window duration must be greater than zero")
    return {
        flight: (label, start, start + duration)
        for flight, (label, start, _end) in windows.items()
    }


def load_velocity_csv(path: Path) -> pd.DataFrame:
    """Load one already-clean velocity CSV and validate its data contract."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_VELOCITY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")

    frame = frame.copy()
    for column in REQUIRED_VELOCITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = ~np.isfinite(frame[REQUIRED_VELOCITY_COLUMNS].to_numpy(dtype=float))
    if invalid.any():
        raise ValueError(f"Non-finite velocity data found in {path}")
    return frame.sort_values("time").reset_index(drop=True)


def load_aligned_pair(est_path: Path, gt_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load estimate/GT data and require their existing timestamps to match."""
    estimate = load_velocity_csv(est_path)
    ground_truth = load_velocity_csv(gt_path)
    if len(estimate) != len(ground_truth) or not np.array_equal(
        estimate["time"].to_numpy(), ground_truth["time"].to_numpy()
    ):
        raise ValueError(
            "Estimate and GT timestamps are not already aligned: "
            f"{est_path} ({len(estimate)} rows), {gt_path} ({len(ground_truth)} rows)"
        )
    return estimate, ground_truth


def clip_aligned_pair(
    estimate: pd.DataFrame,
    ground_truth: pd.DataFrame,
    window_start: float,
    window_end: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select a flight-relative window from an aligned dataframe pair."""
    if window_start < 0 or window_end <= window_start:
        raise ValueError(f"Invalid window: {window_start} -> {window_end}")
    flight_time_zero = float(ground_truth["time"].iloc[0])
    absolute_start = flight_time_zero + window_start
    absolute_end = flight_time_zero + window_end
    mask = estimate["time"].between(absolute_start, absolute_end)
    estimate_window = estimate.loc[mask].copy().reset_index(drop=True)
    ground_truth_window = ground_truth.loc[mask].copy().reset_index(drop=True)
    if estimate_window.empty:
        raise ValueError(f"No samples in window {window_start} -> {window_end} seconds")
    return estimate_window, ground_truth_window


def _rmse(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(array**2)))


def _rmse_xy(err_x: pd.Series, err_y: pd.Series) -> float:
    x = err_x.to_numpy(dtype=float)
    y = err_y.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(x**2 + y**2)))


def calculate_errors(
    estimate: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate raw and first-sample bias-corrected velocity errors."""
    errors = pd.DataFrame({"time": estimate["time"]})
    for axis in "xyz":
        errors[f"err_{axis}"] = (
            estimate[f"vel_{axis}"].to_numpy(dtype=float)
            - ground_truth[f"vel_{axis}"].to_numpy(dtype=float)
        )
        errors[f"err_{axis}_bias"] = (
            errors[f"err_{axis}"] - float(errors[f"err_{axis}"].iloc[0])
        )
    return errors


def calculate_metrics(errors: pd.DataFrame) -> Dict[str, float]:
    """Calculate RMSE and the linear drift fit of horizontal raw error."""
    metrics = {
        "rmse_x": _rmse(errors["err_x"]),
        "rmse_y": _rmse(errors["err_y"]),
        "rmse_z": _rmse(errors["err_z"]),
        "rmse_xy": _rmse_xy(errors["err_x"], errors["err_y"]),
        "rmse_x_bias_corrected": _rmse(errors["err_x_bias"]),
        "rmse_y_bias_corrected": _rmse(errors["err_y_bias"]),
        "rmse_z_bias_corrected": _rmse(errors["err_z_bias"]),
        "rmse_xy_bias_corrected": _rmse_xy(
            errors["err_x_bias"], errors["err_y_bias"]
        ),
    }
    relative_time = errors["time"].to_numpy(dtype=float) - float(errors["time"].iloc[0])
    xy_error = np.hypot(
        errors["err_x"].to_numpy(dtype=float),
        errors["err_y"].to_numpy(dtype=float),
    )
    if len(errors) >= 2:
        drift_rate, drift_intercept = np.polyfit(relative_time, xy_error, 1)
    else:
        drift_rate, drift_intercept = float("nan"), float("nan")
    metrics["drift_rate_mps2"] = float(drift_rate)
    metrics["drift_intercept_mps"] = float(drift_intercept)
    return metrics


def write_rmse_summary(path: Path, metrics: Mapping[str, float]) -> None:
    lines = [
        "RMSE (Not including bias):",
        *(f"RMSE_{axis}={metrics[f'rmse_{axis}']:.6f}" for axis in ("x", "y", "z", "xy")),
        "",
        "RMSE (bias corrected):",
        *(
            f"RMSE_{axis}={metrics[f'rmse_{axis}_bias_corrected']:.6f}"
            for axis in ("x", "y", "z", "xy")
        ),
        "",
        f"Drift Rate={metrics['drift_rate_mps2']:.9f} m/s^2",
        f"Drift Intercept={metrics['drift_intercept_mps']:.9f} m/s",
    ]
    path.write_text("\n".join(lines) + "\n")


def evaluate_flight(
    mode: str,
    flight: str,
    label: str,
    window_start: float,
    window_end: float,
    est_path: Path,
    gt_path: Path,
    out_dir: Path,
) -> Dict[str, object]:
    estimate, ground_truth = load_aligned_pair(est_path, gt_path)
    estimate, ground_truth = clip_aligned_pair(
        estimate, ground_truth, window_start, window_end
    )
    errors = calculate_errors(estimate, ground_truth)
    metrics = calculate_metrics(errors)

    out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = out_dir / f"{flight}_velocity_window.csv"
    ground_truth_path = out_dir / "data_window.csv"
    errors_path = out_dir / f"{flight}_velocity_window_bias_errors.csv"
    rmse_path = out_dir / f"{flight}_velocity_window_rmse_summary.txt"
    estimate.to_csv(estimate_path, index=False)
    ground_truth.to_csv(ground_truth_path, index=False)
    errors.to_csv(errors_path, index=False)
    write_rmse_summary(rmse_path, metrics)

    rows = len(errors)
    actual_duration = float(errors["time"].iloc[-1] - errors["time"].iloc[0])
    print(
        f"Evaluated {mode} {flight}: {rows} samples, "
        f"{actual_duration:.6f} s, RMSE_xy={metrics['rmse_xy']:.6f}, "
        f"drift={metrics['drift_rate_mps2']:.6f} m/s^2",
        flush=True,
    )
    return {
        "mode": mode,
        "flight": flight,
        "label": label,
        "window_start_s": window_start,
        "window_end_s": window_end,
        "rows": rows,
        "actual_duration_s": actual_duration,
        **metrics,
        "out_dir": str(out_dir),
    }


def write_summary(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_csv_group(
    mode: str,
    windows: Windows,
    results_root: Path,
    gt_root: Path,
    run_name: str,
    summary_filename: str,
) -> List[Dict[str, object]]:
    """Evaluate one mode directly from pre-cleaned, aligned source CSVs."""
    source_dir = results_root / mode
    gt_dir = gt_root / mode
    output_root = source_dir / run_name
    rows = [
        evaluate_flight(
            mode=mode,
            flight=flight,
            label=label,
            window_start=start,
            window_end=end,
            est_path=source_dir / f"{flight}_velocity.csv",
            gt_path=gt_dir / flight / "data.csv",
            out_dir=output_root / flight,
        )
        for flight, (label, start, end) in windows.items()
    ]
    summary_path = output_root / summary_filename
    write_summary(summary_path, rows)
    print(f"Saved {mode} summary: {summary_path}", flush=True)
    return rows


def run_dataset_evaluation(
    args: argparse.Namespace,
    windows_by_mode: Mapping[str, Windows],
    dataset_tag: str,
) -> Dict[str, List[Dict[str, object]]]:
    """Run the selected modes and write per-mode and optional combined summaries."""
    results_root = args.results_root.resolve()
    gt_root = args.gt_root.resolve()
    selected_modes = list(windows_by_mode) if args.mode == "ALL" else [args.mode]
    results: Dict[str, List[Dict[str, object]]] = {}
    for mode in selected_modes:
        results[mode] = evaluate_csv_group(
            mode=mode,
            windows=with_duration(windows_by_mode[mode], args.window_duration),
            results_root=results_root,
            gt_root=gt_root,
            run_name=args.run_name,
            summary_filename=f"{mode.lower()}_{dataset_tag}_bias_rmse_drift_summary.csv",
        )

    if len(selected_modes) > 1:
        combined = [row for mode in selected_modes for row in results[mode]]
        combined_path = (
            results_root
            / args.run_name
            / f"combined_{dataset_tag}_bias_rmse_drift_summary.csv"
        )
        write_summary(combined_path, combined)
        print(f"Saved combined summary: {combined_path}", flush=True)
    return results
