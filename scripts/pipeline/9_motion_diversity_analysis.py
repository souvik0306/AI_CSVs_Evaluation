#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


IMU_COLS = ["time", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]
DATA_COLS = ["time", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
ACC_COLS = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
PLOT_COLORS = [
	"#1f77b4",
	"#ff7f0e",
	"#2ca02c",
	"#d62728",
	"#9467bd",
	"#8c564b",
	"#e377c2",
	"#7f7f7f",
	"#bcbd22",
	"#17becf",
	"#003f5c",
	"#ffa600",
	"#665191",
]


def _flight_sort_key(path: Path) -> int:
	try:
		return int(path.parent.name.split("_")[-1])
	except ValueError:
		return 10**9


def _read_csv(path: Path, required_cols: List[str]) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {path}")

	df = df.copy()
	for col in required_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=required_cols).sort_values("time").reset_index(drop=True)
	if df.empty:
		raise ValueError(f"No valid rows after cleaning {path}")
	return df


def _add_magnitudes(df: pd.DataFrame, g: float) -> pd.DataFrame:
	df = df.copy()
	df["acc_mag"] = np.linalg.norm(df[ACC_COLS].to_numpy(dtype=float), axis=1)
	df["gyro_mag"] = np.linalg.norm(df[GYRO_COLS].to_numpy(dtype=float), axis=1)
	df["dynamic_acc_mag"] = np.abs(df["acc_mag"] - g)
	return df


def _pct(mask: np.ndarray) -> float:
	if mask.size == 0:
		return float("nan")
	return float(mask.mean() * 100.0)


def _zscore(series: pd.Series) -> pd.Series:
	values = series.astype(float)
	std = values.std(ddof=0)
	if std == 0 or not np.isfinite(std):
		return values * 0.0
	return (values - values.mean()) / std


def _add_normalized_dynamic_scores(
	summary: pd.DataFrame,
	flight_data: Dict[str, pd.DataFrame],
	bins: int,
) -> pd.DataFrame:
	summary = summary.copy()
	all_dynamic = np.concatenate([df["dynamic_acc_mag"].to_numpy(dtype=float) for df in flight_data.values()])
	all_gyro = np.concatenate([df["gyro_mag"].to_numpy(dtype=float) for df in flight_data.values()])
	dyn_edges = np.linspace(float(np.min(all_dynamic)), float(np.max(all_dynamic)), bins + 1)
	gyro_edges = np.linspace(float(np.min(all_gyro)), float(np.max(all_gyro)), bins + 1)

	entropies = {}
	for flight, df in flight_data.items():
		hist, _, _ = np.histogram2d(
			df["dynamic_acc_mag"].to_numpy(dtype=float),
			df["gyro_mag"].to_numpy(dtype=float),
			bins=[dyn_edges, gyro_edges],
		)
		prob = hist.ravel().astype(float)
		prob = prob[prob > 0]
		entropy = -np.sum(prob / prob.sum() * np.log2(prob / prob.sum()))
		entropies[flight] = float(entropy)

	summary["motion_entropy"] = summary["flight"].map(entropies)
	score_terms = [
		"std_dynamic_acc_mag",
		"p95_dynamic_acc_mag",
		"std_gyro_mag",
		"p95_gyro_mag",
		"motion_entropy",
	]
	for term in score_terms:
		summary[f"z_{term}"] = _zscore(summary[term])
	summary["normalized_dynamic_motion_score"] = summary[[f"z_{term}" for term in score_terms]].sum(axis=1)
	return summary


def _summary_for_flight(
	flight: str,
	imu_df: pd.DataFrame,
	data_df: Optional[pd.DataFrame],
	g: float,
) -> Dict[str, float]:
	acc = imu_df["acc_mag"].to_numpy(dtype=float)
	dynamic_acc = imu_df["dynamic_acc_mag"].to_numpy(dtype=float)
	gyro = imu_df["gyro_mag"].to_numpy(dtype=float)
	duration = float(imu_df["time"].iloc[-1] - imu_df["time"].iloc[0])

	hover_mask = (gyro < 0.2) & (np.abs(acc - g) < 0.3)
	moderate_mask = (gyro >= 0.2) & (gyro < 1.0)
	aggressive_mask = gyro >= 1.0

	row: Dict[str, float] = {
		"flight": flight,
		"samples": int(len(imu_df)),
		"duration_s": duration,
		"mean_ax": float(imu_df["acc_x"].mean()),
		"mean_ay": float(imu_df["acc_y"].mean()),
		"mean_az": float(imu_df["acc_z"].mean()),
		"std_ax": float(imu_df["acc_x"].std(ddof=0)),
		"std_ay": float(imu_df["acc_y"].std(ddof=0)),
		"std_az": float(imu_df["acc_z"].std(ddof=0)),
		"mean_wx": float(imu_df["gyro_x"].mean()),
		"mean_wy": float(imu_df["gyro_y"].mean()),
		"mean_wz": float(imu_df["gyro_z"].mean()),
		"std_wx": float(imu_df["gyro_x"].std(ddof=0)),
		"std_wy": float(imu_df["gyro_y"].std(ddof=0)),
		"std_wz": float(imu_df["gyro_z"].std(ddof=0)),
		"mean_acc_mag": float(np.mean(acc)),
		"std_acc_mag": float(np.std(acc)),
		"p95_acc_mag": float(np.percentile(acc, 95)),
		"max_acc_mag": float(np.max(acc)),
		"mean_dynamic_acc_mag": float(np.mean(dynamic_acc)),
		"std_dynamic_acc_mag": float(np.std(dynamic_acc)),
		"p95_dynamic_acc_mag": float(np.percentile(dynamic_acc, 95)),
		"max_dynamic_acc_mag": float(np.max(dynamic_acc)),
		"mean_gyro_mag": float(np.mean(gyro)),
		"std_gyro_mag": float(np.std(gyro)),
		"p95_gyro_mag": float(np.percentile(gyro, 95)),
		"max_gyro_mag": float(np.max(gyro)),
		"hover_pct": _pct(hover_mask),
		"moderate_pct": _pct(moderate_mask),
		"aggressive_pct": _pct(aggressive_mask),
		"non_hover_low_gyro_pct": _pct((gyro < 0.2) & ~hover_mask),
	}
	row["motion_score"] = (
		row["std_acc_mag"]
		+ row["std_gyro_mag"]
		+ row["mean_gyro_mag"]
		+ row["p95_gyro_mag"]
		+ row["p95_acc_mag"]
	)
	row["dynamic_motion_score"] = (
		row["std_dynamic_acc_mag"]
		+ row["std_gyro_mag"]
		+ row["mean_gyro_mag"]
		+ row["p95_gyro_mag"]
		+ row["p95_dynamic_acc_mag"]
	)

	if data_df is not None:
		vel = data_df[["vel_x", "vel_y", "vel_z"]].to_numpy(dtype=float)
		pos = data_df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
		speed = np.linalg.norm(vel, axis=1)
		step_dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
		row.update(
			{
				"avg_speed": float(np.mean(speed)),
				"max_speed": float(np.max(speed)),
				"total_distance": float(np.sum(step_dist)),
				"trajectory_duration_s": float(data_df["time"].iloc[-1] - data_df["time"].iloc[0]),
			}
		)
	else:
		row.update(
			{
				"avg_speed": float("nan"),
				"max_speed": float("nan"),
				"total_distance": float("nan"),
				"trajectory_duration_s": float("nan"),
			}
		)

	return row


def _plot_magnitude_histograms(
	flight_data: Dict[str, pd.DataFrame],
	out_dir: Path,
	column: str,
	xlabel: str,
	filename: str,
	bins: int,
	log_y: bool = False,
) -> None:
	n = len(flight_data)
	cols = 4
	rows = math.ceil(n / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), squeeze=False)
	all_values = np.concatenate([df[column].to_numpy(dtype=float) for df in flight_data.values()])
	xmin = float(np.min(all_values))
	xmax = float(np.max(all_values))
	if xmin == xmax:
		xmin -= 0.5
		xmax += 0.5

	for idx, (flight, df) in enumerate(flight_data.items()):
		ax = axes[idx // cols][idx % cols]
		values = df[column].to_numpy(dtype=float)
		ax.hist(values, bins=bins, range=(xmin, xmax), color=PLOT_COLORS[idx % len(PLOT_COLORS)], alpha=0.82)
		ax.set_title(flight)
		ax.set_xlabel(xlabel)
		ax.set_ylabel("count")
		if log_y:
			ax.set_yscale("log")
		ax.grid(True, alpha=0.25)

	for idx in range(n, rows * cols):
		axes[idx // cols][idx % cols].axis("off")

	fig.suptitle(f"Per-flight histogram: {xlabel}", fontsize=16)
	fig.tight_layout(rect=(0, 0, 1, 0.97))
	fig.savefig(out_dir / filename, dpi=180)
	plt.close(fig)


def _plot_summary_bars(summary: pd.DataFrame, out_dir: Path) -> None:
	ordered = summary.sort_values("normalized_dynamic_motion_score", ascending=False).reset_index(drop=True)
	x = np.arange(len(ordered))

	fig, ax = plt.subplots(figsize=(16, 7))
	ax.bar(x, ordered["normalized_dynamic_motion_score"], color="#315f72")
	ax.set_xticks(x)
	ax.set_xticklabels(ordered["flight"], rotation=45, ha="right")
	ax.set_ylabel("normalized dynamic motion score")
	ax.set_title("Automatic dynamic motion diversity ranking")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_dir / "normalized_dynamic_motion_score_ranking.png", dpi=180)
	plt.close(fig)

	raw_ordered = summary.sort_values("motion_score", ascending=False).reset_index(drop=True)
	fig, ax = plt.subplots(figsize=(16, 7))
	ax.bar(x, raw_ordered["motion_score"], color="#6f5f3f")
	ax.set_xticks(x)
	ax.set_xticklabels(raw_ordered["flight"], rotation=45, ha="right")
	ax.set_ylabel("raw IMU motion score")
	ax.set_title("Raw IMU motion score ranking")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_dir / "raw_motion_score_ranking.png", dpi=180)
	fig.savefig(out_dir / "motion_score_ranking.png", dpi=180)
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(16, 7))
	bottom = np.zeros(len(ordered))
	for col, label, color in [
		("hover_pct", "hover", "#4c78a8"),
		("moderate_pct", "moderate", "#f58518"),
		("aggressive_pct", "aggressive", "#e45756"),
		("non_hover_low_gyro_pct", "low gyro / non-hover", "#72b7b2"),
	]:
		values = ordered[col].to_numpy(dtype=float)
		ax.bar(x, values, bottom=bottom, label=label, color=color)
		bottom += values
	ax.set_xticks(x)
	ax.set_xticklabels(ordered["flight"], rotation=45, ha="right")
	ax.set_ylabel("percent of flight")
	ax.set_ylim(0, 100)
	ax.set_title("Time spent in motion regimes")
	ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_dir / "motion_regime_percentages.png", dpi=180)
	plt.close(fig)


def _build_windows(
	flight_data: Dict[str, pd.DataFrame],
	window_size: int,
	stride: int,
	max_windows_per_flight: int,
) -> Tuple[np.ndarray, List[str]]:
	windows = []
	labels = []
	for flight, df in flight_data.items():
		values = df[GYRO_COLS + ACC_COLS].to_numpy(dtype=float)
		starts = list(range(0, len(values) - window_size + 1, stride))
		if max_windows_per_flight > 0 and len(starts) > max_windows_per_flight:
			pick = np.linspace(0, len(starts) - 1, max_windows_per_flight).round().astype(int)
			starts = [starts[i] for i in pick]
		for start in starts:
			windows.append(values[start : start + window_size].reshape(-1))
			labels.append(flight)

	if not windows:
		raise ValueError("No PCA windows created. Reduce --window-size or --stride.")
	return np.vstack(windows), labels


def _plot_pca_windows(
	flight_data: Dict[str, pd.DataFrame],
	out_dir: Path,
	window_size: int,
	stride: int,
	max_windows_per_flight: int,
) -> pd.DataFrame:
	x, labels = _build_windows(flight_data, window_size, stride, max_windows_per_flight)
	x = x - np.mean(x, axis=0, keepdims=True)
	scale = np.std(x, axis=0, keepdims=True)
	scale[scale == 0] = 1.0
	x = x / scale

	_, singular_values, vt = np.linalg.svd(x, full_matrices=False)
	components = vt[:2].T
	projected = x @ components
	explained = (singular_values**2) / np.sum(singular_values**2)

	fig, ax = plt.subplots(figsize=(11, 9))
	for idx, flight in enumerate(flight_data.keys()):
		mask = np.array(labels) == flight
		ax.scatter(
			projected[mask, 0],
			projected[mask, 1],
			s=16,
			alpha=0.68,
			label=flight,
			color=PLOT_COLORS[idx % len(PLOT_COLORS)],
			edgecolors="none",
		)
	ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
	ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
	ax.set_title(f"PCA of {window_size}-sample IMU windows")
	ax.grid(True, alpha=0.25)
	ax.legend(ncol=2, fontsize=9, frameon=False)
	fig.tight_layout()
	fig.savefig(out_dir / "pca_imu_windows.png", dpi=180)
	plt.close(fig)
	explained_df = pd.DataFrame(
		{
			"component": np.arange(1, min(21, len(explained) + 1)),
			"explained_variance_ratio": explained[:20],
			"cumulative_explained_variance_ratio": np.cumsum(explained[:20]),
		}
	)
	explained_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)
	return explained_df


def _normalized_wasserstein(a: np.ndarray, b: np.ndarray, scale: float) -> float:
	if scale <= 0 or not np.isfinite(scale):
		scale = 1.0
	return float(wasserstein_distance(a, b) / scale)


def _hist_prob(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
	hist, _ = np.histogram(values, bins=edges)
	prob = hist.astype(float) + 1e-12
	return prob / prob.sum()


def _pairwise_similarity(
	flight_data: Dict[str, pd.DataFrame],
	out_dir: Path,
	bins: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	flights = list(flight_data.keys())
	channels = GYRO_COLS + ACC_COLS
	channel_stats = {}
	standardized_data = {}
	edges = {}
	for channel in channels:
		values = np.concatenate([df[channel].to_numpy(dtype=float) for df in flight_data.values()])
		mean = float(np.mean(values))
		std = float(np.std(values))
		if std == 0 or not np.isfinite(std):
			std = 1.0
		channel_stats[channel] = (mean, std)

	for flight, df in flight_data.items():
		standardized_data[flight] = {}
		for channel in channels:
			mean, std = channel_stats[channel]
			standardized_data[flight][channel] = (df[channel].to_numpy(dtype=float) - mean) / std

	for channel in channels:
		values = np.concatenate([standardized_data[flight][channel] for flight in flights])
		vmin = float(np.min(values))
		vmax = float(np.max(values))
		if vmin == vmax:
			vmin -= 0.5
			vmax += 0.5
		edges[channel] = np.linspace(vmin, vmax, bins + 1)

	wass = pd.DataFrame(0.0, index=flights, columns=flights)
	jsd = pd.DataFrame(0.0, index=flights, columns=flights)
	for i, flight_a in enumerate(flights):
		for j, flight_b in enumerate(flights):
			if j <= i:
				continue
			wass_values = []
			jsd_values = []
			for channel in channels:
				a_values = standardized_data[flight_a][channel]
				b_values = standardized_data[flight_b][channel]
				wass_values.append(float(wasserstein_distance(a_values, b_values)))
				jsd_values.append(
					float(
						jensenshannon(
							_hist_prob(a_values, edges[channel]),
							_hist_prob(b_values, edges[channel]),
							base=2.0,
						)
					)
				)
			wass_value = float(np.mean(wass_values))
			jsd_value = float(np.mean(jsd_values))
			wass.iloc[i, j] = wass.iloc[j, i] = wass_value
			jsd.iloc[i, j] = jsd.iloc[j, i] = jsd_value

	wass.to_csv(out_dir / "pairwise_wasserstein_distance.csv")
	jsd.to_csv(out_dir / "pairwise_jensen_shannon_distance.csv")
	_plot_heatmap(wass, out_dir / "pairwise_wasserstein_heatmap.png", "Pairwise standardized six-channel Wasserstein distance")
	_plot_heatmap(jsd, out_dir / "pairwise_jensen_shannon_heatmap.png", "Pairwise standardized six-channel Jensen-Shannon distance")
	return wass, jsd


def _plot_heatmap(matrix: pd.DataFrame, path: Path, title: str) -> None:
	fig, ax = plt.subplots(figsize=(11, 9))
	im = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis")
	ax.set_xticks(np.arange(len(matrix.columns)))
	ax.set_yticks(np.arange(len(matrix.index)))
	ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
	ax.set_yticklabels(matrix.index)
	ax.set_title(title)
	for i in range(len(matrix.index)):
		for j in range(len(matrix.columns)):
			value = matrix.iloc[i, j]
			ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	fig.tight_layout()
	fig.savefig(path, dpi=180)
	plt.close(fig)


def _split_table(
	summary: pd.DataFrame,
	pairwise_wass: pd.DataFrame,
	assignments: Dict[str, str],
) -> pd.DataFrame:
	rows = []
	for _, row in summary.sort_values("normalized_dynamic_motion_score", ascending=False).iterrows():
		flight = row["flight"]
		split = assignments.get(flight, "training")
		nearest = pairwise_wass.loc[flight].replace(0.0, np.nan).idxmin()
		nearest_distance = float(pairwise_wass.loc[flight, nearest])
		rows.append(
			{
				"flight": flight,
				"split": split,
				"normalized_dynamic_motion_score": row["normalized_dynamic_motion_score"],
				"raw_motion_score": row["motion_score"],
				"hover_pct": row["hover_pct"],
				"aggressive_pct": row["aggressive_pct"],
				"max_gyro_mag": row["max_gyro_mag"],
				"max_acc_mag": row["max_acc_mag"],
				"nearest_flight": nearest,
				"nearest_distance": nearest_distance,
			}
		)
	return pd.DataFrame(rows)


def _balanced_split(summary: pd.DataFrame, pairwise_wass: pd.DataFrame) -> pd.DataFrame:
	assignments = {
		"flight_5": "validation",
		"flight_10": "validation",
		"flight_7": "evaluation",
		"flight_13": "evaluation",
	}
	return _split_table(summary, pairwise_wass, assignments)


def _ood_split(summary: pd.DataFrame, pairwise_wass: pd.DataFrame) -> pd.DataFrame:
	assignments = {
		"flight_9": "validation",
		"flight_12": "validation",
		"flight_6": "evaluation",
		"flight_7": "evaluation",
	}
	return _split_table(summary, pairwise_wass, assignments)


def _per_axis_ranking_table(summary: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
	rows = []
	for label, column in [
		("a_x", "std_ax"),
		("a_y", "std_ay"),
		("a_z", "std_az"),
		("omega_x", "std_wx"),
		("omega_y", "std_wy"),
		("omega_z", "std_wz"),
	]:
		leaders = summary.sort_values(column, ascending=False).head(top_n)
		rows.append(
			{
				"channel": label,
				"largest_spread_flights": ", ".join(leaders["flight"].tolist()),
			}
		)
	return pd.DataFrame(rows)


def _write_report(
	summary: pd.DataFrame,
	balanced_split_df: pd.DataFrame,
	ood_split_df: pd.DataFrame,
	pairwise_wass: pd.DataFrame,
	pca_explained: pd.DataFrame,
	out_dir: Path,
	g: float,
	window_size: int,
	stride: int,
) -> None:
	dynamic_ordered = summary.sort_values("normalized_dynamic_motion_score", ascending=False).reset_index(drop=True)
	raw_ordered = summary.sort_values("motion_score", ascending=False).reset_index(drop=True)
	least_dynamic = summary.sort_values("normalized_dynamic_motion_score", ascending=True).head(3)

	pairs = []
	for i, flight_a in enumerate(pairwise_wass.index):
		for j, flight_b in enumerate(pairwise_wass.columns):
			if j <= i:
				continue
			pairs.append((float(pairwise_wass.loc[flight_a, flight_b]), flight_a, flight_b))
	pairs.sort()
	pairs_df = pd.DataFrame(pairs[:8], columns=["distance", "flight_a", "flight_b"])
	avg_dist = []
	for flight in pairwise_wass.index:
		avg_dist.append(
			{
				"flight": flight,
				"mean_wasserstein_distance": float(pairwise_wass.loc[flight].replace(0.0, np.nan).mean()),
			}
		)
	avg_dist_df = pd.DataFrame(avg_dist).sort_values("mean_wasserstein_distance", ascending=False).head(5)

	def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
		cols = list(df.columns)
		lines = [
			"| " + " | ".join(cols) + " |",
			"| " + " | ".join(["---"] * len(cols)) + " |",
		]
		for _, row in df.iterrows():
			cells = []
			for col in cols:
				value = row[col]
				if isinstance(value, (float, np.floating)):
					cells.append(format(float(value), floatfmt))
				else:
					cells.append(str(value))
			lines.append("| " + " | ".join(cells) + " |")
		return "\n".join(lines)

	lines = [
		"# Motion diversity report",
		"",
		"## Methodology",
		"",
		f"- IMU source: `TL-data/flight_N/imu.csv`, sampled at approximately 200 Hz.",
		"- Accelerometer columns: `acc_x`, `acc_y`, `acc_z` in m/s^2.",
		"- Gyroscope columns: `gyro_x`, `gyro_y`, `gyro_z` in rad/s.",
		f"- Gravity constant for hover and fallback dynamic acceleration: `{g:.5f}` m/s^2.",
		"- Hover: `|omega| < 0.2` and `abs(|a| - g) < 0.3`.",
		"- Moderate angular motion: `0.2 <= |omega| < 1.0`.",
		"- Aggressive angular motion: `|omega| >= 1.0`.",
		"- Low gyro / non-hover: `|omega| < 0.2` but not passing the hover acceleration threshold.",
		"- Raw motion score: `std(|a|) + std(|omega|) + mean(|omega|) + p95(|omega|) + p95(|a|)`.",
		"- Normalized dynamic motion score: z-scored sum of `std(abs(|a|-g))`, `p95(abs(|a|-g))`, `std(|omega|)`, `p95(|omega|)`, and joint motion entropy.",
		"- Histogram ranges use the complete observed min/max. Gyro histograms use a logarithmic count axis to expose rare high-rate tails.",
		"- Pairwise distance is the average of six globally standardized per-channel Wasserstein distances: `gyro_x/y/z` and `acc_x/y/z`.",
		f"- PCA uses flattened `{window_size}`-sample windows with stride `{stride}` and standardized features.",
		"",
		"## Caveats",
		"",
		"The raw acceleration magnitude contains gravity and is concentrated near 9.81 m/s^2. The raw score is kept for traceability, but the normalized dynamic score is the primary ranking for split decisions. `abs(|a|-g)` is only a fallback gravity proxy; full quaternion-based gravity removal would be more informative once the frame convention is locked down.",
		"",
		"## Highest normalized dynamic-score flights",
		"",
		markdown_table(
			dynamic_ordered[
				[
					"flight",
					"normalized_dynamic_motion_score",
					"dynamic_motion_score",
					"hover_pct",
					"aggressive_pct",
					"max_gyro_mag",
				]
			].head(5)
		),
		"",
		"## Lowest normalized dynamic-score flights",
		"",
		markdown_table(
			least_dynamic[
				[
					"flight",
					"normalized_dynamic_motion_score",
					"dynamic_motion_score",
					"hover_pct",
					"aggressive_pct",
					"max_gyro_mag",
				]
			]
		),
		"",
		"## Highest raw-IMU motion-score flights",
		"",
		markdown_table(
			raw_ordered[["flight", "motion_score", "hover_pct", "aggressive_pct", "max_gyro_mag", "max_acc_mag"]].head(5)
		),
		"",
		"## Per-axis spread leaders",
		"",
		markdown_table(_per_axis_ranking_table(summary)),
		"",
		"## Most distinct flights by mean Wasserstein distance",
		"",
		markdown_table(avg_dist_df),
		"",
		"## Most similar flight pairs by Wasserstein distance",
		"",
		markdown_table(pairs_df),
		"",
		"## PCA explained variance",
		"",
		f"The first two PCA components explain {(pca_explained['cumulative_explained_variance_ratio'].iloc[1] * 100):.1f}% of the flattened-window variance.",
		"",
		markdown_table(pca_explained.head(5)),
		"",
		"## Balanced generalization split",
		"",
		markdown_table(
			balanced_split_df[
				[
					"flight",
					"split",
					"normalized_dynamic_motion_score",
					"hover_pct",
					"aggressive_pct",
					"nearest_flight",
				]
			]
		),
		"",
		"## OOD challenge split",
		"",
		"This split intentionally reserves the two strongest high-relative-diversity flights for evaluation. It is useful as a stress test, but it is not the preferred primary fine-tuning split.",
		"",
		markdown_table(
			ood_split_df[
				[
					"flight",
					"split",
					"normalized_dynamic_motion_score",
					"hover_pct",
					"aggressive_pct",
					"nearest_flight",
				]
			]
		),
		"",
	]
	(out_dir / "motion_diversity_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Quantify motion diversity for each TL-data flight and generate split recommendations.",
	)
	parser.add_argument("--input-dir", default="TL-data", help="Directory containing flight_N/imu.csv and optional data.csv")
	parser.add_argument("--output-dir", default="motion_diversity_outputs", help="Directory for tables and plots")
	parser.add_argument("--g", type=float, default=9.80665, help="Gravity magnitude used for hover classification")
	parser.add_argument("--bins", type=int, default=80, help="Histogram bins for plots and distribution distances")
	parser.add_argument("--window-size", type=int, default=200, help="IMU samples per PCA window")
	parser.add_argument("--stride", type=int, default=100, help="Stride in IMU samples between PCA windows")
	parser.add_argument("--max-windows-per-flight", type=int, default=200, help="Cap PCA windows per flight; 0 disables cap")
	args = parser.parse_args()

	input_dir = Path(args.input_dir)
	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	imu_paths = sorted(input_dir.glob("flight_*/imu.csv"), key=_flight_sort_key)
	if not imu_paths:
		raise SystemExit(f"No IMU CSVs found under {input_dir}/flight_*/imu.csv")

	flight_data: Dict[str, pd.DataFrame] = {}
	summary_rows = []
	for imu_path in imu_paths:
		flight = imu_path.parent.name
		imu_df = _add_magnitudes(_read_csv(imu_path, IMU_COLS), args.g)
		data_path = imu_path.parent / "data.csv"
		data_df = _read_csv(data_path, DATA_COLS) if data_path.exists() else None
		flight_data[flight] = imu_df
		summary_rows.append(_summary_for_flight(flight, imu_df, data_df, args.g))

	summary = pd.DataFrame(summary_rows).sort_values("flight", key=lambda col: col.map(lambda x: int(x.split("_")[-1])))
	summary = _add_normalized_dynamic_scores(summary, flight_data, args.bins)
	summary.to_csv(out_dir / "flight_motion_summary.csv", index=False)
	summary.sort_values("normalized_dynamic_motion_score", ascending=False).to_csv(
		out_dir / "flight_dynamic_motion_ranking.csv",
		index=False,
	)
	summary.sort_values("normalized_dynamic_motion_score", ascending=False).to_csv(
		out_dir / "flight_motion_ranking.csv",
		index=False,
	)
	summary.sort_values("motion_score", ascending=False).to_csv(out_dir / "flight_raw_motion_ranking.csv", index=False)

	_plot_magnitude_histograms(flight_data, out_dir, "acc_mag", "|a| (m/s^2)", "acc_magnitude_histograms.png", args.bins)
	_plot_magnitude_histograms(
		flight_data,
		out_dir,
		"dynamic_acc_mag",
		"abs(|a| - g) (m/s^2)",
		"dynamic_acc_magnitude_histograms.png",
		args.bins,
		log_y=True,
	)
	_plot_magnitude_histograms(
		flight_data,
		out_dir,
		"gyro_mag",
		"|omega| (rad/s)",
		"gyro_magnitude_histograms.png",
		args.bins,
		log_y=True,
	)
	_plot_summary_bars(summary, out_dir)
	pca_explained = _plot_pca_windows(flight_data, out_dir, args.window_size, args.stride, args.max_windows_per_flight)
	pairwise_wass, _ = _pairwise_similarity(flight_data, out_dir, args.bins)

	balanced_split_df = _balanced_split(summary, pairwise_wass)
	ood_split_df = _ood_split(summary, pairwise_wass)
	balanced_split_df.to_csv(out_dir / "recommended_split.csv", index=False)
	ood_split_df.to_csv(out_dir / "ood_challenge_split.csv", index=False)
	_write_report(
		summary,
		balanced_split_df,
		ood_split_df,
		pairwise_wass,
		pca_explained,
		out_dir,
		args.g,
		args.window_size,
		args.stride,
	)

	print(f"Analyzed {len(flight_data)} flights.")
	print(f"Saved summary: {out_dir / 'flight_motion_summary.csv'}")
	print(f"Saved dynamic ranking: {out_dir / 'flight_dynamic_motion_ranking.csv'}")
	print(f"Saved raw ranking: {out_dir / 'flight_raw_motion_ranking.csv'}")
	print(f"Saved split: {out_dir / 'recommended_split.csv'}")
	print(f"Saved OOD split: {out_dir / 'ood_challenge_split.csv'}")
	print(f"Saved report: {out_dir / 'motion_diversity_report.md'}")


if __name__ == "__main__":
	main()
