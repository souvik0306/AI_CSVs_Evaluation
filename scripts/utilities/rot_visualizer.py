"""Quaternion position visualizer.

This script loads a CSV with position and quaternion columns, converts the
quaternion to roll/pitch/yaw, and renders a 3D trajectory visualization with
orientation axes sampled along the path.

Expected columns:
	time, pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z

Example:
	python rot_visualizer.py --csv TL-data/flight_1/data.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


REQUIRED_COLUMNS = [
	"pos_x",
	"pos_y",
	"pos_z",
	"quat_w",
	"quat_x",
	"quat_y",
	"quat_z",
]

AXIS_COLORS = ["#ff0000", "#00aa00", "#0000ff"]


def quat_to_rpy(quat_w: float, quat_x: float, quat_y: float, quat_z: float) -> tuple[float, float, float]:
	"""Convert a normalized quaternion to roll, pitch, yaw in radians.

	The quaternion is assumed to follow the Hamilton convention and to be
	ordered as w, x, y, z.
	"""

	norm = math.sqrt(quat_w * quat_w + quat_x * quat_x + quat_y * quat_y + quat_z * quat_z)
	if norm == 0.0:
		raise ValueError("Quaternion norm is zero")

	w = quat_w / norm
	x = quat_x / norm
	y = quat_y / norm
	z = quat_z / norm

	sinr_cosp = 2.0 * (w * x + y * z)
	cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
	roll = math.atan2(sinr_cosp, cosr_cosp)

	sinp = 2.0 * (w * y - z * x)
	if abs(sinp) >= 1.0:
		pitch = math.copysign(math.pi / 2.0, sinp)
	else:
		pitch = math.asin(sinp)

	siny_cosp = 2.0 * (w * z + x * y)
	cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
	yaw = math.atan2(siny_cosp, cosy_cosp)

	return roll, pitch, yaw


def rotation_matrix_from_quat(quat_w: float, quat_x: float, quat_y: float, quat_z: float) -> np.ndarray:
	"""Return the 3x3 rotation matrix for the given quaternion."""

	norm = math.sqrt(quat_w * quat_w + quat_x * quat_x + quat_y * quat_y + quat_z * quat_z)
	if norm == 0.0:
		raise ValueError("Quaternion norm is zero")

	w = quat_w / norm
	x = quat_x / norm
	y = quat_y / norm
	z = quat_z / norm

	return np.array(
		[
			[1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
			[2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
			[2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
		],
		dtype=float,
	)


def load_motion_csv(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {', '.join(missing)}")

	rpy = df.apply(
		lambda row: quat_to_rpy(row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]),
		axis=1,
		result_type="expand",
	)
	rpy.columns = ["roll", "pitch", "yaw"]
	df = pd.concat([df, rpy], axis=1)
	return df


def set_axes_equal(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
	"""Set equal aspect for a 3D axis using the data bounds."""

	x_range = float(np.max(x) - np.min(x))
	y_range = float(np.max(y) - np.min(y))
	z_range = float(np.max(z) - np.min(z))
	max_range = max(x_range, y_range, z_range)
	if max_range == 0.0:
		max_range = 1.0

	x_mid = float((np.max(x) + np.min(x)) * 0.5)
	y_mid = float((np.max(y) + np.min(y)) * 0.5)
	z_mid = float((np.max(z) + np.min(z)) * 0.5)

	half = max_range * 0.5
	ax.set_xlim(x_mid - half, x_mid + half)
	ax.set_ylim(y_mid - half, y_mid + half)
	ax.set_zlim(z_mid - half, z_mid + half)


def clamp_nonnegative_z_axis(ax: plt.Axes, z_values: np.ndarray) -> None:
	"""Prevent negative Z axis limits when all Z samples are non-negative."""

	if float(np.min(z_values)) >= 0.0:
		z_low, z_high = ax.get_zlim()
		if z_low < 0.0:
			ax.set_zlim(0.0, z_high - z_low)


def invert_x_axis(ax: plt.Axes) -> None:
	"""Flip X axis so positive X is shown toward the left in view coordinates."""

	x_low, x_high = ax.get_xlim()
	ax.set_xlim(x_high, x_low)


def build_trajectory_figure(
	df: pd.DataFrame,
	sample_step: int = 25,
	arrow_length: float = 0.08,
) -> tuple[plt.Figure, plt.Axes, np.ndarray, np.ndarray]:
	positions = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)

	fig = plt.figure(figsize=(12, 10))
	ax = fig.add_subplot(111, projection="3d")

	time_values = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df), dtype=float)
	elapsed_time = time_values - time_values[0]
	ax.plot(
		positions[:, 0],
		positions[:, 1],
		positions[:, 2],
		color="#1f77b4",
		linewidth=2.0,
		label="trajectory",
	)
	scatter = ax.scatter(
		positions[:, 0],
		positions[:, 1],
		positions[:, 2],
		c=elapsed_time,
		cmap="viridis",
		s=10,
		alpha=0.75,
		label="samples",
	)

	start = positions[0]
	end = positions[-1]
	ax.scatter(start[0], start[1], start[2], color="#22c55e", s=90, marker="o", label="start")
	ax.scatter(end[0], end[1], end[2], color="#ef4444", s=110, marker="X", label="end")
	ax.text(start[0], start[1], start[2], "  Start", color="#166534", fontsize=9)
	ax.text(end[0], end[1], end[2], "  End", color="#991b1b", fontsize=9)

	for idx in range(0, len(df), max(1, sample_step)):
		row = df.iloc[idx]
		rotation = rotation_matrix_from_quat(row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"])
		origin = np.array([row["pos_x"], row["pos_y"], row["pos_z"]], dtype=float)
		axes = rotation @ np.eye(3)
		for axis_index, color in zip(range(3), AXIS_COLORS):
			vector = axes[:, axis_index] * arrow_length
			ax.quiver(
				origin[0],
				origin[1],
				origin[2],
				vector[0],
				vector[1],
				vector[2],
				color=color,
				linewidth=1.5,
				arrow_length_ratio=0.25,
			)

	ax.set_title("3D Position and Quaternion Orientation Visualizer")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.view_init(elev=10, azim=-90)
	set_axes_equal(ax, positions[:, 0], positions[:, 1], positions[:, 2])
	invert_x_axis(ax)
	clamp_nonnegative_z_axis(ax, positions[:, 2])
	fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7, label="elapsed time (s)")
	ax.legend(loc="upper left")
	fig.tight_layout()
	return fig, ax, positions, elapsed_time


def save_trajectory_gif(
	df: pd.DataFrame,
	output_path: Path,
	sample_step: int = 25,
	arrow_length: float = 0.08,
	frame_step: int = 15,
	fps: int = 20,
) -> None:
	positions = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
	time_values = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df), dtype=float)
	elapsed_time = time_values - time_values[0]
	norm = plt.Normalize(vmin=float(np.min(elapsed_time)), vmax=float(np.max(elapsed_time)))

	fig = plt.figure(figsize=(12, 10))
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title("3D Position and Quaternion Orientation Visualizer")
	time_title = fig.suptitle("Elapsed time: 0.00 s", y=0.98)
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")
	ax.view_init(elev=10, azim=-90)
	set_axes_equal(ax, positions[:, 0], positions[:, 1], positions[:, 2])
	invert_x_axis(ax)
	clamp_nonnegative_z_axis(ax, positions[:, 2])

	trail_line = Line3DCollection([], cmap="viridis", norm=norm, linewidth=2.5, alpha=0.95)
	ax.add_collection3d(trail_line)
	trail_scatter = ax.scatter(
		[positions[0, 0]],
		[positions[0, 1]],
		[positions[0, 2]],
		c=[elapsed_time[0]],
		cmap="viridis",
		norm=norm,
		s=10,
		alpha=0.8,
		label="samples",
	)
	start = positions[0]
	end = positions[-1]
	ax.scatter(start[0], start[1], start[2], color="#22c55e", s=90, marker="o", label="start")
	ax.scatter(end[0], end[1], end[2], color="#ef4444", s=110, marker="X", label="end")
	ax.text(start[0], start[1], start[2], "  Start", color="#166534", fontsize=9)
	ax.text(end[0], end[1], end[2], "  End", color="#991b1b", fontsize=9)
	fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, pad=0.1, shrink=0.7, label="elapsed time (s)")
	ax.legend(loc="upper left")

	marker = ax.scatter([], [], [], color="#ff0000", s=60)
	quivers = []
	indices = list(range(0, len(df), max(1, frame_step)))

	def clear_quivers() -> None:
		nonlocal quivers
		for quiver in quivers:
			quiver.remove()
		quivers = []

	def update(frame_index: int):
		nonlocal quivers
		idx = indices[frame_index]
		clear_quivers()
		trail = positions[: idx + 1]
		if len(trail) >= 2:
			segments = np.stack([trail[:-1], trail[1:]], axis=1)
			trail_line.set_segments(segments)
			trail_line.set_array(elapsed_time[1 : idx + 1])
		else:
			trail_line.set_segments([])
			trail_line.set_array(np.array([], dtype=float))
		trail_scatter._offsets3d = (trail[:, 0], trail[:, 1], trail[:, 2])
		trail_scatter.set_array(elapsed_time[: idx + 1])
		current = positions[idx]
		marker._offsets3d = ([current[0]], [current[1]], [current[2]])
		rotation = rotation_matrix_from_quat(df.iloc[idx]["quat_w"], df.iloc[idx]["quat_x"], df.iloc[idx]["quat_y"], df.iloc[idx]["quat_z"])
		axes = rotation @ np.eye(3)
		for axis_index, color in zip(range(3), AXIS_COLORS):
			vector = axes[:, axis_index] * arrow_length
			quivers.append(
				ax.quiver(
					current[0],
					current[1],
					current[2],
					vector[0],
					vector[1],
					vector[2],
					color=color,
					linewidth=2.0,
					arrow_length_ratio=0.25,
				)
			)
		time_title.set_text(f"Elapsed time: {elapsed_time[idx]:.2f} s")
		return [trail_line, trail_scatter, marker, *quivers]

	anim = FuncAnimation(fig, update, frames=len(indices), interval=1000 / fps, blit=False)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	anim.save(output_path, writer=PillowWriter(fps=fps))
	print(f"Saved {output_path}")
	plt.close(fig)


def save_or_show_figure(fig: plt.Figure, output_path: Path | None, show: bool) -> None:
	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		fig.savefig(output_path, dpi=200, bbox_inches="tight")
		print(f"Saved {output_path}")
	if show:
		fig.show()
	else:
		plt.close(fig)


def plot_rpy(df: pd.DataFrame) -> None:
	fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
	time_values = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df))
	angles_deg = np.degrees(df[["roll", "pitch", "yaw"]].to_numpy(dtype=float))

	axes[0].plot(time_values, angles_deg[:, 0], color="#d62728")
	axes[0].set_ylabel("Roll (deg)")
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(time_values, angles_deg[:, 1], color="#2ca02c")
	axes[1].set_ylabel("Pitch (deg)")
	axes[1].grid(True, alpha=0.3)

	axes[2].plot(time_values, angles_deg[:, 2], color="#1f77b4")
	axes[2].set_ylabel("Yaw (deg)")
	axes[2].set_xlabel("Time")
	axes[2].grid(True, alpha=0.3)

	fig.suptitle("Roll, Pitch, Yaw from Quaternion")
	fig.tight_layout()


def build_position_figure(df: pd.DataFrame) -> plt.Figure:
	fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
	time_values = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df), dtype=float)

	axes[0].plot(time_values, df["pos_x"].to_numpy(dtype=float), color="#d62728")
	axes[0].set_ylabel("X (m)")
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(time_values, df["pos_y"].to_numpy(dtype=float), color="#2ca02c")
	axes[1].set_ylabel("Y (m)")
	axes[1].grid(True, alpha=0.3)

	axes[2].plot(time_values, df["pos_z"].to_numpy(dtype=float), color="#1f77b4")
	axes[2].set_ylabel("Z (m)")
	axes[2].set_xlabel("Time")
	axes[2].grid(True, alpha=0.3)

	fig.suptitle("Position X, Y, Z Over Time")
	fig.tight_layout()
	return fig


def build_z_only_3d_figure(df: pd.DataFrame) -> plt.Figure:
	fig = plt.figure(figsize=(12, 8))
	ax = fig.add_subplot(111, projection="3d")
	time_values = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df), dtype=float)
	elapsed_time = time_values - time_values[0]
	z_values = df["pos_z"].to_numpy(dtype=float)
	y_zeros = np.zeros_like(elapsed_time)

	line = ax.plot(elapsed_time, y_zeros, z_values, color="#1f77b4", linewidth=2.0, label="pos_z")[0]
	scatter = ax.scatter(elapsed_time, y_zeros, z_values, c=elapsed_time, cmap="viridis", s=8, alpha=0.8)

	ax.set_title("3D Z Position Over Time")
	ax.set_xlabel("Elapsed time (s)")
	ax.set_ylabel("Dummy axis")
	ax.set_zlabel("Z (m)")
	ax.set_ylim(-0.5, 0.5)
	fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7, label="elapsed time (s)")
	ax.legend(loc="upper left")
	fig.tight_layout()
	return fig


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize position and quaternion orientation data from CSV.")
	parser.add_argument("--csv", required=True, type=Path, help="Input CSV with position and quaternion columns")
	parser.add_argument(
		"--sample-step",
		type=int,
		default=25,
		help="Plot orientation axes every N samples in the 3D view",
	)
	parser.add_argument(
		"--arrow-length",
		type=float,
		default=0.08,
		help="Length of the orientation axes drawn in the 3D view",
	)
	parser.add_argument(
		"--no-rpy",
		action="store_true",
		help="Skip the separate roll/pitch/yaw time-series plots",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory to save plots; defaults to a rot_visualizer_outputs folder next to the CSV",
	)
	parser.add_argument(
		"--show",
		action="store_true",
		help="Also open the figures interactively after saving",
	)
	parser.add_argument(
		"--no-gif",
		action="store_true",
		help="Skip GIF generation and only save static plots",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	df = load_motion_csv(args.csv)
	output_dir = args.output_dir or args.csv.parent / "rot_visualizer_outputs"
	output_dir.mkdir(parents=True, exist_ok=True)

	print("Loaded", len(df), "samples from", args.csv)
	print(df[["roll", "pitch", "yaw"]].head())

	trajectory_fig, _, _, _ = build_trajectory_figure(df, sample_step=args.sample_step, arrow_length=args.arrow_length)
	save_or_show_figure(trajectory_fig, output_dir / f"{args.csv.stem}_trajectory_3d.png", args.show)
	position_fig = build_position_figure(df)
	save_or_show_figure(position_fig, output_dir / f"{args.csv.stem}_position_xyz.png", args.show)
	rpy_fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
	time_values = df["time"].to_numpy(dtype=float) if "time" in df.columns else np.arange(len(df))
	angles_deg = np.degrees(df[["roll", "pitch", "yaw"]].to_numpy(dtype=float))
	axes[0].plot(time_values, angles_deg[:, 0], color="#d62728")
	axes[0].set_ylabel("Roll (deg)")
	axes[0].grid(True, alpha=0.3)
	axes[1].plot(time_values, angles_deg[:, 1], color="#2ca02c")
	axes[1].set_ylabel("Pitch (deg)")
	axes[1].grid(True, alpha=0.3)
	axes[2].plot(time_values, angles_deg[:, 2], color="#1f77b4")
	axes[2].set_ylabel("Yaw (deg)")
	axes[2].set_xlabel("Time")
	axes[2].grid(True, alpha=0.3)
	rpy_fig.suptitle("Roll, Pitch, Yaw from Quaternion")
	rpy_fig.tight_layout()
	save_or_show_figure(rpy_fig, output_dir / f"{args.csv.stem}_rpy.png", args.show)
	z_only_fig = build_z_only_3d_figure(df)
	save_or_show_figure(z_only_fig, output_dir / f"{args.csv.stem}_z_only_3d.png", args.show)
	if not args.no_gif:
		save_trajectory_gif(
			df,
			output_dir / f"{args.csv.stem}_trajectory.gif",
			sample_step=args.sample_step,
			arrow_length=args.arrow_length,
		)


if __name__ == "__main__":
	main()
