#!/usr/bin/env python3
"""
Run the compiled sensor_fusion binary and plot its estimated trajectory against
data/ground_truth.csv.

Usage:
    python3 plot_results.py

If data/*.csv or the sensor_fusion binary are missing, this regenerates/rebuilds them
first (via generate_sensor_data.py and the g++ command from README.md), then runs the
binary, parses its stdout state estimates (the "t=... x=... y=... vx=... vy=..." lines
Robot::GetState()-derived output in main.cpp prints at each odometry update), and plots
the x-y trajectory plus position error over time against ground truth.

Output: trajectory_plot.png in the repo root.
"""

import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write straight to a file; no display server required
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "sensor_fusion"
GROUND_TRUTH_CSV = ROOT / "data" / "ground_truth.csv"
GENERATOR = ROOT / "generate_sensor_data.py"
BUILD_CMD = [
    "g++", "-std=c++17", "-I/usr/include/eigen3", "-Wall", "-Wextra", "-O2",
    "main.cpp", "-o", "sensor_fusion",
]

# matches lines like: "t=7.9s  x=4.9638  y=2.89761  vx=0.0422152  vy=0.0252074"
LINE_RE = re.compile(
    r"t=(?P<t>[-\d.eE]+)s\s+x=(?P<x>[-\d.eE]+)\s+y=(?P<y>[-\d.eE]+)\s+"
    r"vx=(?P<vx>[-\d.eE]+)\s+vy=(?P<vy>[-\d.eE]+)"
)


def ensure_data() -> None:
    if not GROUND_TRUTH_CSV.exists():
        print("data/ground_truth.csv missing, generating sensor data...")
        subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, check=True)


def ensure_binary() -> None:
    if not BINARY.exists():
        print("sensor_fusion binary missing, building...")
        subprocess.run(BUILD_CMD, cwd=ROOT, check=True)


def run_filter() -> tuple[list[float], list[float], list[float]]:
    """Run the compiled binary and parse its per-odometry-update state estimates."""
    result = subprocess.run([str(BINARY)], cwd=ROOT, check=True, capture_output=True, text=True)
    t: list[float] = []
    x: list[float] = []
    y: list[float] = []
    for line in result.stdout.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        t.append(float(m["t"]))
        x.append(float(m["x"]))
        y.append(float(m["y"]))
    if not t:
        raise RuntimeError(f"no state estimates parsed from sensor_fusion output:\n{result.stdout}")
    return t, x, y


def read_ground_truth() -> tuple[list[float], list[float], list[float]]:
    t: list[float] = []
    x: list[float] = []
    y: list[float] = []
    with GROUND_TRUTH_CSV.open() as f:
        next(f)  # header
        for line in f:
            t_s, x_m, y_m, _theta, _v = line.strip().split(",")
            t.append(float(t_s))
            x.append(float(x_m))
            y.append(float(y_m))
    return t, x, y


def main() -> None:
    ensure_data()
    ensure_binary()

    est_t, est_x, est_y = run_filter()
    true_t, true_x, true_y = read_ground_truth()

    if len(est_t) != len(true_t):
        print(f"warning: {len(est_t)} estimates vs {len(true_t)} ground-truth rows; "
              f"truncating to the shorter series for comparison", file=sys.stderr)
        n = min(len(est_t), len(true_t))
        est_t, est_x, est_y = est_t[:n], est_x[:n], est_y[:n]
        true_t, true_x, true_y = true_t[:n], true_x[:n], true_y[:n]

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(12, 5))

    ax_traj.plot(true_x, true_y, label="ground truth", color="tab:blue", linewidth=2)
    ax_traj.plot(est_x, est_y, label="filter estimate", color="tab:orange",
                 linestyle="--", marker="o", markersize=3)
    ax_traj.scatter([true_x[0]], [true_y[0]], color="green", zorder=5, label="start")
    ax_traj.scatter([true_x[-1]], [true_y[-1]], color="red", zorder=5, label="end")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectory: ground truth vs filter estimate")
    ax_traj.axis("equal")
    ax_traj.legend()
    ax_traj.grid(True, alpha=0.3)

    # Euclidean position error at each odometry tick (both series share those timestamps)
    errors = [((ex - tx) ** 2 + (ey - ty) ** 2) ** 0.5
              for ex, ey, tx, ty in zip(est_x, est_y, true_x, true_y)]
    ax_err.plot(est_t, errors, color="tab:red")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("position error (m)")
    ax_err.set_title("Estimate error vs time")
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = ROOT / "trajectory_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot -> {out_path}")
    print(f"final position error: {errors[-1]:.4f} m  "
          f"(mean: {sum(errors) / len(errors):.4f} m, max: {max(errors):.4f} m)")


if __name__ == "__main__":
    main()
