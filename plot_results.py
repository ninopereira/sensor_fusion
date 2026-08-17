#!/usr/bin/env python3
"""
Run the compiled sensor_fusion binary - once fully fused, and once per individual
correction source - and plot the resulting trajectories against data/ground_truth.csv.

Usage:
    python3 plot_results.py                       # all four sources
    python3 plot_results.py --no-odom              # drop the odometry-only baseline
    python3 plot_results.py --no-odom --no-imu     # fused vs. optical-flow-only only
    python3 plot_results.py --no-all --odom --optflow --no-imu
                                                    # only the two single-sensor baselines

Each data source (--all/--odom/--optflow/--imu, one per SensorMode in main.cpp) can be
toggled on/off independently via --<name>/--no-<name>; all four are on by default. Only
the enabled sources are actually run through the binary. See --help for the full list.

If data/*.csv or the sensor_fusion binary are missing, this regenerates/rebuilds them
first (via generate_sensor_data.py and the g++ command from README.md), then runs the
binary once per enabled SensorMode (see main.cpp: all|odom|optflow|imu), parses its
stdout state estimates (the "t=... x=... y=... vx=... vy=..." lines Robot::GetState()
-derived output in main.cpp prints on a fixed 10 Hz clock, one run per mode), and plots
the x-y trajectories plus position error over time against ground truth.

Output: trajectory_plot.png in the repo root.
"""

import argparse
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

# Sensor modes to run and plot, in main.cpp's SensorMode terms (IMU prediction always runs;
# this picks which correction source(s) feed the filter). Each entry is
# (mode, label, plot style) - the fully-fused "all" run is the headline series; the rest are
# single-sensor baselines, drawn faded/dashed so they read as context rather than competing
# with the main estimate.
RUNS = [
    ("all", "filter estimate (odom + optical flow)",
     dict(color="tab:orange", linestyle="--", marker="o", markersize=3, linewidth=1.5, alpha=1.0)),
    ("odom", "odometry only", dict(color="tab:green", linestyle=":", linewidth=1.3, alpha=0.6)),
    ("optflow", "optical flow only", dict(color="tab:purple", linestyle=":", linewidth=1.3, alpha=0.6)),
    ("imu", "IMU only (dead reckoning)", dict(color="tab:gray", linestyle=":", linewidth=1.3, alpha=0.6)),
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


def run_filter(mode: str) -> tuple[list[float], list[float], list[float]]:
    """Run the compiled binary in the given SensorMode and parse its state estimates."""
    result = subprocess.run([str(BINARY), mode], cwd=ROOT, check=True, capture_output=True, text=True)
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
        raise RuntimeError(f"no state estimates parsed from sensor_fusion {mode!r} output:\n{result.stdout}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sensor_fusion trajectories/error against ground truth, "
                     "with each data source toggleable.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for mode, label, _ in RUNS:
        parser.add_argument(
            f"--{mode}", action=argparse.BooleanOptionalAction, default=True,
            help=f"include '{label}'",
        )
    return parser.parse_args()


def align(t: list[float], x: list[float], y: list[float],
          true_t: list[float], true_x: list[float], true_y: list[float]
          ) -> tuple[list[float], list[float], list[float]]:
    """Truncate a run's series to the shorter of it and ground truth, positionally.

    All SensorMode runs share main.cpp's fixed 10 Hz output clock (see kOutputIntervalS),
    the same clock ground_truth.csv is sampled at, so this is a same-length no-op in the
    normal case; it only kicks in defensively if a run's data doesn't fully cover [0, 8]s.
    """
    n = min(len(t), len(true_t))
    if n != len(t):
        print(f"warning: {len(t)} estimates vs {len(true_t)} ground-truth rows; "
              f"truncating to the shorter series for comparison", file=sys.stderr)
    return t[:n], x[:n], y[:n]


def main() -> None:
    args = parse_args()
    enabled_runs = [(mode, label, style) for mode, label, style in RUNS if getattr(args, mode)]
    if not enabled_runs:
        print("error: all sources disabled (--no-all --no-odom --no-optflow --no-imu), "
              "nothing to plot", file=sys.stderr)
        sys.exit(1)

    ensure_data()
    ensure_binary()

    true_t, true_x, true_y = read_ground_truth()

    runs = {}
    for mode, label, style in enabled_runs:
        t, x, y = run_filter(mode)
        t, x, y = align(t, x, y, true_t, true_x, true_y)
        runs[mode] = (t, x, y, label, style)

    fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(12, 5))

    ax_traj.plot(true_x, true_y, label="ground truth", color="tab:blue", linewidth=2, zorder=4)
    for mode, _, _ in enabled_runs:
        t, x, y, label, style = runs[mode]
        ax_traj.plot(x, y, label=label, **style)
    ax_traj.scatter([true_x[0]], [true_y[0]], color="green", zorder=5, label="start")
    ax_traj.scatter([true_x[-1]], [true_y[-1]], color="red", zorder=5, label="end")
    ax_traj.set_xlabel("x (m)")
    ax_traj.set_ylabel("y (m)")
    ax_traj.set_title("Trajectory: ground truth vs filter estimate")
    ax_traj.axis("equal")
    ax_traj.legend(fontsize=8)
    ax_traj.grid(True, alpha=0.3)

    # Euclidean position error at each output tick (every run shares ground truth's timestamps)
    n = min(len(true_t), *(len(runs[mode][0]) for mode, _, _ in enabled_runs))
    for mode, _, _ in enabled_runs:
        t, x, y, label, style = runs[mode]
        errors = [((ex - tx) ** 2 + (ey - ty) ** 2) ** 0.5
                  for ex, ey, tx, ty in zip(x[:n], y[:n], true_x[:n], true_y[:n])]
        err_style = {k: v for k, v in style.items() if k != "marker"}
        ax_err.plot(t[:n], errors, label=label, **err_style)
        print(f"[{label:38s}] final: {errors[-1]:.4f} m, mean: {sum(errors) / len(errors):.4f} m, "
              f"max: {max(errors):.4f} m")
    ax_err.set_xlabel("time (s)")
    ax_err.set_ylabel("position error (m)")
    ax_err.set_title("Estimate error vs time")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = ROOT / "trajectory_plot.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved plot -> {out_path}")


if __name__ == "__main__":
    main()
