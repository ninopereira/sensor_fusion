#!/usr/bin/env python3
"""
Generate synthetic IMU, wheel-odometry, and optical-flow sensor readings for testing the
sensor-fusion Kalman filter in main.cpp / kalman_filter.hpp.

Ground truth: a 2D robot on a flat floor, following a unicycle motion model
(forward speed v, heading theta, yaw rate omega) through four phases:
  0-2 s: accelerate straight ahead,    a_tan = 0.5 m/s^2,  omega = 0
  2-4 s: turn left at constant speed,  a_tan = 0,          omega = 0.4 rad/s
  4-6 s: cruise straight on new heading, a_tan = 0,        omega = 0
  6-8 s: decelerate back to rest,      a_tan = -0.5 m/s^2, omega = 0

Four files are written:
  data/imu_readings.csv          - t_s, ax_mps2, ay_mps2, gz_radps  @ 1000 Hz
                                    (matches ImuSensor(1000, 150))
  data/odom_readings.csv         - t_s, dx_m, dy_m                  @ 10 Hz
                                    (matches OdometrySensor(10, 0.01))
  data/optical_flow_readings.csv - t_s, dx_m, dy_m                  @ 100 Hz
                                    (matches OpticalFlowSensor(100, 0.001))
  data/ground_truth.csv          - t_s, x_m, y_m, theta_rad, v_mps  @ 10 Hz (for validation/plotting)

Frame conventions (important - these are simplifications the current filter relies on):
  - ax, ay are the WORLD-frame linear acceleration (not body-frame). The filter's
    B/F matrices update [x, y, vx, vy] directly from [ax, ay] with no rotation, so
    this generator applies the theta-dependent rotation itself (ax = d/dt(v cos theta),
    ay = d/dt(v sin theta)) rather than emitting raw accelerometer-frame values.
  - gz (yaw rate) is included for realism (a real IMU has a gyro too) but is NOT
    consumed by the current filter - the state has no theta. It's there for a future
    "rotation handled outside the filter" step, per README.
  - dx, dy (both odometry and optical flow) are WORLD-frame position deltas (true
    displacement + noise), matching the filter's H (absolute position measurement), not
    body-frame sensor output. A real wheel encoder or optical-flow sensor reports
    body-frame displacement that still needs a heading estimate to rotate into world
    frame before hitting this filter's H; that rotation step doesn't exist in this
    repo, so it's folded into the generator instead.

Noise is generated to match the noise models already used in main.cpp:
  - IMU accel noise std per sample:  sigma_a = sqrt(psd / dt_imu)
      where psd = (acc_noise_density_ug_sqrtHz * 9.80665e-6)^2   (see ImuSensor ctor)
  - IMU gyro noise std per sample: a plausible fixed value (no gyro param exists on
    ImuSensor yet, so this isn't derived from it - see GYRO_NOISE_STD_RADPS below).
  - Odometry displacement noise std: pos_noise_std_m, applied directly to dx/dy.
  - Optical-flow displacement noise std: pos_noise_std_m, applied directly to dx/dy, same
    shape as odometry but NOT the same magnitude: optical flow reads 10x more often than
    odometry, so the true displacement between readings is ~10x smaller too. Reusing
    odometry's 0.01 m std at that rate leaves individual readings almost pure noise
    (signal-to-noise ratio ~1.4x, vs. odometry's ~7.7x) and the filter's estimate visibly
    diverges from ground truth. OPTFLOW_POS_NOISE_STD_M is instead odometry's std scaled by
    the update-interval ratio (0.01 m * dt_optflow/dt_odom = 0.001 m), which restores a
    comparable SNR and must stay in sync with OpticalFlowSensor's ctor arg in main.cpp.
"""

import csv
import math
import random
from pathlib import Path

random.seed(42)  # reproducible output

# --- sensor parameters: must match the ImuSensor/OdometrySensor/OpticalFlowSensor
# --- construction in main.cpp ---
IMU_FREQ_HZ = 1000.0
IMU_NOISE_DENSITY_UG_SQRT_HZ = 150.0
ODOM_FREQ_HZ = 10.0
ODOM_POS_NOISE_STD_M = 0.01
OPTFLOW_FREQ_HZ = 100.0
OPTFLOW_POS_NOISE_STD_M = 0.001
GYRO_NOISE_STD_RADPS = 0.01  # plausible low-cost MEMS gyro noise; not modeled by ImuSensor yet

DT_IMU = 1.0 / IMU_FREQ_HZ
DT_ODOM = 1.0 / ODOM_FREQ_HZ
DT_OPTFLOW = 1.0 / OPTFLOW_FREQ_HZ
IMU_TICKS_PER_ODOM_TICK = round(IMU_FREQ_HZ / ODOM_FREQ_HZ)
IMU_TICKS_PER_OPTFLOW_TICK = round(IMU_FREQ_HZ / OPTFLOW_FREQ_HZ)

# accelerometer per-sample noise std, derived the same way main.cpp derives Q's accel_var
acc_noise_density_si = IMU_NOISE_DENSITY_UG_SQRT_HZ * 9.80665 / 1e6  # (m/s^2)/sqrt(Hz)
psd = acc_noise_density_si ** 2                                     # (m/s^2)^2 / Hz
IMU_ACCEL_NOISE_STD = math.sqrt(psd / DT_IMU)                       # m/s^2, per sample

DURATION_S = 8.0


def motion_profile(t: float) -> tuple[float, float]:
    """Ground-truth (a_tan, omega) for the unicycle model: accelerate, turn, cruise, decelerate."""
    if t < 2.0:
        return 0.5, 0.0
    elif t < 4.0:
        return 0.0, 0.4
    elif t < 6.0:
        return 0.0, 0.0
    else:
        return -0.5, 0.0


def simulate() -> tuple[
    list[tuple[float, float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float]],
    list[tuple[float, float, float, float, float]],
]:
    n_imu = round(DURATION_S / DT_IMU)
    t = 0.0
    x = y = theta = v = 0.0
    imu_rows: list[tuple[float, float, float, float]] = []
    odom_true: list[tuple[float, float, float, float, float]] = []  # (t, x, y, theta, v) at each odom tick
    optflow_true: list[tuple[float, float, float, float, float]] = []  # same, at each optical-flow tick

    for i in range(n_imu):
        a_tan, omega = motion_profile(t)

        # world-frame linear acceleration the accelerometer would sense: d/dt(v*cos theta, v*sin theta)
        ax = a_tan * math.cos(theta) - v * omega * math.sin(theta)
        ay = a_tan * math.sin(theta) + v * omega * math.cos(theta)
        meas_ax = ax + random.gauss(0.0, IMU_ACCEL_NOISE_STD)
        meas_ay = ay + random.gauss(0.0, IMU_ACCEL_NOISE_STD)
        meas_gz = omega + random.gauss(0.0, GYRO_NOISE_STD_RADPS)
        imu_rows.append((t, meas_ax, meas_ay, meas_gz))

        # integrate ground truth (simple Euler, fine at 1 kHz for this smooth profile)
        x += v * math.cos(theta) * DT_IMU
        y += v * math.sin(theta) * DT_IMU
        theta += omega * DT_IMU
        v += a_tan * DT_IMU

        t += DT_IMU
        if (i + 1) % IMU_TICKS_PER_ODOM_TICK == 0:
            odom_true.append((t, x, y, theta, v))
        if (i + 1) % IMU_TICKS_PER_OPTFLOW_TICK == 0:
            optflow_true.append((t, x, y, theta, v))

    def deltas_from_truth(
        truth: list[tuple[float, float, float, float, float]], noise_std: float
    ) -> list[tuple[float, float, float]]:
        rows: list[tuple[float, float, float]] = []
        prev_x, prev_y = 0.0, 0.0
        for t_sample, cx, cy, _theta, _v in truth:
            dx = (cx - prev_x) + random.gauss(0.0, noise_std)
            dy = (cy - prev_y) + random.gauss(0.0, noise_std)
            rows.append((t_sample, dx, dy))
            prev_x, prev_y = cx, cy
        return rows

    odom_rows = deltas_from_truth(odom_true, ODOM_POS_NOISE_STD_M)
    optflow_rows = deltas_from_truth(optflow_true, OPTFLOW_POS_NOISE_STD_M)

    return imu_rows, odom_rows, optflow_rows, odom_true


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(f"{v:.6f}" for v in row)


def main():
    imu_rows, odom_rows, optflow_rows, odom_true = simulate()
    write_csv(Path("data/imu_readings.csv"), ["t_s", "ax_mps2", "ay_mps2", "gz_radps"], imu_rows)
    write_csv(Path("data/odom_readings.csv"), ["t_s", "dx_m", "dy_m"], odom_rows)
    write_csv(Path("data/optical_flow_readings.csv"), ["t_s", "dx_m", "dy_m"], optflow_rows)
    write_csv(Path("data/ground_truth.csv"), ["t_s", "x_m", "y_m", "theta_rad", "v_mps"], odom_true)
    print(f"wrote {len(imu_rows)} IMU rows -> data/imu_readings.csv")
    print(f"wrote {len(odom_rows)} odometry rows -> data/odom_readings.csv")
    print(f"wrote {len(optflow_rows)} optical-flow rows -> data/optical_flow_readings.csv")
    print(f"wrote {len(odom_true)} ground-truth rows -> data/ground_truth.csv")


if __name__ == "__main__":
    main()
