# Sensor fusion

This project uses a simple Kalman Filter for a vehicle moving on a flat surface (xy plane).
It separates the fast IMU prediction from the slower odometry correction while keeping the model linear.
It is assumed that orientation is handled separately, and the vehicle does not to perform sharp turns.
For more complex rotational dynamics, use an EKF (Extended Kalman Filter) or UKF (Unscented Kalman Filter).

## Overview

A Kalman filter can fuse IMU and wheel odometry to estimate:
- position
- velocity

### Typical update rates

- IMU: 100 Hz or higher (often 1 kHz in high-rate systems)
- Wheel odometry: 10–20 Hz

> The IMU is usually the fast prediction source, while wheel odometry is the lower-rate correction source.

## Build instructions

Requires the Eigen3 headers (`libeigen3-dev` on Debian/Ubuntu). Compile with:

```bash
g++ -std=c++17 -I/usr/include/eigen3 -Wall -Wextra -O2 main.cpp -o sensor_fusion
```

## Running the simulation

`main()` reads two CSV files of synthetic sensor data and drives the filter through them in
timestamp order. Generate the data, then run the binary from the repo root (it opens
`data/imu_readings.csv` and `data/odom_readings.csv` via relative paths):

```bash
python3 generate_sensor_data.py   # writes data/imu_readings.csv, data/odom_readings.csv, data/ground_truth.csv
./sensor_fusion
```

`generate_sensor_data.py` needs only the Python 3 standard library (no pip install). It
simulates a short trajectory (accelerate, turn, cruise, decelerate) and writes noisy IMU/odometry
readings plus a `ground_truth.csv` of the true state, so the filter's printed output (state
estimate at each odometry update) can be checked against ground truth. See the script's docstring
for the exact motion profile and noise model.

## Kalman filter structure

- **Prediction step** uses IMU acceleration to propagate the state.
- **Update step** uses wheel odometry velocity to correct the state.

## Assumptions

- Orientation is not estimated by the Kalman Filter.
- The filter assumes planar motion on a flat floor.
- Vehicle rotation is handled outside the filter (for example, from IMU gyroscope data).

## State definition

A common planar Kalman filter state is:

```text
x = [x, y, vx, vy]'
```

This is a typical choice for a flat-ground vehicle because it directly represents 2D position and 2D velocity.
IMU acceleration is used to predict motion, and wheel odometry velocity is used to correct the predicted velocity.

## State transition model

For a discrete timestep `dt`, use a constant-velocity process model with acceleration input.

### State transition matrix

```text
F = [1 0 dt 0
     0 1 0 dt
     0 0 1  0
     0 0 0  1]
```

### Control matrix for IMU acceleration

With IMU acceleration input `u = [ax, ay]'`:

```text
B = [0.5*dt*dt  0
     0          0.5*dt*dt
     dt         0
     0          dt]
```

### Prediction step

```text
x_k = F x_{k-1} + B u_k
P_k = F P_{k-1} F^T + Q
```

### The control matrix B

`B` maps a raw accelerometer reading `u = [ax, ay]` onto the state update over one IMU step
(`dt_imu = 1/1000 s`), using ordinary constant-acceleration kinematics:

```text
pos += vel*dt + 0.5*a*dt^2
vel += a*dt
```

`B u_k` injects the measured acceleration into the predicted position and velocity each step.

### The matrix Q contains the IMU noise

`Q` represents only the uncertainty of IMU readings.

The accelerometer's own white noise has power spectral density `psd` (in `(m/s^2)^2/Hz`). Converting that continuous-time PSD to a discrete per-sample noise variance at the IMU's sample rate gives:

```text
accel_var = psd / dt_imu
```

so the process noise covariance is:

```text
Q = B * (accel_var * I) * B^T
```

Note: the x/y accelerometer channels are independent, so `Q_u = accel_var * I` is diagonal.

## Measurement model for wheel odometry

Assuming odometry provides planar velocities in world coordinates:

```text
H = [0 0 1 0
     0 0 0 1]
```

The measurement is:

```text
z = [vx, vy]^T
```

The update step corrects the predicted velocity.

## Notes

This project includes a Kalman filter implementation in C++ inspired by a Python implementation from Zaur Fataliyev (https://github.com/zziz/kalman-filter).
