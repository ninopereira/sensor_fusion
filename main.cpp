#include "kalman_filter.hpp"

#include <cassert>
#include <cstdlib>

class OdometrySensor {
public:
    /**
     * Constructor for the odometry sensor.
     * @param freq update frequency in Hz
     * @param pos_noise_std standard deviation of position noise in meters
     */
    OdometrySensor(const double freq, const double pos_noise_std) : freq_(freq)
    {
        assert(freq > 0.0 && "OdometrySensor: freq must be positive");

        // convert to noise covariance in SI units (m^2)
        pos_noise_cov_ = pos_noise_std * pos_noise_std; // variance in m^2
    }

    double GetFreq() const { return freq_; }

    double GetPosNoiseCov() const { return pos_noise_cov_; }

private:
    double freq_;          ///< update frequency in Hz
    double pos_noise_cov_; ///< position noise covariance in m^2
};

class ImuSensor {
public:
    /**
     * Constructor for the IMU sensor.
     * @param freq update frequency in Hz
     * @param acc_noise_density accelerometer noise density in µg/√Hz
     */
    ImuSensor(const double freq, const double acc_noise_density) : freq_(freq)
    {
        assert(freq > 0.0 && "ImuSensor: freq must be positive");

        //convert to noise density in SI units (m/s^2)/sqrt(Hz)
        const double acc_noise_density_si = (acc_noise_density *  9.80665) / 1e6; // convert µg/√Hz to m/s^2/√Hz
        psd_ = acc_noise_density_si * acc_noise_density_si; // power spectral density (PSD) in (m/s^2)^2/Hz
    }

    double GetFreq() const { return freq_; }

    double GetPSD() const { return psd_; }

private:
    double freq_;                  ///< update frequency in Hz
    double psd_;                   ///< power spectral density
};

class Robot{
public:
    /**
        Provide the initial state of the robot [x, y, vel_x, vel_y], the update frequencies and
        the noise characteristics of the IMU and odometry sensors.
        The Kalman filter will be initialized with these parameters.
        */
    Robot(const Vec& initial_state, const ImuSensor& imu_sensor, const OdometrySensor& odom_sensor)
        : state_(initial_state), kf_(BuildKalmanFilter(imu_sensor, odom_sensor, initial_state))
    {
    }

    /**
     * Update the robot's state with new IMU data.
     * @param imu_data accelerometer measurements [ax, ay]
     */
    void UpdateIMU(const Vec& imu_data)
    {
        kf_.predict(imu_data);
    }

    /**
     * Update the robot's state with new odometry data.
     * @param odom_data position measurements [x, y]
     */
    void UpdateOdometry(const Vec& odom_data)
    {
        kf_.update(odom_data);
    }

    /**
     * Get the current state estimate of the robot.
     * @return state vector [x, y, vel_x, vel_y]
     */
    const Vec& GetState() const
    {
        return kf_.state();
    }

private:
    // Build the Kalman filter from the sensors' update rates and noise characteristics.
    static KalmanFilter BuildKalmanFilter(const ImuSensor& imu_sensor, const OdometrySensor& odom_sensor,
                                    const Vec& initial_state)
    {
        // from imu_hz and odo_hz, we can compute the time step dt for the Kalman filter
        const double dt_imu = 1.0 / imu_sensor.GetFreq();

        // the state transition matrix F
        Mat F = (Mat(4, 4) << 1, 0, dt_imu, 0,
                                0, 1, 0, dt_imu,
                                0, 0, 1, 0,
                                0, 0, 0, 1).finished();

        // the measurement matrix H (from wheel odometry combined with orientation)
        Mat H = (Mat(2, 4) << 1, 0, 0, 0,
                                0, 1, 0, 0).finished();

        // the control input matrix B maps a raw accelerometer reading u = [ax, ay]
        // onto the state update over one IMU step, via constant-acceleration kinematics:
        //   pos += vel*dt + 0.5*a*dt^2
        //   vel += a*dt
        Mat B = (Mat(4, 2) << 0.5*dt_imu*dt_imu, 0,
                                0,                0.5*dt_imu*dt_imu,
                                dt_imu,           0,
                                0,                dt_imu).finished();

        // define the process noise covariance Q (from the IMU's accelerometer)
        // accel_var is the discrete per-sample noise variance, derived from the continuous PSD.
        // Note: we assume the accelerometer noise is white and Gaussian and independent in x and y axes.
        // Also, we assume that the update rate is constant, otherwise the process noise covariance matrix
        // would need to be adjusted for variable dt.
        const double psd = imu_sensor.GetPSD();
        const double accel_var = psd / dt_imu;
        Mat Q = accel_var * (B * B.transpose());

        // define the measurement noise covariance R based on the odometry noise characteristics
        const double pos_noise_cov = odom_sensor.GetPosNoiseCov();
        Mat R = (Mat(2, 2) <<
            pos_noise_cov, 0,
            0, pos_noise_cov
        ).finished();

        // define the state covariance matrix P
        // let's assume an initial std of 0.5m for position and 0.2m/s for velocity
        Mat P = (Mat(4, 4) << 0.5*0.5, 0, 0, 0,
                                0, 0.5*0.5, 0, 0,
                                0, 0, 0.2*0.2, 0,
                                0, 0, 0, 0.2*0.2).finished();

        return KalmanFilter(F, H, Q, R, P, initial_state, B);
    }

    Vec state_; // State vector [x, y, vel_x, vel_y]
    KalmanFilter kf_; // Kalman filter instance
};

int main()
{
    // define an IMU sensor with a noise density of 150 µg/√Hz
    const ImuSensor imu(1000.0, 150.0); // freq = 1000 Hz, noise density = 150 µg/√Hz
    const OdometrySensor odom(10.0, 0.01); // freq = 10 Hz, position noise std = 0.01 m

    // define a robot with initial state [x, y, vel_x, vel_y], IMU, and odometry sensors
    auto wheelies = Robot(Vec::Zero(4), imu, odom);

    wheelies.UpdateIMU(Vec::Zero(2)); // Simulate an IMU update with zero acceleration [ax, ay]

    // Note: we assume the odometry reading here is already processed to give the displacement in the world frame,
    // and that the robot's orientation is known.
    wheelies.UpdateOdometry(Vec::Zero(2)); // Simulate an odometry update
}
