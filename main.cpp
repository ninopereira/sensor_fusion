#include "kalman_filter.hpp"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
     * Wheel odometry naturally reports incremental displacement since the previous reading, so
     * the raw delta is handed straight to the filter, which fuses it against its own last
     * corrected position estimate rather than an independently-accumulated one.
     * @param odom_delta incremental world-frame displacement since the previous odometry
     *                   reading [dx, dy], in meters
     */
    void UpdateOdometry(const Vec& odom_delta)
    {
        kf_.updateDelta(odom_delta);
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

struct ImuSample {
    double t;   ///< timestamp in seconds
    double ax;  ///< x acceleration in m/s^2
    double ay;  ///< y acceleration in m/s^2
    // gz (yaw rate) is present in the CSV but unused: this filter has no orientation state.
};

struct OdomSample {
    double t;   ///< timestamp in seconds
    double dx;  ///< x displacement since the previous odometry reading, in meters
    double dy;  ///< y displacement since the previous odometry reading, in meters
};

// Split a CSV line into its comma-separated fields.
static std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

// Open path or print an error and terminate.
static std::ifstream OpenCsvOrDie(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "error: could not open " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    return file;
}

static std::vector<ImuSample> ReadImuCsv(const std::string& path)
{
    std::ifstream file = OpenCsvOrDie(path);

    std::vector<ImuSample> samples;
    std::string line;
    std::getline(file, line); // discard header (t_s,ax_mps2,ay_mps2,gz_radps)
    while (std::getline(file, line)) {
        const auto fields = SplitCsvLine(line);
        samples.push_back(ImuSample{std::stod(fields[0]), std::stod(fields[1]), std::stod(fields[2])});
    }
    return samples;
}

static std::vector<OdomSample> ReadOdomCsv(const std::string& path)
{
    std::ifstream file = OpenCsvOrDie(path);

    std::vector<OdomSample> samples;
    std::string line;
    std::getline(file, line); // discard header (t_s,dx_m,dy_m)
    while (std::getline(file, line)) {
        const auto fields = SplitCsvLine(line);
        samples.push_back(OdomSample{std::stod(fields[0]), std::stod(fields[1]), std::stod(fields[2])});
    }
    return samples;
}

int main()
{
    // define an IMU sensor with a noise density of 150 µg/√Hz
    const ImuSensor imu(1000.0, 150.0); // freq = 1000 Hz, noise density = 150 µg/√Hz
    const OdometrySensor odom(10.0, 0.01); // freq = 10 Hz, position noise std = 0.01 m

    // define a robot with initial state [x, y, vel_x, vel_y], IMU, and odometry sensors
    auto wheelies = Robot(Vec::Zero(4), imu, odom);

    const std::vector<ImuSample> imu_samples = ReadImuCsv("data/imu_readings.csv");
    const std::vector<OdomSample> odom_samples = ReadOdomCsv("data/odom_readings.csv");

    // Walk both sample streams in timestamp order, feeding each reading to the filter at the
    // moment it actually occurred, rather than assuming a fixed IMU-tick/odometry-tick ratio.
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < imu_samples.size() || j < odom_samples.size()) {
        const bool take_imu = (j >= odom_samples.size()) ||
                               (i < imu_samples.size() && imu_samples[i].t <= odom_samples[j].t);
        if (take_imu) {
            const ImuSample& s = imu_samples[i];
            Vec u(2);
            u << s.ax, s.ay;
            wheelies.UpdateIMU(u);
            ++i;
        } else {
            const OdomSample& s = odom_samples[j];
            Vec delta(2);
            delta << s.dx, s.dy;
            wheelies.UpdateOdometry(delta);
            ++j;

            const Vec& state = wheelies.GetState();
            std::cout << "t=" << s.t << "s  x=" << state(0) << "  y=" << state(1)
                      << "  vx=" << state(2) << "  vy=" << state(3) << "\n";
        }
    }
}
