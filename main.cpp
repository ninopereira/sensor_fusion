#include "kalman_filter.hpp"

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

class OpticalFlowSensor {
public:
    /**
     * Constructor for the optical flow sensor.
     * @param freq update frequency in Hz
     * @param pos_noise_std standard deviation of position noise in meters
     */
    OpticalFlowSensor(const double freq, const double pos_noise_std) : freq_(freq)
    {
        assert(freq > 0.0 && "OpticalFlowSensor: freq must be positive");

        // convert to noise covariance in SI units (m^2)
        pos_noise_cov_ = pos_noise_std * pos_noise_std; // variance in m^2
        // define the measurement noise covariance R based on the optical flow noise characteristics
        r_ = (Mat(2, 2) <<
            pos_noise_cov_, 0,
            0, pos_noise_cov_
        ).finished();
    }

    double GetFreq() const { return freq_; }

    double GetPosNoiseCov() const { return pos_noise_cov_; }

    Mat GetMeasurementNoiseCov() const { return r_; }

private:
    double freq_;          ///< update frequency in Hz
    double pos_noise_cov_; ///< position noise covariance in m^2
    Mat r_;                 ///< measurement noise covariance matrix
};


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
        // define the measurement noise covariance R based on the odometry noise characteristics
        r_ = (Mat(2, 2) <<
            pos_noise_cov_, 0,
            0, pos_noise_cov_
        ).finished();
    }

    double GetFreq() const { return freq_; }

    double GetPosNoiseCov() const { return pos_noise_cov_; }

    Mat GetMeasurementNoiseCov() const { return r_; }

private:
    double freq_;          ///< update frequency in Hz
    double pos_noise_cov_; ///< position noise covariance in m^2
    Mat r_;                 ///< measurement noise covariance matrix

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
    Robot(const Vec& initial_state, const OpticalFlowSensor& optical_flow_sensor, const ImuSensor& imu_sensor, const OdometrySensor& odom_sensor)
        : state_(initial_state), kf_(BuildKalmanFilter(imu_sensor, initial_state)),
          odom_R_(odom_sensor.GetMeasurementNoiseCov()),
          optical_flow_R_(optical_flow_sensor.GetMeasurementNoiseCov()),
          odom_z_ref_(kf_.initialMeasurement()),
          optical_flow_z_ref_(kf_.initialMeasurement())
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
     * Wheel odometry naturally reports incremental displacement since the previous odometry
     * reading, so the raw delta is handed straight to the filter (with odometry's own
     * measurement noise covariance and its own reference point, odom_z_ref_), which fuses it
     * against its own last corrected position estimate rather than an independently-accumulated
     * one. odom_z_ref_ must stay separate from optical_flow_z_ref_ below: odometry's delta is
     * only meaningful relative to odometry's own last reading, not optical flow's.
     * @param odom_delta incremental world-frame displacement since the previous odometry
     *                   reading [dx, dy], in meters
     */
    void UpdateOdometry(const Vec& odom_delta)
    {
        kf_.updateDelta(odom_delta, odom_R_, odom_z_ref_);
    }

    /**
     * Update the robot's state with new optical flow data.
     * Like wheel odometry, optical flow reports incremental displacement since the previous
     * optical flow reading, so the raw delta is handed straight to the filter, tagged with
     * optical flow's own (typically much lower-noise) measurement covariance and its own
     * reference point, optical_flow_z_ref_ (see UpdateOdometry for why it can't share odom's).
     * @param opt_flow_delta incremental world-frame displacement since the previous optical flow
     *                       reading [dx, dy], in meters
     */
    void UpdateOpticalFlow(const Vec& opt_flow_delta)
    {
        kf_.updateDelta(opt_flow_delta, optical_flow_R_, optical_flow_z_ref_);
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
    // Build the Kalman filter from the IMU's update rate and noise characteristics.
    // Measurement noise covariance R is not baked in here: it's supplied per sensor at each
    // UpdateOdometry()/UpdateOpticalFlow() call (see odom_R_, optical_flow_R_ below).
    static KalmanFilter BuildKalmanFilter(const ImuSensor& imu_sensor, const Vec& initial_state)
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

        // define the state covariance matrix P
        // let's assume an initial std of 0.5m for position and 0.2m/s for velocity
        Mat P = (Mat(4, 4) << 0.5*0.5, 0, 0, 0,
                                0, 0.5*0.5, 0, 0,
                                0, 0, 0.2*0.2, 0,
                                0, 0, 0, 0.2*0.2).finished();

        return KalmanFilter(F, H, Q, P, initial_state, B);
    }

    Vec state_; // State vector [x, y, vel_x, vel_y]
    KalmanFilter kf_; // Kalman filter instance
    Mat odom_R_;         // odometry sensor's measurement noise covariance
    Mat optical_flow_R_; // optical flow sensor's measurement noise covariance
    // Per-sensor updateDelta() reference points (see kf_.updateDelta doc for why these must be
    // separate: sharing one between two interleaved delta-reporting sensors corrupts both).
    Vec odom_z_ref_;
    Vec optical_flow_z_ref_;
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

struct OpticalFlowSample {
    double t;   ///< timestamp in seconds
    double dx;  ///< x displacement since the previous optical flow reading, in meters
    double dy;  ///< y displacement since the previous optical flow reading, in meters
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

static std::vector<OpticalFlowSample> ReadOpticalFlowCsv(const std::string& path)
{
    std::ifstream file = OpenCsvOrDie(path);

    std::vector<OpticalFlowSample> samples;
    std::string line;
    std::getline(file, line); // discard header (t_s,dx_m,dy_m)
    while (std::getline(file, line)) {
        const auto fields = SplitCsvLine(line);
        samples.push_back(OpticalFlowSample{std::stod(fields[0]), std::stod(fields[1]), std::stod(fields[2])});
    }
    return samples;
}

// How much of the sensor suite actually corrects the filter. IMU prediction always runs in every
// mode (it's the process model, not a correction source); this only gates which correction(s)
// are applied, so e.g. "imu" mode is pure dead-reckoning drift, useful as a baseline to plot
// alongside the fully-fused estimate.
struct SensorMode {
    bool use_odom;
    bool use_optflow;
};

static SensorMode ParseSensorMode(const std::string& mode)
{
    if (mode == "all") return {true, true};
    if (mode == "odom") return {true, false};
    if (mode == "optflow") return {false, true};
    if (mode == "imu") return {false, false};
    std::cerr << "error: unknown mode '" << mode << "' (expected all|odom|optflow|imu)\n";
    std::exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    const SensorMode mode = ParseSensorMode(argc > 1 ? argv[1] : "all");

    const OpticalFlowSensor optical_flow(100.0, 0.001); // freq = 100 Hz, position noise std = 0.001 m
    // define an IMU sensor with a noise density of 150 µg/√Hz
    const ImuSensor imu(1000.0, 150.0); // freq = 1000 Hz, noise density = 150 µg/√Hz
    const OdometrySensor odom(10.0, 0.01); // freq = 10 Hz, position noise std = 0.01 m

    // define a robot with initial state [x, y, vel_x, vel_y], IMU, and odometry sensors
    auto wheelies = Robot(Vec::Zero(4), optical_flow, imu, odom);

    const std::vector<ImuSample> imu_samples = ReadImuCsv("data/imu_readings.csv");
    const std::vector<OdomSample> odom_samples = ReadOdomCsv("data/odom_readings.csv");
    const std::vector<OpticalFlowSample> optflow_samples = ReadOpticalFlowCsv("data/optical_flow_readings.csv");

    // Print the state at a fixed cadence (matching ground_truth.csv's 10 Hz) rather than
    // whenever a correction happens to fire: correction cadence varies by mode (10 Hz for
    // odometry, 100 Hz for optical flow, never for IMU-only dead reckoning), so a fixed
    // output clock is what makes every mode's trace directly comparable/overlayable.
    constexpr double kOutputIntervalS = 0.1;
    double next_output_t = kOutputIntervalS;

    // Walk all three sample streams in timestamp order, feeding each reading to the filter at
    // the moment it actually occurred, rather than assuming fixed tick ratios between them.
    std::size_t i = 0;
    std::size_t j = 0;
    std::size_t k = 0;
    while (i < imu_samples.size() || j < odom_samples.size() || k < optflow_samples.size()) {
        const double next_imu_t = i < imu_samples.size() ? imu_samples[i].t : std::numeric_limits<double>::infinity();
        const double next_odom_t = j < odom_samples.size() ? odom_samples[j].t : std::numeric_limits<double>::infinity();
        const double next_optflow_t = k < optflow_samples.size() ? optflow_samples[k].t
                                                                   : std::numeric_limits<double>::infinity();

        double current_t;
        if (next_imu_t <= next_odom_t && next_imu_t <= next_optflow_t) {
            const ImuSample& s = imu_samples[i];
            current_t = s.t;
            Vec u(2);
            u << s.ax, s.ay;
            wheelies.UpdateIMU(u);
            ++i;
        } else if (next_odom_t <= next_optflow_t) {
            const OdomSample& s = odom_samples[j];
            current_t = s.t;
            Vec delta(2);
            delta << s.dx, s.dy;
            if (mode.use_odom) wheelies.UpdateOdometry(delta);
            ++j;
        } else {
            const OpticalFlowSample& s = optflow_samples[k];
            current_t = s.t;
            Vec delta(2);
            delta << s.dx, s.dy;
            if (mode.use_optflow) wheelies.UpdateOpticalFlow(delta);
            ++k;
        }

        // Emit every output tick this step has now caught up to or passed (usually exactly one,
        // but the guard is a while loop in case multiple streams land on the same instant).
        while (current_t + 1e-9 >= next_output_t) {
            const Vec& state = wheelies.GetState();
            std::cout << "t=" << next_output_t << "s  x=" << state(0) << "  y=" << state(1)
                      << "  vx=" << state(2) << "  vy=" << state(3) << "\n";
            next_output_t += kOutputIntervalS;
        }
    }
}
