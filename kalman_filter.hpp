#pragma once

#include <Eigen/Dense>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

/// Linear (discrete-time) Kalman filter: x_{k+1} = F x_k + B u_k + w_k,  z_k = H x_k + v_k.
class KalmanFilter {
public:
    /// F, H are required. Q, P default to identity; x0 defaults to zero; B is optional.
    /// Measurement noise covariance R is not fixed here: it's supplied per sensor at each
    /// update()/updateDelta() call, since different sensors (e.g. odometry vs. optical flow)
    /// carry different measurement noise.
    KalmanFilter(const Mat& F,          // state transition matrix
                 const Mat& H,          // measurement matrix
                 const Mat& Q = Mat(),  // process noise covariance
                 const Mat& P = Mat(),  // state covariance
                 const Vec& x0 = Vec(), // initial state
                 const Mat& B = Mat())  // control input matrix
        : n_(F.cols()),
          F_(F), H_(H), B_(B),
          Q_(Q.size() ? Q : Mat::Identity(n_, n_)),
          P_(P.size() ? P : Mat::Identity(n_, n_)),
          x_(x0.size() ? x0 : Vec::Zero(n_)) {}

    /// H*x0, the reference value a fresh delta-measuring sensor should start from (see
    /// updateDelta). Exposed so each sensor can seed its own reference at construction time.
    Vec initialMeasurement() const { return H_ * x_; }

    /// Propagate state and covariance one time step, optionally applying control input u (needs B).
    const Vec& predict(const Vec& u = Vec()) {
        x_ = F_ * x_;
        if (B_.size() && u.size()) x_ += B_ * u;
        P_ = F_ * P_ * F_.transpose() + Q_;
        return x_;
    }

    // Correct state and covariance with an absolute measurement z (z ~= H*x), using the
    // measurement noise covariance R for the sensor that produced z.
    void update(const Vec& z, const Mat& R) {
        correct(z - H_ * x_, R);
    }

    // Correct state and covariance with an incremental measurement delta, i.e. the change in
    // H*x since THIS SENSOR's previous update() / updateDelta() call (e.g. wheel-odometry or
    // optical-flow displacement since that sensor's own last reading), using the measurement
    // noise covariance R for the sensor that produced delta.
    //
    // z_ref, in/out, owned by the caller: H*x at this sensor's own last correction - the
    // reference delta is measured against. Rather than accumulating deltas independently of the
    // filter, it's continuously reset to the filter's own corrected estimate so drift doesn't
    // compound on its own. It must be a SEPARATE Vec per delta-reporting sensor: sharing one
    // reference across two sensors that interleave (e.g. odometry and optical flow both landing
    // on the same timestamp) lets one sensor's correction clobber the other's reference point,
    // silently discarding however much time has passed since that other sensor's own last
    // reading and effectively feeding it its raw, uncorrected delta as if starting from zero.
    void updateDelta(const Vec& delta, const Mat& R, Vec& z_ref) {
        correct(delta - (H_ * x_ - z_ref), R);
        z_ref = H_ * x_;
    }

    const Vec& state() const { return x_; }
    const Mat& covariance() const { return P_; }

private:
    // Correct state and covariance given innovation y and the measurement noise covariance R of
    // the sensor that produced it, using the Joseph-form covariance update. It explicitly
    // includes R and preserves symmetry and positive semi-definiteness better than the simpler
    // form. The Joseph form is typically used when numerical stability and covariance
    // consistency matter.
    void correct(const Vec& y, const Mat& R) {
        const Mat S = R + H_ * P_ * H_.transpose();
        const Mat K = P_ * H_.transpose() * S.inverse();

        x_ += K * y;

        const Mat I_KH = Mat::Identity(n_, n_) - K * H_;
        P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    }

    Eigen::Index n_;     ///< state dimension
    Mat F_, H_, B_;      ///< system, measurement, control matrices
    Mat Q_, P_;          ///< process noise, state covariance
    Vec x_;              ///< state estimate
};
