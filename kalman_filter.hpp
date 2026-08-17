#pragma once

#include <Eigen/Dense>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

/// Linear (discrete-time) Kalman filter: x_{k+1} = F x_k + B u_k + w_k,  z_k = H x_k + v_k.
class KalmanFilter {
public:
    /// F, H are required. Q, P default to identity; x0 defaults to zero; B is optional. F here
    /// is used only to size the state (n_ = F.cols()) - it is NOT retained, since predict()
    /// takes F explicitly on every call (see predict()'s doc for why: a caller whose transition
    /// matrix depends on something that changes over time, e.g. a heading estimate, needs to be
    /// able to pass a fresh F each tick, the same way update()/updateDelta() take R explicitly
    /// per sensor rather than fixing one at construction).
    /// Measurement noise covariance R is not fixed here either: it's supplied per sensor at
    /// each update()/updateDelta() call, since different sensors (e.g. odometry vs. optical
    /// flow) carry different measurement noise.
    KalmanFilter(const Mat& F,          // state transition matrix (used only for sizing; see above)
                 const Mat& H,          // measurement matrix
                 const Mat& Q = Mat(),  // process noise covariance
                 const Mat& P = Mat(),  // state covariance
                 const Vec& x0 = Vec(), // initial state
                 const Mat& B = Mat())  // control input matrix
        : n_(F.cols()),
          H_(H), B_(B),
          Q_(Q.size() ? Q : Mat::Identity(n_, n_)),
          P_(P.size() ? P : Mat::Identity(n_, n_)),
          x_(x0.size() ? x0 : Vec::Zero(n_)) {}

    /// H*x0, the reference value a fresh delta-measuring sensor should start from (see
    /// updateDelta). Exposed so each sensor can seed its own reference at construction time.
    Vec initialMeasurement() const { return H_ * x_; }

    /// Propagate state and covariance one time step using transition matrix F, optionally
    /// applying control input u (needs B, fixed at construction). F is taken per call rather
    /// than fixed at construction so a caller can rebuild it each tick when it depends on
    /// something outside the linear state - e.g. a heading estimate used to rotate a
    /// body-frame-constant sensor bias into this filter's world-frame state (see main.cpp's
    /// Robot::BuildF): passing a fresh, correctly-rotated F each tick keeps that genuinely
    /// linear (F varies with a known/given heading, not with the filter's own uncertain state),
    /// unlike coupling heading into the state itself, which would need an EKF.
    const Vec& predict(const Mat& F, const Vec& u = Vec()) {
        x_ = F * x_;
        if (B_.size() && u.size()) x_ += B_ * u;
        P_ = F * P_ * F.transpose() + Q_;
        return x_;
    }

    // Correct state and covariance with an absolute measurement z (z ~= H*x), using the
    // measurement noise covariance R for the sensor that produced z. See correct() for
    // n_correctable.
    void update(const Vec& z, const Mat& R, Eigen::Index n_correctable = -1) {
        correct(z - H_ * x_, R, n_correctable);
    }

    // Correct state and covariance with an incremental measurement delta, i.e. the change in
    // H*x since THIS SENSOR's previous update() / updateDelta() call (e.g. wheel-odometry or
    // optical-flow displacement since that sensor's own last reading), using the measurement
    // noise covariance R for the sensor that produced delta. See correct() for n_correctable.
    //
    // z_ref, in/out, owned by the caller: H*x at this sensor's own last correction - the
    // reference delta is measured against. Rather than accumulating deltas independently of the
    // filter, it's continuously reset to the filter's own corrected estimate so drift doesn't
    // compound on its own. It must be a SEPARATE Vec per delta-reporting sensor: sharing one
    // reference across two sensors that interleave (e.g. odometry and optical flow both landing
    // on the same timestamp) lets one sensor's correction clobber the other's reference point,
    // silently discarding however much time has passed since that other sensor's own last
    // reading and effectively feeding it its raw, uncorrected delta as if starting from zero.
    void updateDelta(const Vec& delta, const Mat& R, Vec& z_ref, Eigen::Index n_correctable = -1) {
        correct(delta - (H_ * x_ - z_ref), R, n_correctable);
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
    //
    // n_correctable (default -1: every state dimension) caps how many LEADING state dimensions
    // this particular measurement is allowed to update, by zeroing the Kalman gain's remaining
    // rows before applying it to both x_ and P_ - equivalent to telling the filter "treat this
    // measurement as uninformative about those states", a legitimate sub-case of the standard
    // equations (not an approximation to them). This exists for weakly/indirectly-observable
    // augmented states (e.g. an IMU bias, see main.cpp's Robot): a measurement whose own
    // observation window is too short for that state's effect on it to be distinguishable from
    // noise can still, left uncapped, perturb its cross-covariance with everything else on pure
    // noise every time it fires - harmless occasionally, but actively destabilizing at high
    // rate. Capping which sensor is allowed to touch that state is the standard remedy.
    void correct(const Vec& y, const Mat& R, Eigen::Index n_correctable = -1) {
        const Mat S = R + H_ * P_ * H_.transpose();
        Mat K = P_ * H_.transpose() * S.inverse();
        if (n_correctable >= 0 && n_correctable < n_) {
            K.bottomRows(n_ - n_correctable).setZero();
        }

        x_ += K * y;

        const Mat I_KH = Mat::Identity(n_, n_) - K * H_;
        P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
    }

    Eigen::Index n_;     ///< state dimension
    Mat H_, B_;          ///< measurement, control matrices
    Mat Q_, P_;          ///< process noise, state covariance
    Vec x_;              ///< state estimate
};
