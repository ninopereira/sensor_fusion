#pragma once

#include <Eigen/Dense>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

/// Linear (discrete-time) Kalman filter: x_{k+1} = F x_k + B u_k + w_k,  z_k = H x_k + v_k.
class KalmanFilter {
public:
    /// F, H are required. Q, R, P default to identity; x0 defaults to zero; B is optional.
    KalmanFilter(const Mat& F,          // state transition matrix
                 const Mat& H,          // measurement matrix
                 const Mat& Q = Mat(),  // process noise covariance
                 const Mat& R = Mat(),  // measurement noise covariance
                 const Mat& P = Mat(),  // state covariance
                 const Vec& x0 = Vec(), // initial state
                 const Mat& B = Mat())  // control input matrix
        : n_(F.cols()), m_(H.rows()),
          F_(F), H_(H), B_(B),
          Q_(Q.size() ? Q : Mat::Identity(n_, n_)),
          R_(R.size() ? R : Mat::Identity(m_, m_)),
          P_(P.size() ? P : Mat::Identity(n_, n_)),
          x_(x0.size() ? x0 : Vec::Zero(n_)),
          z_ref_(H_ * x_) {}

    /// Propagate state and covariance one time step, optionally applying control input u (needs B).
    const Vec& predict(const Vec& u = Vec()) {
        x_ = F_ * x_;
        if (B_.size() && u.size()) x_ += B_ * u;
        P_ = F_ * P_ * F_.transpose() + Q_;
        return x_;
    }

    // Correct state and covariance with an absolute measurement z (z ~= H*x).
    void update(const Vec& z) {
        correct(z - H_ * x_);
    }

    // Correct state and covariance with an incremental measurement delta, i.e. the change in
    // H*x since the previous update() / updateDelta() call (e.g. wheel-odometry displacement
    // since the last odometry reading). Rather than accumulating deltas independently of the
    // filter, the reference they're measured against (z_ref_) is the filter's own last
    // corrected estimate, so drift is continuously reset to the fused state instead of
    // compounding on its own.
    void updateDelta(const Vec& delta) {
        correct(delta - (H_ * x_ - z_ref_));
        z_ref_ = H_ * x_;
    }

    const Vec& state() const { return x_; }
    const Mat& covariance() const { return P_; }

private:
    // Correct state and covariance given innovation y, using the Joseph-form covariance update.
    // It explicitly includes the measurement noise R and preserves symmetry and positive
    // semi-definiteness better than the simpler form. The Joseph form is typically used when
    // numerical stability and covariance consistency matter.
    void correct(const Vec& y) {
        const Mat S = R_ + H_ * P_ * H_.transpose();
        const Mat K = P_ * H_.transpose() * S.inverse();

        x_ += K * y;

        const Mat I_KH = Mat::Identity(n_, n_) - K * H_;
        P_ = I_KH * P_ * I_KH.transpose() + K * R_ * K.transpose();
    }

    Eigen::Index n_, m_; ///< state and measurement dimensions
    Mat F_, H_, B_;      ///< system, measurement, control matrices
    Mat Q_, R_, P_;      ///< process noise, measurement noise, state covariance
    Vec x_;              ///< state estimate
    Vec z_ref_;           ///< H*x at the last update()/updateDelta(), the reference for updateDelta()
};
