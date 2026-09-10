// The control core, in C++: the Kalman filter, the model-forward predictor, the velocity
// estimator and the LQR law. Line-for-line ports of the Python they are named after
// (`controller/pose/filter.py`, `controller/control/{predictor,simulate_hover}.py`), kept
// in the same evaluation order so the two agree to rounding.
//
// Same contract as `pmw.h`, and for the same reason: **no numeric literal that is a
// tuning constant lives in this file.** Every number arrives in `ControlConfig`, built by
// `controller/control/native_config.py` from the Python module constants, and the loader
// throws on a missing key so a number cannot silently exist twice.
//
// The Python stays as the reference and `controller/pose/native_parity.py --stage filter
// --stage control` holds them to each other. See `control/theory.md` 27.
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <optional>

namespace pmw {

using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

// Fixed sizes throughout. Everything in the control path is at most 6x6, and a
// heap allocation inside a tick that has a 2 ms deadline is the one thing the native
// binary exists to avoid.
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix36d = Eigen::Matrix<double, 3, 6>;
using Matrix63d = Eigen::Matrix<double, 6, 3>;
using Matrix26d = Eigen::Matrix<double, 2, 6>;
using Vector4d = Eigen::Matrix<double, 4, 1>;

struct ControlConfig {
    // filter.py
    double accel_mm_s2, p0_pos, p0_vel, gate_sigma;
    int max_gated;
    double sigma_normal;
    // predictor.py
    double max_coast_s, gravity, tau_vel_s;
    // simulate_hover.py / hover_controller.json
    double ts, f_hover, g, k_lat;
    double mag_max, freq_min, freq_max, freq_slew_hz_per_s;
    double vel_cutoff_hz;
    Matrix26d K;
    // stereo.py's `suspect_mm`, reused as the scale for measurement-noise inflation.
    double suspect_mm;
};

// ---- filter.py::_ConstantVelocity ------------------------------------------------------
//
// 3-channel constant-velocity Kalman filter, state [p(3), v(3)]. Channels are
// block-diagonal, which is exact here: lateral error comes from the ellipse centroid and
// depth error from its size, and the two are uncorrelated.
class ConstantVelocity {
public:
    ConstantVelocity() = default;
    ConstantVelocity(double accel, double p0_pos, double p0_vel) { init(accel, p0_pos, p0_vel); }

    void init(double accel, double p0_pos, double p0_vel) {
        accel_ = accel;
        x_.setZero();
        P_.setZero();
        for (int i = 0; i < 3; ++i) { P_(i, i) = p0_pos; P_(i + 3, i + 3) = p0_vel; }
        P0_ = P_;
        initialised_ = false;
        n_gated_ = 0;
    }

    void reset() {
        x_.setZero();
        P_ = P0_;
        initialised_ = false;
        n_gated_ = 0;
    }

    void predict(double dt) {
        if (dt <= 0.0) return;
        Matrix6d f = Matrix6d::Identity();
        for (int i = 0; i < 3; ++i) f(i, i + 3) = dt;
        // Continuous white-noise acceleration, discretised exactly. Same grouping as the
        // Python so the rounding matches: s = accel^2, then dt^4/4 etc.
        const double s = accel_ * accel_;
        Matrix6d q = Matrix6d::Zero();
        const double q11 = dt * dt * dt * dt / 4.0 * s;
        const double q12 = dt * dt * dt / 2.0 * s;
        const double q22 = dt * dt * s;
        for (int i = 0; i < 3; ++i) {
            q(i, i) = q11;
            q(i, i + 3) = q12;
            q(i + 3, i) = q12;
            q(i + 3, i + 3) = q22;
        }
        x_ = f * x_;
        P_ = f * P_ * f.transpose() + q;
    }

    //: Fuse a measurement with full 3x3 covariance `r`. Returns `accepted`.
    //:
    //: `gate` rejects a measurement whose innovation is more than that many sigmas from
    //: where the filter expected it, in the innovation's own metric, normalised by the
    //: three degrees of freedom. A rejected frame is NOT a lost frame: the state has
    //: already been predicted forward, so the filter keeps the extrapolation, which is
    //: what a constant-velocity model is for.
    //:
    //: Bounded by `max_gated` consecutive rejections, after which the measurement is
    //: taken regardless. Without that, a manoeuvre the model did not anticipate would be
    //: rejected forever -- the filter growing more confident in its extrapolation every
    //: frame while drifting further from the robot. **A gate that can lock itself on is
    //: worse than no gate**, and this is not optional.
    bool update(const Vector3d& z, const Matrix3d& r, double gate, int max_gated) {
        if (!initialised_) {
            x_.head<3>() = z;
            x_.tail<3>().setZero();
            P_.topLeftCorner<3, 3>() = r;
            initialised_ = true;
            n_gated_ = 0;
            return true;
        }
        Matrix36d h = Matrix36d::Zero();
        h.leftCols<3>() = Matrix3d::Identity();
        const Vector3d y = z - h * x_;
        const Matrix3d s = h * P_ * h.transpose() + r;
        if (gate > 0.0 && n_gated_ < max_gated) {
            // `solve` for the statistic and an explicit inverse for the gain, matching
            // the Python's np.linalg.solve / np.linalg.inv exactly. Swapping one for the
            // other is numerically defensible and breaks parity.
            double d;
            Eigen::FullPivLU<Matrix3d> lu(s);
            if (lu.isInvertible()) {
                d = y.dot(lu.solve(y)) / 3.0;
            } else {
                d = 0.0;
            }
            if (d > gate * gate) {
                n_gated_++;
                return false;
            }
        }
        n_gated_ = 0;
        const Matrix63d k = P_ * h.transpose() * s.inverse();
        x_ = x_ + k * y;
        P_ = (Matrix6d::Identity() - k * h) * P_;
        return true;
    }

    Vector3d value() const { return x_.head<3>(); }
    Vector3d rate() const { return x_.tail<3>(); }
    Matrix3d rate_cov() const { return P_.bottomRightCorner<3, 3>(); }
    bool initialised() const { return initialised_; }
    int n_gated() const { return n_gated_; }
    //: The state advanced `dt` without touching it -- for latency compensation.
    Vector3d peek(double dt) const { return x_.head<3>() + x_.tail<3>() * dt; }

private:
    Vector6d x_ = Vector6d::Zero();
    Matrix6d P_ = Matrix6d::Zero(), P0_ = Matrix6d::Zero();
    double accel_ = 0.0;
    bool initialised_ = false;
    int n_gated_ = 0;
};

// ---- predictor.py::StatePredictor -------------------------------------------------------
//
// Model-forward propagation across vision dropouts. Propagates on the frequency we
// COMMANDED rather than coasting kinematically: z_ddot = g*((f_cmd/f_hat)^2 - 1), clamped
// at -g because thrust cannot be negative. Lateral stays constant-velocity -- there is no
// lateral model worth propagating while `k_lat` is a seed guess (control/theory.md 25).
struct Command {
    double f_cmd;   // Hz, the field frequency commanded this tick
    double f_hat;   // Hz, the loop's CURRENT hover estimate; passed, never cached
};

class StatePredictor {
public:
    StatePredictor() = default;
    void init(const ControlConfig& c) {
        max_coast_s_ = c.max_coast_s;
        g_mm_ = 1000.0 * c.gravity;   // the only unit conversion in here
        tau_vel_s_ = c.tau_vel_s;
        reset();
    }

    void reset() {
        xyz_mm_.setZero();
        vel_mm_s_.setZero();
        t_ = 0.0;
        coast_s_ = 0.0;
        initialised_ = false;
        has_meas_ = false;
    }

    bool stale() const { return coast_s_ > max_coast_s_; }
    bool initialised() const { return initialised_; }
    double coast_s() const { return coast_s_; }
    Vector3d xyz_mm() const { return xyz_mm_; }
    Vector3d vel_mm_s() const { return vel_mm_s_; }
    double t() const { return t_; }

    //: A fix arrived: snap to it and clear the coast.
    Vector3d update(const Vector3d& xyz, double t) {
        estimate_vel(xyz, t);
        xyz_mm_ = xyz;
        t_ = t;
        coast_s_ = 0.0;
        initialised_ = true;
        return xyz_mm_;
    }

    //: A fix arrived with a velocity already estimated (from the Kalman filter). Re-seeds
    //: the differencer, so a later frame without one does not difference across the
    //: frames it was supplied for.
    Vector3d update(const Vector3d& xyz, const Vector3d& vel, double t) {
        vel_mm_s_ = vel;
        xyz_meas_ = xyz;
        t_meas_ = t;
        has_meas_ = true;
        xyz_mm_ = xyz;
        t_ = t;
        coast_s_ = 0.0;
        initialised_ = true;
        return xyz_mm_;
    }

    Vector3d accel_mm_s2(const Command& u) const {
        double az;
        if (u.f_hat <= 0.0) az = -g_mm_;
        else az = g_mm_ * ((u.f_cmd / u.f_hat) * (u.f_cmd / u.f_hat) - 1.0);
        return Vector3d(0.0, 0.0, az < -g_mm_ ? -g_mm_ : az);
    }

    //: Propagate `dt` seconds. `a` is held over the interval, so integrate it exactly.
    Vector3d predict(const Command& u, double dt) {
        if (dt <= 0.0) return xyz_mm_;
        const Vector3d a = accel_mm_s2(u);
        xyz_mm_ = xyz_mm_ + vel_mm_s_ * dt + 0.5 * a * dt * dt;
        vel_mm_s_ = vel_mm_s_ + a * dt;
        t_ += dt;
        coast_s_ += dt;
        return xyz_mm_;
    }

private:
    void estimate_vel(const Vector3d& xyz, double t) {
        const bool had = has_meas_;
        const Vector3d prev = xyz_meas_;
        const double prev_t = t_meas_;
        xyz_meas_ = xyz;
        t_meas_ = t;
        has_meas_ = true;
        if (!had || t <= prev_t) return;
        const double dt = t - prev_t;   // spans the whole gap when a fix follows a coast
        const Vector3d raw = (xyz - prev) / dt;
        const double alpha = dt / (tau_vel_s_ + dt);
        vel_mm_s_ = vel_mm_s_ + alpha * (raw - vel_mm_s_);
    }

    Vector3d xyz_mm_ = Vector3d::Zero(), vel_mm_s_ = Vector3d::Zero();
    Vector3d xyz_meas_ = Vector3d::Zero();
    double t_ = 0.0, coast_s_ = 0.0, t_meas_ = 0.0;
    double max_coast_s_ = 0.0, g_mm_ = 0.0, tau_vel_s_ = 0.0;
    bool initialised_ = false, has_meas_ = false;
};

// ---- simulate_hover.py::VelocityEstimator ----------------------------------------------
//
// Finite difference through a 1-pole IIR. `dt` is the interval the position actually moved
// over, NOT the control period: the loop steps at 500 Hz against a ~100 Hz pose pipeline,
// and between fixes the position handed in came from `StatePredictor`, which advanced it
// by exactly v*dt. Differencing that returns the velocity that produced it -- no
// information, only filter lag. A tick with no new fix passes dt <= 0 and the estimate is
// HELD. `alpha` follows `dt` for the same reason: pinning it to `ts` would re-introduce
// the rate dependence from the other side. See control/theory.md 19.13.
class VelocityEstimator {
public:
    void init(double ts, double cutoff_hz) {
        ts_ = ts;
        tau_ = 1.0 / (2.0 * M_PI * cutoff_hz);
        vel_.setZero();
        has_prev_ = false;
    }
    void reset() { vel_.setZero(); has_prev_ = false; }

    Vector2d update(const Vector2d& pos, double dt, bool have_dt) {
        if (!has_prev_) {
            prev_ = pos;
            has_prev_ = true;
            return vel_;
        }
        if (!have_dt || dt <= 0.0) return vel_;   // no new information: hold
        const Vector2d raw = (pos - prev_) / dt;
        prev_ = pos;
        const double alpha = dt / (dt + tau_);
        vel_ = vel_ + alpha * (raw - vel_);
        return vel_;
    }
    Vector2d value() const { return vel_; }

private:
    Vector2d vel_ = Vector2d::Zero(), prev_ = Vector2d::Zero();
    double ts_ = 0.0, tau_ = 0.0;
    bool has_prev_ = false;
};

// ---- simulate_hover.py::DiscreteHoverController -----------------------------------------
//
// u(k) = u_trim + u_ff(k) - K [x_hat - x_ref ; q]. One instance per lateral axis, both
// driven by row 0 of K, exactly as the Python runner does -- the off-block terms of K are
// ~1e-13, so the axes decouple exactly (control/theory.md 19.13).
class HoverController {
public:
    void init(const ControlConfig& c) {
        K_ = c.K;
        ts_ = c.ts;
        f_hover_ = c.f_hover;
        g_ = c.g;
        k_lat_ = c.k_lat;
        mag_max_ = c.mag_max;
        freq_min_ = c.freq_min;
        freq_max_ = c.freq_max;
        freq_slew_ = c.freq_slew_hz_per_s * c.ts;   // Hz per frame
        est_.init(c.ts, c.vel_cutoff_hz);
        reset();
    }

    void reset() {
        q_.setZero();
        prev_f_field_ = f_hover_;
        est_.reset();
    }

    //: Re-anchor the frequency the loop trims about, without disturbing the integrators.
    //: `_anchor` in the Python runner: the ramp reaches a frequency the design did not
    //: pick, and trimming about the wrong one puts a constant offset into the command.
    void anchor(double f_reached) { prev_f_field_ = f_reached; }

    //: One control step. `ref_*` are the profile's position, velocity and acceleration.
    //: `have_dt` false means no new fix this tick and the rate estimate is held.
    Vector2d step(double x_meas, double z_meas, double dt, bool have_dt,
                  const Vector2d& ref_p, const Vector2d& ref_v, const Vector2d& ref_a) {
        const Vector2d pos(x_meas, z_meas);
        const Vector2d vel = est_.update(pos, dt, have_dt);
        Vector4d err;
        err << pos(0) - ref_p(0), vel(0) - ref_v(0), pos(1) - ref_p(1), vel(1) - ref_v(1);
        // Reference-acceleration feedforward: the exact inverse of B's nonzero entries.
        const Vector2d u_ff(ref_a(0) / (g_ * k_lat_), ref_a(1) * f_hover_ / (2.0 * g_));
        Vector6d xa;
        xa << err, q_;
        const Vector2d u = Vector2d(0.0, f_hover_) + u_ff - K_ * xa;

        const double mag = clamp(u(0), -mag_max_, mag_max_);
        const double f_tgt = clamp(u(1), freq_min_, freq_max_);
        const double f_field = clamp(f_tgt, prev_f_field_ - freq_slew_,
                                     prev_f_field_ + freq_slew_);
        prev_f_field_ = f_field;

        // Conditional-integration anti-windup: integrate only when the UNSATURATED
        // command is in range, or when the error points back toward it. Slew limiting is
        // a transient and must NOT freeze the integrator; only hard clamps do.
        integrate(0, err(0), u(0), -mag_max_, mag_max_);
        integrate(1, err(2), u(1), freq_min_, freq_max_);
        return Vector2d(mag, f_field);
    }

    Vector2d q() const { return q_; }
    double f_hover() const { return f_hover_; }
    double prev_f_field() const { return prev_f_field_; }

private:
    static double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    void integrate(int i, double e, double u_unsat, double lo, double hi) {
        if ((lo <= u_unsat && u_unsat <= hi) || (u_unsat > hi && e > 0.0) ||
            (u_unsat < lo && e < 0.0))
            q_(i) += ts_ * e;
    }

    Matrix26d K_ = Matrix26d::Zero();
    Vector2d q_ = Vector2d::Zero();
    VelocityEstimator est_;
    double ts_ = 0.0, f_hover_ = 0.0, g_ = 0.0, k_lat_ = 0.0;
    double mag_max_ = 0.0, freq_min_ = 0.0, freq_max_ = 0.0, freq_slew_ = 0.0;
    double prev_f_field_ = 0.0;
};

}  // namespace pmw
