// scipy.optimize._lsq.trf.trf_no_bounds with tr_solver="exact", loss="cauchy" and
// x_scale="jac", ported statement for statement (scipy 1.18). A faithful port is
// the point: the iterate sequence then matches the Python reference to rounding, so
// parity can be held at 1e-6 mm rather than "the same optimum, roughly"
// (control/theory.md 19.7 makes reproducibility the instrument).
//
// One deliberate omission: scipy evaluates the Jacobian once more at the point it
// terminates on, for the `optimality` field nobody reads. Skipped; x, fun and nfev
// are unaffected.
#include "pmw.h"

namespace pmw {

namespace {

struct Loss {
    // rho0 = f_scale^2 log1p(z), rho1 = 1/(1+z), rho2 = -1/(1+z)^2 / f_scale^2, z = (f/f_scale)^2.
    double fs;
    double cost(const VectorXd& f) const {
        double s = 0.0;
        for (int i = 0; i < f.size(); ++i) {
            double z = (f[i] / fs) * (f[i] / fs);
            s += std::log1p(z);
        }
        return 0.5 * fs * fs * s;
    }
    // scale_for_robust_loss_function, in place.
    void scale(MatrixXd& J, VectorXd& f) const {
        for (int i = 0; i < f.size(); ++i) {
            double z = (f[i] / fs) * (f[i] / fs);
            double t = 1.0 + z;
            double rho1 = 1.0 / t;
            double rho2 = -1.0 / (t * t) / (fs * fs);
            double js = rho1 + 2.0 * rho2 * f[i] * f[i];
            if (js < EPS) js = EPS;
            js = std::sqrt(js);
            f[i] *= rho1 / js;
            J.row(i) *= js;
        }
    }
};

void jac_scale(const MatrixXd& J, VectorXd& scale, VectorXd& scale_inv, bool first) {
    VectorXd s_inv = J.colwise().norm();
    if (first) {
        for (int i = 0; i < s_inv.size(); ++i)
            if (s_inv[i] == 0) s_inv[i] = 1;
    } else {
        s_inv = s_inv.cwiseMax(scale_inv);
    }
    scale_inv = s_inv;
    scale = s_inv.cwiseInverse();
}

// solve_lsq_trust_region (More's method on one SVD).
VectorXd solve_lsq_trust_region(int n, int m, const VectorXd& uf, const VectorXd& s, const MatrixXd& V,
                                double Delta, double& alpha_io, double rtol = 0.01, int max_iter = 10) {
    auto phi_and_derivative = [&](double alpha, const VectorXd& suf, double& phi, double& phi_prime) {
        VectorXd denom = s.array().square() + alpha;
        double p_norm = (suf.array() / denom.array()).matrix().norm();
        phi = p_norm - Delta;
        phi_prime = -(suf.array().square() / denom.array().cube()).sum() / p_norm;
    };
    VectorXd suf = s.cwiseProduct(uf);
    bool full_rank = false;
    if (m >= n) {
        double threshold = EPS * m * s[0];
        full_rank = s[s.size() - 1] > threshold;
    }
    if (full_rank) {
        VectorXd p = -V * (uf.array() / s.array()).matrix();
        if (p.norm() <= Delta) {
            alpha_io = 0.0;
            return p;
        }
    }
    double alpha_upper = suf.norm() / Delta;
    double alpha_lower = 0.0;
    if (full_rank) {
        double phi, phi_prime;
        phi_and_derivative(0.0, suf, phi, phi_prime);
        alpha_lower = -phi / phi_prime;
    }
    double alpha;
    if (!full_rank && alpha_io == 0)
        alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
    else
        alpha = alpha_io;
    for (int it = 0; it < max_iter; ++it) {
        if (alpha < alpha_lower || alpha > alpha_upper)
            alpha = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
        double phi, phi_prime;
        phi_and_derivative(alpha, suf, phi, phi_prime);
        if (phi < 0) alpha_upper = alpha;
        double ratio = phi / phi_prime;
        alpha_lower = std::max(alpha_lower, alpha - ratio);
        alpha -= (phi + Delta) * ratio / Delta;
        if (std::abs(phi) < rtol * Delta) break;
    }
    VectorXd denom = s.array().square() + alpha;
    VectorXd p = -V * (suf.array() / denom.array()).matrix();
    p *= Delta / p.norm();
    alpha_io = alpha;
    return p;
}

}  // namespace

TrfResult trf_cauchy(const std::function<VectorXd(const VectorXd&)>& fun,
                     const std::function<MatrixXd(const VectorXd&)>& jac, const VectorXd& x0,
                     double ftol, double xtol, double gtol, int max_nfev, double f_scale) {
    Loss loss{f_scale};
    VectorXd x = x0;
    VectorXd f = fun(x);
    if (!f.allFinite()) throw std::domain_error("Residuals are not finite in the initial point.");
    VectorXd f_true = f;
    int nfev = 1;
    MatrixXd J = jac(x);
    int m = (int)J.rows(), n = (int)J.cols();
    double cost = loss.cost(f);
    loss.scale(J, f);
    VectorXd g = J.transpose() * f;
    VectorXd scale, scale_inv;
    jac_scale(J, scale, scale_inv, true);
    double Delta = x0.cwiseProduct(scale_inv).norm();
    if (Delta == 0) Delta = 1.0;
    double alpha = 0.0;
    int status = -1;   // scipy's None
    double actual_reduction = -1, step_norm = 0, cost_new = 0;
    VectorXd x_new, f_new;

    while (true) {
        double g_norm = g.lpNorm<Eigen::Infinity>();
        if (g_norm < gtol) status = 1;
        if (status != -1 || nfev == max_nfev) break;

        const VectorXd& d = scale;
        VectorXd g_h = d.cwiseProduct(g);
        MatrixXd J_h = J * d.asDiagonal();
        Eigen::JacobiSVD<MatrixXd> svd(J_h, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const MatrixXd& U = svd.matrixU();
        const VectorXd& s = svd.singularValues();
        const MatrixXd& V = svd.matrixV();
        VectorXd uf = U.transpose() * f;

        actual_reduction = -1;
        while (actual_reduction <= 0 && nfev < max_nfev) {
            VectorXd step_h = solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha);
            VectorXd Js = J_h * step_h;
            double predicted_reduction = -(0.5 * Js.dot(Js) + g_h.dot(step_h));
            VectorXd step = d.cwiseProduct(step_h);
            x_new = x + step;
            f_new = fun(x_new);
            nfev += 1;
            double step_h_norm = step_h.norm();
            if (!f_new.allFinite()) {
                Delta = 0.25 * step_h_norm;
                continue;
            }
            cost_new = loss.cost(f_new);
            actual_reduction = cost - cost_new;
            // update_tr_radius
            double ratio;
            if (predicted_reduction > 0) ratio = actual_reduction / predicted_reduction;
            else if (predicted_reduction == 0 && actual_reduction == 0) ratio = 1;
            else ratio = 0;
            double Delta_new = Delta;
            if (ratio < 0.25) Delta_new = 0.25 * step_h_norm;
            else if (ratio > 0.75 && step_h_norm > 0.95 * Delta) Delta_new = Delta * 2.0;
            step_norm = step.norm();
            // check_termination
            bool ftol_ok = actual_reduction < ftol * cost && ratio > 0.25;
            bool xtol_ok = step_norm < xtol * (xtol + x.norm());
            if (ftol_ok && xtol_ok) status = 4;
            else if (ftol_ok) status = 2;
            else if (xtol_ok) status = 3;
            if (status != -1) break;
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0) {
            x = x_new;
            f = f_new;
            f_true = f;
            cost = cost_new;
            if (status != -1) break;   // scipy would evaluate one more Jacobian here
            J = jac(x);
            loss.scale(J, f);
            g = J.transpose() * f;
            jac_scale(J, scale, scale_inv, false);
        } else {
            step_norm = 0;
            actual_reduction = 0;
        }
    }
    if (status == -1) status = 0;
    return {x, f_true, nfev, status};
}

}  // namespace pmw
