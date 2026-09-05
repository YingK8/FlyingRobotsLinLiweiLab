// stereo.py refine(mode="image") with the analytic Jacobian (control/theory.md 19.12),
// plus the projection helpers it uses: _rim_shape, _project_ideal, _distort_points,
// the centre-cal displacement cache.
#include "pmw.h"

#include <opencv2/calib3d.hpp>

#include <stdexcept>

namespace pmw {

namespace {

// Rim offsets from the centre for one normal: r * (cos phi u + sin phi v).
MatrixXd rim_shape(const Vector3d& normal, double radius, const VectorXd& phis) {
    auto [u, v] = tangent_basis(normal);
    MatrixXd out(phis.size(), 3);
    for (int i = 0; i < phis.size(); ++i) {
        double c = std::cos(phis[i]), s = std::sin(phis[i]);
        for (int k = 0; k < 3; ++k) out(i, k) = radius * (c * u[k] + s * v[k]);
    }
    return out;
}

// _project_ideal: pinhole pixels, the visibility mask, the camera-frame points and the
// guarded depth.
void project_ideal(const MatrixXd& pts_world, const Camera& cam, MatrixXd& px, std::vector<bool>& seen,
                   MatrixXd& pc, VectorXd& z) {
    int n = (int)pts_world.rows();
    pc.resize(n, 3);
    px.resize(n, 2);
    z.resize(n);
    seen.assign(n, true);
    double fx = cam.K(0, 0), fy = cam.K(1, 1), cx = cam.K(0, 2), cy = cam.K(1, 2);
    for (int i = 0; i < n; ++i) {
        Vector3d d = pts_world.row(i).transpose() - cam.T;
        // (X - T) @ R
        for (int k = 0; k < 3; ++k) pc(i, k) = d[0] * cam.R(0, k) + d[1] * cam.R(1, k) + d[2] * cam.R(2, k);
        double zz = pc(i, 2);
        bool behind = zz < 1e-6;
        if (behind) { zz = 1.0; seen[i] = false; }
        z[i] = zz;
        px(i, 0) = fx * pc(i, 0) / zz + cx;
        px(i, 1) = fy * pc(i, 1) / zz + cy;
    }
}

// _distort_points: ideal pinhole pixels back to raw sensor pixels through projectPoints.
MatrixXd distort_points(const MatrixXd& pts, const Camera& cam) {
    if (!cam.has_dist()) return pts;
    int n = (int)pts.rows();
    double fx = cam.K(0, 0), fy = cam.K(1, 1), cx = cam.K(0, 2), cy = cam.K(1, 2);
    std::vector<cv::Point3d> obj(n);
    for (int i = 0; i < n; ++i) obj[i] = cv::Point3d((pts(i, 0) - cx) / fx, (pts(i, 1) - cy) / fy, 1.0);
    std::vector<cv::Point2d> img;
    cv::Mat zero = cv::Mat::zeros(3, 1, CV_64F);
    cv::projectPoints(obj, zero, zero, cam.Kcv, cam.distcv, img);
    MatrixXd out(n, 2);
    for (int i = 0; i < n; ++i) { out(i, 0) = img[i].x; out(i, 1) = img[i].y; }
    return out;
}

void clip_far(MatrixXd& pts, double far) {
    for (int i = 0; i < pts.size(); ++i) pts(i) = std::min(far, std::max(-far, pts(i)));
}

Matrix3d predict_image_conic(const Vector3d& c_w, const Vector3d& n_w, const Camera& cam, double radius) {
    auto [c_cam, n_cam] = cam.to_camera(c_w, n_w);
    Matrix3d q = cone_from_circle(c_cam, n_cam, radius);
    return cam.Kinv.transpose() * q * cam.Kinv;
}

}  // namespace

std::optional<Refinement> refine_image(const std::vector<cv::Mat>& maps, const std::vector<Camera>& cams,
                                       const Vector3d& c0_in, const Vector3d& n0_in, const CentreCal& cc,
                                       const Config& cfg, const Vector3d& reference, RefineDebug* dbg) {
    const int n = cfg.ring_samples;
    const int nv = (int)maps.size();
    const int total = n * nv;
    const double radius = cfg.radius_mm;
    const double far = cfg.far_px;
    auto [t1, t2] = tangent_basis(n0_in);
    Vector3d n0 = unit(n0_in);
    Vector3d c0 = c0_in;
    VectorXd phis(n);
    for (int i = 0; i < n; ++i) phis[i] = i * (2.0 * M_PI / n);

    auto unpack = [&](const VectorXd& p, Vector3d& c, Vector3d& nrm) {
        c = p.head<3>();
        nrm = unit(n0 + p[3] * t1 + p[4] * t2);
    };

    // The centre-cal displacement, cached on a quantised pose per camera (theory.md 16.9).
    const bool use_cc = !cc.identity();
    using Key = std::array<long long, 6>;
    std::vector<std::map<Key, std::optional<Vector2d>>> disp_cache(nv);
    auto displacement = [&](const Vector3d& c, const Vector3d& nrm, int vi) -> std::optional<Vector2d> {
        Key key;
        for (int k = 0; k < 3; ++k) key[k] = (long long)pyround(c[k] / cfg.cal_disp_tol_mm);
        for (int k = 0; k < 3; ++k) key[3 + k] = (long long)pyround(nrm[k] / cfg.cal_disp_tol_n);
        auto& cache = disp_cache[vi];
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        std::optional<Vector2d> d;
        try {
            const Camera& cam = cams[vi];
            Ellipse ideal = normalise_ellipse(ellipse_from_conic(predict_image_conic(c, nrm, cam, radius)));
            // _predict_ellipse with tilt_cal identity and this centre_cal.
            Ellipse corr = ideal;
            auto [c_cam, n_cam] = cam.to_camera(c, nrm);
            auto dir = projected_axis_dir(c_cam, n_cam, cam.K);
            if (dir) {
                double tilt = line_angle_deg(n_cam, Vector3d(0.0, 0.0, 1.0));
                double shift = cc.offset(tilt) * corr.major;
                corr.cx = corr.cx + shift * (*dir)[0];
                corr.cy = corr.cy + shift * (*dir)[1];
            }
            d = Vector2d(corr.cx - ideal.cx, corr.cy - ideal.cy);
        } catch (const std::domain_error&) {
            d = std::nullopt;
        }
        cache[key] = d;
        return d;
    };

    auto evidence = [&](const VectorXd& p, bool record = false) {
        Vector3d c, nrm;
        unpack(p, c, nrm);
        MatrixXd rim = rim_shape(nrm, radius, phis);
        rim.rowwise() += c.transpose();
        VectorXd out(total);
        for (int vi = 0; vi < nv; ++vi) {
            MatrixXd px, pc;
            VectorXd z;
            std::vector<bool> seen;
            project_ideal(rim, cams[vi], px, seen, pc, z);
            if (use_cc) {
                auto d = displacement(c, nrm, vi);
                if (d) px.rowwise() += d->transpose();
            }
            clip_far(px, far);
            MatrixXd raw = distort_points(px, cams[vi]);
            for (int i = 0; i < n; ++i)
                if (!seen[i]) { raw(i, 0) = -far; raw(i, 1) = -far; }
            // Row-major interleaved (x, y) for the remap.
            Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> xy = raw;
            if (record && dbg) dbg->pts0.push_back(raw);
            sample_map(maps[vi], xy.data(), n, out.data() + vi * n);
        }
        return out;
    };

    double ref;
    VectorXd p0(5);
    p0 << c0[0], c0[1], c0[2], 0.0, 0.0;
    try {
        VectorXd e0 = evidence(p0, true);
        std::vector<double> v(e0.data(), e0.data() + e0.size());
        ref = percentile_linear(v, cfg.ref_percentile);
        if (dbg) dbg->e0 = e0;
    } catch (const std::domain_error&) {
        return std::nullopt;
    }
    ref = std::max(ref, 1e-6);
    double f_scale = std::sqrt(ref / 2.0);
    double sqrt_ref = std::sqrt(ref);

    if (dbg) { dbg->ref = ref; dbg->f_scale = f_scale; }
    auto residual = [&](const VectorXd& p) {
        if (dbg) dbg->trace.push_back(p);
        VectorXd e = evidence(p);
        for (int i = 0; i < e.size(); ++i) e[i] = std::sqrt(std::max(ref - e[i], 0.0));
        return e;
    };

    auto jac = [&](const VectorXd& p) {
        Vector3d c, nrm;
        unpack(p, c, nrm);
        MatrixXd rim = rim_shape(nrm, radius, phis);
        rim.rowwise() += c.transpose();
        // D: (n, 5, 3) world-space d(rim)/d(param). Centre columns are the identity.
        std::vector<MatrixXd> D(5, MatrixXd::Zero(n, 3));
        for (int i = 0; i < 3; ++i) D[i].col(i).setOnes();
        MatrixXd base = rim_shape(nrm, radius, phis);
        for (int j = 3; j < 5; ++j) {
            double h = cfg.jac_rel_step * (p[j] >= 0 ? 1.0 : -1.0) * std::max(std::abs(p[j]), 1.0);
            VectorXd q = p;
            q[j] += h;
            Vector3d cq, nq;
            unpack(q, cq, nq);
            D[j] = (rim_shape(nq, radius, phis) - base) / h;
        }
        MatrixXd out = MatrixXd::Zero(total, 5);
        const double hp = cfg.jac_distort_step_px, hg = cfg.jac_grad_step_px;
        for (int vi = 0; vi < nv; ++vi) {
            const Camera& cam = cams[vi];
            MatrixXd px, pc;
            VectorXd z;
            std::vector<bool> seen;
            project_ideal(rim, cam, px, seen, pc, z);
            if (use_cc) {
                auto d = displacement(c, nrm, vi);
                if (d) px.rowwise() += d->transpose();
            }
            clip_far(px, far);
            // d(ideal pixel)/d(param) through the camera frame: dc = D @ R.
            MatrixXd du(n, 5), dv(n, 5);
            for (int j = 0; j < 5; ++j) {
                MatrixXd dc = D[j] * cam.R;
                for (int i = 0; i < n; ++i) {
                    double zi = 1.0 / z[i];
                    du(i, j) = cam.K(0, 0) * (dc(i, 0) - pc(i, 0) * dc(i, 2) * zi) * zi;
                    dv(i, j) = cam.K(1, 1) * (dc(i, 1) - pc(i, 1) * dc(i, 2) * zi) * zi;
                }
            }
            // d(raw pixel)/d(ideal pixel), differenced in one call over 3n points.
            MatrixXd stack(3 * n, 2);
            stack.topRows(n) = px;
            stack.middleRows(n, n) = px;
            stack.middleRows(n, n).col(0).array() += hp;
            stack.bottomRows(n) = px;
            stack.bottomRows(n).col(1).array() += hp;
            MatrixXd sd = distort_points(stack, cam);
            MatrixXd pd = sd.topRows(n);
            for (int i = 0; i < n; ++i)
                if (!seen[i]) { pd(i, 0) = -far; pd(i, 1) = -far; }
            // Five reads in one remap: pd, pd +- (hg, 0), pd +- (0, hg).
            Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> xy(5 * n, 2);
            xy.topRows(n) = pd;
            xy.middleRows(n, n) = pd;     xy.middleRows(n, n).col(0).array() += hg;
            xy.middleRows(2 * n, n) = pd; xy.middleRows(2 * n, n).col(0).array() -= hg;
            xy.middleRows(3 * n, n) = pd; xy.middleRows(3 * n, n).col(1).array() += hg;
            xy.bottomRows(n) = pd;        xy.bottomRows(n).col(1).array() -= hg;
            VectorXd g(5 * n);
            sample_map(maps[vi], xy.data(), 5 * n, g.data());
            // The Python Jacobian's gradient, residual and live mask are computed in
            // FLOAT32 -- `sample_map` returns float32 and a Python float is weak under
            // NEP 50 -- while `residual` is float64. Mirrored exactly: one rounding
            // difference here is amplified by 1/(2r) on near-zero residual samples.
            const float two_hg = (float)(2.0 * hg);
            const float ref32 = (float)ref;
            const float live_floor = (float)(cfg.jac_min_resid_frac * sqrt_ref);
            for (int i = 0; i < n; ++i) {
                double d_du_x = (sd(n + i, 0) - sd(i, 0)) / hp, d_du_y = (sd(n + i, 1) - sd(i, 1)) / hp;
                double d_dv_x = (sd(2 * n + i, 0) - sd(i, 0)) / hp, d_dv_y = (sd(2 * n + i, 1) - sd(i, 1)) / hp;
                float g0 = (float)g[i], g1 = (float)g[n + i], g2 = (float)g[2 * n + i];
                float g3 = (float)g[3 * n + i], g4 = (float)g[4 * n + i];
                double gx = (g1 - g2) / two_hg;
                double gy = (g3 - g4) / two_hg;
                float r = std::sqrt(std::max(ref32 - g0, 0.0f));
                bool live = r > live_floor;
                double denom = 2.0f * r;
                for (int j = 0; j < 5; ++j) {
                    double ddu = d_du_x * du(i, j) + d_dv_x * dv(i, j);
                    double ddv = d_du_y * du(i, j) + d_dv_y * dv(i, j);
                    double dE = gx * ddu + gy * ddv;
                    out(vi * n + i, j) = live ? -dE / denom : 0.0;
                }
            }
        }
        return out;
    };

    if (dbg) {
        dbg->J0 = jac(p0);
        for (const auto& x : dbg->x_eval) { dbg->f_eval.push_back(residual(x)); dbg->J_eval.push_back(jac(x)); }
        dbg->trace.clear();
    }
    TrfResult sol;
    try {
        sol = trf_cauchy(residual, jac, p0, cfg.refine_tol, cfg.refine_tol, cfg.refine_tol,
                         cfg.max_refine_iter * 6, f_scale);
    } catch (const std::domain_error&) {
        return std::nullopt;
    }
    Vector3d c, nrm;
    unpack(sol.x, c, nrm);
    VectorXd ev = evidence(sol.x);
    Refinement r;
    r.center = c;
    r.normal = orient(nrm, reference);
    r.rms_px = std::sqrt(sol.f.squaredNorm() / (double)sol.f.size());
    r.n_iter = sol.nfev;
    r.converged = sol.status > 0;
    r.samples.resize(nv, n);
    for (int vi = 0; vi < nv; ++vi)
        for (int i = 0; i < n; ++i) r.samples(vi, i) = ev[vi * n + i];
    r.evidence = ev.mean();
    return r;
}

}  // namespace pmw
