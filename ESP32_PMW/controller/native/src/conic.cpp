// conic.py, plus the small vector helpers from stereo.py (_unit, _tangent_basis,
// orient, line_angle_deg, projected_axis_dir) and the numpy reductions the port
// has to reproduce bit-for-bit (percentile, median).
#include "pmw.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace pmw {

bool Camera::has_dist() const {
    for (double d : dist)
        if (d != 0.0) return true;
    return false;
}

double CentreCal::offset(double tilt_deg) const {
    // np.interp: clamped at both ends, linear between knots.
    if (identity()) return 0.0;
    const auto& xp = knots;
    const auto& fp = offsets;
    if (tilt_deg <= xp.front()) return fp.front();
    if (tilt_deg >= xp.back()) return fp.back();
    size_t j = std::upper_bound(xp.begin(), xp.end(), tilt_deg) - xp.begin();
    double x0 = xp[j - 1], x1 = xp[j], y0 = fp[j - 1], y1 = fp[j];
    double slope = (y1 - y0) / (x1 - x0);
    return slope * (tilt_deg - x0) + y0;
}

Vector3d unit(const Vector3d& v) {
    double n = v.norm();
    if (n < 1e-12) throw std::domain_error("zero-length direction");
    return v / n;
}

std::pair<Vector3d, Vector3d> tangent_basis(const Vector3d& n_in) {
    Vector3d n = unit(n_in);
    double nx = n[0], ny = n[1], nz = n[2];
    Vector3d t1 = std::abs(nx) < 0.9 ? Vector3d(0.0, nz, -ny) : Vector3d(-nz, 0.0, nx);
    t1 = unit(t1);
    double ax = t1[0], ay = t1[1], az = t1[2];
    return {t1, Vector3d(ny * az - nz * ay, nz * ax - nx * az, nx * ay - ny * ax)};
}

Vector3d orient(const Vector3d& normal, const Vector3d& reference) {
    Vector3d n = unit(normal);
    return n.dot(reference) < 0 ? Vector3d(-n) : n;
}

double line_angle_deg(const Vector3d& u, const Vector3d& v) {
    double c = std::abs(unit(u).dot(unit(v)));
    return std::acos(std::min(1.0, std::max(0.0, c))) * 180.0 / M_PI;
}

Matrix3d cone_from_circle(const Vector3d& c, const Vector3d& n_in, double radius) {
    Vector3d n = n_in / n_in.norm();
    double d = n.dot(c);
    return d * d * Matrix3d::Identity() - d * (n * c.transpose() + c * n.transpose()) +
           (c.dot(c) - radius * radius) * (n * n.transpose());
}

Ellipse normalise_ellipse(const Ellipse& e) {
    double a = e.major, b = e.minor, ang = e.ang;
    if (b > a) {
        std::swap(a, b);
        ang = ang + 90.0;
    }
    return {e.cx, e.cy, a, b, pymod(ang, 180.0)};
}

Matrix3d conic_from_ellipse(const Ellipse& e) {
    double a = e.major / 2.0, b = e.minor / 2.0;
    if (a <= 0 || b <= 0) throw std::domain_error("degenerate ellipse axes");
    double t = e.ang * M_PI / 180.0;
    Eigen::Matrix2d rot;
    rot << std::cos(t), -std::sin(t), std::sin(t), std::cos(t);
    Eigen::Matrix2d diag = Eigen::Vector2d(1.0 / (a * a), 1.0 / (b * b)).asDiagonal();
    Eigen::Matrix2d m = (rot * diag) * rot.transpose();
    Eigen::Vector2d ctr(e.cx, e.cy);
    Matrix3d c = Matrix3d::Identity();
    c.block<2, 2>(0, 0) = m;
    Eigen::Vector2d mc = m * ctr;
    c.block<2, 1>(0, 2) = -mc;
    c.block<1, 2>(2, 0) = -mc.transpose();
    c(2, 2) = ctr.dot(m * ctr) - 1.0;
    return c;
}

Ellipse ellipse_from_conic(const Matrix3d& conic) {
    double a = conic(0, 0), b = conic(0, 1), c = conic(1, 1);
    double bx = conic(0, 2), by = conic(1, 2);
    double det = a * c - b * b;
    if (std::abs(det) < 1e-300) throw std::domain_error("degenerate conic (singular quadratic part)");
    double cx = (-bx * c + by * b) / det;
    double cy = (-by * a + bx * b) / det;
    double k = (a * cx * cx + 2.0 * b * cx * cy + c * cy * cy) + 2.0 * (bx * cx + by * cy) + conic(2, 2);
    if (std::abs(k) < 1e-15) throw std::domain_error("degenerate conic (zero scale)");
    double s = -1.0 / k;
    double qa = a * s, qb = b * s, qc = c * s;
    double half = 0.5 * (qa + qc);
    double h = 0.5 * (qa - qc);
    double disc = std::sqrt(std::max(0.0, h * h + qb * qb));
    double l_hi = half + disc, l_lo = half - disc;
    if (l_lo <= 0.0) throw std::domain_error("conic is not a real ellipse");
    double vx, vy;
    if (std::abs(qb) > 1e-300) {
        vx = qb;
        vy = l_lo - qa;
    } else {
        if (qa <= qc) { vx = 1.0; vy = 0.0; } else { vx = 0.0; vy = 1.0; }
    }
    return {cx, cy, 2.0 / std::sqrt(l_lo), 2.0 / std::sqrt(l_hi), std::atan2(vy, vx) * 180.0 / M_PI};
}

// _normalise_cone: canonical (+, +, -) signature. Returns false when not an elliptic cone.
static bool normalise_cone(const Matrix3d& cone, Vector3d& w_out, Matrix3d& v_out) {
    Matrix3d q = 0.5 * (cone + cone.transpose());
    double scale = q.cwiseAbs().maxCoeff();
    if (!std::isfinite(scale) || scale < 1e-300) return false;
    q /= scale;
    Eigen::SelfAdjointEigenSolver<Matrix3d> es(q);   // ascending, like np.linalg.eigh
    Vector3d w = es.eigenvalues();
    Matrix3d v = es.eigenvectors();
    int npos = 0;
    for (int i = 0; i < 3; ++i) npos += w[i] > 0;
    if (npos == 1) {   // signature is (-, -, +): flip it
        Vector3d w2(-w[2], -w[1], -w[0]);
        Matrix3d v2;
        v2.col(0) = v.col(2); v2.col(1) = v.col(1); v2.col(2) = v.col(0);
        w = w2; v = v2;
        npos = 0;
        for (int i = 0; i < 3; ++i) npos += w[i] > 0;
    }
    if (npos != 2) return false;
    std::vector<int> pos, neg;
    for (int i = 0; i < 3; ++i) {
        if (w[i] > 0) pos.push_back(i);
        else if (w[i] < 0) neg.push_back(i);
    }
    if (neg.size() != 1) return false;
    // l1 >= l2 among the positive pair. np.argsort(w[pos])[::-1].
    if (w[pos[0]] < w[pos[1]]) std::swap(pos[0], pos[1]);
    int idx[3] = {pos[0], pos[1], neg[0]};
    for (int i = 0; i < 3; ++i) {
        w_out[i] = w[idx[i]];
        v_out.col(i) = v.col(idx[i]);
    }
    return true;
}

static Vector3d cross(const Vector3d& a, const Vector3d& b) {
    return Vector3d(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

static std::optional<Vector3d> circle_on_cone(const Vector3d& lam, const Vector3d& n_e, double radius,
                                              double isotropy_tol = 1e-6) {
    Vector3d seed = std::abs(n_e[1]) < 0.9 ? Vector3d(0.0, 1.0, 0.0) : Vector3d(1.0, 0.0, 0.0);
    Vector3d u = cross(n_e, seed);
    u /= u.norm();
    Vector3d w = cross(n_e, u);
    Vector3d lu = lam.cwiseProduct(u), lw = lam.cwiseProduct(w), ln = lam.cwiseProduct(n_e);
    double a_uu = u.dot(lu), a_ww = w.dot(lw), a_uw = u.dot(lw);
    double scale = std::max({std::abs(a_uu), std::abs(a_ww), 1e-300});
    if (std::abs(a_uu - a_ww) / scale > isotropy_tol || std::abs(a_uw) / scale > isotropy_tol)
        return std::nullopt;
    if (std::abs(a_uu) < 1e-300) return std::nullopt;
    double p_u = ln.dot(u), p_w = ln.dot(w), p_n = ln.dot(n_e);
    double r1_sq = (p_u * p_u + p_w * p_w) / (a_uu * a_uu) - p_n / a_uu;
    if (r1_sq <= 0) return std::nullopt;
    double d = radius / std::sqrt(r1_sq);
    return Vector3d(d * n_e - (d * p_u / a_uu) * u - (d * p_w / a_uu) * w);
}

static bool allclose(const Vector3d& a, const Vector3d& b, double atol, double rtol = 1e-5) {
    for (int i = 0; i < 3; ++i)
        if (std::abs(a[i] - b[i]) > atol + rtol * std::abs(b[i])) return false;
    return true;
}

std::vector<CirclePose> backproject(const Matrix3d& cone, double radius, double verify_tol) {
    Vector3d lam;
    Matrix3d v;
    std::vector<CirclePose> out;
    if (!normalise_cone(cone, lam, v)) return out;
    double l1 = lam[0], l2 = lam[1], l3 = lam[2];
    double denom = l1 - l3;
    if (denom <= 0) return out;
    double h = std::sqrt(std::max(0.0, (l1 - l2) / denom));
    double g = std::sqrt(std::max(0.0, (l2 - l3) / denom));
    for (double sz : {+1.0, -1.0}) {
        Vector3d n_e(h, 0.0, sz * g);
        double nn = n_e.norm();
        if (nn < 1e-12) continue;
        n_e /= nn;
        auto c_e = circle_on_cone(lam, n_e, radius);
        if (!c_e) continue;
        Vector3d n = v * n_e;
        Vector3d c = v * (*c_e);
        if (c[2] < 0) c = -c;
        if (c[2] <= 0) continue;
        if (n.dot(c) > 0) n = -n;
        if (verify_tol >= 0) {
            Matrix3d rebuilt = cone_from_circle(c, n, radius);
            Matrix3d a = cone / cone.cwiseAbs().maxCoeff();
            Matrix3d b = rebuilt / rebuilt.cwiseAbs().maxCoeff();
            double d1 = (a - b).cwiseAbs().maxCoeff(), d2 = (a + b).cwiseAbs().maxCoeff();
            if (std::min(d1, d2) > verify_tol) continue;
        }
        // _dedupe
        bool dup = false;
        for (const auto& q : out)
            if (allclose(c, q.center, 1e-9) && allclose(n, q.normal, 1e-9)) dup = true;
        if (!dup) out.push_back({c, n});
    }
    return out;
}

std::vector<CirclePose> backproject_ellipse(const Ellipse& e, const Matrix3d& K, double radius,
                                            double verify_tol) {
    Matrix3d conic = conic_from_ellipse(e);
    Matrix3d cone = K.transpose() * conic * K;
    return backproject(cone, radius, verify_tol);
}

double ambiguity_margin_deg(const std::vector<CirclePose>& poses) {
    if (poses.size() < 2) return 0.0;
    double d = std::min(1.0, std::max(-1.0, poses[0].normal.dot(poses[1].normal)));
    return std::acos(d) * 180.0 / M_PI;
}

std::optional<Matrix3d> fit_conic_weighted(const std::vector<Vector2d>& pts,
                                           const std::vector<double>* weights) {
    size_t n = pts.size();
    if (n < 5) return std::nullopt;
    Vector2d mu = Vector2d::Zero();
    for (const auto& p : pts) mu += p;
    mu /= (double)n;
    double ss = 0.0;
    for (const auto& p : pts) ss += (p - mu).squaredNorm();
    double scale = std::sqrt(ss / (double)n);
    if (!std::isfinite(scale) || scale < 1e-12) return std::nullopt;
    std::vector<double> w(n, 1.0);
    if (weights) {
        if (weights->size() != n) return std::nullopt;
        double wsum = 0.0, wmax = -INFINITY;
        for (double x : *weights) {
            if (!std::isfinite(x)) return std::nullopt;
            wsum += x;
            wmax = std::max(wmax, x);
        }
        if (wsum <= 0) return std::nullopt;
        for (size_t i = 0; i < n; ++i) w[i] = (*weights)[i] / wmax;
    }
    Matrix3d s1 = Matrix3d::Zero(), s2 = Matrix3d::Zero(), s3 = Matrix3d::Zero();
    for (size_t i = 0; i < n; ++i) {
        double x = (pts[i][0] - mu[0]) / scale, y = (pts[i][1] - mu[1]) / scale;
        Vector3d d1(x * x, x * y, y * y), d2(x, y, 1.0);
        s1 += d1 * (w[i] * d1).transpose();
        s2 += d1 * (w[i] * d2).transpose();
        s3 += d2 * (w[i] * d2).transpose();
    }
    Eigen::PartialPivLU<Matrix3d> lu(s3);
    if (lu.determinant() == 0.0) return std::nullopt;
    Matrix3d t = -lu.solve(s2.transpose());
    Matrix3d m = s1 + s2 * t;
    Matrix3d mc;
    mc.row(0) = m.row(2) / 2.0;
    mc.row(1) = -m.row(1);
    mc.row(2) = m.row(0) / 2.0;
    Eigen::EigenSolver<Matrix3d> es(mc);
    if (es.info() != Eigen::Success) return std::nullopt;
    auto evecs = es.eigenvectors();
    int valid = -1;
    for (int j = 0; j < 3 && valid < 0; ++j) {
        std::complex<double> e0 = evecs(0, j), e1 = evecs(1, j), e2 = evecs(2, j);
        std::complex<double> cond = 4.0 * e0 * e2 - e1 * e1;
        if (std::isfinite(cond.real()) && cond.real() > 0) valid = j;
    }
    if (valid < 0) return std::nullopt;
    Vector3d a1 = evecs.col(valid).real();
    double a = a1[0], b = a1[1], c = a1[2];
    Vector3d def = t * a1;
    double d = def[0], e = def[1], f = def[2];
    double s = scale, mx = mu[0], my = mu[1];
    Matrix3d conic;
    conic(0, 0) = a / (s * s);
    conic(0, 1) = b / (2 * s * s);
    conic(0, 2) = (d / s - (2 * a * mx + b * my) / (s * s)) / 2.0;
    conic(1, 0) = b / (2 * s * s);
    conic(1, 1) = c / (s * s);
    conic(1, 2) = (e / s - (b * mx + 2 * c * my) / (s * s)) / 2.0;
    conic(2, 0) = conic(0, 2);
    conic(2, 1) = conic(1, 2);
    conic(2, 2) = (a * mx * mx + b * mx * my + c * my * my) / (s * s) - (d * mx + e * my) / s + f;
    for (int i = 0; i < 9; ++i)
        if (!std::isfinite(conic(i / 3, i % 3))) return std::nullopt;
    return conic;
}

std::optional<Vector2d> projected_axis_dir(const Vector3d& c, const Vector3d& n_in, const Matrix3d& K,
                                           double delta_mm) {
    Vector3d n = unit(n_in);
    Vector3d p0 = K * c;
    Vector3d p1 = K * (c + delta_mm * n);
    if (std::abs(p0[2]) < 1e-9 || std::abs(p1[2]) < 1e-9) return std::nullopt;
    Vector2d d = p1.head<2>() / p1[2] - p0.head<2>() / p0[2];
    double m = d.norm();
    if (m < 1e-9) return std::nullopt;
    return Vector2d(d / m);
}

std::vector<double> sampson(const Matrix3d& C, const std::vector<Vector2d>& pts, bool keep_sign) {
    std::vector<double> out(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        Vector3d ph(pts[i][0], pts[i][1], 1.0);
        double alg = ph.dot(C * ph);
        Vector3d cp = C.transpose() * ph;
        double den = std::hypot(2.0 * cp[0], 2.0 * cp[1]);
        double num = keep_sign ? alg : std::abs(alg);
        out[i] = num / std::max(den, 1e-12);
    }
    return out;
}

double rms(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s / (double)v.size());
}

double percentile_linear(std::vector<double> v, double q) {
    // np.percentile(..., method="linear"), including numpy's own lerp rule.
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double qq = q / 100.0;
    double virt = (double)n * qq + (1.0 + qq * (1.0 - 1.0 - 1.0)) - 1.0;
    double prev = std::floor(virt);
    size_t lo = (size_t)std::max(0.0, prev);
    size_t hi = std::min(lo + 1, n - 1);
    double t = virt - prev;
    double a = v[lo], b = v[hi];
    double diff = b - a;
    double r = a + diff * t;
    if (t >= 0.5) r = b - diff * (1.0 - t);
    return r;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2) return v[n / 2];
    return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

}  // namespace pmw
