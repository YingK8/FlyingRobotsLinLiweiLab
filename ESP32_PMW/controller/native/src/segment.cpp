// segment.py: background_mask, threshold_mask, silhouette_hull and its grouping,
// fit_ellipse and its three reweighting passes, ring_seed / segment_ring, and
// undistort_ellipse.
#include "pmw.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <numeric>
#include <set>

namespace pmw {

static cv::Mat kernel_ellipse(int k) {
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
}

static Ellipse from_rect(const cv::RotatedRect& r) {
    // RotatedRect holds floats; the Python side sees the same float32 values.
    return normalise_ellipse({(double)r.center.x, (double)r.center.y, (double)r.size.width,
                              (double)r.size.height, (double)r.angle});
}

Ellipse fit_ellipse_direct(const std::vector<Vector2d>& pts) {
    std::vector<cv::Point2f> p(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) p[i] = cv::Point2f((float)pts[i][0], (float)pts[i][1]);
    return from_rect(cv::fitEllipseDirect(p));
}

static std::vector<double> sampson_ellipse(const Ellipse& e, const std::vector<Vector2d>& pts,
                                           bool keep_sign) {
    return sampson(conic_from_ellipse(e), pts, keep_sign);
}

// -- fit_ellipse -------------------------------------------------------------------

static std::optional<std::vector<double>> axial_weights(const std::vector<Vector2d>& pts,
                                                        const Ellipse& e, double power, double floor) {
    if (e.major <= 0) return std::nullopt;
    double t = e.ang * M_PI / 180.0;
    Vector2d u(std::cos(t), std::sin(t));
    std::vector<double> w(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        double s = std::abs((pts[i] - Vector2d(e.cx, e.cy)).dot(u)) / (e.major / 2.0);
        w[i] = floor + (1.0 - floor) * std::pow(std::min(1.0, std::max(0.0, s)), power);
    }
    return w;
}

static std::optional<std::vector<double>> outward_weights(const std::vector<Vector2d>& pts,
                                                          const Ellipse& e, double w_out, double power) {
    if (e.major <= 0) return std::nullopt;
    std::vector<double> signed_d;
    try {
        signed_d = sampson_ellipse(e, pts, true);
    } catch (const std::domain_error&) {
        return std::nullopt;
    }
    double t = e.ang * M_PI / 180.0;
    Vector2d u(std::cos(t), std::sin(t));
    std::vector<double> w(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        double s = std::abs((pts[i] - Vector2d(e.cx, e.cy)).dot(u)) / (e.major / 2.0);
        double axial_w = std::pow(std::min(1.0, std::max(0.0, s)), power);
        w[i] = (signed_d[i] > 0.0 ? w_out : 1.0) * axial_w;
    }
    return w;
}

static double wsum(const std::vector<double>& w) {
    double s = 0.0;
    for (double x : w) s += x;
    return s;
}

// np.histogram(np.arctan2(v, u), bins, range=(-pi, pi)): fraction of non-empty bins.
static double angular_coverage(const Ellipse& e, const std::vector<Vector2d>& pts, int bins) {
    double t = e.ang * M_PI / 180.0;
    double ct = std::cos(t), st = std::sin(t);
    std::vector<double> edges(bins + 1);
    double step = (2.0 * M_PI) / bins;
    for (int i = 0; i <= bins; ++i) edges[i] = i * step + (-M_PI);
    edges[bins] = M_PI;
    double first = edges[0], last = edges[bins];
    std::vector<int> h(bins, 0);
    for (const auto& p : pts) {
        double dx = p[0] - e.cx, dy = p[1] - e.cy;
        double u = dx * ct + dy * st, v = dx * (-st) + dy * ct;
        double a = std::atan2(v, u);
        if (!(a >= first && a <= last)) continue;
        double f = ((a - first) / (last - first)) * bins;
        int idx = (int)f;
        if (idx == bins) idx -= 1;
        if (a < edges[idx]) idx -= 1;
        if (a >= edges[idx + 1] && idx != bins - 1) idx += 1;
        h[idx] += 1;
    }
    int nonzero = 0;
    for (int c : h) nonzero += c > 0;
    return (double)nonzero / (double)bins;
}

static bool refit(const std::vector<Vector2d>& pts, const std::vector<double>& w, double plain_angle,
                  Ellipse& ellipse) {
    auto c = fit_conic_weighted(pts, &w);
    if (!c) return false;
    try {
        Ellipse e = normalise_ellipse(ellipse_from_conic(*c));
        ellipse = {e.cx, e.cy, e.major, e.minor, plain_angle};
    } catch (const std::domain_error&) {
        return false;
    }
    return true;
}

std::optional<std::pair<Ellipse, double>> fit_ellipse(const std::vector<Vector2d>& pts,
                                                      const Config& cfg) {
    if ((int)pts.size() < cfg.min_contour_pts) return std::nullopt;
    Ellipse ellipse;
    try {
        ellipse = fit_ellipse_direct(pts);
    } catch (const cv::Exception&) {
        return std::nullopt;
    }
    double plain_angle = ellipse.ang;
    bool axial = cfg.axial;
    if (axial && cfg.axial_weight_iters && ellipse.minor / ellipse.major <= cfg.axial_skip_ratio) {
        for (int it = 0; it < cfg.axial_weight_iters; ++it) {
            auto w = axial_weights(pts, ellipse, cfg.axial_weight_power, cfg.axial_weight_floor);
            if (!w || wsum(*w) <= 0) break;
            if (!refit(pts, *w, plain_angle, ellipse)) break;
        }
    }
    if (axial && cfg.one_sided_weight > 0.0) {
        for (int it = 0; it < cfg.one_sided_iters; ++it) {
            auto w = outward_weights(pts, ellipse, cfg.one_sided_weight, cfg.axial_weight_power);
            if (!w || wsum(*w) <= 0) break;
            if (!refit(pts, *w, plain_angle, ellipse)) break;
        }
    }
    if (axial && cfg.trim_fraction > 0.0 && pts.size() >= 16) {
        bool covered = false;
        try {
            covered = angular_coverage(ellipse, pts, cfg.trim_coverage_bins) >= cfg.trim_min_coverage;
        } catch (const std::domain_error&) {
            covered = false;
        }
        if (covered) {
            int k = (int)pyround(cfg.trim_fraction * (double)pts.size());
            for (int it = 0; it < cfg.trim_iters; ++it) {
                if (k < 1 || (int)pts.size() - k < 8) break;
                std::vector<double> d;
                try {
                    d = sampson_ellipse(ellipse, pts, true);
                } catch (const std::domain_error&) {
                    break;
                }
                // np.argsort(d)[:-k]: the len-k smallest, in ascending order.
                std::vector<size_t> order(pts.size());
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(),
                                 [&](size_t a, size_t b) { return d[a] < d[b]; });
                std::vector<Vector2d> kept;
                kept.reserve(pts.size() - k);
                for (size_t i = 0; i + k < order.size(); ++i) kept.push_back(pts[order[i]]);
                std::vector<double> ones(kept.size(), 1.0);
                if (!refit(kept, ones, plain_angle, ellipse)) break;
            }
        }
    }
    if (!(std::isfinite(ellipse.major) && std::isfinite(ellipse.minor)) || ellipse.major <= 0 ||
        ellipse.minor <= 0)
        return std::nullopt;
    return std::make_pair(ellipse, rms(sampson_ellipse(ellipse, pts, false)));
}

// -- silhouette_hull ---------------------------------------------------------------

struct Components {
    int n;
    cv::Mat labels, stats, centroids;
    int area(int l) const { return stats.at<int>(l, cv::CC_STAT_AREA); }
    int left(int l) const { return stats.at<int>(l, cv::CC_STAT_LEFT); }
    int top(int l) const { return stats.at<int>(l, cv::CC_STAT_TOP); }
    int width(int l) const { return stats.at<int>(l, cv::CC_STAT_WIDTH); }
    int height(int l) const { return stats.at<int>(l, cv::CC_STAT_HEIGHT); }
    Vector2d centroid(int l) const {
        return Vector2d(centroids.at<double>(l, 0), centroids.at<double>(l, 1));
    }
};

static std::vector<Vector2d> contour_points(const cv::Mat& bin) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Vector2d> pts;
    for (const auto& c : contours)
        for (const auto& p : c) pts.emplace_back((double)p.x, (double)p.y);
    return pts;
}

static std::vector<Vector2d> convex_hull_f32(const std::vector<Vector2d>& pts) {
    std::vector<cv::Point2f> p(pts.size()), hull;
    for (size_t i = 0; i < pts.size(); ++i) p[i] = cv::Point2f((float)pts[i][0], (float)pts[i][1]);
    cv::convexHull(p, hull);
    std::vector<Vector2d> out(hull.size());
    for (size_t i = 0; i < hull.size(); ++i) out[i] = Vector2d(hull[i].x, hull[i].y);
    return out;
}

static std::optional<std::vector<Vector2d>> group_hull(const Components& cc, const std::vector<int>& members,
                                                       const Config& cfg) {
    std::vector<uint8_t> lut(cc.n, 0);
    for (int m : members) lut[m] = 255;
    int x0 = 0, y0 = 0, x1 = cc.labels.cols, y1 = cc.labels.rows;
    if (!members.empty()) {
        x0 = INT_MAX; y0 = INT_MAX; x1 = INT_MIN; y1 = INT_MIN;
        for (int m : members) {
            x0 = std::min(x0, cc.left(m));
            y0 = std::min(y0, cc.top(m));
            x1 = std::max(x1, cc.left(m) + cc.width(m));
            y1 = std::max(y1, cc.top(m) + cc.height(m));
        }
    }
    cv::Mat view(y1 - y0, x1 - x0, CV_8U);
    for (int r = 0; r < view.rows; ++r) {
        const int* lr = cc.labels.ptr<int>(y0 + r) + x0;
        uint8_t* vr = view.ptr<uint8_t>(r);
        for (int c = 0; c < view.cols; ++c) vr[c] = lut[lr[c]];
    }
    auto pts = contour_points(view);
    if (pts.empty() || (int)pts.size() < cfg.min_contour_pts) return std::nullopt;
    auto hull = convex_hull_f32(pts);
    if ((int)hull.size() < cfg.min_contour_pts) return std::nullopt;
    for (auto& h : hull) h += Vector2d(x0, y0);
    return hull;
}

static std::vector<int> regrow(std::vector<int> members, const std::vector<int>& keep,
                               const Components& cc, const Config& cfg) {
    for (int it = 0; it < cfg.ring_regrow_iters; ++it) {
        auto hull = group_hull(cc, members, cfg);
        if (!hull) return members;
        Ellipse e;
        try {
            e = fit_ellipse_direct(*hull);
        } catch (const cv::Exception&) {
            return members;
        }
        if (!std::isfinite(e.major) || e.major <= 0) return members;
        // _on_ring over the kept centroids, then np.union1d (sorted, unique).
        std::set<int> grown(members.begin(), members.end());
        double a = e.major / 2.0, b = e.minor / 2.0;
        if (a > 0 && b > 0) {
            double th = e.ang * M_PI / 180.0, c = std::cos(th), sn = std::sin(th);
            for (int k : keep) {
                Vector2d p = cc.centroid(k);
                double dx = p[0] - e.cx, dy = p[1] - e.cy;
                double x = dx * c + dy * sn, y = -dx * sn + dy * c;
                if (std::abs(std::hypot(x / a, y / b) - 1.0) < cfg.ring_band) grown.insert(k);
            }
        }
        if (grown.size() == members.size()) break;
        members.assign(grown.begin(), grown.end());
    }
    return members;
}

static std::optional<std::vector<int>> best_group(const std::vector<int>& keep, const Components& cc,
                                                  const Config& cfg) {
    // keep[np.argsort(areas[keep])[::-1]][:max_anchors]
    std::vector<int> order = keep;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cc.area(a) < cc.area(b); });
    std::reverse(order.begin(), order.end());
    if ((int)order.size() > cfg.max_anchors) order.resize(cfg.max_anchors);
    std::vector<std::pair<double, std::vector<int>>> admissible;
    std::optional<std::vector<int>> best;
    double best_score = INFINITY;
    for (int anchor : order) {
        double radius = 0.5 * std::hypot((double)cc.width(anchor), (double)cc.height(anchor));
        if (radius <= 0) continue;
        Vector2d ca = cc.centroid(anchor);
        std::vector<int> members;
        for (int k : keep) {
            Vector2d d = cc.centroid(k) - ca;
            if (std::hypot(d[0], d[1]) <= cfg.ring_max_spread * radius) members.push_back(k);
        }
        members = regrow(members, keep, cc, cfg);
        auto hull = group_hull(cc, members, cfg);
        if (!hull) continue;
        Ellipse e;
        try {
            e = fit_ellipse_direct(*hull);
        } catch (const cv::Exception&) {
            continue;
        }
        if (!std::isfinite(e.major) || e.major <= 0) continue;
        double score;
        try {
            score = rms(sampson_ellipse(e, *hull, false)) / e.major;
        } catch (const std::domain_error&) {
            continue;
        }
        if (score <= cfg.shape_tol) admissible.emplace_back(e.major, members);
        if (score < best_score) {
            best = members;
            best_score = score;
        }
    }
    if (!admissible.empty()) {
        auto it = std::max_element(admissible.begin(), admissible.end(),
                                   [](const auto& a, const auto& b) { return a.first < b.first; });
        return it->second;
    }
    return best;
}

static std::pair<std::optional<std::vector<Vector2d>>, double> silhouette_hull(const cv::Mat& mask,
                                                                                bool spread,
                                                                                const Config& cfg) {
    Components cc;
    cc.n = cv::connectedComponentsWithStats(mask, cc.labels, cc.stats, cc.centroids, 8, CV_32S);
    if (cc.n <= 1) return {std::nullopt, 0.0};
    int amax = 0;
    for (int l = 1; l < cc.n; ++l) amax = std::max(amax, cc.area(l));
    double floor = std::max(1.0, cfg.blob_keep_fraction * (double)amax);
    std::vector<int> keep;
    for (int l = 1; l < cc.n; ++l)
        if ((double)cc.area(l) >= floor) keep.push_back(l);
    if (keep.empty()) return {std::nullopt, 0.0};
    if (spread && keep.size() > 1) {
        auto g = best_group(keep, cc, cfg);
        if (!g) return {std::nullopt, 0.0};
        keep = *g;
    }
    cv::Mat kept;
    if ((int)keep.size() == cc.n - 1) {
        kept = mask;
    } else {
        std::vector<uint8_t> lut(cc.n, 0);
        for (int k : keep) lut[k] = 255;
        kept.create(mask.size(), CV_8U);
        for (int r = 0; r < kept.rows; ++r) {
            const int* lr = cc.labels.ptr<int>(r);
            uint8_t* kr = kept.ptr<uint8_t>(r);
            for (int c = 0; c < kept.cols; ++c) kr[c] = lut[lr[c]];
        }
    }
    auto pts = contour_points(kept);
    if (pts.empty() || (int)pts.size() < cfg.min_contour_pts) return {std::nullopt, 0.0};
    auto hull = convex_hull_f32(pts);
    double area = 0.0;
    for (int k : keep) area += (double)cc.area(k);
    return {hull, area};
}

// -- segment -----------------------------------------------------------------------

static cv::Mat open_close(const cv::Mat& mask) {
    cv::Mat a, b;
    cv::morphologyEx(mask, a, cv::MORPH_OPEN, kernel_ellipse(3));
    cv::morphologyEx(a, b, cv::MORPH_CLOSE, kernel_ellipse(7));
    return b;
}

std::optional<Segmentation> segment(const cv::Mat& gray, const cv::Mat* plate, const Config& cfg) {
    // Bright appearance with a plate: background_mask gates the threshold.
    cv::Mat region;
    std::string region_from;
    if (plate && plate->size() == gray.size()) {
        cv::Mat diff;
        cv::absdiff(gray, *plate, diff);
        cv::threshold(diff, region, cfg.bg_diff_thresh, 255, cv::THRESH_BINARY);
        region_from = "background";
    }
    cv::Mat channel = gray;
    if (!region.empty()) cv::bitwise_and(gray, region, channel);
    cv::Mat mask;
    cv::threshold(channel, mask, cfg.level, 255, cv::THRESH_BINARY);
    mask = open_close(mask);

    auto [hull, area] = silhouette_hull(mask, plate != nullptr, cfg);
    if (!hull || area < cfg.min_area) return std::nullopt;
    auto fit = fit_ellipse(*hull, cfg);
    if (!fit) return std::nullopt;
    Segmentation s;
    s.mask = mask;
    s.hull = *hull;
    s.ellipse = fit->first;
    s.area_px = area;
    s.n_points = (int)hull->size();
    s.fit_rms_px = fit->second;
    s.threshold = cfg.level;
    s.valid = region;
    s.valid_from = region_from;
    return s;
}

std::optional<Segmentation> segment_ring(const cv::Mat& gray, const cv::Mat& weight, const Config& cfg) {
    // ring_seed: level from the map's own high percentile, on every fourth pixel.
    std::vector<double> sub;
    sub.reserve((weight.rows / 4 + 1) * (weight.cols / 4 + 1));
    for (int r = 0; r < weight.rows; r += 4) {
        const float* row = weight.ptr<float>(r);
        for (int c = 0; c < weight.cols; c += 4) sub.push_back(row[c]);
    }
    // numpy on a float32 array: the neighbour difference is taken in float32, the rest in float64.
    std::sort(sub.begin(), sub.end());
    size_t n = sub.size();
    double qq = cfg.ring_seed_percentile / 100.0;
    double virt = (double)n * qq + (1.0 + qq * (1.0 - 1.0 - 1.0)) - 1.0;
    double prev = std::floor(virt);
    size_t lo = (size_t)std::max(0.0, prev), hi = std::min(lo + 1, n - 1);
    double t = virt - prev;
    float a = (float)sub[lo], b = (float)sub[hi];
    float diff = b - a;
    double pct = (double)a + (double)diff * t;
    if (t >= 0.5) pct = (double)b - (double)diff * (1.0 - t);
    double level = cfg.ring_seed_fraction * pct;
    // `weight >= level` compares in float32 (a Python float is weak under NEP 50).
    float lvl = (float)std::max(level, 1e-6);
    cv::Mat mask(weight.size(), CV_8U);
    for (int r = 0; r < weight.rows; ++r) {
        const float* w = weight.ptr<float>(r);
        uint8_t* m = mask.ptr<uint8_t>(r);
        for (int c = 0; c < weight.cols; ++c) m[c] = w[c] >= lvl ? 255 : 0;
    }
    mask = open_close(mask);
    auto [hull, area] = silhouette_hull(mask, true, cfg);
    if (!hull || area < cfg.min_area) return std::nullopt;
    auto fit = fit_ellipse(*hull, cfg);
    if (!fit) return std::nullopt;
    Segmentation s;
    s.mask = mask;
    s.hull = *hull;
    s.ellipse = fit->first;
    s.area_px = area;
    s.n_points = (int)hull->size();
    s.fit_rms_px = rms(sampson_ellipse(fit->first, *hull, false));
    s.threshold = 0;
    return s;
}

Ellipse undistort_ellipse(const Ellipse& e, const Camera& cam, int n_samples) {
    if (!cam.has_dist()) return e;
    auto pts = ellipse_points(e, n_samples);
    std::vector<cv::Point2d> src(pts.size()), dst;
    for (size_t i = 0; i < pts.size(); ++i) src[i] = cv::Point2d(pts[i][0], pts[i][1]);
    cv::undistortPoints(src, dst, cam.Kcv, cam.distcv, cv::noArray(), cam.Kcv);
    std::vector<Vector2d> d(dst.size());
    for (size_t i = 0; i < dst.size(); ++i) d[i] = Vector2d(dst[i].x, dst[i].y);
    // A double fit, as in Python: cv::fitEllipseDirect differs between OpenCV builds by
    // a float32 ulp on these points (segment.undistort_ellipse, theory.md 21.3).
    auto c = fit_conic_weighted(d, nullptr);
    if (!c) throw std::domain_error("undistorted rim did not fit an ellipse");
    return normalise_ellipse(ellipse_from_conic(*c));
}

}  // namespace pmw
