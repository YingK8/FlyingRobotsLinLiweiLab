// Python surface of the compiled pose core. Everything numeric lives in the other
// translation units; this file only converts. Frames and plates cross as numpy
// buffers wrapped in cv::Mat headers -- never as cv::Mat objects, because the cv2
// wheel and Homebrew's OpenCV are different builds with different ABIs.
// See controller/pose/theory.md 21.
#include "pmw.h"
#include "tracker.h"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;
using namespace pmw;

namespace {

using Gray = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Map32 = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Pts = nb::ndarray<const double, nb::shape<-1, 2>, nb::c_contig, nb::device::cpu>;
using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Mat = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

cv::Mat as_mat(const Gray& a) {
    return cv::Mat((int)a.shape(0), (int)a.shape(1), CV_8U, const_cast<uint8_t*>(a.data()));
}
cv::Mat as_mat(const Map32& a) {
    return cv::Mat((int)a.shape(0), (int)a.shape(1), CV_32F, const_cast<float*>(a.data()));
}
std::vector<Vector2d> as_points(const Pts& p) {
    std::vector<Vector2d> out(p.shape(0));
    for (size_t i = 0; i < p.shape(0); ++i) out[i] = Vector2d(p(i, 0), p(i, 1));
    return out;
}
Vector3d as_vec3(const Vec& v) {
    if (v.shape(0) != 3) throw std::invalid_argument("expected a 3-vector");
    return Vector3d(v(0), v(1), v(2));
}
Matrix3d as_mat3(const Mat& m) {
    if (m.shape(0) != 3 || m.shape(1) != 3) throw std::invalid_argument("expected a 3x3 matrix");
    Matrix3d out;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out(i, j) = m(i, j);
    return out;
}

// A numpy array that owns a heap copy of its data.
template <class T>
nb::ndarray<nb::numpy, T> own(std::vector<T>&& v, std::initializer_list<size_t> shape) {
    auto* p = new std::vector<T>(std::move(v));
    nb::capsule owner(p, [](void* q) noexcept { delete static_cast<std::vector<T>*>(q); });
    return nb::ndarray<nb::numpy, T>(p->data(), shape, owner);
}
nb::object mat_u8(const cv::Mat& m) {
    if (m.empty()) return nb::none();
    std::vector<uint8_t> v(m.rows * m.cols);
    for (int r = 0; r < m.rows; ++r) std::copy(m.ptr<uint8_t>(r), m.ptr<uint8_t>(r) + m.cols, v.data() + r * m.cols);
    return nb::cast(own(std::move(v), {(size_t)m.rows, (size_t)m.cols}));
}
nb::object mat_f32(const cv::Mat& m) {
    std::vector<float> v(m.rows * m.cols);
    for (int r = 0; r < m.rows; ++r) std::copy(m.ptr<float>(r), m.ptr<float>(r) + m.cols, v.data() + r * m.cols);
    return nb::cast(own(std::move(v), {(size_t)m.rows, (size_t)m.cols}));
}
nb::object vec3(const Vector3d& v) { return nb::cast(own(std::vector<double>{v[0], v[1], v[2]}, {3})); }
nb::object points(const std::vector<Vector2d>& p) {
    std::vector<double> v;
    v.reserve(2 * p.size());
    for (const auto& q : p) { v.push_back(q[0]); v.push_back(q[1]); }
    return nb::cast(own(std::move(v), {p.size(), (size_t)2}));
}
nb::object matrix(const MatrixXd& m) {
    std::vector<double> v(m.size());
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j) v[i * m.cols() + j] = m(i, j);
    return nb::cast(own(std::move(v), {(size_t)m.rows(), (size_t)m.cols()}));
}
nb::object ellipse_py(const Ellipse& e) {
    return nb::make_tuple(nb::make_tuple(e.cx, e.cy), nb::make_tuple(e.major, e.minor), e.ang);
}
Ellipse ellipse_cpp(nb::handle e) {
    auto c = nb::cast<std::pair<double, double>>(e[0]);
    auto a = nb::cast<std::pair<double, double>>(e[1]);
    return {c.first, c.second, a.first, a.second, nb::cast<double>(e[2])};
}
nb::object pose_py(const CirclePose& p) { return nb::make_tuple(vec3(p.center), vec3(p.normal)); }
std::vector<CirclePose> poses_cpp(nb::handle seq) {
    std::vector<CirclePose> out;
    for (nb::handle item : seq) {
        auto c = nb::cast<Vec>(item[0]);
        auto n = nb::cast<Vec>(item[1]);
        out.push_back({as_vec3(c), as_vec3(n)});
    }
    return out;
}

Camera camera_cpp(nb::handle d) {
    Camera c;
    c.name = nb::cast<std::string>(d["name"]);
    c.K = as_mat3(nb::cast<Mat>(d["K"]));
    c.Kinv = c.K.inverse();
    auto T = nb::cast<Mat>(d["T_world_cam"]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) c.R(i, j) = T(i, j);
        c.T[i] = T(i, 3);
    }
    if (!d["dist"].is_none()) {
        auto dist = nb::cast<Vec>(d["dist"]);
        c.dist.assign(dist.data(), dist.data() + dist.shape(0));
    }
    c.Kcv = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) c.Kcv.at<double>(i, j) = c.K(i, j);
    c.distcv = cv::Mat((int)c.dist.size(), 1, CV_64F);
    for (size_t i = 0; i < c.dist.size(); ++i) c.distcv.at<double>((int)i, 0) = c.dist[i];
    return c;
}
std::vector<Camera> cameras_cpp(nb::handle seq) {
    std::vector<Camera> out;
    for (nb::handle d : seq) out.push_back(camera_cpp(d));
    return out;
}
CentreCal centre_cal_cpp(nb::handle d) {
    CentreCal c;
    c.knots = nb::cast<std::vector<double>>(d["tilt_knots_deg"]);
    c.offsets = nb::cast<std::vector<double>>(d["offset_over_major"]);
    return c;
}

Config config_cpp(nb::dict d) {
    Config c;
    auto get = [&](const char* k) -> nb::handle {
        if (!d.contains(k)) throw std::invalid_argument(std::string("native config missing key: ") + k);
        return d[k];
    };
    auto D = [&](const char* k) { return nb::cast<double>(get(k)); };
    auto I = [&](const char* k) { return nb::cast<int>(get(k)); };
    auto B = [&](const char* k) { return nb::cast<bool>(get(k)); };
    // A None means "gate off", carried as NaN.
    auto DN = [&](const char* k) { nb::handle h = get(k); return h.is_none() ? NAN : nb::cast<double>(h); };
    c.ring_ksize = I("ring_ksize");
    c.ring_blur_sigma = D("ring_blur_sigma");
    c.ring_plate_weight = D("ring_plate_weight");
    c.ring_samples = I("ring_samples");
    c.ring_coverage_floor = D("ring_coverage_floor");
    c.roi_margin = D("roi_margin");
    c.bg_diff_thresh = I("bg_diff_thresh");
    c.blob_keep_fraction = D("blob_keep_fraction");
    c.ring_max_spread = D("ring_max_spread");
    c.shape_tol = D("shape_tol");
    c.ring_band = D("ring_band");
    c.ring_regrow_iters = I("ring_regrow_iters");
    c.max_anchors = I("max_anchors");
    c.min_contour_pts = I("min_contour_pts");
    c.axial = B("axial");
    c.axial_weight_iters = I("axial_weight_iters");
    c.axial_weight_power = D("axial_weight_power");
    c.axial_weight_floor = D("axial_weight_floor");
    c.axial_skip_ratio = D("axial_skip_ratio");
    c.one_sided_weight = D("one_sided_weight");
    c.one_sided_iters = I("one_sided_iters");
    c.trim_fraction = D("trim_fraction");
    c.trim_min_coverage = D("trim_min_coverage");
    c.trim_iters = I("trim_iters");
    c.trim_coverage_bins = I("trim_coverage_bins");
    c.ring_seed_fraction = D("ring_seed_fraction");
    c.ring_seed_percentile = D("ring_seed_percentile");
    c.undistort_samples = I("undistort_samples");
    c.refine_tol = D("refine_tol");
    c.max_refine_iter = I("max_refine_iter");
    c.ref_percentile = D("ref_percentile");
    c.jac_rel_step = D("jac_rel_step");
    c.jac_distort_step_px = D("jac_distort_step_px");
    c.jac_grad_step_px = D("jac_grad_step_px");
    c.jac_min_resid_frac = D("jac_min_resid_frac");
    c.far_px = D("far_px");
    c.cal_disp_tol_mm = D("cal_disp_tol_mm");
    c.cal_disp_tol_n = D("cal_disp_tol_n");
    c.max_discrepancy_mm = DN("max_discrepancy_mm");
    c.suspect_mm = D("suspect_mm");
    c.min_ridge = D("min_ridge");
    c.max_fit_rms_rel = DN("max_fit_rms_rel");
    c.dropout_s = D("dropout_s");
    c.window_frames = I("window_frames");
    c.min_window_support = I("min_window_support");
    c.prior_sigma_deg = D("prior_sigma_deg");
    c.sigma_normal_rad = D("sigma_normal_rad");
    c.accel_mm_s2 = D("accel_mm_s2");
    c.sigma_lat_mm = D("sigma_lat_mm");
    c.sigma_depth_mm = D("sigma_depth_mm");
    c.radius_mm = D("radius_mm");
    c.level = I("level");
    c.min_area = D("min_area");
    c.verify_tol = DN("verify_tol");
    if (std::isnan(c.verify_tol)) c.verify_tol = -1.0;
    c.require_stereo = B("require_stereo");
    return c;
}

nb::object seg_py(const std::optional<Segmentation>& s) {
    if (!s) return nb::none();
    nb::dict d;
    d["mask"] = mat_u8(s->mask);
    d["contour"] = points(s->hull);
    d["ellipse"] = ellipse_py(s->ellipse);
    d["area_px"] = s->area_px;
    d["n_points"] = s->n_points;
    d["fit_rms_px"] = s->fit_rms_px;
    d["threshold"] = s->threshold;
    d["valid"] = mat_u8(s->valid);
    d["valid_from"] = s->valid_from.empty() ? nb::none() : nb::cast(s->valid_from);
    return d;
}

nb::object result_py(const std::optional<PoseResult>& r) {
    if (!r) return nb::none();
    nb::dict d;
    d["center"] = vec3(r->center);
    d["normal"] = vec3(r->normal);
    nb::list segs, cands, ellipses;
    for (const auto& v : r->views) {
        segs.append(seg_py(v.seg));
        nb::list c;
        for (const auto& p : v.cands) c.append(pose_py(p));
        cands.append(c);
        ellipses.append(v.ellipse ? ellipse_py(*v.ellipse) : nb::none());
    }
    d["segs"] = segs;
    d["candidates"] = cands;
    d["ellipses"] = ellipses;
    d["views_used"] = r->usable;
    nb::list mposes;
    for (const auto& p : r->match.poses) mposes.append(pose_py(p));
    d["match_poses"] = mposes;
    d["match_indices"] = r->match.indices;
    d["discrepancy_mm"] = r->match.discrepancy_mm;
    d["margin"] = r->match.margin;
    d["refine_rms_px"] = r->refine_rms_px;
    d["refine_iters"] = r->refine_iters;
    d["jump_deg"] = r->jump_deg;
    d["t_seg_ms"] = r->t_seg_ms;
    d["t_est_ms"] = r->t_est_ms;
    d["skew_ms"] = r->skew_ms;
    d["union_coverage"] = r->union_coverage;
    d["alive"] = r->alive ? matrix(*r->alive) : nb::none();
    d["n_solutions"] = r->n_solutions;
    d["ambiguity_margin_deg"] = r->ambiguity_margin_deg;
    return d;
}

// The same dict, plus what only the tracker knows: the sequence number that says
// whether this pose is new, and the pair's own capture stamps.
nb::object tracker_result_py(std::uint64_t seq, const std::optional<PoseResult>& r,
                             double t, const std::vector<double>& stamps) {
    if (!r) return nb::make_tuple(seq, nb::none());
    nb::object d = result_py(r);
    nb::cast<nb::dict>(d)["t"] = t;
    nb::cast<nb::dict>(d)["stamps"] = stamps;
    return nb::make_tuple(seq, d);
}

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double pos = q * (v.size() - 1);
    const size_t lo = (size_t)std::floor(pos);
    const size_t hi = (size_t)std::ceil(pos);
    std::nth_element(v.begin(), v.begin() + hi, v.end());
    const double vhi = v[hi];
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    return v[lo] + (vhi - v[lo]) * (pos - lo);
}

nb::object stats_py(const TrackerStats& s) {
    nb::dict d;
    const double span = s.t_last_pose > s.t_start ? s.t_last_pose - s.t_start : 0.0;
    d["n_grabbed"] = nb::make_tuple(s.n_grabbed[0], s.n_grabbed[1]);
    // Camera rate and pose rate are different numbers and conflating them is the single
    // most common way to misdiagnose this pipeline (`camera/theory.md` 1.1).
    d["fps_grabbed"] = nb::make_tuple(span > 0 ? s.n_grabbed[0] / span : 0.0,
                                      span > 0 ? s.n_grabbed[1] / span : 0.0);
    d["age_ms"] = nb::make_tuple(s.age_ms[0], s.age_ms[1]);
    d["n_pose"] = s.n_pose;
    d["n_lost"] = s.n_lost;
    d["n_detected"] = s.n_detected;
    d["n_rejected"] = s.n_rejected;              // cross-view discrepancy gate
    d["n_rejected_fit"] = s.n_rejected_fit;      // hull fit-quality gate
    d["n_rejected_mono"] = s.n_rejected_mono;    // fewer than two usable views
    d["n_rearbitrated"] = s.n_rearbitrated;
    d["event"] = nb::make_tuple(s.event[0], s.event[1]);
    d["n_wrong_size"] = nb::make_tuple(s.n_wrong_size[0], s.n_wrong_size[1]);
    d["n_blank_raw"] = nb::make_tuple(s.n_blank_raw[0], s.n_blank_raw[1]);
    d["n_blank_out"] = nb::make_tuple(s.n_blank_out[0], s.n_blank_out[1]);
    d["n_skew_dropped"] = s.n_skew_dropped;
    d["fps_pose"] = span > 0 ? s.n_pose / span : 0.0;
    d["elapsed_s"] = span;
    if (!s.skews.empty()) {
        d["skew_median_ms"] = percentile(s.skews, 0.5) * 1e3;
        d["skew_p95_ms"] = percentile(s.skews, 0.95) * 1e3;
        d["skew_max_ms"] = *std::max_element(s.skews.begin(), s.skews.end()) * 1e3;
    }
    return d;
}

}  // namespace

NB_MODULE(pmw_pose, m) {
    m.attr("__version__") = "0.1.0";
    m.def("opencv_threads", [](int n) { if (n >= 0) cv::setNumThreads(n); return cv::getNumThreads(); },
          "n"_a = -1);
    m.def("opencv_build", []() { return std::string(cv::getBuildInformation()); });
    // background.RunningPlate.update's arithmetic: bg += step * sign(f - bg) in float32, then
    // the uint8 cast. In place on the caller's float32 plate; returns the uint8 view of it.
    m.def("running_plate_update",
          [](nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> bg, Gray gray, double step) {
              size_t H = bg.shape(0), W = bg.shape(1);
              if (gray.shape(0) != H || gray.shape(1) != W) throw std::invalid_argument("plate/frame shape mismatch");
              float st = (float)step;
              float* b = bg.data();
              const uint8_t* g = gray.data();
              std::vector<uint8_t> out(H * W);
              for (size_t i = 0; i < H * W; ++i) {
                  float d = (float)g[i] - b[i];
                  float sgn = d > 0 ? 1.0f : (d < 0 ? -1.0f : 0.0f);
                  b[i] += st * sgn;
                  out[i] = (uint8_t)b[i];
              }
              return own(std::move(out), {H, W});
          });

    // ---- stage (a): evidence
    m.def("ring_weight",
          [](Gray gray, std::optional<Gray> plate, std::optional<std::tuple<double, double, double, double>> roi,
             int ksize, double sigma, double plate_weight) {
              cv::Mat g = as_mat(gray), p;
              if (plate) p = as_mat(*plate);
              std::optional<Roi> r;
              if (roi) {
                  auto [x, y, w, h] = *roi;
                  r = Roi{(int)x, (int)y, (int)w, (int)h};
              }
              cv::Mat out = ring_weight(g, plate ? &p : nullptr, ksize, sigma, plate_weight, r);
              return mat_f32(out);
          },
          "gray"_a, "plate"_a.none(), "roi"_a.none(), "ksize"_a, "sigma"_a, "plate_weight"_a);
    m.def("ellipse_roi", [](nb::handle e, int H, int W, double margin) -> nb::object {
        auto r = ellipse_roi(e.is_none() ? std::nullopt : std::optional<Ellipse>(ellipse_cpp(e)), H, W, margin);
        if (!r) return nb::none();
        return nb::make_tuple(r->x, r->y, r->w, r->h);
    });
    m.def("sample_map", [](Map32 w, Pts pts) {
        cv::Mat m = as_mat(w);
        int n = (int)pts.shape(0);
        std::vector<double> out(n);
        sample_map(m, pts.data(), n, out.data());
        return own(std::move(out), {(size_t)n});
    });
    m.def("ellipse_points", [](nb::handle e, int n) { return points(ellipse_points(ellipse_cpp(e), n)); });

    // ---- stage (b): refine
    m.def("refine",
          [](std::vector<Map32> maps, nb::handle cams, Vec c0, Vec n0, nb::handle centre_cal, nb::dict cfg,
             Vec reference, bool debug, std::vector<Vec> x_eval) -> nb::object {
              std::vector<cv::Mat> ms;
              for (auto& w : maps) ms.push_back(as_mat(w));
              RefineDebug dbg;
              for (auto& x : x_eval) {
                  VectorXd v(x.shape(0));
                  for (size_t i = 0; i < x.shape(0); ++i) v[i] = x(i);
                  dbg.x_eval.push_back(v);
              }
              auto r = refine_image(ms, cameras_cpp(cams), as_vec3(c0), as_vec3(n0), centre_cal_cpp(centre_cal),
                                    config_cpp(cfg), as_vec3(reference), debug ? &dbg : nullptr);
              if (!r) return nb::none();
              nb::dict d;
              if (debug) {
                  nb::list pts, trace;
                  for (const auto& p : dbg.pts0) pts.append(matrix(p));
                  for (const auto& x : dbg.trace) trace.append(matrix(x));
                  d["pts0"] = pts;
                  d["e0"] = matrix(dbg.e0);
                  d["ref"] = dbg.ref;
                  d["f_scale"] = dbg.f_scale;
                  d["J0"] = matrix(dbg.J0);
                  d["trace"] = trace;
                  nb::list fe, je;
                  for (const auto& f : dbg.f_eval) fe.append(matrix(f));
                  for (const auto& J : dbg.J_eval) je.append(matrix(J));
                  d["f_eval"] = fe;
                  d["J_eval"] = je;
              }
              d["center"] = vec3(r->center);
              d["normal"] = vec3(r->normal);
              d["rms_px"] = r->rms_px;
              d["n_iter"] = r->n_iter;
              d["converged"] = r->converged;
              d["samples"] = matrix(r->samples);
              d["evidence"] = r->evidence;
              return d;
          }, "maps"_a, "cams"_a, "c0"_a, "n0"_a, "centre_cal"_a, "cfg"_a, "reference"_a, "debug"_a = false, "x_eval"_a = std::vector<Vec>());

    // ---- stage (c): segment
    m.def("segment", [](Gray gray, std::optional<Gray> plate, nb::dict cfg) {
        cv::Mat g = as_mat(gray), p;
        if (plate) p = as_mat(*plate);
        return seg_py(segment(g, plate ? &p : nullptr, config_cpp(cfg)));
    }, "gray"_a, "plate"_a.none(), "cfg"_a);
    m.def("segment_ring", [](Gray gray, Map32 weight, nb::dict cfg) {
        cv::Mat g = as_mat(gray), w = as_mat(weight);
        return seg_py(segment_ring(g, w, config_cpp(cfg)));
    });
    m.def("fit_ellipse", [](Pts pts, nb::dict cfg) -> nb::object {
        auto r = fit_ellipse(as_points(pts), config_cpp(cfg));
        if (!r) return nb::none();
        return nb::make_tuple(ellipse_py(r->first), r->second);
    });
    m.def("undistort_ellipse", [](nb::handle e, nb::handle cam, int n) {
        return ellipse_py(undistort_ellipse(ellipse_cpp(e), camera_cpp(cam), n));
    });

    // ---- stage (d): geometry
    m.def("backproject_ellipse", [](nb::handle e, Mat K, double radius, double verify_tol) {
        nb::list out;
        for (const auto& p : backproject_ellipse(ellipse_cpp(e), as_mat3(K), radius, verify_tol)) out.append(pose_py(p));
        return out;
    });
    m.def("match", [](nb::handle cands, nb::handle cams, nb::dict cfg, std::optional<Vec> prior) -> nb::object {
        std::vector<std::vector<CirclePose>> c;
        for (nb::handle view : cands) c.push_back(poses_cpp(view));
        std::optional<Vector3d> pr;
        if (prior) pr = as_vec3(*prior);
        auto m = match(c, cameras_cpp(cams), config_cpp(cfg), pr ? &*pr : nullptr);
        if (!m) return nb::none();
        nb::list poses;
        for (const auto& p : m->poses) poses.append(pose_py(p));
        return nb::make_tuple(poses, m->indices, m->discrepancy_mm, m->margin);
    }, "cands"_a, "cams"_a, "cfg"_a, "prior"_a.none());
    m.def("fuse", [](nb::handle poses, nb::handle cams, nb::dict cfg, Vec reference, std::optional<std::vector<double>> stamps,
                     std::optional<Vec> velocity, std::optional<Mat> vel_cov) {
        std::optional<Vector3d> v;
        std::optional<Matrix3d> vc;
        if (velocity) v = as_vec3(*velocity);
        if (vel_cov) vc = as_mat3(*vel_cov);
        Fused f = fuse(poses_cpp(poses), cameras_cpp(cams), config_cpp(cfg), as_vec3(reference),
                       stamps ? &*stamps : nullptr, v ? &*v : nullptr, vc ? &*vc : nullptr);
        return nb::make_tuple(vec3(f.center), vec3(f.normal), matrix(f.cov));
    }, "poses"_a, "cams"_a, "cfg"_a, "reference"_a, "stamps"_a.none(), "velocity"_a.none(), "vel_cov"_a.none());

    // ---- the estimator
    nb::class_<Estimator>(m, "Estimator")
        .def("__init__",
             [](Estimator* self, nb::handle cams, nb::dict cfg, nb::handle centre_cal, Vec reference) {
                 new (self) Estimator(cameras_cpp(cams), config_cpp(cfg), centre_cal_cpp(centre_cal), as_vec3(reference));
             })
        .def("reset", &Estimator::reset)
        .def_ro("n_detected", &Estimator::n_detected)
        .def_ro("n_lost", &Estimator::n_lost)
        .def_ro("n_rejected", &Estimator::n_rejected)
        .def_ro("n_rejected_fit", &Estimator::n_rejected_fit)
        .def_ro("n_rejected_mono", &Estimator::n_rejected_mono)
        .def_ro("n_rearbitrated", &Estimator::n_rearbitrated)
        .def("prev_ellipse", [](const Estimator& e, int i) -> nb::object {
            return e.prev_ellipse[i] ? ellipse_py(*e.prev_ellipse[i]) : nb::none();
        })
        .def("update",
             [](Estimator& e, Gray frameA, Gray frameB, std::optional<Gray> plateA, std::optional<Gray> plateB,
                long long versionA, long long versionB, double now, std::optional<std::vector<double>> stamps,
                std::optional<Vec> velocity, std::optional<Mat> vel_cov) {
                 cv::Mat fa = as_mat(frameA), fb = as_mat(frameB), pa, pb;
                 if (plateA) pa = as_mat(*plateA);
                 if (plateB) pb = as_mat(*plateB);
                 std::optional<Vector3d> v;
                 std::optional<Matrix3d> vc;
                 if (velocity) v = as_vec3(*velocity);
                 if (vel_cov) vc = as_mat3(*vel_cov);
                 std::optional<PoseResult> r;
                 {
                     nb::gil_scoped_release release;
                     r = e.update(fa, fb, plateA ? &pa : nullptr, plateB ? &pb : nullptr, versionA, versionB, now,
                                  stamps ? &*stamps : nullptr, v ? &*v : nullptr, vc ? &*vc : nullptr);
                 }
                 return result_py(r);
             },
             "frameA"_a, "frameB"_a, "plateA"_a.none(), "plateB"_a.none(), "versionA"_a, "versionB"_a, "now"_a, "stamps"_a.none(),
             "velocity"_a.none(), "vel_cov"_a.none())
        .def("set_frame_seq", &Estimator::set_frame_seq, "seqA"_a, "seqB"_a)
        .def("set_thresh", &Estimator::set_thresh, "level"_a);

    // ---- capture
    m.def("list_cameras", [] {
        nb::list out;
        for (const auto& c : list_cameras()) {
            nb::dict d;
            d["unique_id"] = c.unique_id;
            d["name"] = c.name;
            out.append(d);
        }
        return out;
    }, "Every video device AVFoundation can see: unique_id and name.");

    m.def("camera_formats", [](const std::string& id) {
        nb::list out;
        for (const auto& f : camera_formats(id)) {
            nb::dict d;
            d["width"] = f.width;
            d["height"] = f.height;
            d["fourcc"] = f.fourcc;
            d["min_fps"] = f.min_fps;
            d["max_fps"] = f.max_fps;
            out.append(d);
        }
        return out;
    }, "unique_id"_a, "Every mode one camera offers. What to look at when open refuses.");

    // ---- the live tracker
    //
    // Every method that returns pose data copies out of the tracker under its mutex and
    // builds the Python object AFTER releasing it (see `Tracker::latest`): allocating a
    // nanobind object needs the GIL, and holding the tracker mutex across a GIL
    // acquisition deadlocks against a Python thread already holding the GIL and blocked
    // on that same mutex.
    nb::class_<Tracker>(m, "Tracker")
        .def("__init__",
             [](Tracker* self, nb::handle cams, nb::dict cfg, nb::handle centre_cal, Vec reference,
                std::vector<std::string> ids, int width, int height, double fps, bool rotate180,
                const std::string& pair_mode, double max_skew_s, double plate_step,
                int plate_warmup, long long plate_refresh) {
                 if (ids.size() != 2)
                     throw std::invalid_argument("Tracker needs exactly two camera ids");
                 std::vector<CameraSpec> specs;
                 for (const auto& id : ids)
                     specs.push_back(CameraSpec{id, width, height, fps, rotate180});
                 PairMode mode;
                 if (pair_mode == "interleave") mode = PairMode::Interleave;
                 else if (pair_mode == "both") mode = PairMode::Both;
                 else throw std::invalid_argument("pair_mode must be 'interleave' or 'both'");
                 new (self) Tracker(cameras_cpp(cams), config_cpp(cfg), centre_cal_cpp(centre_cal),
                                    as_vec3(reference), std::move(specs), mode, max_skew_s,
                                    plate_step, plate_warmup, plate_refresh);
             },
             "cams"_a, "cfg"_a, "centre_cal"_a, "reference"_a, "ids"_a, "width"_a, "height"_a,
             "fps"_a, "rotate180"_a, "pair_mode"_a, "max_skew_s"_a, "plate_step"_a,
             "plate_warmup"_a, "plate_refresh"_a)
        .def("start", [](Tracker& t) { nb::gil_scoped_release r; t.start(); })
        .def("stop", [](Tracker& t) { nb::gil_scoped_release r; t.stop(); })
        .def("reset", [](Tracker& t) { nb::gil_scoped_release r; t.reset(); })
        .def("latest", [](Tracker& t) {
            std::uint64_t seq = 0;
            std::optional<PoseResult> r;
            double tm = 0.0;
            std::vector<double> stamps;
            {
                nb::gil_scoped_release rel;
                PoseResult tmp;
                if (t.latest(&seq, &tmp, &tm, &stamps)) r = std::move(tmp);
            }
            return tracker_result_py(seq, r, tm, stamps);
        }, "Newest (seq, pose dict or None). Non-blocking.")
        .def("wait", [](Tracker& t, std::uint64_t after, double timeout_s) {
            std::uint64_t seq = 0;
            std::optional<PoseResult> r;
            double tm = 0.0;
            std::vector<double> stamps;
            {
                nb::gil_scoped_release rel;
                PoseResult tmp;
                if (t.wait(after, timeout_s, &seq, &tmp, &tm, &stamps)) r = std::move(tmp);
            }
            return tracker_result_py(seq, r, tm, stamps);
        }, "after"_a, "timeout_s"_a,
           "Block until the pose seq passes `after`. (0, None) on timeout.")
        .def("frames", [](Tracker& t) -> nb::object {
            cv::Mat a, b;
            double tm = 0.0, sma = -1.0, smb = -1.0;
            std::vector<double> stamps;
            {
                nb::gil_scoped_release rel;
                t.frames(&a, &b, &tm, &stamps, &sma, &smb);
            }
            if (a.empty() || b.empty()) return nb::none();
            return nb::make_tuple(tm, mat_u8(a), mat_u8(b), stamps,
                                  nb::make_tuple(sma, smb));
        }, "Newest (t, frameA, frameB, stamps), or None before both cameras have run.")
        .def("set_motion", [](Tracker& t, std::optional<Vec> velocity, std::optional<Mat> vel_cov) {
            std::optional<Vector3d> v;
            std::optional<Matrix3d> vc;
            if (velocity) v = as_vec3(*velocity);
            if (vel_cov) vc = as_mat3(*vel_cov);
            nb::gil_scoped_release rel;
            t.set_motion(v ? &*v : nullptr, vc ? &*vc : nullptr);
        }, "velocity"_a.none(), "vel_cov"_a.none())
        .def("push_frame", [](Tracker& t, int ci, Gray frame, double stamp) {
            cv::Mat f = as_mat(frame);
            nb::gil_scoped_release rel;
            t.push_frame(ci, f, stamp);
        }, "ci"_a, "frame"_a, "t"_a,
           "Feed a frame as if a camera had delivered it. For the offline self-check.")
        .def("pump", [](Tracker& t) {
            nb::gil_scoped_release rel;
            return t.pump();
        }, "Run one pairing step synchronously. Pairs with push_frame.")
        .def("set_thresh", [](Tracker& t, int level) {
            nb::gil_scoped_release rel;
            t.set_thresh(level);
        }, "level"_a)
        .def("stats", [](Tracker& t) {
            TrackerStats s;
            {
                nb::gil_scoped_release rel;
                s = t.stats();
            }
            return stats_py(s);
        });
}
