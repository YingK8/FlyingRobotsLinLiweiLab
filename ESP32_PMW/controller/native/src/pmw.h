// The compiled stereo pose core. Every function here is a line-for-line port of the
// Python it is named after (controller/pose/{segment,conic,stereo}.py), kept in the
// same evaluation order so the two agree to rounding. The Python is the reference and
// stays in the tree; `controller/pose/native_parity.py` holds them to each other.
// See controller/pose/theory.md 21.
//
// No numeric literal that is a tuning constant lives in this file or its .cpp files:
// every one arrives in `Config`, built by `stereo_native.native_config()` from the
// Python module constants, so there is one home for each number.
#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pmw {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

// np.finfo(float).eps, which scipy's trust region and the Halir-Flusser fit both use.
constexpr double EPS = 2.220446049250313e-16;

// Python's round() and np.round(): half to even. std::nearbyint under the default
// FE_TONEAREST mode is exactly that; std::round is half away from zero and is wrong here.
inline double pyround(double v) { return std::nearbyint(v); }
// Python's `%` on floats: the result takes the sign of the divisor.
inline double pymod(double a, double m) {
    double r = std::fmod(a, m);
    return r < 0 ? r + m : r;
}

// OpenCV's ((cx, cy), (major, minor), angle_deg), major first once normalised.
struct Ellipse {
    double cx, cy, major, minor, ang;
};

// `calib/rig.py` Camera: K, distortion, and T_world_cam.
struct Camera {
    std::string name;
    Matrix3d K, Kinv, R;   // R = T_world_cam[:3, :3], camera -> world
    Vector3d T;            // T_world_cam[:3, 3]
    std::vector<double> dist;  // empty or all-zero means no distortion
    cv::Mat Kcv, distcv;
    bool has_dist() const;
    Vector3d optical_axis() const { return R.col(2); }
    std::pair<Vector3d, Vector3d> to_world(const Vector3d& c, const Vector3d& n) const {
        return {R * c + T, R * n};
    }
    std::pair<Vector3d, Vector3d> to_camera(const Vector3d& c, const Vector3d& n) const {
        return {R.transpose() * (c - T), R.transpose() * n};
    }
};

// `calib/shape.py` CentreCalibration: np.interp over tilt knots.
struct CentreCal {
    std::vector<double> knots, offsets;
    bool identity() const { return knots.size() < 2; }
    double offset(double tilt_deg) const;
};

// Every tuning constant, by the name it has in Python. Filled by `native_config()`;
// the loader throws on a missing key rather than defaulting, so a number cannot
// silently exist twice.
struct Config {
    // segment.py
    int ring_ksize;              // already rescaled to the frame, odd, >= 3
    double ring_blur_sigma, ring_plate_weight;
    int ring_samples;
    double ring_coverage_floor, roi_margin;
    int bg_diff_thresh;
    double blob_keep_fraction, ring_max_spread, shape_tol, ring_band;
    int ring_regrow_iters, max_anchors, min_contour_pts;
    bool axial;
    int axial_weight_iters;
    double axial_weight_power, axial_weight_floor, axial_skip_ratio, one_sided_weight;
    int one_sided_iters;
    double trim_fraction, trim_min_coverage;
    int trim_iters, trim_coverage_bins;
    double ring_seed_fraction, ring_seed_percentile;
    int undistort_samples;
    // stereo.py
    double refine_tol;
    int max_refine_iter;
    double ref_percentile, jac_rel_step, jac_distort_step_px, jac_grad_step_px,
        jac_min_resid_frac, far_px, cal_disp_tol_mm, cal_disp_tol_n;
    double max_discrepancy_mm;   // NaN: gate off
    double suspect_mm, min_ridge;
    double max_fit_rms_rel;      // NaN: gate off
    double dropout_s;
    int window_frames, min_window_support;
    double prior_sigma_deg, sigma_normal_rad;
    double accel_mm_s2;          // filter.py
    double sigma_lat_mm, sigma_depth_mm, radius_mm;
    int level;                   // the threshold level `segment` is handed
    double min_area;             // already rescaled
    double verify_tol;           // < 0: skip the reprojection check
    bool require_stereo;
};

// ---- evidence.cpp -----------------------------------------------------------------
struct Roi {
    int x, y, w, h;
};
Roi clamp_roi(double x, double y, double w, double h, int H, int W, int pad);
std::optional<Roi> ellipse_roi(const std::optional<Ellipse>& e, int H, int W, double margin);
// segment.ring_weight's plate-response cache, one per camera: the plate's own top-hat is
// reused while (version, box) holds. Python keys it on the running plate's frame
// counter // PLATE_REFRESH_FRAMES, so the response is up to 30 frames stale by design
// (control/theory.md 19.2); a saved plate has no version and is recomputed each frame.
struct PlateCache {
    bool valid = false;
    long long version = 0;
    bool has_box = false;
    Roi box{0, 0, 0, 0};
    cv::Mat response;
};
cv::Mat ring_weight(const cv::Mat& gray, const cv::Mat* plate, int k, double sigma,
                    double plate_weight, const std::optional<Roi>& roi,
                    PlateCache* cache = nullptr, long long version = -1);
std::vector<Vector2d> ellipse_points(const Ellipse& e, int n);
// Bilinear reads at interleaved (x, y) doubles, zero outside the frame. Hand-rolled to
// match the cv2 wheel's (OpenCV 5) remap kernel bit for bit; see evidence.cpp.
void sample_map(const cv::Mat& w, const double* xy, int n, double* out);

// ---- conic.cpp --------------------------------------------------------------------
Vector3d unit(const Vector3d& v);  // throws std::domain_error on a zero vector
std::pair<Vector3d, Vector3d> tangent_basis(const Vector3d& n);
Vector3d orient(const Vector3d& n, const Vector3d& reference);
double line_angle_deg(const Vector3d& u, const Vector3d& v);
Matrix3d cone_from_circle(const Vector3d& c, const Vector3d& n, double radius);
Ellipse normalise_ellipse(const Ellipse& e);
Matrix3d conic_from_ellipse(const Ellipse& e);  // throws std::domain_error
Ellipse ellipse_from_conic(const Matrix3d& c);   // throws std::domain_error
struct CirclePose {
    Vector3d center, normal;
};
std::vector<CirclePose> backproject(const Matrix3d& cone, double radius, double verify_tol);
std::vector<CirclePose> backproject_ellipse(const Ellipse& e, const Matrix3d& K, double radius,
                                            double verify_tol);
double ambiguity_margin_deg(const std::vector<CirclePose>& poses);
std::optional<Matrix3d> fit_conic_weighted(const std::vector<Vector2d>& pts,
                                           const std::vector<double>* weights);
std::optional<Vector2d> projected_axis_dir(const Vector3d& c_cam, const Vector3d& n_cam,
                                           const Matrix3d& K, double delta_mm = 5.0);
// Sampson distance of points to a conic; signed keeps the sign (positive outside).
std::vector<double> sampson(const Matrix3d& C, const std::vector<Vector2d>& pts, bool keep_sign);
double rms(const std::vector<double>& v);
double percentile_linear(std::vector<double> v, double q);  // np.percentile, 'linear'
double median(std::vector<double> v);

// ---- segment.cpp ------------------------------------------------------------------
struct Segmentation {
    cv::Mat mask;                 // empty for the tracked-ellipse fallback
    std::vector<Vector2d> hull;   // the convex hull, float64
    Ellipse ellipse;
    double area_px;
    int n_points;
    double fit_rms_px;
    int threshold;
    cv::Mat valid;                // the region, or empty
    std::string valid_from;       // "background" or ""
};
Ellipse fit_ellipse_direct(const std::vector<Vector2d>& pts);  // throws cv::Exception
std::optional<std::pair<Ellipse, double>> fit_ellipse(const std::vector<Vector2d>& pts,
                                                      const Config& cfg);
std::optional<Segmentation> segment(const cv::Mat& gray, const cv::Mat* plate, const Config& cfg);
std::optional<Segmentation> segment_ring(const cv::Mat& gray, const cv::Mat& weight,
                                         const Config& cfg);
Ellipse undistort_ellipse(const Ellipse& e, const Camera& cam, int n_samples);

// ---- trf.cpp ----------------------------------------------------------------------
// scipy.optimize.least_squares(method="trf", loss="cauchy", x_scale="jac",
// tr_solver="exact") without bounds, ported call for call.
struct TrfResult {
    VectorXd x, f;   // f: the raw residual at x (scipy's `fun`)
    int nfev;
    int status;      // scipy's termination status; > 0 is success
};
TrfResult trf_cauchy(const std::function<VectorXd(const VectorXd&)>& fun,
                     const std::function<MatrixXd(const VectorXd&)>& jac, const VectorXd& x0,
                     double ftol, double xtol, double gtol, int max_nfev, double f_scale);

// ---- refine.cpp -------------------------------------------------------------------
struct Refinement {
    Vector3d center, normal;
    double rms_px;
    int n_iter;
    bool converged;
    MatrixXd samples;   // (n_views, ring_samples) evidence at the solution
    double evidence;
};
// Intermediates for `native_parity`, filled when a pointer is passed.
struct RefineDebug {
    std::vector<MatrixXd> pts0;   // per view: the (n, 2) raw pixels sampled at the seed
    VectorXd e0;                  // evidence at the seed
    double ref = 0, f_scale = 0;
    MatrixXd J0;                  // Jacobian at the seed
    std::vector<VectorXd> trace;  // every x the solver evaluated, in order
    std::vector<VectorXd> x_eval; // in: points to evaluate f and J at
    std::vector<VectorXd> f_eval;
    std::vector<MatrixXd> J_eval;
};
std::optional<Refinement> refine_image(const std::vector<cv::Mat>& maps,
                                       const std::vector<Camera>& cams, const Vector3d& c0,
                                       const Vector3d& n0, const CentreCal& centre_cal,
                                       const Config& cfg, const Vector3d& reference,
                                       RefineDebug* dbg = nullptr);

// ---- stereo.cpp -------------------------------------------------------------------
struct Match {
    std::vector<CirclePose> poses;
    std::vector<int> indices;
    double discrepancy_mm, margin;
};
std::optional<Match> match(const std::vector<std::vector<CirclePose>>& cands,
                           const std::vector<Camera>& cams, const Config& cfg,
                           const Vector3d* prior_normal);
struct Fused {
    Vector3d center, normal;
    Matrix3d cov;
};
Fused fuse(const std::vector<CirclePose>& poses, const std::vector<Camera>& cams,
           const Config& cfg, const Vector3d& reference, const std::vector<double>* stamps,
           const Vector3d* velocity, const Matrix3d* vel_cov);

struct ViewResult {
    std::optional<Segmentation> seg;
    std::vector<CirclePose> cands;
    std::optional<Ellipse> ellipse;   // undistorted
    cv::Mat weight;
};

struct PoseResult {
    Vector3d center, normal;   // world, before zeroing
    std::vector<ViewResult> views;
    std::vector<int> usable;
    Match match;
    double refine_rms_px;
    int refine_iters;
    double jump_deg, t_seg_ms, t_est_ms, skew_ms, union_coverage;
    std::optional<MatrixXd> alive;   // (n_views, n) as 0/1
    int n_solutions;
    double ambiguity_margin_deg;
};

class Estimator {
public:
    Estimator(std::vector<Camera> cams, Config cfg, CentreCal centre_cal, Vector3d reference);
    std::optional<PoseResult> update(const cv::Mat& frameA, const cv::Mat& frameB,
                                     const cv::Mat* plateA, const cv::Mat* plateB,
                                     long long versionA, long long versionB, double now,
                                     const std::vector<double>* stamps, const Vector3d* velocity,
                                     const Matrix3d* vel_cov);
    void reset();
    ViewResult view_candidates(const cv::Mat& gray, const cv::Mat* plate, long long version, int cam_index);

    //: Per-view reuse for the interleaved tracker, where one view is unchanged on every
    //: call. Pass the grabber's frame sequence number for each view; a view whose seq
    //: matches the previous call reuses its `ViewResult` verbatim instead of segmenting
    //: again. 0 (the default) means "no seq known", which always recomputes -- so every
    //: existing caller, `stereo_native.py` and `native_parity.py` included, is unaffected.
    //:
    //: Safe because `view_candidates` touches only `prev_ellipse[ci]` and
    //: `plate_cache_[ci]`, both per-view, and writes `prev_ellipse[ci]` from its own
    //: segmentation rather than from the refined pose. A view with no new frame has
    //: nothing to advance. Keyed on the seq and NEVER on the `Mat`'s data pointer: the
    //: cache `pose/theory.md` 19.1 dissects missed 100% of the time because it keyed on
    //: `id(img)` of an array reallocated every frame. Version the content, not the buffer.
    void set_frame_seq(std::uint64_t seqA, std::uint64_t seqB);

    //: The segmentation threshold, live.
    //:
    //: It reaches the core as `Config.level`, which is read once at construction, and
    //: `stereo_native._ensure_native` only rebuilds on a frame-scale change -- so
    //: `live_viz`'s `est.thresh = viz.thresh` has done nothing since the native core
    //: became the default. Tuning against a recording means dragging the slider and
    //: watching the same pass again, which is the whole point of replaying.
    void set_thresh(int level) { cfg_.level = level; }

    int n_detected = 0, n_lost = 0, n_rejected = 0, n_rejected_fit = 0, n_rejected_mono = 0,
        n_rearbitrated = 0;
    std::vector<std::optional<Ellipse>> prev_ellipse;

private:
    std::optional<Vector3d> window_normal(double now) const;
    std::vector<Camera> cams_;
    Config cfg_;
    CentreCal centre_cal_;
    Vector3d reference_;
    std::deque<std::pair<double, Vector3d>> window_;
    std::vector<PlateCache> plate_cache_;
    std::optional<Vector3d> prev_normal_;
    //: The view cache `set_frame_seq` drives. `seq_[i] == 0` disables it for that view.
    std::uint64_t seq_[2] = {0, 0}, seq_done_[2] = {0, 0};
    ViewResult view_cache_[2];
};

}  // namespace pmw
