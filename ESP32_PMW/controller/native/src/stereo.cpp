// stereo.py: _pair_information, _agreement, match, fuse, _window_normal, and the
// per-frame flow of StereoPoseEstimator.update for the live configuration
// (direct=True, do_refine=True, undistort=True, tilt_cal identity, the centre
// correction in the forward model only).
#include "pmw.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <thread>

namespace pmw {

namespace {

Matrix3d view_cov(const Camera& cam, double sl, double sd) {
    Vector3d d = cam.optical_axis();
    return (sl * sl) * Matrix3d::Identity() + (sd * sd - sl * sl) * (d * d.transpose());
}

double agreement(const CirclePose& pa, const CirclePose& pb, const Camera& ca, const Camera& cb,
                 const Matrix3d& info, double sigma_normal_rad) {
    auto [cwa, nwa] = ca.to_world(pa.center, pa.normal);
    auto [cwb, nwb] = cb.to_world(pb.center, pb.normal);
    Vector3d d = cwa - cwb;
    double pos = d.dot(info * d);
    double c = std::min(1.0, std::abs(unit(nwa).dot(unit(nwb))));
    double sin_ang = std::sqrt(std::max(0.0, 1.0 - c * c));
    return pos + (sin_ang / sigma_normal_rad) * (sin_ang / sigma_normal_rad);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::optional<Match> match(const std::vector<std::vector<CirclePose>>& cands,
                           const std::vector<Camera>& cams, const Config& cfg,
                           const Vector3d* prior_normal) {
    for (const auto& c : cands)
        if (c.empty()) return std::nullopt;
    if (cands.size() == 1) return Match{{cands[0][0]}, {0}, 0.0, 0.0};
    const Camera& ca = cams[0];
    const Camera& cb = cams[1];
    Matrix3d total = view_cov(ca, cfg.sigma_lat_mm, cfg.sigma_depth_mm) +
                     view_cov(cb, cfg.sigma_lat_mm, cfg.sigma_depth_mm);
    Matrix3d info = total.inverse();
    std::optional<Vector3d> prior;
    if (prior_normal) prior = unit(*prior_normal);
    double inv_sig = 1.0 / (std::max(cfg.prior_sigma_deg, 1e-3) * M_PI / 180.0);
    struct Scored {
        double score;
        int i, j;
    };
    std::vector<Scored> scored;
    for (int i = 0; i < (int)cands[0].size(); ++i) {
        for (int j = 0; j < (int)cands[1].size(); ++j) {
            const CirclePose& pa = cands[0][i];
            const CirclePose& pb = cands[1][j];
            double score = agreement(pa, pb, ca, cb, info, cfg.sigma_normal_rad);
            if (prior) {
                Vector3d nw = ca.to_world(pa.center, pa.normal).second;
                double a = line_angle_deg(nw, *prior) * M_PI / 180.0 * inv_sig;
                score += a * a;
            }
            scored.push_back({score, i, j});
        }
    }
    std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        if (a.score != b.score) return a.score < b.score;
        if (a.i != b.i) return a.i < b.i;
        return a.j < b.j;
    });
    const Scored& best = scored[0];
    double runner_up = scored.size() > 1 ? scored[1].score : INFINITY;
    Vector3d cwa = ca.to_world(cands[0][best.i].center, cands[0][best.i].normal).first;
    Vector3d cwb = cb.to_world(cands[1][best.j].center, cands[1][best.j].normal).first;
    return Match{{cands[0][best.i], cands[1][best.j]}, {best.i, best.j}, (cwa - cwb).norm(),
                 runner_up - best.score};
}

Fused fuse(const std::vector<CirclePose>& poses, const std::vector<Camera>& cams, const Config& cfg,
           const Vector3d& reference, const std::vector<double>* stamps, const Vector3d* velocity,
           const Matrix3d* vel_cov) {
    Matrix3d info = Matrix3d::Zero();
    Vector3d info_c = Vector3d::Zero(), n_acc = Vector3d::Zero();
    std::optional<Vector3d> ref_n;
    std::vector<double> offsets(poses.size(), 0.0);
    if (stamps) {
        double mean = 0.0;
        for (double s : *stamps) mean += s;
        mean /= (double)stamps->size();
        for (size_t i = 0; i < poses.size(); ++i) offsets[i] = mean - (*stamps)[i];
    }
    for (size_t i = 0; i < poses.size(); ++i) {
        const Camera& cam = cams[i];
        auto [c_w, n_w] = cam.to_world(poses[i].center, poses[i].normal);
        Vector3d d = cam.optical_axis();
        Matrix3d cov = view_cov(cam, cfg.sigma_lat_mm, cfg.sigma_depth_mm);
        double dt = offsets[i];
        if (dt != 0.0) {
            if (velocity) c_w = c_w + (*velocity) * dt;
            if (vel_cov) cov = cov + (*vel_cov) * (dt * dt);
            double a = 0.5 * cfg.accel_mm_s2 * dt * dt;
            cov = cov + (a * a) * Matrix3d::Identity();
        }
        Matrix3d inv = cov.inverse();
        info += inv;
        info_c += inv * c_w;
        double nd = n_w.dot(d);
        double w = 1.0 - nd * nd;
        if (!ref_n) ref_n = n_w;
        n_acc += w * (n_w.dot(*ref_n) >= 0 ? n_w : Vector3d(-n_w));
    }
    Fused out;
    out.cov = info.inverse();
    out.center = out.cov * info_c;
    out.normal = orient(n_acc.norm() > 1e-12 ? n_acc : *ref_n, reference);
    return out;
}

Estimator::Estimator(std::vector<Camera> cams, Config cfg, CentreCal centre_cal, Vector3d reference)
    : cams_(std::move(cams)), cfg_(std::move(cfg)), centre_cal_(std::move(centre_cal)),
      reference_(std::move(reference)) {
    prev_ellipse.assign(cams_.size(), std::nullopt);
    plate_cache_.resize(cams_.size());
}

void Estimator::reset() {
    prev_normal_.reset();
    prev_ellipse.assign(cams_.size(), std::nullopt);
    window_.clear();
    // The view cache holds evidence maps and candidates from before the reset. Dropping
    // the SEQS is what matters -- a stale `seq_done_` would let the next call reuse a
    // view from the run we just reset away.
    for (int i = 0; i < 2; ++i) { seq_[i] = seq_done_[i] = 0; view_cache_[i] = ViewResult(); }
}

void Estimator::set_frame_seq(std::uint64_t seqA, std::uint64_t seqB) {
    seq_[0] = seqA;
    seq_[1] = seqB;
}

std::optional<Vector3d> Estimator::window_normal(double now) const {
    if ((int)window_.size() < cfg_.min_window_support) return std::nullopt;
    double t_last = window_.back().first;
    double gap = now - t_last;
    if (!(0.0 <= gap && gap <= cfg_.dropout_s)) return std::nullopt;
    const Vector3d& last = window_.back().second;
    std::vector<double> comp[3];
    for (const auto& [t, nrm] : window_) {
        double sgn = nrm.dot(last) < 0.0 ? -1.0 : 1.0;
        for (int k = 0; k < 3; ++k) comp[k].push_back(nrm[k] * sgn);
    }
    Vector3d med(median(comp[0]), median(comp[1]), median(comp[2]));
    double norm = med.norm();
    if (norm < 1e-9) return std::nullopt;
    return Vector3d(med / norm);
}

ViewResult Estimator::view_candidates(const cv::Mat& gray, const cv::Mat* plate, long long version, int ci) {
    ViewResult out;
    const Camera& cam = cams_[ci];
    auto roi = ellipse_roi(prev_ellipse[ci], gray.rows, gray.cols, cfg_.roi_margin);
    out.weight = ring_weight(gray, plate, cfg_.ring_ksize, cfg_.ring_blur_sigma, cfg_.ring_plate_weight, roi,
                             &plate_cache_[ci], version);
    std::optional<Segmentation> seg =
        plate ? segment(gray, plate, cfg_) : segment_ring(gray, out.weight, cfg_);
    if (!seg && prev_ellipse[ci]) {
        // The rim was here a frame ago: seed from it, and let the joint solve measure
        // the pose against this frame's evidence map.
        Segmentation s;
        s.ellipse = *prev_ellipse[ci];
        auto pts = ellipse_points(s.ellipse, cfg_.ring_samples);
        s.hull = pts;
        s.area_px = 0.0;
        s.n_points = cfg_.ring_samples;
        s.fit_rms_px = NAN;
        s.threshold = 0;
        seg = s;
    }
    if (!seg) {
        prev_ellipse[ci] = std::nullopt;
        return out;
    }
    prev_ellipse[ci] = seg->ellipse;
    Ellipse ellipse = seg->ellipse;
    if (cam.has_dist()) {
        try {
            ellipse = undistort_ellipse(ellipse, cam, cfg_.undistort_samples);
        } catch (const std::exception&) {
            ellipse = seg->ellipse;   // keep the distorted fit rather than lose the view
        }
    }
    out.seg = seg;
    out.ellipse = ellipse;
    try {
        out.cands = backproject_ellipse(ellipse, cam.K, cfg_.radius_mm, cfg_.verify_tol);
    } catch (const std::domain_error&) {
        out.cands.clear();
    }
    return out;
}

std::optional<PoseResult> Estimator::update(const cv::Mat& frameA, const cv::Mat& frameB,
                                            const cv::Mat* plateA, const cv::Mat* plateB,
                                            long long versionA, long long versionB, double now,
                                            const std::vector<double>* stamps, const Vector3d* velocity,
                                            const Matrix3d* vel_cov) {
    double skew_ms = 0.0;
    if (stamps && !stamps->empty())
        skew_ms = (*std::max_element(stamps->begin(), stamps->end()) -
                   *std::min_element(stamps->begin(), stamps->end())) * 1e3;

    auto t_seg0 = std::chrono::steady_clock::now();
    std::vector<ViewResult> views(2);
    // Reuse a view whose frame has not changed since the last call. Only the interleaved
    // tracker sets a seq; everything else leaves them 0 and both views recompute, which
    // is the behaviour every existing caller has. See `set_frame_seq`.
    const bool hit[2] = {seq_[0] != 0 && seq_[0] == seq_done_[0],
                         seq_[1] != 0 && seq_[1] == seq_done_[1]};
    if (hit[0] && hit[1]) {
        // Nothing moved in either view. The caller should not have called, but reusing
        // both is still the right answer and costs nothing.
        views[0] = view_cache_[0];
        views[1] = view_cache_[1];
    } else if (hit[0] || hit[1]) {
        // One fresh view: run it on THIS thread. Spawning a second thread to hand back a
        // cached struct costs more than the copy -- and `pose/theory.md` 19.4's rule that
        // thread creation must not be per-frame is why the fresh path below is the only
        // place that still spawns one.
        const int fresh = hit[0] ? 1 : 0;
        views[1 - fresh] = view_cache_[1 - fresh];
        views[fresh] = fresh == 0 ? view_candidates(frameA, plateA, versionA, 0)
                                  : view_candidates(frameB, plateB, versionB, 1);
    } else {
        // The two views share nothing, so one runs on its own thread (theory.md 19.7).
        std::thread other([&] { views[1] = view_candidates(frameB, plateB, versionB, 1); });
        views[0] = view_candidates(frameA, plateA, versionA, 0);
        other.join();
    }
    for (int i = 0; i < 2; ++i) {
        if (seq_[i] == 0) continue;
        view_cache_[i] = views[i];
        seq_done_[i] = seq_[i];
    }
    double t_seg_ms = ms_since(t_seg0);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<int> usable;
    for (int i = 0; i < 2; ++i)
        if (!views[i].cands.empty()) usable.push_back(i);
    if (usable.empty()) { n_lost++; return std::nullopt; }
    if (cfg_.require_stereo && usable.size() < 2) { n_rejected_mono++; n_lost++; return std::nullopt; }

    std::vector<Camera> sub;
    std::vector<std::vector<CirclePose>> cands;
    for (int i : usable) { sub.push_back(cams_[i]); cands.push_back(views[i].cands); }
    auto m = match(cands, sub, cfg_, nullptr);
    if (!m) { n_lost++; return std::nullopt; }

    auto prior = window_normal(now);
    if (prior && usable.size() > 1 && m->discrepancy_mm > cfg_.suspect_mm) {
        auto alt = match(cands, sub, cfg_, &*prior);
        if (alt && alt->indices != m->indices) { n_rearbitrated++; m = alt; }
    }
    if (std::isfinite(cfg_.max_discrepancy_mm) && usable.size() > 1 &&
        m->discrepancy_mm > cfg_.max_discrepancy_mm) {
        n_rejected++; n_lost++; return std::nullopt;
    }

    std::vector<double> sub_stamps;
    if (stamps) for (int i : usable) sub_stamps.push_back((*stamps)[i]);
    Fused fused = fuse(m->poses, sub, cfg_, reference_, stamps ? &sub_stamps : nullptr, velocity, vel_cov);
    Vector3d centre = fused.center, normal = fused.normal;

    // Fit-quality gate on the per-view outline fits (MAX_FIT_RMS_REL). `ridge` is NaN on
    // this path, so the ridge half never fires; the max() below keeps Python's own
    // semantics for a NaN entry (a comparison with NaN is false, so it is skipped
    // unless it comes first).
    if (std::isfinite(cfg_.max_fit_rms_rel)) {
        bool first = true;
        double worst = 0.0;
        for (int i : usable) {
            const Segmentation& s = *views[i].seg;
            double v = s.fit_rms_px / std::max(s.ellipse.major, 1e-9);
            if (first) { worst = v; first = false; }
            else if (v > worst) worst = v;
        }
        if (worst > cfg_.max_fit_rms_rel) { n_rejected_fit++; n_lost++; return std::nullopt; }
    }

    double rms_px = NAN;
    int iters = 0;
    double union_cov = NAN;
    std::optional<MatrixXd> alive;
    {
        std::vector<cv::Mat> maps;
        for (int i : usable) maps.push_back(views[i].weight);
        auto r = refine_image(maps, sub, centre, normal, centre_cal_, cfg_, reference_);
        if (r) {
            centre = r->center;
            normal = r->normal;
            rms_px = r->rms_px;
            iters = r->n_iter;
            int nv = (int)r->samples.rows(), n = (int)r->samples.cols();
            MatrixXd a(nv, n);
            for (int vi = 0; vi < nv; ++vi) {
                std::vector<double> row(r->samples.row(vi).data(), r->samples.row(vi).data() + n);
                std::vector<double> rowv(n);
                for (int k = 0; k < n; ++k) rowv[k] = r->samples(vi, k);
                double med = median(rowv);
                double floor = cfg_.ring_coverage_floor * std::max(med, 1e-6);
                for (int k = 0; k < n; ++k) a(vi, k) = r->samples(vi, k) >= floor ? 1.0 : 0.0;
            }
            int any = 0;
            for (int k = 0; k < n; ++k) any += a.col(k).maxCoeff() > 0;
            union_cov = (double)any / (double)n;
            alive = a;
        }
    }

    double jump = NAN;
    if (prev_normal_) jump = line_angle_deg(normal, *prev_normal_);
    prev_normal_ = normal;
    window_.emplace_back(now, unit(normal));
    while ((int)window_.size() > cfg_.window_frames) window_.pop_front();
    n_detected++;

    PoseResult out;
    out.center = centre;
    out.normal = normal;
    out.views = std::move(views);
    out.usable = usable;
    out.match = *m;
    out.refine_rms_px = rms_px;
    out.refine_iters = iters;
    out.jump_deg = jump;
    out.t_seg_ms = t_seg_ms;
    out.skew_ms = skew_ms;
    out.union_coverage = union_cov;
    out.alive = alive;
    const auto& ref_cands = out.views[usable[0]].cands;
    out.n_solutions = (int)ref_cands.size();
    out.ambiguity_margin_deg = ambiguity_margin_deg(ref_cands);
    out.t_est_ms = ms_since(t0);
    return out;
}

}  // namespace pmw
