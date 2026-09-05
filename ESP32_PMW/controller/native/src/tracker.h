// `Tracker`: two cameras, a pose worker, and a latest-pose slot. The whole live path.
//
// ## Why the frame slot is not consumed
//
// `sources.MonoCamera.read` takes the frame out of its slot and blocks until the next
// one arrives, so a stereo read is bounded by the SLOWER camera and the pair rate can
// never exceed one camera's rate. Here the slot always holds the newest frame and
// `seq` -- not the slot's emptiness -- is what says "new", the same rule
// `hover_controller_runner._PoseFeed` states at :235.
//
// That is what makes **interleaving** possible. The two cameras free-run with
// uncorrelated phase (`camera/theory.md` 1.4), so firing a pose on every frame from
// *either* camera, paired with the other's newest, roughly doubles the observation rate
// at 640x400 without touching the resolution, the FOV or the calibration. The partner
// view is then at most one frame period old -- which is not a new problem: the live path
// already runs `max_skew_s=None`, so today's pairs carry the same uniform(0, T) skew,
// and `fuse` already moves both views to a common instant from their per-view stamps
// (`pose/theory.md` 17).
//
// ## The guard that replaces the timeout
//
// A non-consuming slot has one failure mode a consuming one does not: a camera that
// stops delivering leaves its last frame in the slot forever, and the tracker would go
// on producing confident poses against a frozen view. `MonoCamera.read`'s 2 s timeout is
// what does that job today. Here it is `max_skew_s`: a dead camera makes the pair's skew
// grow without bound, every pair is refused, poses stop, and `wait()` times out into the
// caller's existing "camera was unplugged" path. `stats()["n_grabbed"]` per camera says
// which one died.

#pragma once

#include "capture.h"
#include "pmw.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <string>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace pmw {

namespace {
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

//: `background.RunningPlate`, in the worker.
//:
//: The same four pieces of state (`background.py:98-102`) and the same arithmetic as the
//: `running_plate_update` binding. Two semantics that are easy to lose in a port and
//: change which branch the segmenter takes, so they are spelled out:
//:
//:  - `update` returns nothing until `n >= warmup` (WARMUP_FRAMES = 5). A null plate
//:    sends `view_candidates` down `segment_ring`; a non-null one sends it down
//:    `segment`. The branch flips at each camera's 5th frame.
//:  - the cache version is `n / PLATE_REFRESH_FRAMES` taken **before** the tick, which
//:    is what `stereo_native.py:226` does.
class RunningPlate {
public:
    RunningPlate(double step, int warmup, long long refresh)
        : step_(step), warmup_(warmup), refresh_(refresh) {}

    long long version() const { return n_ / refresh_; }

    //: Fold one frame in. Returns null until warm. The returned Mat is a fresh buffer
    //: every call, never a view of `bg_`, so the segmenter cannot read a plate the next
    //: frame is midway through mutating.
    const cv::Mat* update(const cv::Mat& gray) {
        if (bg_.empty() || bg_.size() != gray.size()) {
            gray.convertTo(bg_, CV_32F);
        } else {
            // np.sign then a scaled add, NOT a clipped difference: the step must not
            // depend on how far off the pixel is, or it becomes a mean (background.py:152).
            cv::Mat f;
            gray.convertTo(f, CV_32F);
            cv::Mat d = f - bg_;
            cv::Mat sgn(d.size(), CV_32F);
            // sign(x) as (x>0) - (x<0), which matches np.sign including sign(0) == 0.
            cv::Mat pos, neg;
            cv::compare(d, 0.0, pos, cv::CMP_GT);
            cv::compare(d, 0.0, neg, cv::CMP_LT);
            pos.convertTo(pos, CV_32F, 1.0 / 255.0);
            neg.convertTo(neg, CV_32F, 1.0 / 255.0);
            sgn = pos - neg;
            bg_ += step_ * sgn;
        }
        n_++;
        if (n_ < warmup_) return nullptr;
        bg_.convertTo(out_, CV_8U);   // truncating, as `self.bg.astype(np.uint8)` is
        return &out_;
    }

    void reset() { bg_.release(); out_.release(); n_ = 0; }

private:
    double step_;
    long long warmup_, refresh_, n_ = 0;
    cv::Mat bg_, out_;
};

enum class PairMode { Interleave, Both };

struct TrackerStats {
    std::uint64_t n_grabbed[2] = {0, 0};
    std::uint64_t n_pose = 0, n_lost = 0, n_skew_dropped = 0;
    // The estimator's own gate counters, copied out by `stats()`. `n_lost` alone says a
    // frame produced no pose; these say WHICH stage refused it, which is the difference
    // between diagnosing and guessing.
    int n_detected = 0, n_rejected = 0, n_rejected_fit = 0, n_rejected_mono = 0,
        n_rearbitrated = 0;
    std::string event[2];        // what AVFoundation last said about each session
    std::uint64_t n_wrong_size[2] = {0, 0};   // frames dropped for the wrong resolution
    std::uint64_t n_blank_raw[2] = {0, 0}, n_blank_out[2] = {0, 0};
    double t_start = 0.0, t_last_pose = 0.0;
    double age_ms[2] = {-1.0, -1.0};   // filled by stats(), not accumulated
    std::vector<double> skews;   // seconds, capped -- see `note_skew`
};

class Tracker : public FrameSink {
public:
    Tracker(std::vector<Camera> cams, Config cfg, CentreCal centre_cal, Vector3d reference,
            std::vector<CameraSpec> specs, PairMode mode, double max_skew_s,
            double plate_step, int plate_warmup, long long plate_refresh)
        : est_(std::move(cams), std::move(cfg), std::move(centre_cal), std::move(reference)),
          specs_(std::move(specs)), mode_(mode), max_skew_s_(max_skew_s) {
        if (specs_.size() != 2) throw std::runtime_error("Tracker needs exactly two cameras");
        for (int i = 0; i < 2; ++i)
            plates_.emplace_back(plate_step, plate_warmup, plate_refresh);
    }

    ~Tracker() override { stop(); }

    // ---- FrameSink -----------------------------------------------------------------
    //
    // On AVFoundation's delivery queue, one per camera, so both can be here at once.
    // Does as little as possible: store, bump seq, wake the worker. The plate tick and
    // the pose live on the worker, so a slow solve cannot back up a capture queue.
    void deliver(int ci, cv::Mat gray, double t, double pts) override {
        {
            std::lock_guard<std::mutex> lk(m_);
            Frame& f = slot_[ci];
            f.gray = std::move(gray);
            f.t = t;
            f.pts = pts;
            f.seq++;
        }
        cv_.notify_all();
    }

    // ---- lifecycle -----------------------------------------------------------------
    void start() {
        if (running_) return;
        running_ = true;
        stats_.t_start = now_s();
        // Cameras first, then the worker: a worker that wakes before any frame exists
        // simply waits, but a camera delivering into a tracker with no worker would
        // silently fill and overwrite the slot.
        for (int i = 0; i < 2; ++i) {
            cams_[i] = open_camera(specs_[i], i, this);
            cams_[i]->start();
        }
        // Check what actually ARRIVES, not what was configured. A session preset can
        // override `activeFormat` as the input is added and leave the format reading
        // correctly while another size is delivered -- the undetectable substitution
        // `camera/theory.md` 1.2 is about, which a format check cannot catch. Measured
        // once for real (640x400 asked, 1280x800 delivered), so this is a fix, not a
        // precaution. Fail loudly here rather than let the rig fly on frames whose
        // scale silently multiplies every distance downstream.
        for (int i = 0; i < 2; ++i) {
            cv::Size got;
            for (int k = 0; k < 300 && got.width == 0; ++k) {
                got = cams_[i]->delivered_size();
                if (got.width == 0) std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (got.width == 0) {
                stop_cameras();
                throw std::runtime_error("camera " + std::to_string(i) + " (" +
                    specs_[i].unique_id + ") delivered no frame in 3 s");
            }
            if (got != cv::Size(specs_[i].width, specs_[i].height)) {
                stop_cameras();
                throw std::runtime_error(
                    "camera " + std::to_string(i) + " (" + specs_[i].unique_id +
                    ") delivers " + std::to_string(got.width) + "x" +
                    std::to_string(got.height) + " but " +
                    std::to_string(specs_[i].width) + "x" +
                    std::to_string(specs_[i].height) + " was configured -- the session "
                    "overrode activeFormat");
            }
        }
        worker_ = std::thread([this] { run(); });
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        // Cameras down first so nothing new arrives, THEN wake and join the worker,
        // THEN let the cameras destruct. A camera released while its delivery queue is
        // still in `deliver` writes into a dead slot.
        stop_cameras();
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        for (auto& c : cams_) c.reset();
    }

    // ---- what Python reads ---------------------------------------------------------
    //: Newest pose and its sequence number. Copies out under the lock and returns; the
    //: caller builds the Python object AFTER releasing it. Holding this mutex across a
    //: GIL acquisition would deadlock against a Python thread already holding the GIL
    //: and waiting here.
    bool latest(std::uint64_t* seq, PoseResult* out, double* t, std::vector<double>* stamps) {
        std::lock_guard<std::mutex> lk(m_);
        if (pose_seq_ == 0 || !pose_) return false;
        *seq = pose_seq_;
        *out = *pose_;
        *t = pose_t_;
        *stamps = pose_stamps_;
        return true;
    }

    //: Block until the pose sequence passes `after`, or the timeout expires.
    bool wait(std::uint64_t after, double timeout_s, std::uint64_t* seq, PoseResult* out,
              double* t, std::vector<double>* stamps) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                     [&] { return !running_ || (pose_seq_ > after && pose_); });
        if (pose_seq_ <= after || !pose_) return false;
        *seq = pose_seq_;
        *out = *pose_;
        *t = pose_t_;
        *stamps = pose_stamps_;
        return true;
    }

    //: The newest frame pair, for the recorder and the viz. Clones, so the caller can
    //: hold them while the cameras run on.
    static double sampled_mean(const cv::Mat& m) {
        if (m.empty()) return -1.0;
        long sum = 0; int cnt = 0;
        for (int y = 0; y < m.rows; y += 8) {
            const std::uint8_t* row = m.ptr<std::uint8_t>(y);
            for (int x = 0; x < m.cols; x += 8) { sum += row[x]; ++cnt; }
        }
        return cnt ? double(sum) / cnt : -1.0;
    }

    void frames(cv::Mat* a, cv::Mat* b, double* t, std::vector<double>* stamps,
                double* slot_mean_a = nullptr, double* slot_mean_b = nullptr) {
        std::lock_guard<std::mutex> lk(m_);
        // Measured BEFORE the clone, so a blank slot can be told from a blank copy.
        if (slot_mean_a) *slot_mean_a = sampled_mean(slot_[0].gray);
        if (slot_mean_b) *slot_mean_b = sampled_mean(slot_[1].gray);
        *a = slot_[0].gray.clone();
        *b = slot_[1].gray.clone();
        *stamps = {slot_[0].t, slot_[1].t};
        *t = 0.5 * (slot_[0].t + slot_[1].t);
    }

    //: The filter's current velocity, pushed back in from Python.
    //:
    //: Under the lock because a torn `Matrix3d` read could be non-PSD and `fuse` inverts
    //: it. The staleness itself is not new: `live_viz.py:1503` passes `filt.pos` and
    //: only updates the filter at :1504, so the estimator has always seen the previous
    //: pose's velocity.
    void set_motion(const Vector3d* v, const Matrix3d* c) {
        std::lock_guard<std::mutex> lk(motion_m_);
        has_motion_ = v != nullptr;
        if (v) vel_ = *v;
        if (c) vel_cov_ = *c;
        has_cov_ = c != nullptr;
    }

    //: The segmentation threshold, live.
    //:
    //: Deliberately NOT under `m_`: the worker runs `est_.update` outside that lock, so
    //: taking it here would look like synchronisation without providing any. What this
    //: really is, is a write to one aligned `int` racing a read of it -- which cannot
    //: tear, and whose worst case is one frame segmented at the previous threshold.
    //: That is exactly what a slider wants.
    void set_thresh(int level) { est_.set_thresh(level); }

    TrackerStats stats() {
        std::lock_guard<std::mutex> lk(m_);
        TrackerStats s = stats_;
        const double now = now_s();
        for (int i = 0; i < 2; ++i) {
            s.n_grabbed[i] = cams_[i] ? cams_[i]->n_grabbed() : 0;
            // Age of each camera's newest frame. THE number that says which camera
            // stopped: with a slot that is never consumed, a dead camera is invisible
            // in `n_grabbed` (which only stops rising) and shows up downstream as
            // every pair failing the skew guard -- a symptom that names neither camera.
            s.age_ms[i] = slot_[i].seq == 0 ? -1.0 : (now - slot_[i].t) * 1e3;
            s.event[i] = cams_[i] ? cams_[i]->last_event() : std::string();
            s.n_wrong_size[i] = cams_[i] ? cams_[i]->n_wrong_size() : 0;
            s.n_blank_raw[i] = cams_[i] ? cams_[i]->n_blank_raw() : 0;
            s.n_blank_out[i] = cams_[i] ? cams_[i]->n_blank_out() : 0;
        }
        s.n_detected = est_.n_detected;
        s.n_rejected = est_.n_rejected;
        s.n_rejected_fit = est_.n_rejected_fit;
        s.n_rejected_mono = est_.n_rejected_mono;
        s.n_rearbitrated = est_.n_rearbitrated;
        return s;
    }

    void reset() {
        std::lock_guard<std::mutex> lk(m_);
        est_.reset();
        for (auto& p : plates_) p.reset();
        pose_.reset();
        pose_seq_ = 0;
        done_[0] = done_[1] = 0;
    }

    //: Feed a frame as if a camera had delivered it. The offline self-check drives the
    //: whole slot/pairing/worker/cache path from a recording's mp4s with no camera and
    //: no AVFoundation, which is the one thing writing our own capture layer cost us.
    void push_frame(int ci, const cv::Mat& gray, double t) { deliver(ci, gray.clone(), t, t); }

    //: Run one pairing step synchronously, for `push_frame` callers. Returns whether a
    //: pose came out. Identical code to the worker's body -- see `step`.
    bool pump() { return step(); }

    Estimator& estimator() { return est_; }

private:
    void stop_cameras() { for (auto& c : cams_) if (c) c->stop(); }

    void run() {
        while (running_) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return !running_ || ready(); });
                if (!running_) return;
            }
            step();
        }
    }

    //: Is there a pair worth solving? Both cameras must have produced something, and
    //: then it depends on the mode.
    bool ready() const {
        if (slot_[0].seq == 0 || slot_[1].seq == 0) return false;
        const bool a = slot_[0].seq != done_[0], b = slot_[1].seq != done_[1];
        return mode_ == PairMode::Interleave ? (a || b) : (a && b);
    }

    //: One pose. Snapshot under the lock, solve outside it, publish under it again.
    bool step() {
        Frame a, b;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (!ready()) return false;
            // **Deep copies, not header copies.** `a = slot_[0]` shares the pixel
            // buffer with the live slot, and the estimator then holds that buffer while
            // the grabber goes on writing -- so the frame under the solve is not the
            // frame it was handed. Measured: the slot read back blank on 40% of samples
            // while the delegate handed over 0 blanks in 2138 frames. The pose worker
            // must own the pixels it is reasoning about.
            a.t = slot_[0].t; a.pts = slot_[0].pts; a.seq = slot_[0].seq;
            b.t = slot_[1].t; b.pts = slot_[1].pts; b.seq = slot_[1].seq;
            a.gray = slot_[0].gray.clone();
            b.gray = slot_[1].gray.clone();
            done_[0] = a.seq;
            done_[1] = b.seq;
        }

        const double skew = std::abs(a.t - b.t);
        if (skew > max_skew_s_) {
            // The accuracy gate and the dead-camera guard at once: a stopped camera
            // makes this grow without bound and every pair is refused from then on.
            std::lock_guard<std::mutex> lk(m_);
            stats_.n_skew_dropped++;
            note_skew(skew);
            return false;
        }

        // The plate ticks once per frame this worker actually CONSUMES, not once per
        // pose: an unchanged view has no new pixels to fold in, and ticking it again
        // would walk the plate at twice the camera's rate. Frames the worker is too far
        // behind to see are skipped here too, which is right -- the plate should
        // describe the frames the estimator used.
        long long ver[2];
        const cv::Mat* plate[2];
        const Frame* f[2] = {&a, &b};
        for (int i = 0; i < 2; ++i) {
            ver[i] = plates_[i].version();
            plate[i] = (f[i]->seq != plate_seq_[i]) ? plates_[i].update(f[i]->gray)
                                                    : plate_last_[i];
            plate_last_[i] = plate[i];
            plate_seq_[i] = f[i]->seq;
        }

        Vector3d vel;
        Matrix3d vcov;
        bool has_v, has_c;
        {
            std::lock_guard<std::mutex> lk(motion_m_);
            has_v = has_motion_;
            has_c = has_cov_;
            vel = vel_;
            vcov = vel_cov_;
        }

        const std::vector<double> stamps = {a.t, b.t};
        const double t_mean = 0.5 * (a.t + b.t);
        est_.set_frame_seq(a.seq, b.seq);
        auto r = est_.update(a.gray, b.gray, plate[0], plate[1], ver[0], ver[1], t_mean,
                             &stamps, has_v ? &vel : nullptr, has_c ? &vcov : nullptr);

        std::lock_guard<std::mutex> lk(m_);
        note_skew(skew);
        stats_.t_last_pose = now_s();
        if (!r) { stats_.n_lost++; return false; }
        pose_ = std::move(r);
        pose_t_ = t_mean;
        pose_stamps_ = stamps;
        pose_seq_++;
        stats_.n_pose++;
        cv_.notify_all();
        return true;
    }

    //: Keep a bounded sample of skews for the percentiles `stats()` reports. Bounded
    //: because a flight is minutes long at hundreds of hertz and nobody needs every one.
    void note_skew(double s) {
        if (stats_.skews.size() < 20000) stats_.skews.push_back(s);
    }

    Estimator est_;
    std::vector<CameraSpec> specs_;
    PairMode mode_;
    double max_skew_s_;

    std::vector<RunningPlate> plates_;
    const cv::Mat* plate_last_[2] = {nullptr, nullptr};
    std::uint64_t plate_seq_[2] = {0, 0};

    std::unique_ptr<CameraSource> cams_[2];
    std::thread worker_;
    std::atomic<bool> running_{false};

    mutable std::mutex m_;
    std::condition_variable cv_;
    Frame slot_[2];
    std::uint64_t done_[2] = {0, 0};

    std::optional<PoseResult> pose_;
    std::uint64_t pose_seq_ = 0;
    double pose_t_ = 0.0;
    std::vector<double> pose_stamps_;
    TrackerStats stats_;

    std::mutex motion_m_;
    Vector3d vel_ = Vector3d::Zero();
    Matrix3d vel_cov_ = Matrix3d::Zero();
    bool has_motion_ = false, has_cov_ = false;
};

}  // namespace pmw
