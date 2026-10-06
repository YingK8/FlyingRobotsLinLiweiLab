// AVFoundation behind `capture.h`. The only Objective-C++ in the module.
//
// The argument for being here at all is in `capture.h`. What is here is the mechanics.

#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>

#include "capture.h"

#include <opencv2/imgproc.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace pmw {
namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string fourcc_str(OSType c) {
    char s[5] = {char(c >> 24), char(c >> 16), char(c >> 8), char(c), 0};
    return std::string(s);
}

//: Run `body`, converting an NSException into a std::runtime_error.
//:
//: AVFoundation raises `NSException` for misuse -- an unsupported frame duration, a
//: format that is not the device's, a `commitConfiguration` out of order. That is not a
//: `std::exception`, so nanobind cannot translate it and Python sees only "exception
//: could not be translated!" with the cause thrown away. Every ObjC call that can raise
//: goes through here.
template <class F>
static auto objc_guard(const char* what, F&& body) -> decltype(body()) {
    @try {
        return body();
    } @catch (NSException* e) {
        throw std::runtime_error(std::string(what) + ": " +
                                 std::string(e.name.UTF8String) + " -- " +
                                 std::string(e.reason ? e.reason.UTF8String : "no reason"));
    }
}

//: External cameras first, so the ELPs sort ahead of the built-in one. Order is only a
//: fallback for reporting -- cameras are selected by id, never by position.
NSArray<AVCaptureDevice *> *discover() {
    AVCaptureDeviceDiscoverySession *ds = [AVCaptureDeviceDiscoverySession
        discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeExternal,
                                          AVCaptureDeviceTypeBuiltInWideAngleCamera]
                              mediaType:AVMediaTypeVideo
                               position:AVCaptureDevicePositionUnspecified];
    return ds.devices;
}

//: The grayscale plane of a captured buffer, as a CV_8U `cv::Mat` that owns its pixels.
//:
//: The sensor is monochrome (`camera/theory.md` 1.5: `max(BGR) - min(BGR)` is identically
//: zero), so for every YCbCr layout the luma plane **is** the image and chroma is
//: constant. Taking Y is not a conversion, it is a projection onto the only channel that
//: carries anything -- which is why this is cheaper than, and identical to, the
//: `cvtColor(BGR2GRAY)` it replaces.
cv::Mat gray_of(CVImageBufferRef img) {
    OSType fmt = CVPixelBufferGetPixelFormatType(img);
    CVPixelBufferLockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
    const int w = int(CVPixelBufferGetWidth(img));
    const int h = int(CVPixelBufferGetHeight(img));
    cv::Mat out;

    if (CVPixelBufferIsPlanar(img)) {
        // 420v / 420f / x420: plane 0 is a full-resolution 8-bit luma plane. Clone,
        // because the buffer goes back to the pool the moment we return.
        auto* p = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(img, 0));
        size_t stride = CVPixelBufferGetBytesPerRowOfPlane(img, 0);
        cv::Mat(h, w, CV_8UC1, p, stride).copyTo(out);
    } else if (fmt == kCVPixelFormatType_422YpCbCr8_yuvs ||
               fmt == kCVPixelFormatType_422YpCbCr8) {
        // Packed 4:2:2. `yuvs` is Y Cb Y Cr (luma on even bytes), `2vuy` is Cb Y Cr Y
        // (odd). Either way luma is every other byte, so a 2-channel view and
        // `extractChannel` reads it with the right stride and no arithmetic here.
        auto* p = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(img));
        size_t stride = CVPixelBufferGetBytesPerRow(img);
        const int y_ch = (fmt == kCVPixelFormatType_422YpCbCr8_yuvs) ? 0 : 1;
        cv::extractChannel(cv::Mat(h, w, CV_8UC2, p, stride), out, y_ch);
    } else if (fmt == kCVPixelFormatType_32BGRA) {
        auto* p = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(img));
        size_t stride = CVPixelBufferGetBytesPerRow(img);
        cv::cvtColor(cv::Mat(h, w, CV_8UC4, p, stride), out, cv::COLOR_BGRA2GRAY);
    } else if (fmt == kCVPixelFormatType_OneComponent8) {
        auto* p = static_cast<std::uint8_t*>(CVPixelBufferGetBaseAddress(img));
        cv::Mat(h, w, CV_8UC1, p, CVPixelBufferGetBytesPerRow(img)).copyTo(out);
    } else {
        CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
        throw std::runtime_error("unhandled pixel format '" + fourcc_str(fmt) +
                                 "'; add a case in capture_avf.mm gray_of()");
    }
    CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
    return out;
}

}  // namespace
}  // namespace pmw

//: The delegate. Holds no ownership of the sink: `AvfCamera::stop` tears the session
//: down and waits for the queue to drain before the sink can go away.
@interface PmwGrabber : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
@end

@implementation PmwGrabber {
@public
    pmw::FrameSink* sink;
    int ci;
    bool rotate180;
    std::atomic<std::uint64_t>* n_grabbed;
    std::atomic<bool>* running;
    std::atomic<int>* got_w;
    std::atomic<int>* got_h;
    std::atomic<std::uint64_t>* n_wrong_size;
    std::atomic<std::uint64_t>* n_blank_raw;
    std::atomic<std::uint64_t>* n_blank_out;
    int want_w;
    int want_h;
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)buf
       fromConnection:(AVCaptureConnection *)conn {
    if (!running->load(std::memory_order_acquire)) return;
    CVImageBufferRef img = CMSampleBufferGetImageBuffer(buf);
    if (!img) return;
    // Taken here rather than after the copy: this is the closest to the shutter we get,
    // and it is the clock every pairing decision downstream is made on.
    const double t = pmw::now_s();
    CMTime cmt = CMSampleBufferGetPresentationTimeStamp(buf);
    const double pts = CMTIME_IS_VALID(cmt) ? CMTimeGetSeconds(cmt)
                                            : std::numeric_limits<double>::quiet_NaN();
    // Sampled brightness of the RAW buffer, before anything of ours touches it. If this
    // is blank the camera sent us nothing; if only the delivered Mat is blank, we did it.
    {
        CVPixelBufferLockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
        auto* p0 = (const std::uint8_t*)CVPixelBufferGetBaseAddressOfPlane(img, 0);
        if (p0) {
            const size_t stride = CVPixelBufferGetBytesPerRowOfPlane(img, 0);
            const int H = (int)CVPixelBufferGetHeight(img), W = (int)CVPixelBufferGetWidth(img);
            long sum = 0; int cnt = 0;
            for (int y = 0; y < H; y += 8) for (int x = 0; x < W; x += 8) { sum += p0[y*stride+x]; ++cnt; }
            if (cnt && double(sum)/cnt < 1.0) n_blank_raw->fetch_add(1, std::memory_order_relaxed);
        }
        CVPixelBufferUnlockBaseAddress(img, kCVPixelBufferLock_ReadOnly);
    }
    const int gw = int(CVPixelBufferGetWidth(img)), gh = int(CVPixelBufferGetHeight(img));
    got_w->store(gw, std::memory_order_relaxed);
    got_h->store(gh, std::memory_order_relaxed);
    // A wrong-size frame must never reach the estimator. The C++ core has no
    // `_match_scale` -- its intrinsics are fixed at construction -- so a frame at another
    // resolution is not merely unusual, it is silently in the wrong coordinate system.
    if (want_w && (gw != want_w || gh != want_h)) {
        n_wrong_size->fetch_add(1, std::memory_order_relaxed);
        return;
    }
    try {
        cv::Mat g = pmw::gray_of(img);
        // A pure pixel permutation, exact, no resampling -- and it belongs here because
        // this is the only place frames enter the system, so calibration and the live
        // loop agree by construction (sources.py:202-206).
        //
        // **Out of place, deliberately.** `cv::rotate(g, g, ROTATE_180)` is `flip` with
        // src == dst, which OpenCV does not guarantee: for a both-axes flip the kernel
        // walks rows from each end and reads what it has already written. It does not
        // fail, it corrupts, and it corrupts intermittently under load -- which showed
        // up as blank and flickering frames from the tracker while a bare delegate on
        // the same camera saw 0.0% blanks and 0.4% brightness variation.
        if (rotate180) {
            cv::Mat r;
            cv::rotate(g, r, cv::ROTATE_180);
            g = std::move(r);
        }
        {   // and the same measure on what we are about to hand over
            long sum = 0; int cnt = 0;
            for (int y = 0; y < g.rows; y += 8) {
                const std::uint8_t* row = g.ptr<std::uint8_t>(y);
                for (int x = 0; x < g.cols; x += 8) { sum += row[x]; ++cnt; }
            }
            if (cnt && double(sum)/cnt < 1.0) n_blank_out->fetch_add(1, std::memory_order_relaxed);
        }
        n_grabbed->fetch_add(1, std::memory_order_relaxed);
        sink->deliver(ci, std::move(g), t, pts);
    } catch (const std::exception&) {
        // A frame we cannot read is a dropped frame, not a dead flight. It shows up as
        // `n_grabbed` failing to advance, which is what `stats()` reports.
    }
}
@end

namespace pmw {
namespace {

class AvfCamera : public CameraSource {
public:
    AvfCamera(AVCaptureDevice* dev, AVCaptureSession* s, PmwGrabber* g,
              cv::Size size, std::string uid, AVCaptureDeviceFormat* fmt, double fps)
        : dev_(dev), session_(s), grabber_(g), size_(size), uid_(std::move(uid)),
          fmt_(fmt), fps_(fps) {}

    ~AvfCamera() override {
        stop();
        for (id o in observers_) [[NSNotificationCenter defaultCenter] removeObserver:o];
        [observers_ removeAllObjects];
    }

    void start() override {
        if (started_) return;
        running_.store(true, std::memory_order_release);
        objc_guard("startRunning", [&] { [session_ startRunning]; return 0; });
        // **The format is applied HERE, after the session is running, and that is the
        // only ordering that works on macOS.** Measured on this bench across all four
        // orderings, asking 640x400 of the ELP:
        //
        //     activeFormat inside beginConfiguration   -> 1280x800 @ 119 fps
        //     activeFormat after commitConfiguration   -> 1280x800 @ 118
        //     activeFormat after startRunning          ->  640x400 @ 207   <-
        //     videoSettings width/height keys          ->  640x400 @ 200, but SCALED
        //                                                  from 1280x800, not the
        //                                                  sensor's own mode
        //
        // The session's preset wins over `activeFormat` until it is running.
        // `AVCaptureSessionPresetInputPriority`, which says "the format is mine", is
        // iOS-only. So this cannot be checked at open time and `Tracker::start` checks
        // the delivered buffer instead.
        // **Retry until the DELIVERED size is right.** Applying the format after
        // `startRunning` is the only ordering that works (see the table above) but it is
        // also a race: the call succeeds and `activeFormat` reads back correctly while
        // the session is still bringing the stream up, and the camera then runs at the
        // session's size instead. Measured on this bench: one camera of the pair
        // silently streaming 1280x800 while the other ran 640x400 -- and because the C++
        // estimator has no `_match_scale`, that view's ellipse is in the wrong pixel
        // coordinates, the two views' poses disagree, and the discrepancy gate rejects
        // every frame. It looks exactly like a scene or lighting fault.
        for (int attempt = 0; attempt < 10; ++attempt) {
            apply_format();
            for (int k = 0; k < 40; ++k) {          // up to 400 ms for a frame to land
                if (got_w_.load(std::memory_order_relaxed) != 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (delivered_size() == size_) break;
            got_w_.store(0, std::memory_order_relaxed);   // re-check a fresh delivery
            got_h_.store(0, std::memory_order_relaxed);
        }
        started_ = true;
    }

    void stop() override {
        if (!started_) return;
        // Order matters. Clear the flag first so a sample already in flight on the
        // delivery queue returns without touching the sink, THEN stop the session,
        // which blocks until the queue has drained. The reverse races a delegate
        // callback against the sink's destructor.
        running_.store(false, std::memory_order_release);
        @try { [session_ stopRunning]; } @catch (NSException*) {}   // teardown never throws
        started_ = false;
    }

    std::uint64_t n_grabbed() const override {
        return n_grabbed_.load(std::memory_order_relaxed);
    }
    cv::Size size() const override { return size_; }
    cv::Size delivered_size() const override {
        return cv::Size(got_w_.load(std::memory_order_relaxed),
                        got_h_.load(std::memory_order_relaxed));
    }
    std::string unique_id() const override { return uid_; }
    std::uint64_t n_wrong_size() const override {
        return n_wrong_size_.load(std::memory_order_relaxed);
    }
    std::uint64_t n_blank_raw() const override { return n_blank_raw_.load(std::memory_order_relaxed); }
    std::uint64_t n_blank_out() const override { return n_blank_out_.load(std::memory_order_relaxed); }
    std::string last_event() const override {
        std::lock_guard<std::mutex> lk(ev_m_);
        return last_event_;
    }
    void note_event(std::string e) {
        std::lock_guard<std::mutex> lk(ev_m_);
        last_event_ = std::move(e);
    }

    std::atomic<std::uint64_t> n_grabbed_{0};
    std::atomic<bool> running_{false};
    std::atomic<int> got_w_{0}, got_h_{0};
    std::atomic<std::uint64_t> n_wrong_size_{0}, n_blank_raw_{0}, n_blank_out_{0};
    mutable std::mutex ev_m_;
    std::string last_event_;
    NSMutableArray* observers_ = [NSMutableArray array];

private:
    void apply_format() {
        NSError* err = nil;
        if (![dev_ lockForConfiguration:&err]) return;
        objc_guard("setting activeFormat", [&] { dev_.activeFormat = fmt_; return 0; });
        if (fps_ > 0) {
            CMTime dur = CMTimeMake(1000, int32_t(std::llround(fps_ * 1000)));
            for (AVFrameRateRange* r in fmt_.videoSupportedFrameRateRanges) {
                if (CMTimeCompare(dur, r.minFrameDuration) < 0) dur = r.minFrameDuration;
                if (CMTimeCompare(dur, r.maxFrameDuration) > 0) dur = r.maxFrameDuration;
            }
            objc_guard("setting frame duration", [&] {
                dev_.activeVideoMinFrameDuration = dur;
                dev_.activeVideoMaxFrameDuration = dur;
                return 0;
            });
        }
        [dev_ unlockForConfiguration];
    }

    AVCaptureDevice* dev_;
    AVCaptureSession* session_;
    PmwGrabber* grabber_;
    cv::Size size_;
    std::string uid_;
    AVCaptureDeviceFormat* fmt_;
    double fps_;
    bool started_ = false;
};

//: Match by exact id, then by substring either way round.
//:
//: The rig stores `elp_ids` like `0x11000032e49281` -- a USB location plus VID/PID --
//: and AVFoundation's `uniqueID` for a UVC device is built the same way, but the two
//: have not been checked against each other on this bench with the ELPs attached. The
//: substring fallback is what makes a prefix or a case difference recoverable instead of
//: fatal, and the exception below prints the real list so the mismatch is one line to
//: diagnose rather than a guess.
AVCaptureDevice* find_camera(NSArray<AVCaptureDevice *>* devs, const std::string& id) {
    NSString* want = [NSString stringWithUTF8String:id.c_str()];
    for (AVCaptureDevice* d in devs)
        if ([d.uniqueID isEqualToString:want]) return d;
    for (AVCaptureDevice* d in devs)
        if ([d.uniqueID rangeOfString:want options:NSCaseInsensitiveSearch].location != NSNotFound ||
            [want rangeOfString:d.uniqueID options:NSCaseInsensitiveSearch].location != NSNotFound)
            return d;
    return nil;
}

//: The format delivering exactly `w x h` with the highest frame rate, or nil.
//:
//: Exactly: a format that substitutes a size changes the intrinsics' scale and every
//: distance downstream is then wrong by a fixed factor nothing can detect
//: (`camera/theory.md` 1.2). `640x480` is a *crop* of this sensor, not a rescale, so
//: accepting it "because it is close" is the specific trap.
AVCaptureDeviceFormat* best_format(AVCaptureDevice* dev, int w, int h, double* out_fps) {
    AVCaptureDeviceFormat* best = nil;
    double best_fps = -1;
    for (AVCaptureDeviceFormat* f in dev.formats) {
        CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
        if (d.width != w || d.height != h) continue;
        for (AVFrameRateRange* r in f.videoSupportedFrameRateRanges)
            if (r.maxFrameRate > best_fps) { best_fps = r.maxFrameRate; best = f; }
    }
    if (out_fps) *out_fps = best_fps;
    return best;
}

std::string device_list_msg(NSArray<AVCaptureDevice *>* devs) {
    std::string s;
    for (AVCaptureDevice* d in devs)
        s += "\n    " + std::string(d.uniqueID.UTF8String) + "  " +
             std::string(d.localizedName.UTF8String);
    return s.empty() ? std::string("\n    (none)") : s;
}

}  // namespace

std::vector<CameraFormat> camera_formats(const std::string& id) {
    @autoreleasepool {
        NSArray<AVCaptureDevice *>* devs = discover();
        AVCaptureDevice* dev = find_camera(devs, id);
        if (!dev)
            throw std::runtime_error("no camera with id '" + id + "'. Devices:" +
                                     device_list_msg(devs));
        std::vector<CameraFormat> out;
        for (AVCaptureDeviceFormat* f in dev.formats) {
            CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
            CameraFormat cf;
            cf.width = d.width;
            cf.height = d.height;
            cf.fourcc = fourcc_str(CMFormatDescriptionGetMediaSubType(f.formatDescription));
            cf.min_fps = 1e9;
            cf.max_fps = 0;
            for (AVFrameRateRange* r in f.videoSupportedFrameRateRanges) {
                cf.max_fps = std::max(cf.max_fps, r.maxFrameRate);
                cf.min_fps = std::min(cf.min_fps, r.minFrameRate);
            }
            out.push_back(cf);
        }
        return out;
    }
}

std::vector<CameraInfo> list_cameras() {
    @autoreleasepool {
        std::vector<CameraInfo> out;
        for (AVCaptureDevice* d in discover())
            out.push_back({std::string(d.uniqueID.UTF8String),
                           std::string(d.localizedName.UTF8String)});
        return out;
    }
}

std::unique_ptr<CameraSource> open_camera(const CameraSpec& spec, int ci, FrameSink* sink) {
    @autoreleasepool {
        NSArray<AVCaptureDevice *>* devs = discover();
        AVCaptureDevice* dev = find_camera(devs, spec.unique_id);
        if (!dev)
            throw std::runtime_error("no camera with id '" + spec.unique_id +
                                     "'. Devices AVFoundation can see:" +
                                     device_list_msg(devs));

        double native_fps = 0;
        AVCaptureDeviceFormat* fmt = best_format(dev, spec.width, spec.height, &native_fps);
        if (!fmt) {
            std::string have;
            for (AVCaptureDeviceFormat* f in dev.formats) {
                CMVideoDimensions d = CMVideoFormatDescriptionGetDimensions(f.formatDescription);
                have += " " + std::to_string(d.width) + "x" + std::to_string(d.height);
            }
            throw std::runtime_error(
                std::string("camera '") + dev.localizedName.UTF8String + "' has no " +
                std::to_string(spec.width) + "x" + std::to_string(spec.height) +
                " format. It offers:" + have +
                "\n  A substituted size rescales every distance downstream and nothing "
                "can detect it (camera/theory.md 1.2), so this refuses rather than "
                "taking the nearest.");
        }

        NSError* err = nil;
        AVCaptureDeviceInput* in = [AVCaptureDeviceInput deviceInputWithDevice:dev error:&err];
        if (!in)
            throw std::runtime_error(std::string("deviceInput failed: ") +
                                     (err ? err.localizedDescription.UTF8String : "?"));

        AVCaptureSession* session = [[AVCaptureSession alloc] init];
        objc_guard("beginConfiguration", [&] { [session beginConfiguration]; return 0; });

        // **Order is load-bearing here and getting it wrong fails silently.** A session
        // carries a preset (`...PresetHigh` by default) and ADDING AN INPUT re-applies
        // it, overwriting `activeFormat`. Measured 2026-09-04: configuring the device
        // first and then adding the input delivered 1280x800 at 94 fps for a requested
        // 640x400, while `dev.activeFormat` still read 640x400 -- so a check on the
        // format cannot see it, which is exactly the undetectable size substitution
        // `camera/theory.md` 1.2 is about. `InputPriority` tells the session the format
        // is ours; the format is set after the input is in; and the check that counts is
        // on the first delivered BUFFER, below.
        // (`AVCaptureSessionPresetInputPriority`, which states this explicitly, is
        // iOS-only. On macOS the ordering below is the whole mechanism, which is why
        // the delivered-buffer check exists rather than being belt and braces.)
        if (![session canAddInput:in]) throw std::runtime_error("canAddInput said no");
        objc_guard("addInput", [&] { [session addInput:in]; return 0; });

        AVCaptureVideoDataOutput* out = [[AVCaptureVideoDataOutput alloc] init];
        // Drop-oldest, inside the framework. Same rule as `MonoCamera`'s single-frame
        // slot: for feedback control a stale frame is worse than no frame, so a queue
        // that buffered backlog would be actively harmful (camera/theory.md 1.1).
        out.alwaysDiscardsLateVideoFrames = YES;
        // Biplanar 420: plane 0 is the full-resolution luma, which for this mono sensor
        // is the image. See `gray_of`.
        out.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey:
                              @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)};

        PmwGrabber* g = [[PmwGrabber alloc] init];
        const double want = spec.fps > 0 ? std::min(spec.fps, native_fps) : native_fps;
        auto cam = std::make_unique<AvfCamera>(dev, session, g,
                                               cv::Size(spec.width, spec.height),
                                               std::string(dev.uniqueID.UTF8String),
                                               fmt, want);
        g->sink = sink;
        g->ci = ci;
        g->rotate180 = spec.rotate180;
        g->n_grabbed = &cam->n_grabbed_;
        g->running = &cam->running_;
        g->got_w = &cam->got_w_;
        g->got_h = &cam->got_h_;
        g->n_wrong_size = &cam->n_wrong_size_;
        g->n_blank_raw = &cam->n_blank_raw_;
        g->n_blank_out = &cam->n_blank_out_;
        g->want_w = spec.width;
        g->want_h = spec.height;

        // One serial queue per camera, so the two cameras deliver independently and
        // neither can be held up behind the other.
        dispatch_queue_t q = dispatch_queue_create(
            ("pmw.grab." + std::to_string(ci)).c_str(), DISPATCH_QUEUE_SERIAL);
        [out setSampleBufferDelegate:g queue:q];
        if (![session canAddOutput:out]) throw std::runtime_error("canAddOutput said no");
        objc_guard("addOutput / commitConfiguration", [&] {
            [session addOutput:out];
            [session commitConfiguration];
            return 0;
        });

        // Observe what AVFoundation says goes wrong. Without these a stalled session is
        // silent: `n_grabbed` simply stops rising and nothing distinguishes an
        // interruption, a runtime error and a hang. The block captures a raw pointer,
        // which is safe because the camera owns the session and removes these observers
        // in its destructor.
        {
            AvfCamera* raw = cam.get();
            NSNotificationCenter* nc = [NSNotificationCenter defaultCenter];
            auto watch = [&](NSNotificationName name, const char* label) {
                id o = [nc addObserverForName:name object:session queue:nil
                        usingBlock:^(NSNotification* note) {
                    std::string msg(label);
                    id err = note.userInfo[AVCaptureSessionErrorKey];
                    if (err) msg += std::string(": ") +
                        [[err localizedDescription] UTF8String];
                    // (AVCaptureSessionInterruptionReasonKey is iOS-only, so on macOS
                    // an interruption arrives without a machine-readable reason.)
                    raw->note_event(msg);
                }];
                [raw->observers_ addObject:o];
            };
            watch(AVCaptureSessionRuntimeErrorNotification, "runtime error");
            watch(AVCaptureSessionWasInterruptedNotification, "interrupted");
            watch(AVCaptureSessionInterruptionEndedNotification, "interruption ended");
            watch(AVCaptureSessionDidStopRunningNotification, "session stopped running");
        }

        // No size assert here: the format is not applied until `start()` (see
        // `apply_format`), so nothing to check yet. `Tracker::start` checks the first
        // delivered BUFFER, which is the only thing that catches a session preset
        // overriding the format anyway.
        return cam;
    }
}

}  // namespace pmw
