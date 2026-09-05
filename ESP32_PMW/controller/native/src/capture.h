// Live camera capture, straight onto AVFoundation.
//
// **Not OpenCV's videoio, deliberately.** Homebrew's `libopencv_videoio.4.11.0.dylib`
// links `/opt/homebrew/opt/ffmpeg/lib/libavcodec.61.dylib` while ffmpeg 8.1.1
// (`libavcodec.62`) is what is installed, so the library does not load at all -- and
// linking it here would break the import of a pose core that works. Repairing it is
// worse than the fault: `brew` would install OpenCV **5.0**, the version bump
// `pose/theory.md` 21.2 records as moving `fitEllipseDirect` and `remap`, which is the
// one thing the parity harness exists to stop happening by accident.
//
// Measured before this was written (`probe_avf2.mm`): an `AVCaptureSession` creates,
// configures, starts and delivers frames entirely from a worker thread with no
// `NSRunLoop`. So the whole capture path is off the main thread and nothing here needs
// to be called from it.
//
// Three things this buys over the videoio port it replaces, none of them incidental:
//
//  1. **A and B are resolved by identity.** `camera/identify.py` opens each index and
//     checks the delivered size, because "neither macOS listing enumerates in OpenCV's
//     order" and a unique-id "cannot be tied to an OpenCV index". Addressed directly,
//     it can: the rig's own `elp_ids` say which camera is A. See `find_camera`.
//  2. **The format is chosen, not requested and then checked.** `AVCaptureDeviceFormat`
//     enumerates exact size/fourcc/max-fps up front, so `activeFormat` is *set*. The
//     silent 640x400 -> 640x480 substitution `camera/theory.md` 1.2 warns about cannot
//     happen; `open_camera` still asserts the result rather than trusting itself.
//  3. **The Y plane is the grayscale.** A biplanar 420 buffer's plane 0 is exactly the
//     mono image this sensor produces, so the `cvtColor(BGR2GRAY)` that
//     `sources.MonoCamera._grab_loop` pays per frame per camera is gone.

#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pmw {

//: One captured frame. `seq` starts at 1; 0 means "this camera has produced nothing".
struct Frame {
    cv::Mat gray;              // CV_8U, owned (a clone of the sample buffer)
    double t = 0.0;            // arrival, steady_clock seconds -- the clock everything pairs on
    double pts = 0.0;          // the buffer's own presentation timestamp, NaN if it had none
    std::uint64_t seq = 0;
};

//: Where a running camera puts its frames. `Tracker` implements it.
//:
//: Called on AVFoundation's delivery queue, one queue per camera, so two cameras can be
//: inside `deliver` at once and the implementation must be thread-safe.
struct FrameSink {
    virtual ~FrameSink() = default;
    virtual void deliver(int ci, cv::Mat gray, double t, double pts) = 0;
};

//: What `list_cameras` reports. `unique_id` is the string the rig stores in `elp_ids`.
struct CameraInfo {
    std::string unique_id;
    std::string name;
};

//: One mode the device offers, as `camera_formats` reports it.
struct CameraFormat {
    int width = 0, height = 0;
    std::string fourcc;
    double min_fps = 0, max_fps = 0;
};

struct CameraSpec {
    std::string unique_id;   // exact match first, then substring -- see `find_camera`
    int width = 640;
    int height = 400;
    double fps = 210.0;
    bool rotate180 = true;   // the cameras are mounted inverted (sources.py:202-206)
};

//: Every video device AVFoundation can see, in its order.
std::vector<CameraInfo> list_cameras();

//: Every mode one camera offers. What to look at when `open_camera` refuses a size.
std::vector<CameraFormat> camera_formats(const std::string& unique_id);

//: A camera that runs itself and pushes into a sink. Opaque so that nothing outside
//: `capture_avf.mm` has to be Objective-C++.
class CameraSource {
public:
    virtual ~CameraSource() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    //: Frames this camera has delivered. Camera loss, unlike `MonoCamera.n_dropped`,
    //: which counts consumer loss (`camera/theory.md` 1.1).
    virtual std::uint64_t n_grabbed() const = 0;
    //: The size asked for, which `open_camera` checked against `activeFormat`.
    virtual cv::Size size() const = 0;
    //: The size of the last buffer actually DELIVERED, or 0x0 before the first frame.
    //:
    //: Not the same question as `size()`, and the difference is not theoretical: a
    //: session's preset can override `activeFormat` when the input is added, leaving
    //: `activeFormat` reading 640x400 while 1280x800 arrives. This is the only check
    //: that sees that, so it is the one `Tracker::start` makes.
    virtual cv::Size delivered_size() const = 0;
    virtual std::string unique_id() const = 0;
    //: The last thing AVFoundation said went wrong with this session, or "".
    //:
    //: A capture that stops has, until now, been silent about why: `n_grabbed` merely
    //: stops rising. AVFoundation does post `AVCaptureSessionRuntimeError`,
    //: `...WasInterrupted` and `...DidStopRunning`, and without observing them a stall
    //: is indistinguishable from a hang. Observed so the next one names itself.
    virtual std::string last_event() const = 0;
    //: Frames dropped for arriving at the wrong resolution. Non-zero means the session
    //: won the race against `activeFormat` -- see `AvfCamera::start`.
    virtual std::uint64_t n_wrong_size() const = 0;
    //: Blank frames counted as the buffer ARRIVED, and as it was handed on. If the two
    //: differ, the blanking is ours; if they match, the camera sent nothing.
    virtual std::uint64_t n_blank_raw() const = 0;
    virtual std::uint64_t n_blank_out() const = 0;
};

//: Open one camera and bind it to `sink` as index `ci`. Throws with the device list in
//: the message if the id matches nothing, or if no format delivers the requested size.
std::unique_ptr<CameraSource> open_camera(const CameraSpec& spec, int ci, FrameSink* sink);

}  // namespace pmw
