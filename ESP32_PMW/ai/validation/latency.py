r"""Measure latency, which is not the same thing as throughput.

    uv run python controller/pose/validation/latency.py
    uv run python controller/pose/validation/latency.py --camera --width 640 --height 480

Throughput is how many frames a second the estimator can chew through; latency is
how stale the answer is by the time a controller sees it.  A pipeline can have
excellent throughput and useless latency, and for feedback control it is latency
that sets the achievable bandwidth -- `ai/design_hover_lqr.py` hard-fails on
closed-loop poles above `rate/6` precisely because of this.

Budget, capture to pose available:

    t_expose ---> t_transfer ---> t_queue ---> t_segment ---> t_solve
    \______ camera and driver ______/  \___ measured here ___/

Everything from the grab timestamp onward is measured directly.  Exposure and
USB transfer are *not* measurable from the host without hardware timestamps --
`cv2` stamps a frame when the driver hands it over, which is already after both.
That unmeasured head is reported as such rather than quietly omitted; on a UVC
camera it is typically one to two frame periods.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import cv2  # noqa: E402
import sources  # noqa: E402
from estimator import PoseEstimator, load_intrinsics  # noqa: E402

FPS_TARGETS = (240, 420)


def percentiles(name, v_ms, indent="  "):
    if len(v_ms) == 0:
        print(f"{indent}{name:<26s} (no samples)")
        return
    a = np.asarray(v_ms)
    print(f"{indent}{name:<26s} {a.mean():7.3f} {np.median(a):7.3f} "
          f"{np.percentile(a, 95):7.3f} {np.percentile(a, 99):7.3f} {a.max():7.3f}")


def header(title):
    print(f"\n{title}")
    print(f"  {'stage':<26s} {'mean':>7s} {'med':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s}   (ms)")


def run(source, est, n, scale=1.0, warmup=20):
    """Collect per-frame stage timings and end-to-end pipeline latency."""
    seg_ms, est_ms, tot_ms, pipe_ms, gaps = [], [], [], [], []
    seen = n_lost = 0
    prev_done = None

    while seen < n + warmup:
        item = source.read()
        if item is None:
            break
        t_capture, frame = item
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        t_in = time.monotonic()
        pose = est.update(frame, t=t_capture)
        t_done = time.monotonic()

        seen += 1
        if pose is None:
            n_lost += 1
        if seen <= warmup or pose is None:
            prev_done = t_done
            continue

        seg_ms.append(pose.t_seg_ms)
        est_ms.append(pose.t_est_ms)
        tot_ms.append((t_done - t_in) * 1e3)
        # Grab timestamp to answer-ready: includes any wait in the grabber slot.
        pipe_ms.append((t_done - t_capture) * 1e3)
        if prev_done is not None:
            gaps.append((t_done - prev_done) * 1e3)
        prev_done = t_done

    return dict(seg=seg_ms, est=est_ms, total=tot_ms, pipeline=pipe_ms, gap=gaps,
                n_seen=seen, n_lost=n_lost)


def report(r, label, live):
    header(f"{label}")
    if r["n_lost"] == r["n_seen"]:
        print(f"  NO DETECTIONS in {r['n_seen']} frames -- the robot is below the minimum")
        print("  blob size at this scale, so there is no timing to report. Not a timing")
        print("  result: at this resolution the estimator simply cannot see the target.")
        return
    percentiles("segmentation", r["seg"])
    percentiles("back-projection", r["est"])
    percentiles("compute, capture-free", r["total"])
    if live:
        percentiles("grab -> pose ready", r["pipeline"])
    percentiles("frame-to-frame interval", r["gap"])

    tot = np.asarray(r["total"])
    if len(tot) == 0:
        return
    print(f"\n  sustained {1e3/np.median(tot):.0f} Hz median, "
          f"{1e3/np.percentile(tot, 95):.0f} Hz at p95")
    for fps in FPS_TARGETS:
        budget = 1e3 / fps
        over = float((tot > budget).mean())
        print(f"  vs {fps:3d} fps ({budget:.2f} ms/frame): {over:6.2%} of frames over budget "
              f"[{'OK' if over < 0.01 else 'OVER'}]")

    if live:
        pipe = np.asarray(r["pipeline"])
        print(f"\n  measured latency (grab -> pose): median {np.median(pipe):.2f} ms, "
              f"p95 {np.percentile(pipe, 95):.2f} ms")
        print("  NOT included: sensor exposure and USB transfer, which happen before the")
        print("  driver hands over a frame and are unmeasurable without hardware timestamps.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None,
                    help="video/dir; default is the repo's 240 fps footage")
    ap.add_argument("--camera", action="store_true", help="use a live camera instead")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--scales", default="1.0,0.75,0.5",
                    help="comma-separated frame scales to sweep")
    args = ap.parse_args(argv)

    K, dist = load_intrinsics()
    default_video = HERE.parents[3] / "writeup" / "two_channel.mp4"

    print("=" * 78)
    print("pose pipeline latency")
    print("=" * 78)

    if args.camera:
        kw = {k: v for k, v in (("width", args.width), ("height", args.height))
              if v is not None}
        src = sources.CameraSource(**kw)
        print(f"live camera -> {src.actual}")
        est = PoseEstimator(camera_matrix=K, dist_coeffs=dist)
        try:
            r = run(src, est, args.frames)
        finally:
            src.close()
        report(r, "live camera", live=True)
        print(f"\n  grabber: {src.n_grabbed} grabbed, {src.n_dropped} dropped "
              f"({src.n_dropped/max(1,src.n_grabbed):.1%}) -- drops mean the estimator "
              f"is slower than the camera")
        return 0

    path = args.source or str(default_video)
    for scale in [float(s) for s in args.scales.split(",")]:
        src = sources.open_source(path)
        w, h = int(src._cap.get(3) * scale), int(src._cap.get(4) * scale)
        Ks = K.copy()
        Ks[:2, :] *= scale
        est = PoseEstimator(camera_matrix=Ks, dist_coeffs=dist)
        try:
            r = run(src, est, args.frames, scale=scale)
        finally:
            src.close()
        report(r, f"{Path(path).name} at scale {scale:g}  ({w}x{h})", live=False)

    print("\nNote: file playback has no capture latency to measure, so only compute is")
    print("shown. Run with --camera for the grab-to-pose figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
