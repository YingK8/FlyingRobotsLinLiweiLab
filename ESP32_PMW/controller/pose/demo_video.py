#!/usr/bin/env python3
"""
Render a recording to a video with every overlay on it, for looking at a whole flight.

    python demo_video.py                       # the newest flight
    python demo_video.py <flight> [-o out.mp4] [--gated]

Both views side by side. Per view: the evidence map's mask, the seed ellipse in red, the
direct fit in green, the rotor axis in cyan from the ellipse centre, and that view's
`ridge`. Across the bottom: the fused pose, cross-view discrepancy, and whether the frame
would survive the gates.

Runs `never_reject` by default, so every frame carries a pose and the overlay shows what
the gates would have thrown away and why -- which is the point of watching it rather than
reading a solve rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera"),
                str(HERE.parent / "viz")]

import background as bgmod  # noqa: E402
import rig as rigmod  # noqa: E402
import segment as segmod  # noqa: E402
import stereo as st  # noqa: E402
from estimator import RADIUS_BENCH_MM  # noqa: E402
from filter import PoseFilter  # noqa: E402
from live_viz import normal_segment_px  # noqa: E402
from record import latest_flight, open_recording  # noqa: E402
from shape import CentreCalibration  # noqa: E402

FLIGHTS = HERE.parents[1] / "results" / "flights"
SCALE = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _rim_overlay(out, pose, cam, view):
    """Mark the rim where this view has no evidence for it.

        The dead arcs are what the two-view fit is for, and a solve rate does not show
        them. Drawn from the *refined world pose*, so both panels mark the same physical
        points of the rim: an arc dark in one panel and lit in the other is the case the
        union covers, and one dark in both is the case nothing can.
    """

    alive = pose.extra.get("alive") if pose is not None else None
    if alive is None or view >= len(alive):
        return
    centre, normal = pose.extra["world"]
    phis = np.linspace(0.0, 2.0 * np.pi, alive.shape[1], endpoint=False)
    pts, seen = st._project_ideal(
        st._rim_points(centre, normal, RADIUS_BENCH_MM, phis), cam)
    pts = st._distort_points(pts, cam)
    for (x, y), ok, vis in zip(pts, alive[view], seen):
        if not vis or not np.isfinite(x) or max(abs(x), abs(y)) >= st._FAR_PX:
            continue
        cv2.circle(out, (int(x), int(y)), 3,
                   (90, 220, 90) if ok else (60, 60, 255), -1)


def _panel(frame, seg, pose, cam, zero, tag, show_mask=True, view=0):
    """One view, with its mask, both ellipses, the axis, its dead arcs, and its ridge."""

    npx = None
    if pose is not None and seg is not None:
        npx = normal_segment_px(pose, cam, zero, centre_px=seg.ellipse[0])
    mask = seg.mask if (show_mask and seg is not None) else None
    out = segmod.draw(frame, seg, normal_px=npx, mask=mask)
    # One ellipse now. There used to be two -- the mask fit in red and a per-view
    # refinement of it in green -- and the refinement was removed (`theory.md` 16.24).
    _rim_overlay(out, pose, cam, view)
    label = f"{tag}"
    if seg is None:
        label += "  no fit"
    elif seg.mask is None:
        label += "  seed from the last frame"
    else:
        label += f"  {seg.area_px / 1000:.0f}k px"
    cv2.putText(out, label, (12, out.shape[0] - 16), FONT, 0.9, (255, 255, 255), 2)
    return out


def render(flight, out_path=None, gated=False, radius_mm=RADIUS_BENCH_MM, scale=SCALE):
    flight = Path(flight)
    out_path = Path(out_path or flight / "demo.mp4")
    rig = rigmod.StereoRig.load(rigmod.DEFAULT_PATH)
    cams = list(rig.cameras)
    est = st.StereoPoseEstimator(
        rig, radius_mm=radius_mm,
        backgrounds={c.name: bgmod.RunningPlate() for c in rig.cameras},
        centre_cal=CentreCalibration.load(), direct=True, never_reject=not gated,
    )
    # The same `PoseFilter` the viser path runs, so what this shows and what `live_viz`
    # shows are the same signal. Without it the overlay draws the raw per-frame estimate,
    # which is visibly jitterier than anything downstream ever sees -- 11.7 deg p90
    # frame-to-frame on `2026-08-28_131552` against 1.9 for the filtered track.
    filt = PoseFilter()
    caps, _ = open_recording(flight)
    writer, n, solved = None, 0, 0
    try:
        while True:
            got = [c.read() for c in caps]
            if not all(ok for ok, _ in got):
                break
            frames = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                      for _, f in got]
            pose = est.update(frames, t=n / 60.0, frame_index=n)
            got = filt.update(pose, t=n / 60.0)
            if pose is not None and got is not None:
                # Drawn from the filtered track, reported alongside the raw numbers so
                # the smoothing cannot hide a bad frame: `discrepancy_mm` and `ridge`
                # below are still this frame's own.
                pose.xyz_mm, _, pose.normal = got
            # `update` has already segmented both views and carries the result. Calling
            # `_view_candidates` again to draw them ran the whole per-view pipeline
            # twice, for the same answer.
            segs = list(pose.per_view) if pose is not None else [
                est._view_candidates(f, c)[0] for f, c in zip(frames, cams)]
            solved += pose is not None

            panels = [_panel(f, s, pose, c, est.zero, t, view=i)
                      for i, (f, s, c, t) in enumerate(zip(frames, segs, cams, "AB"))]
            canvas = np.hstack(panels)
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale)

            bar = np.zeros((70, canvas.shape[1], 3), np.uint8)
            if pose is not None:
                keep = pose.discrepancy_mm <= st.MAX_DISCREPANCY_MM
                cov = ""
                if np.isfinite(pose.union_coverage):
                    per = "/".join(f"{v:.2f}"
                                   for v in pose.extra["alive"].mean(axis=1))
                    cov = f"   rim seen {pose.union_coverage:.2f} (A/B {per})"
                cv2.putText(bar, f"frame {n:5d}   xyz "
                            f"({pose.xyz_mm[0]:7.1f},{pose.xyz_mm[1]:7.1f},{pose.xyz_mm[2]:7.1f}) mm"
                            f"   discrepancy {pose.discrepancy_mm:7.2f} mm{cov}",
                            (12, 28), FONT, 0.6, (255, 255, 255), 1)
                cv2.putText(bar, "PASSES the gates" if keep else
                            f"would be REJECTED (gate {st.MAX_DISCREPANCY_MM:.0f} mm)",
                            (12, 56), FONT, 0.6,
                            (120, 255, 120) if keep else (120, 120, 255), 2)
            else:
                cv2.putText(bar, f"frame {n:5d}   no pose", (12, 40), FONT, 0.7,
                            (120, 120, 255), 2)
            canvas = np.vstack([canvas, bar])

            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out_path),
                                         cv2.VideoWriter_fourcc(*"avc1"), 30.0, (w, h))
            writer.write(canvas)
            n += 1
    finally:
        for c in caps:
            c.release()
        if writer is not None:
            writer.release()
    print(f"{out_path}  {n} frames, {solved} with a pose ({100 * solved / max(n, 1):.1f}%)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flight", nargs="?", default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--gated", action="store_true",
                    help="run the real gates, so rejected frames show as 'no pose'")
    a = ap.parse_args()
    render(a.flight or latest_flight(FLIGHTS), a.out, gated=a.gated)
