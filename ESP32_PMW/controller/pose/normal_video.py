#!/usr/bin/env python3
"""Demo video of a tilt-sweep take: both views side by side, the disc ellipse and the
stereo rotor normal drawn on each, the 30% duty drop called out in the footer.

    uv run python controller/pose/normal_video.py <take_dir> [--stride 7]   # one take
    uv run python controller/pose/normal_video.py --all                     # every take
    uv run python controller/pose/normal_video.py --all --seg               # segmentation demo

`--seg` writes `seg_demo.mp4` instead: the disc mask tinted red on top of everything
above, and each view's mask area and fit rms in the footer. It never *solves* the stereo
pose -- it draws one only if the take already has `stereo_pose.csv` -- so it still runs
before any solve, on the segmenter alone. `controller/report.py` runs it after the solve,
which is how one video carries the mask, the ellipse and the normal at once.

The normal is the world-frame `nx,ny,nz` from the take's `stereo_pose.csv`
(`live_viz.from_recording` + `disc_pose.DiscStereoEstimator`); `--all` runs that pass
first for any take that lacks the file. Direction only is drawn: the arrow's tail is
pinned to the disc centre re-segmented in each view and its length is fixed in pixels,
because the stereo *position* is scaled by a rim radius this rotor does not have
(`disc_pose.py` header) and projecting it would float the arrow off the robot
(`live_viz.normal_segment_px`). The sign is chosen so the arrow points toward world -y,
which is camera A's up; the ellipse alone cannot tell rotor-up from rotor-down.

`--stride 7` keeps every 7th frame and writes at 30 fps, so a 208 fps take plays in real
time. `--stride 1` writes every frame (7x slow motion).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from controller.calib import rig as rigmod
from controller.camera.record import open_recording
from controller.pose import disc_pose

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FLIGHTS = ROOT / "results" / "flights"
SWEEPS = ROOT / "results" / "tilt_sweep"
FONT = cv2.FONT_HERSHEY_SIMPLEX
ARROW_PX = 70
UP_WORLD = np.array([0.0, -1.0, 0.0])   # camera A's -y, the rig's nearest thing to "up"


def flight_of(take_dir):
    """`20260901_210758_drone1` -> `results/flights/2026-09-01_210758`."""

    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{6})", Path(take_dir).name)
    return FLIGHTS / f"{m.group(1)}-{m.group(2)}-{m.group(3)}_{m.group(4)}"


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def ensure_pose(take_dir, flight_dir):
    """Run the disc stereo pass into `stereo_pose.csv` if the take does not have one."""

    out = Path(take_dir) / "stereo_pose.csv"
    if out.exists():
        return out
    from controller.viz import live_viz

    rig = rigmod.StereoRig.load(rigmod.DEFAULT_PATH)
    est = disc_pose.DiscStereoEstimator(rig, backgrounds=disc_pose.plates_for_flight(flight_dir))
    print(f"{take_dir.name}: solving {flight_dir.name} -> stereo_pose.csv")
    # Written aside and renamed only on success. A pass killed part-way (out of memory,
    # 2026-09-04) otherwise leaves a truncated CSV that this function's `exists()` accepts
    # as a cache forever -- 45003 of 48119 frames, silently, with no way to tell.
    tmp = out.with_name(out.name + ".part")
    live_viz.from_recording(flight_dir, rig=rig, est=est, viz=live_viz.NullViz(),
                            speed=0, zero=None, csv_out=str(tmp))
    tmp.replace(out)
    return out


def _arrow_px(cam, centre_world, n_world, tail_px):
    """Pixel direction of the normal in this view, as a unit 2-vector, or None."""

    c_cam, n_cam = cam.to_camera(centre_world, n_world)
    p0 = np.asarray(c_cam, float)
    p1 = p0 + 5.0 * np.asarray(n_cam, float)
    if p0[2] <= 1e-6 or p1[2] <= 1e-6:
        return None
    uv = (cam.K @ np.stack([p0, p1]).T).T
    uv = uv[:, :2] / uv[:, 2:3]
    d = uv[1] - uv[0]
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 1e-9 else None


def render(take_dir, out_path=None, stride=7, fps=30.0, max_frames=None, seg=False):
    """``seg=True`` adds the disc mask tinted and the per-view area and fit rms to the
    footer. It never solves the pose, only draws one the take already has, so it still
    runs on a take with none -- and after a solve it is the everything-at-once video."""

    take_dir = Path(take_dir)
    flight_dir = flight_of(take_dir)
    out_path = Path(out_path or take_dir / ("seg_demo.mp4" if seg else "normal_demo.mp4"))
    rig = rigmod.StereoRig.load(rigmod.DEFAULT_PATH)
    cams = list(rig.cameras)          # rescaled to the frame size on the first frame

    pose_csv = take_dir / "stereo_pose.csv" if seg else ensure_pose(take_dir, flight_dir)
    poses = {int(r["frame"]): r for r in _read(pose_csv)} if pose_csv.exists() else {}
    # Timeline and per-frame tilt are optional dressing; the video stands without them.
    pts, tilt = [], {}
    if (take_dir / "sweep.log").exists():
        from controller.pose import body_angle
        pts = body_angle.timeline(take_dir / "sweep.log")
    if (take_dir / "normal_angle.csv").exists():
        tilt = {int(r["frame"]): float(r["tilt_deg"]) for r in _read(take_dir / "normal_angle.csv")}

    plates = disc_pose.plates_for_flight(flight_dir)
    caps, stamps = open_recording(flight_dir)
    writer, i, n_out = None, 0, 0
    try:
        while True:
            got = [c.read() for c in caps]
            if not all(ok for ok, _ in got) or (max_frames is not None and i >= max_frames):
                break
            if i % stride:
                i += 1
                continue
            row = poses.get(i)
            panels, segtxt = [], []
            n_w = c_w = None
            if row is not None:
                n_w = np.array([float(row["nx"]), float(row["ny"]), float(row["nz"])])
                c_w = np.array([float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])])
            # Segment every view first: the rod in each view feeds the fused axis
            # (`disc_pose.fused_axis`), which is what the arrow shows -- the same axis the
            # report plots, not the raw disc normal.
            views = []
            for cam, (_, frame) in zip(cams, got):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                if cam.K[0, 2] > gray.shape[1]:            # rig calibrated at a larger mode
                    scale = gray.shape[1] / (2.0 * cam.K[0, 2])
                    cams = [c.scaled(scale) for c in cams]
                    cam = cams[len(views)]
                sg = disc_pose.segment_disc(gray, plate=plates.get(cam.name))
                mast = disc_pose.find_mast(gray, sg, plate=plates.get(cam.name)) if sg is not None else None
                views.append((cam, gray, sg, mast))
            lines = [(m[0], m[1]) for _, _, _, m in views if m is not None]
            axis, code = None, None
            if n_w is not None:
                mast_w = disc_pose.mast_direction(lines, [c for c, _, _, m in views if m is not None],
                                                  up=disc_pose.MAST_UP) if len(lines) == 2 else None
                planes = ([disc_pose.mast_plane(lines[0], [c for c, _, _, m in views if m is not None][0])]
                          if len(lines) == 1 else [])
                ratios = [sg.ellipse[1][1] / sg.ellipse[1][0] for _, _, sg, _ in views
                          if sg is not None and sg.ellipse[1][0] > 0]
                got_axis = disc_pose.fused_axis(n_w, mast=mast_w, planes=planes,
                                                ratio_min=min(ratios) if ratios else None,
                                                ref=np.array(UP_WORLD))
                if got_axis is not None:
                    axis, code = got_axis
                    if axis @ UP_WORLD < 0:
                        axis = -axis
            for cam, gray, sg, mast in views:
                out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if sg is not None:
                    (cx, cy), (ma, mi), ang = sg.ellipse
                    if seg:
                        sel = sg.mask > 0
                        out[sel] = (0.6 * out[sel] + np.array([0, 0, 100])).astype(np.uint8)
                        segtxt.append(f"{cam.name}: {int(sg.area_px)} px, rms {sg.fit_rms_px:.2f}")
                    if mast is not None:
                        cv2.circle(out, tuple(int(v) for v in mast[0]), 5, (255, 0, 255), 1)
                    cv2.ellipse(out, ((cx, cy), (ma, mi), ang), (0, 200, 0), 1)
                    if axis is not None:
                        d = _arrow_px(cam, c_w, axis, (cx, cy))
                        if d is not None:
                            tip = (int(round(cx + ARROW_PX * d[0])), int(round(cy + ARROW_PX * d[1])))
                            col = {1: (255, 220, 0), 2: (255, 160, 60), 0: (160, 160, 160)}[code]
                            cv2.arrowedLine(out, (int(round(cx)), int(round(cy))), tip, col, 2,
                                            tipLength=0.25)
                cv2.putText(out, cam.name or "?", (8, 20), FONT, 0.6, (255, 255, 255), 1)
                panels.append(out)
            canvas = np.hstack(panels)

            t = float(np.mean(stamps[i])) if stamps is not None and i < len(stamps) else i / fps
            bar = np.zeros((44, canvas.shape[1], 3), np.uint8)
            txt = f"frame {i:6d}"
            p = next((p for p in pts if p["t_start"] <= t <= p["t_end"]), None)
            col = (255, 255, 255)
            if p is not None:
                rel = t - p["t_drop"]
                txt += f"   {p['freq']} Hz point   t from ch0 30% drop {rel:+7.2f} s"
                if rel >= 0:
                    txt += "   CH0 AT 30%"
                    col = (80, 200, 255)
            if i in tilt:
                txt += f"   tilt from rest {tilt[i]:5.1f} deg"
            if seg:
                txt += "   " + "   ".join(segtxt or ["no disc"])
            if row is None:
                txt += "   (no stereo pose)"
            elif axis is None:
                txt += "   axis REFUSED (disc and rod disagree)"
            else:
                txt += "   axis: " + {1: "disc+rod blended", 2: "disc in one rod plane", 0: "disc only"}[code]
            cv2.putText(bar, txt, (10, 29), FONT, 0.6, col, 1)
            canvas = np.vstack([canvas, bar])
            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
            writer.write(canvas)
            n_out += 1
            i += 1
    finally:
        for c in caps:
            c.release()
        if writer is not None:
            writer.release()
    print(f"{out_path}  {n_out} frames written from {i} read (stride {stride})")
    return out_path


def render_all(stride=7, seg=False):
    for take in sorted(SWEEPS.iterdir()):
        if not take.is_dir() or not re.match(r"\d{8}_\d{6}", take.name):
            continue
        flight = flight_of(take)
        if not (flight / "meta.json").exists():
            print(f"{take.name}: no meta.json in {flight.name}, recording never finalised; skipped")
            continue
        render(take, stride=stride, seg=seg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?", help="results/tilt_sweep/<take>")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seg", action="store_true", help="tint the disc mask too (seg_demo.mp4)")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    if a.all:
        render_all(stride=a.stride, seg=a.seg)
    elif a.take:
        render(a.take, stride=a.stride, max_frames=a.max_frames, seg=a.seg)
    else:
        ap.error("give a take dir or --all")
