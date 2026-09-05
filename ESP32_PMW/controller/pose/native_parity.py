"""
Hold the compiled pose core to the Python reference, stage by stage.

    uv run python controller/pose/native_parity.py --stage all
    uv run python controller/pose/native_parity.py --stage segment --plates saved

    Each stage feeds the same recorded frames through the Python function and its
    `pmw_pose` port and asserts the outputs agree. The evidence map, the segmenter and
    the match are held to rounding; the solve is held to identical iteration counts and
    a loose bound on the pose, because the trust region amplifies sqrt(eps) noise in the
    Jacobian's forward-differenced columns into 1e-4 mm on the odd frame (p95 is 1e-6).
    A difference beyond that is a bug or an OpenCV version skew, and either wants
    finding here rather than on the bench. See `theory.md` 21.3.

    ``evidence``  ring_weight / sample_map / ellipse_points, on the frames' own ROIs.
    ``segment``   segment() on saved plates: hull, area, ellipse, None-agreement.
    ``refine``    the image-mode solve on the exact inputs the Python estimator built.
    ``solve``     both estimators end to end on identical frames, stamps and motion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from controller.pose import background as bgmod
from controller.pose import segment as segmod
from controller.pose import stereo
from controller.pose import stereo_native as nat
from controller.pose.filter import PoseFilter
from controller.viz import live_viz

HERE = Path(__file__).resolve().parent
DEFAULT_RECORDING = HERE.parents[1] / "results/flights/New Folder With Items/2026-08-29_231418"


def frames_of(rec_dir, n, scale):
    """``(index, [gray per camera], stamps)`` for the first ``n`` stereo pairs, rescaled
    exactly as `live_viz.from_recording` does."""

    from controller.camera.record import open_recording

    caps, stamps = open_recording(rec_dir)
    try:
        for i in range(n):
            got = [c.read() for c in caps]
            if not all(ok for ok, _ in got):
                return
            fr = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, f in got]
            if scale != 1.0:
                fr = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                      for f in fr]
            row = stamps[i] if stamps is not None and i < len(stamps) else None
            yield i, fr, (None if row is None else list(row))
    finally:
        for c in caps:
            c.release()


def plates_of(kind, tags, scale):
    """Per-camera plates: fresh `RunningPlate`s, or the bench's saved plates rescaled."""

    if kind == "running":
        return {t: bgmod.RunningPlate() for t in tags}
    plates = bgmod.load_stereo(tags)
    if scale != 1.0:
        plates = {k: cv2.resize(v, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                  for k, v in plates.items()}
    return plates


def _estimators(plates):
    rig, py = live_viz._stereo_estimator(backgrounds=dict(plates), native=False)
    return rig, py


def _report(name, diffs, tol):
    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    mx = float(d.max()) if len(d) else 0.0
    p95 = float(np.percentile(d, 95)) if len(d) else 0.0
    print(f"  {name:28s} n={len(d):4d}  max {mx:.3e}  p95 {p95:.3e}  tol {tol:.0e}"
          f"  {'ok' if mx <= tol else 'FAIL'}")
    return mx <= tol


# ---------------------------------------------------------------------------- stages


def stage_evidence(args):
    print("evidence: ring_weight / sample_map / ellipse_points")
    plates = plates_of(args.plates, "AB", args.scale)
    rig, py = _estimators(plates)
    map_diff, samp_diff, pts_diff = [], [], []
    rng = np.random.default_rng(0)
    for i, fr, row in frames_of(args.recording, args.frames, args.scale):
        py.update(fr, t=i / 60.0, frame_index=i, stamps=row)
        cfg = nat.native_config(py)
        for g, cam in zip(fr, rig.cameras):
            plate = py.backgrounds.get(cam.name)
            plate_arr = plate.bg.astype(np.uint8) if hasattr(plate, "bg") and plate.ready else (
                None if hasattr(plate, "update") else plate)
            roi = segmod.ellipse_roi(py._prev_ellipse.get(cam.name), g.shape)
            k = cfg["ring_ksize"]
            w_py = segmod.ring_weight(g, background=plate_arr, roi=roi, ksize=k)
            w_c = nat.pmw_pose.ring_weight(g, plate_arr, roi, k, cfg["ring_blur_sigma"],
                                           cfg["ring_plate_weight"])
            map_diff.append(np.abs(w_py - w_c).max())
            pts = np.column_stack([rng.uniform(-50, g.shape[1] + 50, 900),
                                   rng.uniform(-50, g.shape[0] + 50, 900)])
            samp_diff.append(np.abs(segmod.sample_map(w_py, pts) - nat.pmw_pose.sample_map(w_py, pts)).max())
            e = py._prev_ellipse.get(cam.name)
            if e is not None:
                pts_diff.append(np.abs(segmod.ellipse_points(e) - nat.pmw_pose.ellipse_points(e, 180)).max())
    ok = _report("ring_weight max|dw|", map_diff, 1e-3)
    ok &= _report("sample_map max|ds|", samp_diff, 0.0)
    ok &= _report("ellipse_points max|dp|", pts_diff, 1e-9)
    return ok


def stage_segment(args):
    print("segment: threshold -> silhouette_hull -> fit_ellipse, on saved plates")
    plates = plates_of("saved", "AB", args.scale)
    rig, py = _estimators(plates)
    agree, hull_diff, area_diff, ell_diff, n_none = 0, [], [], [], 0
    for i, fr, row in frames_of(args.recording, args.frames, args.scale):
        py._match_scale(fr[0])
        cfg = nat.native_config(py)
        for g, cam in zip(fr, rig.cameras):
            plate = plates[cam.name]
            s_py = segmod.segment(g, thresh=py.thresh, min_area=py.min_area, background=plate)
            s_c = nat.pmw_pose.segment(g, plate, cfg)
            if (s_py is None) != (s_c is None):
                print(f"  frame {i} {cam.name}: python {'None' if s_py is None else 'seg'}, "
                      f"native {'None' if s_c is None else 'seg'}")
                continue
            agree += 1
            if s_py is None:
                n_none += 1
                continue
            a = np.array(sorted(map(tuple, s_py.contour)))
            b = np.array(sorted(map(tuple, s_c["contour"])))
            hull_diff.append(np.abs(a - b).max() if a.shape == b.shape else np.inf)
            area_diff.append(abs(s_py.area_px - s_c["area_px"]))
            (cx, cy), (M, m), ang = s_py.ellipse
            (cx2, cy2), (M2, m2), ang2 = s_c["ellipse"]
            ell_diff.append(max(abs(cx - cx2), abs(cy - cy2), abs(M - M2), abs(m - m2),
                                abs((ang - ang2 + 90) % 180 - 90)))
    print(f"  None/None agreement on {n_none} views; {agree} views compared")
    ok = _report("hull points max|dp|", hull_diff, 1e-9)
    ok &= _report("area_px |da|", area_diff, 0.0)
    ok &= _report("ellipse max|de| (px, deg)", ell_diff, 1e-6)
    return ok


def stage_refine(args):
    print("refine: image-mode solve on the Python estimator's own inputs")
    plates = plates_of(args.plates, "AB", args.scale)
    rig, py = _estimators(plates)
    captured = []
    orig = stereo.refine

    def spy(hulls, rig_, seed_c, seed_n, radius_mm, **kw):
        r = orig(hulls, rig_, seed_c, seed_n, radius_mm, **kw)
        captured.append((list(rig_.cameras), np.array(seed_c), np.array(seed_n), kw["weights"], r))
        return r

    stereo.refine = spy
    try:
        for i, fr, row in frames_of(args.recording, args.frames, args.scale):
            py.update(fr, t=i / 60.0, frame_index=i, stamps=row)
    finally:
        stereo.refine = orig
    cfg = nat.native_config(py)
    cc = nat.centre_cal_dict(py.centre_cal)
    ref = np.ascontiguousarray(py.reference, dtype=np.float64)
    dc, dn, dnfev, drms, none_mismatch = [], [], [], [], 0
    for cams, c0, n0, weights, r_py in captured:
        r_c = nat.pmw_pose.refine([np.ascontiguousarray(w, dtype=np.float32) for w in weights],
                                  [nat.camera_dict(c) for c in cams], c0, n0, cc, cfg, ref)
        if (r_py is None) != (r_c is None):
            none_mismatch += 1
            continue
        if r_py is None:
            continue
        dc.append(np.abs(r_py.center - r_c["center"]).max())
        dn.append(stereo.line_angle_deg(r_py.normal, r_c["normal"]))
        dnfev.append(abs(r_py.n_iter - r_c["n_iter"]))
        drms.append(abs(r_py.rms_px - r_c["rms_px"]))
    print(f"  {len(captured)} solves captured, {none_mismatch} None mismatches")
    ok = none_mismatch == 0
    ok &= _report("centre max|dc| (mm)", dc, 1e-2)
    ok &= _report("normal angle (deg)", dn, 1e-1)
    ok &= _report("nfev |dn|", dnfev, 0.0)
    ok &= _report("rms_px |dr|", drms, 1e-3)
    return ok


def stage_solve(args):
    print("solve: both estimators end to end on identical frames, stamps and motion")
    plates_py = plates_of(args.plates, "AB", args.scale)
    plates_c = plates_of(args.plates, "AB", args.scale)
    rig, py = _estimators(plates_py)
    cn = nat.NativeStereoPoseEstimator.from_python(py)
    cn.backgrounds = dict(plates_c)
    filt = PoseFilter()
    dxyz, dang, diters, ddisc, dmarg, mismatch = [], [], [], [], [], 0
    n_py = n_c = 0
    t_py = t_c = 0.0
    for i, fr, row in frames_of(args.recording, args.frames, args.scale):
        t = float(np.mean(row)) if row is not None else i / 60.0
        motion = filt.pos if filt.pos.initialised else None
        p_py = py.update(fr, t=t, frame_index=i, stamps=row, motion=motion)
        p_c = cn.update(fr, t=t, frame_index=i, stamps=row, motion=motion)
        filt.update(p_py, t=t)
        if (p_py is None) != (p_c is None):
            mismatch += 1
            print(f"  frame {i}: python {'None' if p_py is None else 'pose'}, native "
                  f"{'None' if p_c is None else 'pose'}")
            continue
        if p_py is None:
            continue
        n_py += 1
        n_c += 1
        t_py += p_py.t_seg_ms + p_py.t_est_ms
        t_c += p_c.t_seg_ms + p_c.t_est_ms
        dxyz.append(np.abs(p_py.xyz_mm - p_c.xyz_mm).max())
        dang.append(stereo.line_angle_deg(p_py.normal, p_c.normal))
        diters.append(abs(p_py.refine_iters - p_c.refine_iters))
        ddisc.append(abs(p_py.discrepancy_mm - p_c.discrepancy_mm))
        dmarg.append(abs(p_py.margin - p_c.margin))
    print(f"  solved: python {n_py}, native {n_c}, {mismatch} None mismatches")
    if n_py:
        print(f"  mean seg+est ms/pair: python {t_py / n_py:.2f}  native {t_c / n_c:.2f}")
    ok = mismatch == 0
    ok &= _report("xyz max|d| (mm)", dxyz, 1e-2)
    ok &= _report("normal angle (deg)", dang, 1e-1)
    ok &= _report("refine_iters |dn|", diters, 0.0)
    ok &= _report("discrepancy_mm |d|", ddisc, 1e-4)
    ok &= _report("margin |d|", dmarg, 1e-4)
    return ok


STAGES = {"evidence": stage_evidence, "segment": stage_segment, "refine": stage_refine,
          "solve": stage_solve}


def _self_check(args):
    ok = True
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        ok &= STAGES[name](args)
    assert ok, "native parity FAILED (see the FAIL rows above)"
    print("native parity ok")


if __name__ == "__main__":
    if not nat.available():
        sys.exit("pmw_pose is not built; run `uv sync --extra native`")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    ap.add_argument("--recording", default=str(DEFAULT_RECORDING))
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--plates", choices=["running", "saved"], default="running")
    _self_check(ap.parse_args())
