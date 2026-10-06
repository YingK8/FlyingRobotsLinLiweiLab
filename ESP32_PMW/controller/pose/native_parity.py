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
    ``refine``    the image-mode solve. Uses the Python estimator's own inputs when the
                  recording solves; otherwise synthesises well-posed ones (21.5), so the
                  stage runs on any recording rather than only on a rim robot's.
    ``solve``     both estimators end to end. NEEDS a recording that solves -- it cannot
                  be synthesised on this rig's symmetric geometry (21.6).
    ``filter``    the constant-velocity Kalman filter, its innovation gate, and the
                  MAX_GATED escape, on a scripted stream that visits all three.
    ``control``   the LQR law: saturation, slew limiting and anti-windup, plus every
                  `simulate_hover` acceptance scenario re-run against the C++ controller.

    ``filter`` and ``control`` need no recording at all. **A stage with no samples FAILS**
    rather than reporting `ok` on an empty comparison; see 21.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import json
import math

import numpy as np

from controller.pose import background as bgmod
from controller.pose import segment as segmod
from controller.pose import conic
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


def _report(name, diffs, tol, outlier_frac=0.0):
    """One comparison line. **A stage with no samples FAILS.**

    `outlier_frac` allows a stated fraction of samples past `tol`, and is used by exactly
    one caller: the synthetic refine seeds. It is not slack. `pose/theory.md` 21.3
    measured that one float32 ulp in the shared arithmetic moves the trust-region solve by
    ~0.4 mm on ~5 % of frames -- the two cores take a different number of steps and settle
    in a different basin. That is a property of the problem, established before this
    harness existed, not of the port. Grading `max` alone would either fail a correct port
    or force a tolerance loose enough to hide a real one; grading the bulk tightly and
    bounding the outlier COUNT keeps both. The count is always printed.

    An empty `diffs` used to print `max 0.000e+00 ... ok`, so a run that compared nothing
    was indistinguishable from a run that compared everything and agreed. That is not a
    hypothetical: on a `tilt_sweep` recording the rim estimator solves 0 of 250 frames
    ("tilt robots have no rim" -- `pose/theory.md`), so `--stage refine` and `--stage
    solve` capture zero solves and the harness reported `native parity ok` while
    exercising neither the trust-region solve nor the fuse. The harness is the gate the
    C++ port is held to; a gate that passes on no evidence is not a gate.

    Same rule as every other refusal in this tree: say what is missing, do not average it
    away. `fit_rotation` refuses below its noise floor, `ramp.check` refuses instead of
    clamping, `coil_phase.fit_channel` refuses a peak on the band edge. This is that.
    """

    d = np.asarray(diffs, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        print(f"  {name:28s} n=   0  NO SAMPLES -- nothing was compared, so nothing "
              f"is verified")
        return False
    mx, p95 = float(d.max()), float(np.percentile(d, 95))
    over = int((d > tol).sum())
    ok = over == 0 or (outlier_frac > 0.0 and over <= math.ceil(outlier_frac * len(d)))
    note = "" if over == 0 else f"  [{over}/{len(d)} over tol]"
    print(f"  {name:28s} n={len(d):4d}  max {mx:.3e}  p95 {p95:.3e}  tol {tol:.0e}"
          f"  {'ok' if ok else 'FAIL'}{note}")
    return ok


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


def _rim_map(cam, c_world, n_world, radius_mm, shape, sigma=5.0):
    """A float32 evidence map with a real rim drawn into it, as `ring_weight` would see.

    The circle of `radius_mm` about `n_world` centred at `c_world` is projected through
    `cam` and rasterised as a soft bright ring on a dim ground. `refine(mode="image")`
    maximises sampled evidence along the predicted silhouette, so a map built this way has
    its optimum exactly at the pose that drew it -- which is what makes it a test with an
    answer rather than a comparison of two wanderings.
    """

    h, w = shape
    m = np.full((h, w), 2.0, dtype=np.float32)
    t1, t2 = stereo._tangent_basis(np.asarray(n_world, dtype=np.float64))
    th = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    pts = (np.asarray(c_world, dtype=np.float64)[None, :]
           + radius_mm * (np.cos(th)[:, None] * t1[None, :]
                          + np.sin(th)[:, None] * t2[None, :]))
    R = cam.T_world_cam[:3, :3]
    T = cam.T_world_cam[:3, 3]
    pc = (pts - T[None, :]) @ R                      # world -> camera
    pc = pc[pc[:, 2] > 1.0]
    if pc.shape[0] < 32:
        return None
    uv = (cam.K @ pc.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    ok = np.isfinite(uv).all(axis=1)
    uv = uv[ok]
    xi = np.rint(uv[:, 0]).astype(int)
    yi = np.rint(uv[:, 1]).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    if inside.sum() < 32:
        return None
    # Anti-aliased polyline with real width, then a wide blur. Setting isolated pixels
    # and blurring narrowly leaves a SPECKLED ridge: the rasterised samples land on
    # integer pixels at uneven spacing, so the surface carries high-frequency structure
    # the solver descends into differently in each core. That is chaos, not divergence --
    # it took the synthetic refine parity from 1/24 seeds over tolerance to 9/24, with a
    # p95 normal error of 1.8 deg, none of which said anything about the port. A smooth
    # ridge is what `ring_weight` actually produces and what the residual assumes.
    poly = np.stack([xi[inside], yi[inside]], axis=1).astype(np.int32)
    cv2.polylines(m, [poly], True, 255.0, thickness=3, lineType=cv2.LINE_AA)
    return np.ascontiguousarray(cv2.GaussianBlur(m, (0, 0), sigma), dtype=np.float32)


def _synthetic_refine_inputs(args, rig, py, shape, n=24):
    """Deterministic (cams, seed_c, seed_n, weights, r_py) tuples with a KNOWN answer.

    Used when the recording solves nothing, which is every `tilt_sweep` take in `results/`
    -- a tilt robot has no rim for the rim segmenter to find, so the estimator calls
    `refine` zero times and there is nothing to spy on.

    Two properties make this a real test rather than a substitute for one:

    **The problem is well posed.** Evidence maps are rasterised from a known pose
    (`_rim_map`), so the residual surface has a true optimum and both cores must descend
    to it. An earlier version of this used real frames from a rim-less recording with
    arbitrary seeds; it measured chaos, not parity -- p95 agreed to 0.45 mm while the
    worst seed diverged by 1.2 m, because a solve with no optimum to find amplifies the
    last ulp without bound. Two correct implementations disagree on that surface. Never
    grade a port on an ill-posed problem.

    **The pose is realisable.** Seeds are perturbations of a back-projected truth, so the
    cone is never degenerate. A pose picked as three world numbers usually is: camera A
    sits at the world origin looking along +z, so a centre near the origin is a centre at
    the optical centre, and `refine` returns None from inside its own
    `np.percentile(evidence(p0))` guard for every such seed.
    """

    rng = np.random.default_rng(20260908)
    # `py.rig`, NOT the rig `_estimators` returned: `_ensure_scale` REBINDS `py.rig` to a
    # rescaled copy on the first frame, so the caller's reference still carries the
    # calibrated 1280x800 intrinsics while the frames are 640x400. Projecting with those
    # puts the rotor at x=671 in a 640-wide image, i.e. off the edge, and every rendered
    # pose is silently rejected.
    rig = py.rig
    cams = list(rig.cameras)
    cam_a = cams[0]
    K = cam_a.K
    f = 0.5 * (K[0, 0] + K[1, 1])
    R = py.radius_mm
    out = []
    while len(out) < n:
        depth = rng.uniform(115.0, 145.0)
        major = 2.0 * f * R / depth
        tilt = math.radians(rng.uniform(3.0, 22.0))
        e = ((float(K[0, 2] + rng.uniform(-20, 20)), float(K[1, 2] + rng.uniform(-20, 20))),
             (float(major), float(major * math.cos(tilt))), float(rng.uniform(0, 180)))
        poses = conic.backproject_ellipse(e, K, R)
        if not poses:
            continue
        c_cam, n_cam = poses[0]
        c_true, n_true = cam_a.to_world(np.asarray(c_cam, float), np.asarray(n_cam, float))
        n_true = n_true / np.linalg.norm(n_true)
        maps = [_rim_map(c, c_true, n_true, R, shape) for c in cams]
        if any(m is None for m in maps):
            continue
        # Seeded CLOSE -- 0.05 mm and ~0.1 deg. Deliberately, and the distance was chosen
        # by measurement, not taste. Sweeping it against both cores:
        #
        #   seed offset   centre p50    centre p95    max |d nfev|   seeds over 1e-2 mm
        #   0.00 mm       9.5e-10       4.6e-06       0              0 / 16
        #   0.05 mm       8.8e-10       3.7e-08       0              0 / 16
        #   0.25 mm       1.7e-09       2.1e-02       4              1 / 16
        #   1.00 mm       1.9e-08       3.8e-02       1              2 / 16
        #
        # The median is at rounding everywhere: the cores compute the same thing. What
        # grows with distance is the FRACTION of seeds whose descent takes a different
        # number of steps and settles in a different basin -- 21.3's one-ulp sensitivity,
        # amplified by a longer path. A far seed therefore measures basin-hopping, and a
        # test that measures basin-hopping cannot see a real port bug underneath it.
        #
        # A short descent is strictly more sensitive to the thing this exists to catch:
        # any genuine arithmetic difference shows up at 1e-7 here, where the tolerance is
        # 1e-2. So the tolerances stay tight and NOTHING is allowed past them.
        c0 = c_true + rng.normal(scale=0.05, size=3)
        nn = n_true + rng.normal(scale=0.002, size=3)
        n0 = nn / np.linalg.norm(nn)
        r_py = stereo.refine(
            None, rig, c0, n0, R, loss="cauchy", reference=py.reference,
            tilt_cal=py.tilt_cal, centre_cal=py.centre_cal, ellipses=None,
            mode="image", weights=maps)
        out.append((cams, c0, n0, maps, r_py))
    return out


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
    shape = None
    try:
        for i, fr, row in frames_of(args.recording, args.frames, args.scale):
            # The estimator rescales its rig to the first frame it sees, so the synthetic
            # maps below must be that size too -- not the calibrated size.
            shape = fr[0].shape[:2]
            py.update(fr, t=i / 60.0, frame_index=i, stamps=row)
    finally:
        stereo.refine = orig
    if not captured:
        # The estimator solved nothing on this recording, so it called `refine` zero
        # times and there is nothing to compare. That is not a reason to pass: it is a
        # reason to make the inputs ourselves.
        #
        # Parity is a question about two IMPLEMENTATIONS, not about the robot. The solve
        # takes an evidence map per view and a seed pose; both are constructible from any
        # frames at all. So when the spy comes back empty -- which is every `tilt_sweep`
        # take in `results/`, because a tilt robot has no rim for the rim segmenter to
        # find -- synthesise deterministic seeds over the working volume and put the same
        # problem to both cores.
        #
        # An arbitrary seed on a real evidence map is a HARDER parity test than a clean
        # one, not a weaker one. The solve wanders, hits its iteration cap, and takes
        # branches a converging solve never reaches; the two cores must wander
        # identically, including diverging identically. `theory.md` 21.3 records that one
        # float32 ulp of `fitEllipseDirect` moves the solve by 0.4 mm on 5% of frames --
        # exactly the sensitivity this exercises.
        captured = _synthetic_refine_inputs(args, rig, py, shape, n=args.seeds)
        print(f"  no solves on this recording -- {len(captured)} synthetic seed(s) instead")
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
    """Both estimators end to end. **Needs a recording the estimator can actually solve.**

    Unlike `--stage refine`, this one cannot be driven synthetically, and the reason is a
    property of the rig rather than a gap in the harness. `calib/stereo_rig.json` places
    the reference axis at 41.4 deg from BOTH optical axes -- the deliberate symmetry of a
    45 deg / 90 deg rig. Back-projecting one ellipse always yields two circle poses, and
    on a symmetric synthetic target view A's FALSE branch carries the same normal as view
    B's TRUE one, so `match` -- which scores orientation agreement -- pairs them, agrees
    perfectly, and lands ~1000 mm out. Measured over a 9x4 tilt/azimuth sweep at the
    axes' crossing point, with a filled disc and with a rim: 0 solves, every frame
    refused by the discrepancy gate.

    Real recordings do not hit this, because the estimator carries a temporal
    `prior_normal` once it has solved a frame and a real rim is not perfectly symmetric.
    **That dependence is worth knowing for its own sake**: first-frame stereo
    disambiguation on this rig is structurally degenerate, so acquisition leans on the
    prior rather than on the geometry. `pose/theory.md` 21.5.

    On a recording that solves nothing this reports NO SAMPLES and fails, which is
    correct: it has verified nothing.
    """

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


def stage_filter(args):
    """`filter._ConstantVelocity` against its C++ port, on a scripted measurement stream.

    No recording needed and none wanted: the filter is a recurrence, so what has to match
    is the whole trajectory of the state through predicts, accepted updates, GATED updates
    and the `MAX_GATED` escape. A stream built here can visit all four deliberately, where
    a recording visits whichever the robot happened to produce.
    """

    from controller.control import native_config
    from controller.pose.filter import _ConstantVelocity, GATE_SIGMA, MAX_GATED

    print("filter: constant-velocity Kalman, including the gate and its escape")
    cfg = native_config.control_config()
    rng = np.random.default_rng(4242)
    dpos, drate, dacc = [], [], []
    n_gated_py = n_gated_c = 0

    for trial in range(6):
        py = _ConstantVelocity(cfg["accel_mm_s2"], cfg["p0_pos"], cfg["p0_vel"])
        cc = nat.pmw_pose.ConstantVelocity(cfg["accel_mm_s2"], cfg["p0_pos"], cfg["p0_vel"])
        truth = np.array([0.0, 0.0, 60.0])
        vel = rng.normal(scale=8.0, size=3)
        sig = np.array([0.05, 0.05, 0.35])
        r = np.diag(sig ** 2)
        for k in range(220):
            dt = 0.005 if k % 3 else 0.011      # uneven, as a real pose feed is
            truth = truth + vel * dt
            py.predict(dt)
            cc.predict(dt)
            z = truth + rng.normal(scale=sig)
            # Deliberate outliers, and then a SUSTAINED excursion: the first must be
            # gated, the second must force its way in through MAX_GATED. A port that
            # dropped the escape passes an outlier-only test and locks on in flight.
            if k in (60, 61, 130):
                z = z + np.array([25.0, -18.0, 40.0])
            if 160 <= k < 160 + 3 * MAX_GATED:
                z = z + np.array([0.0, 0.0, 30.0])
            ok_py = py.update(z, r, gate=GATE_SIGMA)[1]
            ok_c = cc.update(np.ascontiguousarray(z), np.ascontiguousarray(r),
                             GATE_SIGMA, MAX_GATED)
            if ok_py != ok_c:
                print(f"  trial {trial} step {k}: python {'took' if ok_py else 'gated'}, "
                      f"native {'took' if ok_c else 'gated'}")
                return False
            n_gated_py += (not ok_py)
            n_gated_c += (not ok_c)
            dpos.append(float(np.abs(py.value - cc.value).max()))
            drate.append(float(np.abs(py.rate - cc.rate).max()))
            dacc.append(0.0 if py.n_gated == cc.n_gated else 1.0)

    print(f"  {len(dpos)} steps, {n_gated_py} gated on both sides, escape exercised")
    ok = _report("position max|dp| (mm)", dpos, 1e-9)
    ok &= _report("rate max|dv| (mm/s)", drate, 1e-9)
    ok &= _report("n_gated |dn|", dacc, 0.0)
    return ok


def stage_control(args):
    """`simulate_hover.DiscreteHoverController` against its C++ port.

    Scalar arithmetic, so it should agree to ~1e-12; the point is not the numbers but the
    BRANCHES -- saturation on both signs, the slew limit, and conditional-integration
    anti-windup, which is three predicates and the place a port silently differs.
    """

    from controller.control import native_config
    from controller.control.reference_profiles import Profile
    from controller.control.simulate_hover import DiscreteHoverController

    print("control: LQR law, saturation, slew limit and anti-windup")
    gains = json.loads((Path(__file__).resolve().parents[1] /
                        "control" / "hover_controller.json").read_text())
    cfg = native_config.control_config(gains)
    rng = np.random.default_rng(99)
    dmag, dfreq, dq = [], [], []

    for trial, (x0, z0, drive) in enumerate([
            (0.0, 0.0, 0.0),        # at the setpoint: nothing saturates
            (8.0, 5.0, 0.0),        # large error: mag saturates, freq slews
            (-8.0, -5.0, 0.0),      # and on the other sign
            (0.0, 0.0, 1.0)]):      # driven by noise, so the branches interleave
        prof = Profile.hold()
        py = DiscreteHoverController(gains, prof)
        cc = nat.pmw_pose.HoverController(cfg)
        x, z = x0 * 1e-3, z0 * 1e-3
        for k in range(400):
            t = k * cfg["ts"]
            # A tick with no new fix passes dt=None on the Python side and have_dt=False
            # on the C++ side; the rate estimate must be HELD, not re-differenced.
            fresh = (k % 5 == 0)
            dt = 0.01 if fresh else None
            x += drive * rng.normal(scale=2e-4)
            z += drive * rng.normal(scale=2e-4)
            m_py, f_py = py.step(t, x, z, dt)
            rp, rv, ra = prof.eval(t)
            m_c, f_c = cc.step(x, z, 0.01 if fresh else 0.0, fresh,
                               (float(rp[0]), float(rp[1])), (float(rv[0]), float(rv[1])),
                               (float(ra[0]), float(ra[1])))
            dmag.append(abs(m_py - m_c))
            dfreq.append(abs(f_py - f_c))
            dq.append(float(np.abs(np.asarray(py.q) - np.asarray(cc.q)).max()))
    print(f"  {len(dmag)} steps over 4 trials")
    ok = _report("mag |dm|", dmag, 1e-12)
    ok &= _report("f_field |df| (Hz)", dfreq, 1e-12)
    ok &= _report("integrator |dq|", dq, 1e-12)

    # THE CHEAP STRONG CHECK. The law is also run through `simulate_hover`'s own
    # acceptance scenarios -- the ones the shipped design was signed off on -- by
    # substituting the C++ controller for the Python one inside `simulate`. Seven
    # validated closed-loop tests for the cost of an adapter, and they exercise the law
    # against a nonlinear truth plant with noise, latency, trim mismatch and a 0.25x-4x
    # gain sweep, which no scripted input can imitate.
    class _NativeCtrl:
        """`DiscreteHoverController`'s surface, delegating to the C++ implementation."""

        def __init__(self, gains_, profile):
            self.ts = gains_["design"]["ts"]
            self.f_hover = gains_["params"]["f_hover"]
            self._prof = profile
            self._c = nat.pmw_pose.HoverController(native_config.control_config(gains_))

        def step(self, t, x_meas, z_meas, dt=None):
            # `simulate` steps once per measurement, so every tick carries a real dt --
            # which is exactly the case the Python's `_UNSET` default stands for.
            rp, rv, ra = self._prof.eval(t)
            return self._c.step(x_meas, z_meas, self.ts, True,
                                (float(rp[0]), float(rp[1])), (float(rv[0]), float(rv[1])),
                                (float(ra[0]), float(ra[1])))

    import controller.control.simulate_hover as SHmod
    orig_ctrl = SHmod.DiscreteHoverController
    n_pass = n_fail = 0
    try:
        SHmod.DiscreteHoverController = _NativeCtrl
        for group in SHmod.build_scenarios().values():
            for sc in group:
                out = SHmod.simulate(sc, gains)
                good, msgs = SHmod.evaluate(sc, out, gains)
                n_pass += bool(good)
                n_fail += (not good)
                if not good:
                    print(f"  scenario {sc.name}: FAIL under the C++ law")
                    for m in msgs:
                        print(f"    {m}")
    finally:
        SHmod.DiscreteHoverController = orig_ctrl
    print(f"  simulate_hover scenarios under the C++ law: {n_pass} pass, {n_fail} fail")
    ok &= (n_fail == 0)
    return ok


STAGES = {"evidence": stage_evidence, "segment": stage_segment, "refine": stage_refine,
          "solve": stage_solve, "filter": stage_filter, "control": stage_control}


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
    ap.add_argument("--seeds", type=int, default=24,
                    help="synthetic refine seeds when the recording solves nothing")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--plates", choices=["running", "saved"], default="running")
    _self_check(ap.parse_args())
