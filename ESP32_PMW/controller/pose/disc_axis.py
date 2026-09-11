#!/usr/bin/env python3
"""Rotor axis per frame, offline, into the `axis.csv` + `tilt_A/B.csv` take layout.

    uv run python controller/pose/disc_axis.py                      # self-check, no camera
    uv run python controller/pose/disc_axis.py results/flights/<take>

WHY THIS FILE EXISTS
--------------------
`results/rim/` holds 56 takes in this layout and `results/alignment_rate/summary.csv` was
computed from them, but the script that wrote them is in no branch of this repo -- commit
`7f638cf` landed the output and not the producer. This is that pass, rebuilt from the
columns it left behind. It is deliberately the SAME measurement, so old and new takes
concatenate.

TWO ESTIMATORS, ON PURPOSE
--------------------------
`theta_deg` is the legacy per-view channel, `acos(minor / major)` -- the weak-perspective
axis-ratio fit. That it is exactly what the missing script computed was established by
reading its output back: over all 7065 frames of `2026-09-08_205108`, `theta_deg` matches
`degrees(arccos(d1 / d2))` to 0.0008 deg. Keeping it keeps those 56 takes comparable.

`nx,ny,nz` instead comes from `conic.backproject_ellipse` -- the exact cone decomposition,
which needs `K` and gets the answer right off-axis. Measured against the axis-ratio form at
this rig's intrinsics (see `theory.md`), the two have IDENTICAL noise -- 0.020 deg sd apiece
at 0.2 px of contour noise -- and differ only in bias, 0.25 deg against 0.004. So the exact
one is free and the fused axis may as well have it.

Neither choice is what limits this pipeline. Measured second-difference noise on the shipped
takes is 2.03 deg on view A after the coherent 1-3x motion is removed, which is ~100x either
estimator's floor and corresponds to ~20 px of contour error. That is SEGMENTATION -- the
ellipse being fit to a different shape frame to frame -- which is why this uses
`disc_pose.segment_disc` (hysteresis threshold, round-hole fill, an opening that deletes the
mast and guy wires, blade recovery, an area cap) rather than a plain threshold.

`agree_deg` here is view A against view B. Note this is NOT the `agree_deg` of
`tilt_report.py`, which is the disc against the triangulated mast -- two sensors with no
shared failure mode. A-vs-B share the segmenter, so it is the weaker of the two measures and
reads ~6.5 deg median on the existing takes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from controller.calib import rig as rigmod
from controller.camera import record
from controller.pose import conic, disc_pose, stereo

#: Sign convention for the reported axis, matching the existing takes (their `nz` sits near
#: -0.35). A global flip changes no physics -- every tilt downstream is an angle from a datum
#: -- but matching means old and new `axis.csv` can be plotted on one pair of axes.
#:
#: THIS ONLY SEEDS THE FIRST FRAME. An ellipse fixes the axis as a line and nothing more, so
#: the sign is imposed by `v @ ref`, and that test is decided by the component of the axis
#: ALONG the reference. A reference near perpendicular to where the axis actually lies is
#: therefore decided by noise, and the vector flips end-for-end between adjacent frames --
#: azimuth jumps 180 deg and any rate differenced across the flip is garbage. `(0, 0, -1)`
#: IS near perpendicular here: over 399890 frames of `results/flights/*/axis.csv` the axis
#: comes within 0.3 deg of the plane perpendicular to it, and the 2026-09-09 campaign sits a
#: median 64 deg off. No fixed vector fixes this -- the rig has moved between campaigns, and
#: the best single reference for one of them is perpendicular to another.
#:
#: So the sign is carried FORWARD instead: each frame is oriented to agree with the previous
#: one, which cannot be ambiguous because the axis moves far less than 90 deg in 5 ms. The
#: constant is used only when there is no previous frame to agree with.
ORIENT_REF = (0.0, 0.0, -1.0)

#: Only the direction out of `backproject_ellipse` is used, so the radius is a free scale.
NOMINAL_RADIUS_MM = 10.0

#: Morphological opening applied to the mask before the ellipse is fitted, in pixels.
#:
#: THE MAST PROJECTS ALONG THE MINOR AXIS. The rotor axis and the mast are the same physical
#: direction, so anything thin left attached to the silhouette stretches the ellipse along
#: exactly the axis the measurement reads -- the minor one. A minor axis that is too long
#: makes `b/a` too large and `acos(b/a)` too small, i.e. it under-reports tilt.
#:
#: Measured on `2026-09-09_194514`, sweeping the kernel:
#:
#:     kernel   theta_A   theta_B   2nd-diff A   A-vs-B world
#:          0     41.85     49.64        1.052         15.15
#:          5     46.15     49.33        0.876         13.61
#:          7     46.46     49.26        0.474         13.38
#:          9     46.41     49.05        0.529         13.61
#:         13     46.37     49.16        0.757         13.44
#:
#: View A was reading 4.5 deg low and its frame-to-frame noise halves. View B barely moves,
#: so only A had the contamination. The PLATEAU from 5 to 13 is what says this removes a
#: distinct feature about 4-5 px wide rather than eroding the disc: an isotropic erosion of
#: the disc itself would walk theta monotonically, since (b-2r)/(a-2r) != b/a.
#:
#: `disc_pose.segment_disc` already opens at OPEN_PX=11 to delete the mast, then calls
#: `_recover_blades` to give thinned blade tips back -- and that recovery is what returns the
#: thin structure this removes again. The opening belongs here rather than in the segmenter
#: so `disc_pose.py` stays the upstream file it was cherry-picked as.
OPEN_PX = 7

AXIS_COLS = ["frame", "t", "nx", "ny", "nz", "agree_deg"]
#: The minor-axis variant carries the triangulated centre and the unforeshortened radius as
#: well, because that construction naturally produces them: the MAJOR axis is the disc's true
#: diameter (it is the one direction that is not foreshortened) and the ellipse centre
#: back-projects to a 3-D point. `axis_b3.csv` in `results/rim/` is the same idea.
MINOR_COLS = ["frame", "t", "nx", "ny", "nz", "vs_conic_deg", "x_mm", "y_mm", "z_mm",
              "radius_mm"]
VIEW_COLS = ["frame", "t", "theta_deg", "area_px", "cx", "cy", "d1", "d2", "ang_deg"]


def _clean_contour(seg, open_px=OPEN_PX):
    """The silhouette outline with thin features stripped. See `OPEN_PX`.

    Falls back to the segmenter's own contour if the mask is unusable or the opening leaves
    nothing -- a degraded measurement beats no measurement, and `view_row` still guards the
    degenerate cases.
    """

    m = getattr(seg, "mask", None)
    if m is None or open_px <= 0:
        return np.asarray(seg.contour, dtype=np.float64).reshape(-1, 1, 2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    opened = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k)
    cs, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.asarray(seg.contour, dtype=np.float64).reshape(-1, 1, 2)
    biggest = max(cs, key=cv2.contourArea)
    if len(biggest) < 5:
        return np.asarray(seg.contour, dtype=np.float64).reshape(-1, 1, 2)
    # Convex hull: the outline of a 4-blade rotor is only a disc at its extremities, and the
    # hull is what makes the fit invariant to where the blades happen to be this frame.
    return cv2.convexHull(biggest.astype(np.float32)).astype(np.float64)


def view_row(seg, cam, undistort=True):
    """One camera's ellipse measurements, plus both backprojected axis branches.

    Returns ``(row_without_frame_or_t, [world_normal, ...])``, or ``None`` if the ellipse is
    degenerate. ``cx``/``cy`` are relative to the principal point, which is the convention
    the existing takes use (view A sits near -54, view B near +136).
    """

    pts = _clean_contour(seg)
    if pts is None:
        return None
    if undistort:
        # Costs nothing and removes a real bias: view B's k1/k2/k3 are -0.338/0.652/-5.38,
        # worth -0.16 deg of tilt at the radius its disc actually sits at. Mapping back
        # through K leaves an ideal-pinhole ellipse, which is what the conic wants.
        pts = cv2.undistortPoints(pts.astype(np.float32), cam.K, cam.dist, P=cam.K)
        pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 5:
        return None
    (ex, ey), (major, minor), ang = conic.normalise_ellipse(
        cv2.fitEllipse(pts.astype(np.float32)))
    if not (minor > 0.0 and major > 0.0):
        return None

    row = [np.degrees(np.arccos(np.clip(minor / major, -1.0, 1.0))),
           float(seg.area_px), ex - cam.K[0, 2], ey - cam.K[1, 2], minor, major, ang]

    poses = conic.backproject_ellipse(((ex, ey), (major, minor), ang), cam.K,
                                      NOMINAL_RADIUS_MM)
    normals = [cam.to_world(p.center, p.normal)[1] for p in poses]
    return row, normals


def axis_from_minor(rows, cams, prior=None):
    """Rotor axis from the minor axis treated as the projected normal. DIRECTIONS ONLY.

    The rotor axis and the mast are the same physical direction, and that direction projects
    into the image along the ellipse's MINOR axis. An image line plus the lens centre span a
    plane containing the true axis, so two views give two planes and the axis is their
    intersection -- `disc_pose.mast_direction` already does exactly this for the mast.

    The point of it: **no length enters the orientation.** The conic backprojection uses all
    five ellipse parameters and therefore depends on the minor axis LENGTH, which is the one
    number a thin feature left in the silhouette inflates (`OPEN_PX`). This construction
    cannot be biased that way, because it reads only the minor axis's DIRECTION.

    The lengths still have jobs, just not this one: the major axis is the disc's true
    diameter (the unforeshortened direction) and so gives its size, and the ellipse centre
    back-projects to a 3-D point. Both are returned.
    """

    lines, centres = [], []
    for (row, _), cam in zip(rows, cams):
        _, _, cx, cy, _, major, ang = row
        ex, ey = cx + cam.K[0, 2], cy + cam.K[1, 2]
        a = np.radians(ang + 90.0)                       # minor axis: major angle + 90
        lines.append(((ex, ey), (np.cos(a), np.sin(a))))
        centres.append((ex, ey))
    axis = disc_pose.mast_direction(lines, cams, up=ORIENT_REF if prior is None else prior)
    if axis is None:
        return None
    centre = disc_pose.triangulate_centre(centres, cams)
    xyz = None if centre is None else np.asarray(centre[0] if isinstance(centre, tuple)
                                                 else centre, float).reshape(3)
    # Radius from the major axis, which is not foreshortened, at the triangulated depth.
    radius = float("nan")
    if xyz is not None:
        cam0 = cams[0]
        depth = float(np.linalg.norm(xyz - cam0.T_world_cam[:3, 3]))
        radius = 0.5 * rows[0][0][5] * depth / cam0.K[0, 0]
    return axis, xyz, radius


#: Weight on the temporal prior when choosing branches. The two views' agreement is the
#: primary evidence; the previous frame breaks the ties it cannot.
PRIOR_W = 0.6
#: A prior older than this (seconds) is not evidence about this frame any more.
PRIOR_MAX_AGE_S = 0.25


def fuse(normals_a, normals_b, prior=None):
    """Pick the branch from each view that the other view agrees with.

    A single ellipse cannot tell a disc leaning toward the camera from one leaning away --
    `backproject_ellipse` returns both, always. Two views resolve it: of the (usually four)
    combinations, the physical one is the pair that agrees. Compared as LINES, because the
    normal's sign is a separate ambiguity that no ellipse carries.

    Returns ``(axis_world, agree_deg)``.
    """

    best = None
    for na in normals_a:
        for nb in normals_b:
            d = stereo.line_angle_deg(na, nb)
            score = d
            if prior is not None:
                # Agreement alone is ambiguous when the two cone branches are close, which is
                # what happens as the disc turns face-on: both combinations agree about
                # equally well and noise picks between them frame to frame. Measured on the
                # 60-70 Hz chunks: line jumps of 30-70 deg at ~1% of frames, which propagate
                # into rotations of 25 deg with a sample SD of 20. The previous frame is the
                # tie-breaker -- the axis cannot move 30 deg in 5 ms.
                mid = na + (nb if float(na @ nb) >= 0 else -nb)
                n = np.linalg.norm(mid)
                if n > 1e-12:
                    score += PRIOR_W * stereo.line_angle_deg(mid / n, prior)
            if best is None or score < best[0]:
                best = (score, d, na, nb)
    _score, d, na, nb = best
    nb = nb if float(na @ nb) >= 0.0 else -nb        # sign-align before averaging
    axis = na + nb
    n = np.linalg.norm(axis)
    axis = na if n < 1e-12 else axis / n
    # Sign it against the PREVIOUS frame, not against a fixed vector -- see ORIENT_REF.
    # `prior` is the last accepted axis and `solve` already refuses one older than
    # PRIOR_MAX_AGE_S, so a stale prior cannot pin the sign of a frame it knows nothing about.
    return stereo.orient(axis, ORIENT_REF if prior is None else prior), d


def solve(take_dir, out_dir=None, rig_path=None, progress=True, stride=1):
    """Segment, fit and fuse a recording. Returns the output directory.

    ``stride`` solves every Nth frame. **KEEP IT AT 1 FOR ALIGNMENT-RATE TAKES.** The rule is
    Nyquist against the ROTOR, not against the transient:

        solved rate must exceed 2 * drive_hz

    because the once-per-rev wobble sits AT the drive frequency and is large -- measured 18.2
    deg in azimuth at 40 Hz, against a ~45 deg swing -- so `alignment_rate.rev_window` has to
    null it with a boxcar of a whole number of revolutions. That null only exists if the
    revolution is resolved. Below Nyquist the wobble folds to |f - n*fs| and the boxcar
    removes a frequency the data no longer contains.

    The cameras deliver 210 fps each and the tracker fires on either view, so a full solve
    runs ~230 Hz: 5.8 samples per revolution at 40 Hz, and Nyquist 115 Hz covers the whole
    sweep. Stride 4 cuts that to 51 Hz -- 1.3 samples per revolution -- and folds 40 Hz onto
    11.1 Hz, whose 92 ms period is comparable to the ramp being measured.

    This docstring previously said the measurement is "a ~3 s transient, which 210 Hz
    oversamples by a factor of hundreds", and that "a strided take and a full one give the
    same answer". Both were wrong and together they cost a campaign. The ramp is ~150 ms, not
    3 s; and on take 2026-09-10_012419 at 40 Hz, stride 1 against stride 4 moves trace
    roughness 9.30 -> 0.96 deg and the fitted rate 1262 -> 597 deg/s. Stride is safe only for
    something that reads the trace at frequencies far below the rotor -- the `frame` and `t`
    columns do still carry true indices and stamps, which is what made the error invisible.
    """

    take_dir = record.latest_flight(Path(take_dir))
    out = Path(out_dir or take_dir)
    out.mkdir(parents=True, exist_ok=True)
    rig = rigmod.StereoRig.load(rig_path) if rig_path else rigmod.StereoRig.load()
    caps, stamps = record.open_recording(take_dir)
    if len(caps) != 2:
        raise SystemExit(f"{take_dir}: need two videos, found {len(caps)}")

    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Scale from the rig's declared `image_size`, NOT from the principal point.
    #
    # The first version inferred the calibrated width as 2 * pp_x, which assumes the
    # principal point sits at the image centre. It does not: the 2026-09-09 rig has camera B
    # at pp_x = 269.1 in a 640-wide frame, 51 px left of centre, so the heuristic decided the
    # calibration was for a 538 px image and rescaled B's intrinsics by 1.19 -- silently
    # corrupting every solve that used that rig.
    calib = json.loads(Path(rig_path or rigmod.DEFAULT_PATH).read_text()).get("image_size")
    scale = 1.0
    if calib and len(calib) == 2 and calib[0]:
        if abs(w / calib[0] - h / calib[1]) > 1e-3:
            raise SystemExit(f"rig calibrated at {calib[0]}x{calib[1]}, video is {w}x{h} -- "
                             f"not a uniform rescale, so fx and fy would need different "
                             f"factors. Re-calibrate at the mode you fly.")
        scale = w / float(calib[0])
    cams = [c if abs(scale - 1.0) < 1e-6 else c.scaled(scale) for c in rig.cameras]
    if abs(scale - 1.0) > 1e-6:
        print(f"rig calibrated at {calib[0]}x{calib[1]}, video {w}x{h} -- scaled {scale:.4f}x")

    # Written under `.partial` names and renamed only after the whole take is solved. Every
    # reader (`alignment_rate.campaign`, `rate_scatter`, `sweep_report`, `settle_rows`)
    # decides "solved" by `axis.csv` EXISTING, and the `finally` below closes the files even
    # on an exception -- so writing the final names directly let a solve in progress, or one
    # that crashed, pass as finished. On 2026-09-11 `campaign` read an in-flight take's
    # still-empty axis.csv and died on `rows[0]`; a half-written one would have been worse,
    # silently analysed as a truncated take.
    names = ["axis_minor.csv", "axis.csv", "tilt_A.csv", "tilt_B.csv"]
    part = {n: out / (n + ".partial") for n in names}
    fm = open(part["axis_minor.csv"], "w", newline="")
    wm = csv.writer(fm)
    wm.writerow(MINOR_COLS)
    fa = open(part["axis.csv"], "w", newline="")
    fv = [open(part[f"tilt_{t}.csv"], "w", newline="") for t in "AB"]
    wa, wv = csv.writer(fa), [csv.writer(f) for f in fv]
    wa.writerow(AXIS_COLS)
    for x in wv:
        x.writerow(VIEW_COLS)

    n_frame = n_axis = 0
    prior, prior_t = None, None
    try:
        while True:
            grabbed = [c.read() for c in caps]
            if not all(ok for ok, _ in grabbed):
                break
            i = n_frame
            n_frame += 1
            if stride > 1 and (i % stride):
                continue
            t = stamps[i] if stamps is not None and i < len(stamps) else (i, i)

            rows, norms = [], []
            for k, (_, frame) in enumerate(grabbed):
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                seg = disc_pose.segment_disc(gray)
                got = view_row(seg, cams[k]) if seg is not None else None
                rows.append(None if got is None else got[0])
                norms.append(None if got is None else got[1])
                if got is not None:
                    wv[k].writerow([i, f"{t[k]:.6f}"]
                                   + [f"{v:.5g}" for v in got[0]])

            if norms[0] is not None and norms[1] is not None:
                use_prior = (prior is not None and prior_t is not None
                             and 0 <= (t[0] - prior_t) <= PRIOR_MAX_AGE_S)
                axis, agree = fuse(norms[0], norms[1], prior if use_prior else None)
                prior, prior_t = axis, t[0]
                wa.writerow([i, f"{t[0]:.6f}"] + [f"{v:.6f}" for v in axis]
                            + [f"{agree:.3f}"])
                n_axis += 1
                # Signed against THIS frame's conic axis, not against its own previous
                # frame: the two are the same physical direction a few degrees apart, so
                # `axis` is the tighter prior, and it keeps the two CSVs in one hemisphere
                # where the whole point of axis_minor.csv is comparing them.
                got = axis_from_minor(list(zip(rows, norms)), cams, prior=axis)
                if got is not None:
                    m_ax, xyz, radius = got
                    wm.writerow([i, f"{t[0]:.6f}"] + [f"{v:.6f}" for v in m_ax]
                                + [f"{stereo.line_angle_deg(axis, m_ax):.3f}"]
                                + ([f"{v:.3f}" for v in xyz] if xyz is not None
                                   else ["", "", ""])
                                + [f"{radius:.3f}"])
            if progress and n_frame % 500 == 0:
                print(f"  {n_frame} frames, {n_axis} solved", flush=True)
    finally:
        for c in caps:
            c.release()
        fa.close()
        fm.close()
        for f in fv:
            f.close()
    # Reached only when the loop finished: an exception skips this and leaves `.partial`.
    # `axis.csv` last, since it is the name every reader tests.
    for n in [x for x in names if x != "axis.csv"] + ["axis.csv"]:
        part[n].replace(out / n)

    print(f"{take_dir.name}: {n_axis}/{n_frame} frames solved -> {out}")
    return out


# --------------------------------------------------------------------------------------


def _synthetic_rig(sep_deg=82.0, baseline_mm=184.0):
    """Two cameras straddling the origin, close to the real rig's geometry."""

    K = np.array([[1375.0, 0.0, 320.0], [0.0, 1375.0, 200.0], [0.0, 0.0, 1.0]])
    r = baseline_mm / (2.0 * np.sin(np.radians(sep_deg) / 2.0))
    cams = []
    for name, az in (("A", -sep_deg / 2.0), ("B", sep_deg / 2.0)):
        eye = r * np.array([np.sin(np.radians(az)), 0.0, -np.cos(np.radians(az))])
        cams.append(rigmod.Camera(K=K, dist=np.zeros(5), name=name,
                                  T_world_cam=rigmod.look_at(eye, (0, 0, 0), (0, 1, 0))))
    return rigmod.StereoRig(cameras=tuple(cams))


def _render(cam, normal, radius=12.0, size=(400, 640)):
    """A filled disc of the given world normal, as this camera sees it."""

    n = np.asarray(normal, float) / np.linalg.norm(normal)
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    e1 = np.cross(n, [0.0, 1.0, 0.0]) if np.linalg.norm(e1) < 1e-6 else e1
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    a = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    P = radius * (np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2)
    pts, _ = cv2.projectPoints((P - cam.T_world_cam[:3, 3]) @ cam.R,
                               np.zeros(3), np.zeros(3), cam.K, np.zeros(5))
    img = np.zeros(size, np.uint8)
    cv2.fillPoly(img, [np.round(pts).astype(np.int32).reshape(-1, 2)], 255)
    return cv2.GaussianBlur(img, (5, 5), 1.0)


def _self_check():
    rig = _synthetic_rig()

    # 1. the legacy channel is the formula the shipped takes used. These three numbers are
    #    row 0 of results/rim/2026-09-08_205108/tilt_A.csv, copied verbatim: the shipped
    #    producer's own minor, major and the theta it wrote for them.
    d1, d2, theta_shipped = 54.525, 133.353, 65.866
    assert abs(np.degrees(np.arccos(d1 / d2)) - theta_shipped) < 0.001, \
        "acos(minor/major) is no longer what the shipped takes were computed with"

    # 2. the opening must strip a thin spur without moving the disc it is attached to
    disc = np.zeros((400, 640), np.uint8)
    cv2.ellipse(disc, ((320, 200), (240, 90), 30.0), 255, -1)
    spur = disc.copy()
    cv2.line(spur, (320, 200), (320, 40), 255, 3)          # a 3 px mast
    a_clean, b_clean = cv2.fitEllipse(_clean_contour(
        type("S", (), {"mask": disc, "contour": None})())[:, 0, :].astype(np.float32))[1]
    a_spur, b_spur = cv2.fitEllipse(_clean_contour(
        type("S", (), {"mask": spur, "contour": None})())[:, 0, :].astype(np.float32))[1]
    assert abs(min(a_spur, b_spur) - min(a_clean, b_clean)) < 4.0, \
        f"the spur survived the opening: minor {min(a_clean, b_clean):.1f} -> {min(a_spur, b_spur):.1f}"

    # 3. a known axis comes back, through segmentation and both backprojections
    for truth in ((0.0, 0.0, 1.0), (0.20, 0.0, 0.98), (0.0, -0.25, 0.97)):
        t = np.asarray(truth, float)
        t /= np.linalg.norm(t)
        got = []
        for cam in rig.cameras:
            seg = disc_pose.segment_disc(_render(cam, t))
            assert seg is not None, f"nothing segmented for {cam.name} at {truth}"
            r = view_row(seg, cam)
            assert r is not None, f"degenerate ellipse for {cam.name} at {truth}"
            got.append(r[1])
        axis, agree = fuse(*got)
        err = stereo.line_angle_deg(axis, t)
        assert err < 2.0, f"axis off by {err:.2f} deg at {truth}"
        assert agree < 5.0, f"views disagree by {agree:.2f} deg on synthetic data"

    # 4. branch resolution actually chooses, rather than taking the first
    a = [np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.0, 0.8])]
    b = [np.array([0.61, 0.0, 0.79]), np.array([0.0, 1.0, 0.0])]
    axis, agree = fuse(a, b)
    assert agree < 2.0, agree
    assert abs(float(axis @ stereo.orient(np.array([0.6, 0.0, 0.8]), ORIENT_REF))) > 0.99

    # 5. an axis sweeping THROUGH the plane perpendicular to ORIENT_REF must not flip.
    #    This is the failure the prior exists to remove: with a fixed reference the sign is
    #    decided by a dot product that passes through zero, and the vector reverses.
    ref = np.asarray(ORIENT_REF, float)
    perp = stereo._unit(np.cross(ref, [1.0, 0.0, 0.0]))
    prior = None
    flips = 0
    for deg in np.arange(-20.0, 20.5, 0.5):          # crosses perpendicular at deg = 0
        a = np.radians(deg)
        truth = stereo._unit(np.cos(a) * perp + np.sin(a) * ref)
        got = stereo.orient(truth * (1 if deg % 2 else -1),   # arbitrary incoming sign
                            ORIENT_REF if prior is None else prior)
        if prior is not None and float(got @ prior) < 0.0:
            flips += 1
        prior = got
    assert flips == 0, f"{flips} sign flips sweeping through the ORIENT_REF singularity"

    print("disc_axis: self-check passed (legacy formula, thin-feature opening, axis "
          "recovery, branch pick, sign continuity)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?", help="a recording directory (results/flights/<take>)")
    ap.add_argument("--out", default=None, help="where the CSVs go (default: the take)")
    ap.add_argument("--rig", default=None)
    ap.add_argument("--stride", type=int, default=1, help="solve every Nth frame")
    a = ap.parse_args()
    if a.take is None:
        _self_check()
        sys.exit()
    solve(a.take, out_dir=a.out, rig_path=a.rig, stride=a.stride)
