#!/usr/bin/env python3
"""Overlay what the segmenter saw on the footage it saw it in.

    uv run python controller/pose/disc_video.py                       # self-check
    uv run python controller/pose/disc_video.py results/flights/<take>
    uv run python controller/pose/disc_video.py <take> --stride 3 --max-frames 900

Writes `<take>/overlay.mp4`: the two views side by side, the `disc_pose.segment_disc` mask
tinted, the fitted ellipse drawn, its minor axis marked (that is the projected rotor axis),
and a per-frame readout of the numbers `disc_axis` derives from it.

WHY THIS EXISTS
---------------
Every number in `tilt_A.csv` comes from one ellipse per view per frame, and every failure
mode of this pipeline is a failure of that ellipse -- a mask that swallowed the mast, a blade
tip lost to the opening, a silhouette that changed shape as the rotor turned. None of that is
visible in the CSV, where a wrong ellipse and a right one are both six finite numbers.

It is also the fastest way to answer a question the schedule cannot: `tilt_sweep` reports
`outcome=ok` when the SCHEDULE completed, which says nothing about whether the ROTOR ever
turned. A stationary rotor and a spinning one produce identical logs and very different
films.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from controller.calib import rig as rigmod
from controller.camera import record
from controller.pose import disc_axis, disc_pose

MASK_TINT = (0, 90, 0)          #: BGR added inside the mask
ELLIPSE_BGR = (60, 200, 255)    #: the fitted ellipse
MINOR_BGR = (255, 140, 40)      #: its minor axis == the projected rotor axis
HULL_BGR = (200, 80, 200)       #: the points the ellipse is actually fitted to
MAJOR_BGR = (60, 200, 255)      #: the fitted major axis -- the disc's true diameter
AXIS3D_BGR = (40, 220, 40)      #: the reconstructed 3-D rotor axis, reprojected
CONIC_BGR = (208, 111, 47)      #: axis from conic backprojection (blue)
PLANE_BGR = (26, 143, 26)       #: axis from minor-axis plane intersection (green)
TEXT_BGR = (240, 240, 240)
AXIS_PX = 90                    #: drawn length of a projected axis


def _project_dir(axis_world, cam):
    """A world direction as a unit image-plane direction, weak perspective."""

    d = cam.R.T @ np.asarray(axis_world, float)
    v = np.array([d[0], d[1]], float)
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def _draw(gray, seg, cam, label, axes=()):
    """One view, annotated. Returns a BGR frame.

    ``axes`` is ``[(world_axis, bgr, name), ...]`` -- each is projected onto the image plane
    and drawn from the ellipse centre, so the two reconstructions can be compared against the
    silhouette they both came from.
    """

    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    txt = [label]
    if seg is not None:
        # Tint the OPENED mask, not the raw one: that is the outline the ellipse is actually
        # fitted to (`disc_axis.OPEN_PX`), and an overlay that shows a different mask from
        # the one being measured is a diagnostic that lies.
        m = None
        if seg.mask is not None:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (disc_axis.OPEN_PX, disc_axis.OPEN_PX))
            m = cv2.morphologyEx(seg.mask.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
        if m is not None and m.shape == gray.shape:
            img[m] = np.clip(img[m].astype(np.int16) + MASK_TINT, 0, 255).astype(np.uint8)
        # Draw the points the fit actually sees. The ellipse is a direct algebraic fit to
        # the mask's CONVEX HULL -- the pose is derived from it, not the other way round --
        # and a hull BRIDGES from the disc to anything protruding, notably the mounting
        # bearing sitting on the rotor axis. When that happens the hull runs straight out to
        # the bearing along the minor axis and the fit follows. The mask is a single
        # connected component, so no opening detaches it (7 px is already the jitter optimum;
        # 25 px is 65% worse), which makes seeing it the next best thing to fixing it.
        pts = disc_axis._clean_contour(seg)
        if pts is not None and len(pts) >= 3:
            cv2.polylines(img, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], True,
                          HULL_BGR, 1)
        got = disc_axis.view_row(seg, cam)
        if got is not None:
            theta, area, cx, cy, d1, d2, ang = got[0]
            # back to absolute pixels: view_row reports the centre against the principal
            # point, which is what the CSV stores and not where to draw
            ex, ey = cx + cam.K[0, 2], cy + cam.K[1, 2]
            # The two axes, not the ellipse outline. They are what the measurement reads:
            # the MAJOR axis is the disc's true diameter (the one direction that is not
            # foreshortened) and the MINOR is the rotor axis projected into the image, whose
            # ratio to the major gives theta. Drawing the outline shows a curve that is
            # mostly interpolation between them.
            for length, angle_deg, bgr in ((d2, ang, MAJOR_BGR),
                                           (d1, ang + 90.0, MINOR_BGR)):
                a = np.radians(angle_deg)
                dx, dy = 0.5 * length * np.cos(a), 0.5 * length * np.sin(a)
                cv2.line(img, (int(ex - dx), int(ey - dy)), (int(ex + dx), int(ey + dy)),
                         bgr, 2, cv2.LINE_AA)
            cv2.circle(img, (int(ex), int(ey)), 3, (255, 255, 255), -1)
            txt.append(f"major {d2:5.1f}  minor {d1:5.1f}")
            txt.append(f"area {area:6.0f}  ang {ang:5.1f}")
            for ax_w, bgr, name in axes:
                u = _project_dir(ax_w, cam)
                if u is None:
                    continue
                q1 = (int(ex + AXIS_PX * u[0]), int(ey + AXIS_PX * u[1]))
                cv2.arrowedLine(img, (int(ex), int(ey)), q1, bgr, 3, cv2.LINE_AA,
                                tipLength=0.18)
                txt.append(name)
        else:
            txt.append("ellipse degenerate")
    else:
        txt.append("NO SEGMENT")
    for i, t in enumerate(txt):
        cv2.putText(img, t, (6, 16 + 15 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_BGR, 1,
                    cv2.LINE_AA)
    return img


def _both_axes(segs, rig):
    """The two reconstructions of the rotor axis, for side-by-side drawing.

    * **conic** -- `conic.backproject_ellipse` in each view, branches matched across views.
      Uses all five ellipse parameters, so it depends on the minor axis LENGTH.
    * **planes** -- the minor axis treated as the projected normal: its image line and the
      lens centre span a plane containing the true axis, and two such planes intersect in it.
      Uses only DIRECTIONS, so an over-long minor axis cannot bias it.
    """

    from controller.pose import disc_axis as da

    if any(s is None for s in segs) or len(segs) != 2:
        return []
    rows, lines = [], []
    for k, seg in enumerate(segs):
        r = da.view_row(seg, rig.cameras[k])
        if r is None:
            return []
        rows.append(r)
        _, _, cx, cy, _, _, ang = r[0]
        a = np.radians(ang + 90.0)
        lines.append(((cx + rig.cameras[k].K[0, 2], cy + rig.cameras[k].K[1, 2]),
                      (np.cos(a), np.sin(a))))
    out = []
    ca, agree = da.fuse(rows[0][1], rows[1][1])
    out.append((ca, CONIC_BGR, f"conic (A-vs-B {agree:.1f})"))
    pa = disc_pose.mast_direction(lines, rig.cameras, up=da.ORIENT_REF)
    if pa is not None:
        from controller.pose import stereo
        out.append((pa, PLANE_BGR, f"planes (vs conic {stereo.line_angle_deg(ca, pa):.1f})"))
    return out


def render(take_dir, out_path=None, stride=3, max_frames=None, fps=30.0, rig_path=None,
           compare=True):
    """Write the annotated side-by-side film. Returns its path."""

    take = record.latest_flight(Path(take_dir))
    out = Path(out_path or (take / "overlay.mp4"))
    rig = rigmod.StereoRig.load(rig_path) if rig_path else rigmod.StereoRig.load()
    caps, stamps = record.open_recording(take)
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))

    i = n = 0
    try:
        while max_frames is None or n < max_frames:
            grabbed = [c.read() for c in caps]
            if not all(ok for ok, _ in grabbed):
                break
            if i % stride == 0:
                grays = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                         for _, f in grabbed]
                segs = [disc_pose.segment_disc(g) for g in grays]
                axes = _both_axes(segs, rig) if compare else []
                panes = []
                for k, g in enumerate(grays):
                    t = stamps[i][k] - stamps[0][0] if stamps is not None and i < len(stamps) \
                        else i / 200.0
                    panes.append(_draw(g, segs[k], rig.cameras[k],
                                       f"{'AB'[k]}  t={t:6.2f}s", axes))
                vw.write(np.hstack(panes))
                n += 1
            i += 1
    finally:
        for c in caps:
            c.release()
        vw.release()
    print(f"{take.name}: {n} frames (every {stride}) -> {out}")
    return out


def _angle_panel(t_rel, radial, azimuth, width, height=320, dpi=100):
    """Render the angle traces ONCE as an image, plus a time -> pixel-x mapping.

    Drawing a matplotlib figure per output frame costs ~50 ms and would dominate the render.
    The traces do not change, so the panel is rasterised once and each frame only gets a
    cursor drawn onto a copy of it with `cv2` -- microseconds instead of milliseconds.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(width / dpi, height / dpi), dpi=dpi,
                                 sharex=True, facecolor="white",
                                 gridspec_kw={"hspace": 0.08})
    a1.plot(t_rel, radial, color="#2f6fd0", lw=1.4)
    a1.set_ylabel("radial (deg)\nfrom rest", fontsize=7.5, color="#2f6fd0")
    a2.plot(t_rel, azimuth, color="#e0721a", lw=1.4)
    if np.isnan(azimuth).any():
        # Show WHERE it is undefined rather than leaving an unexplained gap.
        band = np.isnan(azimuth)
        lo, hi = np.nanmin(azimuth), np.nanmax(azimuth)
        pad = 0.1 * max(hi - lo, 1.0)
        a2.fill_between(t_rel, lo - pad, hi + pad, where=band, color="#eef1f4", lw=0)
        mid = 0.5 * (lo + hi)
        if band.any():
            a2.text(float(np.mean(t_rel[band])), mid, "lean < %.0f deg:\ndirection undefined"
                    % AZIMUTH_MIN_RADIAL_DEG, fontsize=6.5, color="#9aa0a6",
                    ha="center", va="center")
    a2.set_ylabel("lean azimuth (deg)", fontsize=7.5, color="#e0721a")
    a2.set_xlabel("time from the coil cut (s)", fontsize=8, color="#6b7280")
    for a in (a1, a2):
        a.axvline(0.0, color="#1a8f6a", lw=1.3, ls="--")
        a.grid(True, color="#dfe3e8", lw=0.6)
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#dfe3e8")
        a.tick_params(labelsize=7, colors="#6b7280")
        a.set_xlim(t_rel[0], t_rel[-1])
    a1.annotate("coils A,C cut", (0.0, a1.get_ylim()[1]), xytext=(4, -6),
                textcoords="offset points", fontsize=7.5, color="#1a8f6a", va="top")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.16)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3][..., ::-1].copy()   # RGBA -> BGR

    # data x -> pixel x, taken from the axes transform so it cannot drift from the drawing
    x0, x1 = a2.transData.transform([(t_rel[0], 0), (t_rel[-1], 0)])[:, 0]
    tops = [int(round(fig.bbox.height - a.bbox.y1)) for a in (a1, a2)]
    bots = [int(round(fig.bbox.height - a.bbox.y0)) for a in (a1, a2)]
    plt.close(fig)

    def x_of(t):
        if t_rel[-1] == t_rel[0]:
            return int(x0)
        return int(round(x0 + (x1 - x0) * (t - t_rel[0]) / (t_rel[-1] - t_rel[0])))

    return img, x_of, list(zip(tops, bots))


#: Window drawn around the cut, in seconds. The interesting motion is the first second or
#: two; a whole take is mostly ramp and rest.
WINDOW_S = (-2.0, 6.0)
#: Below this much lean the azimuth has no meaning -- the direction of a ~0 deg tilt is
#: undefined and atan2 just tracks noise, winding through hundreds of degrees. Blanked rather
#: than drawn, because a plotted line implies a measurement.
AZIMUTH_MIN_RADIAL_DEG = 1.0


def render_with_angles(take_dir, log_path=None, out_path=None, stride=4, fps=30.0,
                       rig_path=None, max_frames=None, spin_hz=None):
    """Segmentation overlay over a live trace of the radial and azimuthal angles.

    The two camera views with their masks and fitted ellipses sit above a pair of angle
    traces with a cursor at the current instant, so a wobble in the plot can be checked
    against the silhouette that produced it in the same glance. Requires the take to be
    solved (`disc_axis`), because the angles come from `axis.csv`.
    """

    from controller.control import alignment_rate as ar

    take = record.latest_flight(Path(take_dir))
    out = Path(out_path or (take / "overlay_angles.mp4"))
    rig = rigmod.StereoRig.load(rig_path) if rig_path else rigmod.StereoRig.load()

    t, axis, _q = ar.load(take)
    tk = None
    if log_path and Path(log_path).exists():
        pts = ar.timeline(log_path)
        kills = [k for p in pts for k in p["kills"]]
        tk = kills[0] if kills else None
        if spin_hz is None and pts:
            spin_hz = pts[0]["freq"]
    if tk is None:                                   # no log: find the step in the data
        up = ar._hemisphere(axis, axis[0]).mean(0)
        up /= np.linalg.norm(up)
        found = ar.find_kills(t, ar.tilt_from(axis, up))
        tk = found[0] if found else float(t[len(t) // 3])
    # Reference: the COILS-OFF REST ATTITUDE, not the pre-cut mean axis.
    #
    # Using the pre-cut mean as the reference is circular -- it makes the pre-cut lean zero
    # by construction, so its azimuth is the direction of a ~0 deg vector and is pure noise.
    # The rest attitude is an independent physical zero (the robot hanging with nothing
    # driving it), so the initial pose has a real tilt with a definite azimuth and BOTH
    # angles are defined for the whole record. Same datum `alignment_rate` uses, and the
    # reason the schedule keeps a coils-off tail.
    pre = (t >= tk - 1.0) & (t < tk)
    ref = None
    if log_path and Path(log_path).exists():
        try:
            ref = ar.datum(t, axis, ar.timeline(log_path))[0]
        except SystemExit:
            ref = None
    if ref is None:
        seed = axis[pre][0] if pre.sum() else axis[0]
        ref = ar._hemisphere(axis[pre] if pre.sum() else axis, seed).mean(0)
    ref = np.asarray(ref, float) / np.linalg.norm(ref)
    # Azimuth from the SMOOTHED lean vector, not from the instantaneous axis.
    #
    # `ar.angles` unwraps atan2 of the raw axis, and the raw axis has been flipped into one
    # hemisphere (a rotor axis is a line, so its sign is arbitrary). Every hemisphere crossing
    # is then a 180 deg jump, `unwrap` turns each into a real step, and they accumulate: the
    # 40 Hz take displayed 25000 deg of "precession" that was entirely this artefact.
    #
    # Smoothing the two tangent-plane components on a whole number of revolutions nulls the
    # 1x cone (`ar.rev_window`), leaving the MEAN lean vector. Its angle is the direction the
    # robot leaned -- bounded, single-valued, and the quantity worth watching beside the
    # radial step. `pose/theory.md` 20.7 argues the tangent plane is the right coordinate
    # here for the same reason.
    e1 = np.cross(ref, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ref, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ref, e1)
    # Sign resolved by continuity from the rest attitude, not folded into a hemisphere:
    # the drone never inverts, so the sign is physical and angles past 90 deg are real.
    aligned = ar.orient_continuous(axis, ref)
    l1, l2 = np.degrees(aligned @ e1), np.degrees(aligned @ e2)
    radial = np.degrees(np.arccos(np.clip(aligned @ ref, -1.0, 1.0)))
    t_rel = t - tk

    # Trim to the transient. Over a whole take the unwrapped azimuth winds through the
    # coning precession -- one turn per revolution of the rotor, so ~20 turns a second at
    # 20 Hz and -20000 deg over an 18 s take. True, and useless on a plot.
    keep = (t_rel >= WINDOW_S[0]) & (t_rel <= WINDOW_S[1])
    if keep.sum() > 20:
        t_rel, radial, l1, l2 = t_rel[keep], radial[keep], l1[keep], l2[keep]

    # Smooth both on a whole number of rotor revolutions, which nulls the 1x cone exactly
    # (`alignment_rate.rev_window`), then take out the residual precession rate fitted on
    # the pre-cut window. What is left in the azimuth is the change of lean DIRECTION at the
    # cut, which is the thing worth watching next to the radial step.
    span = ar.rev_window(spin_hz)
    radial = ar._smooth(t_rel, radial, span)
    azimuth = np.degrees(np.arctan2(ar._smooth(t_rel, l2, span),
                                    ar._smooth(t_rel, l1, span)))
    # Fit the precession on the POST-cut window, not the pre-cut one. Before the cut the
    # axis sits only ~3 deg off the reference, where azimuth is ill-conditioned and does not
    # wind coherently; after it, the axis is ~12 deg out and the azimuth advances steadily at
    # the coning rate. Fitting the flat pre-cut segment and extrapolating it left 25000 deg of
    # un-removed winding on screen.
    pre_w = t_rel < 0
    fit_w = t_rel > 0.5
    # Mask FIRST, then unwrap only what survives.
    #
    # A wrapped angle draws a spurious vertical line at every +-180 crossing, and unwrapping
    # before masking lets the undefined stretch drag the whole trace away. So: blank the
    # samples where the lean is too small to have a direction, unwrap across the remaining
    # run, and reference it to the settled post-cut heading so 0 means "the way it ended up".
    ok = radial >= AZIMUTH_MIN_RADIAL_DEG
    azimuth = np.where(ok, azimuth, np.nan)
    if ok.sum() > 20:
        azimuth[ok] = np.degrees(np.unwrap(np.radians(azimuth[ok])))
        post = ok & (t_rel > 0.5)
        if post.sum() > 20:
            azimuth = azimuth - np.median(azimuth[post])

    caps, stamps = record.open_recording(take)
    w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    panel, x_of, bands = _angle_panel(t_rel, radial, azimuth, 2 * w)
    ph = panel.shape[0]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h + ph))

    i = n = 0
    try:
        while max_frames is None or n < max_frames:
            grabbed = [c.read() for c in caps]
            if not all(ok for ok, _ in grabbed):
                break
            now_full = ((stamps[i][0] if stamps is not None and i < len(stamps) else t[0])
                        - tk)
            if now_full < WINDOW_S[0] or now_full > WINDOW_S[1]:
                i += 1
                continue                                  # only film the window we plot
            if i % stride == 0:
                grays = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                         for _, f in grabbed]
                segs = [disc_pose.segment_disc(g) for g in grays]
                now = now_full
                # The 3-D axis, reprojected into each view. This is the answer the analysis
                # uses: the minor-axis direction in each image back-projects to a plane
                # containing the true axis, and the two planes intersect in it
                # (`disc_axis.axis_from_minor`). No length and no acos(minor/major) anywhere
                # in it -- which is why it is drawn and the ratio is not called a result.
                j = int(np.clip(np.searchsorted(t, stamps[i][0]
                                                if stamps is not None and i < len(stamps)
                                                else t[0]), 0, len(axis) - 1))
                axes3 = [(axis[j], AXIS3D_BGR, "3D axis (two-plane intersection)")]
                panes = [_draw(g, segs[k], rig.cameras[k], f"{'AB'[k]}  t={now:+6.2f}s",
                               axes3)
                         for k, g in enumerate(grays)]
                strip = panel.copy()
                cx = x_of(now)
                for top, bot in bands:
                    cv2.line(strip, (cx, top), (cx, bot), (60, 60, 60), 1)
                j = int(np.searchsorted(t_rel, now))
                if 0 <= j < len(t_rel):
                    for (top, bot), val, series in zip(bands, (radial[j], azimuth[j]),
                                                       (radial, azimuth)):
                        if not np.isfinite(val):
                            continue
                        lo = float(np.nanmin(series))
                        hi = float(np.nanmax(series))
                        y = bot if hi == lo else int(round(bot - (bot - top)
                                                           * (val - lo) / (hi - lo)))
                        cv2.circle(strip, (cx, int(np.clip(y, top, bot))), 4,
                                   (40, 40, 220), -1)
                vw.write(np.vstack([np.hstack(panes), strip]))
                n += 1
            i += 1
    finally:
        for c in caps:
            c.release()
        vw.release()
    print(f"{take.name}: {n} frames -> {out}")
    return out


def spin_check(take_dir, n=400, stride=2):
    """Is the ROTOR turning? Reports the frame-to-frame change in the silhouette.

    A spinning rotor sweeps its blades past the camera, so consecutive frames differ inside
    the mask even when the body is still. A stationary one gives a near-constant image. This
    separates "the schedule ran" from "the robot did something", which the logs cannot.
    """

    take = record.latest_flight(Path(take_dir))
    caps, _ = record.open_recording(take)
    prev, diffs, areas = None, [], []
    i = 0
    try:
        while len(diffs) < n:
            grabbed = [c.read() for c in caps]
            if not all(ok for ok, _ in grabbed):
                break
            if i % stride == 0:
                f = grabbed[0][1]
                g = f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                seg = disc_pose.segment_disc(g)
                areas.append(float(seg.area_px) if seg is not None else np.nan)
                if prev is not None:
                    diffs.append(float(np.mean(np.abs(g.astype(np.int16) - prev))))
                prev = g.astype(np.int16)
            i += 1
    finally:
        for c in caps:
            c.release()
    d = np.asarray(diffs, float)
    a = np.asarray(areas, float)
    return {"n": len(d), "mean_abs_diff": float(np.nanmean(d)),
            "p95_abs_diff": float(np.nanpercentile(d, 95)),
            "area_mean": float(np.nanmean(a)), "area_cv": float(np.nanstd(a) / np.nanmean(a)),
            "segment_fail_frac": float(np.mean(~np.isfinite(a)))}


def _self_check():
    """A synthetic disc must draw, and a moving scene must read as moving."""

    K = np.array([[1375.0, 0.0, 320.0], [0.0, 1375.0, 200.0], [0.0, 0.0, 1.0]])
    cam = rigmod.Camera(K=K, dist=np.zeros(5), name="A")
    img = np.zeros((400, 640), np.uint8)
    # 200, not 255: a saturated disc clips the tint away and the check below silently
    # passes on a frame where nothing was drawn.
    cv2.ellipse(img, ((320, 200), (240, 90), 30.0), 200, -1)
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    seg = disc_pose.segment_disc(img)
    assert seg is not None, "synthetic disc did not segment"
    out = _draw(img, seg, cam, "A t=0")
    assert out.shape == (400, 640, 3), out.shape
    # the tint has to be measured against the untinted conversion, over the mask only --
    # the ellipse and axis are drawn in other hues and swamp a whole-frame channel mean
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    m = seg.mask.astype(bool)
    lift = (out[m][:, 1].astype(int) - base[m][:, 1].astype(int)).mean()
    assert lift > 10.0, f"mask tint did not land: green lifted {lift:.1f}"

    # a frame that never changes must read as not moving, and a shifting one as moving
    still = np.mean(np.abs(img.astype(np.int16) - img.astype(np.int16)))
    moved = np.mean(np.abs(np.roll(img, 12, axis=1).astype(np.int16) - img.astype(np.int16)))
    assert still == 0.0 and moved > 1.0, (still, moved)

    # the degenerate path must annotate rather than raise
    assert _draw(np.zeros((400, 640), np.uint8), None, cam, "A").shape == (400, 640, 3)
    print("disc_video: self-check passed (draw, tint, motion metric, no-segment path)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--spin-check", action="store_true", help="is the rotor turning?")
    ap.add_argument("--angles", action="store_true",
                    help="composite with a live radial/azimuth trace")
    ap.add_argument("--log", default=None, help="sweep.log, for the kill instant")
    a = ap.parse_args()
    if a.take is None:
        _self_check()
        sys.exit()
    if a.angles:
        render_with_angles(a.take, log_path=a.log, out_path=a.out, stride=a.stride,
                           max_frames=a.max_frames)
    elif a.spin_check:
        for k, v in spin_check(a.take).items():
            print(f"  {k:20s} {v}")
    else:
        render(a.take, a.out, stride=a.stride, max_frames=a.max_frames)
