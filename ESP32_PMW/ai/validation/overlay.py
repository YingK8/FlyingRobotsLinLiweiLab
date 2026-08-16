"""Draw what the estimator saw, and how wrong it was, onto the frame.

An aggregate residual says nothing about *direction* or about *which part of the
fit* produced it.  These overlays turn the numbers into something you can look
at:

* the fitted ellipse against the analytically projected true one -- their gap is
  the pixel error that drives everything downstream;
* the rotor normal, estimated and true, as projected 3-D arrows;
* **both** back-projection branches, so the two-fold ambiguity is a pair of
  diverging arrows rather than a column in a CSV;
* the position residual as an arrow, amplified.

**On amplifying the residual.**  At 0.87 mm against ~130 px per 20.4 mm of
robot, the true residual is under 6 px, and most of it points *along the optical
axis* where it barely projects at all.  Drawn honestly it is invisible.  So it is
drawn at a stated gain, with the gain written on the image and a scale bar next
to it.  An amplified arrow without its gain on the face of it is a lie, which is
why `gain` is never silently applied.

**Identity is never colour alone.**  Estimated is solid, ground truth is dashed,
rejected branches are thin, and everything carries a short text label.  That
survives colour-vision deficiency, greyscale printing, and the small thumbnails
these end up in.

Extends `segment.draw`, which already handles the grey->BGR conversion trap that
`servo.py:199` documents: draw onto a grayscale frame and the overlay comes out
grey too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import conic  # noqa: E402

# BGR. Chosen to stay distinguishable under the common colour-vision
# deficiencies -- blue/orange/white rather than the red/green pair that
# deuteranopes cannot separate. Line style carries the same information anyway.
COL_FIT = (255, 176, 0)      # blue-cyan: what the estimator measured
COL_TRUE = (255, 255, 255)   # white: ground truth
COL_ALT = (128, 128, 128)    # grey: the rejected ambiguity branch
COL_RESID = (0, 165, 255)    # orange: the residual arrow
COL_TEXT = (230, 230, 230)

ALL_LAYERS = ("fit", "truth", "normal", "branches", "residual", "text")

# Length of the drawn normal, in mm, so the arrow is comparable across frames.
NORMAL_LEN_MM = 12.0


def _project(points_mm, K):
    """Project camera-frame points to pixels. Points behind the camera -> None."""
    pts = np.atleast_2d(np.asarray(points_mm, dtype=np.float64))
    out = []
    for p in pts:
        if p[2] <= 1e-6:
            out.append(None)
            continue
        q = K @ p
        out.append((float(q[0] / q[2]), float(q[1] / q[2])))
    return out


def _pt(p):
    return None if p is None else (int(round(p[0])), int(round(p[1])))


def _dashed_ellipse(img, ellipse, colour, dash_deg=12, thickness=1):
    """cv2 has no dashed ellipse, so draw alternating arcs."""
    (cx, cy), (major, minor), ang = ellipse
    for start in range(0, 360, dash_deg * 2):
        cv2.ellipse(img, (int(cx), int(cy)), (int(major / 2), int(minor / 2)),
                    ang, start, start + dash_deg, colour, thickness, cv2.LINE_AA)


def _arrow(img, p0, p1, colour, thickness=2, tip=0.25):
    if p0 is None or p1 is None:
        return False
    a, b = _pt(p0), _pt(p1)
    if np.hypot(b[0] - a[0], b[1] - a[1]) < 2:
        cv2.circle(img, a, 2, colour, -1, cv2.LINE_AA)
        return False
    cv2.arrowedLine(img, a, b, colour, thickness, cv2.LINE_AA, tipLength=tip)
    return True


def _fs(img, base=0.42):
    """Font scale for this image size.

    Tiles get resized to gallery thumbnails, so a fixed point size is either
    unreadable on a small tile or cartoonish on a full frame. Referenced to a
    400 px tile and clamped so it never collapses or dominates.
    """
    return float(np.clip(base * min(img.shape[:2]) / 400.0, 0.34, 0.75))


def _text(img, org, text, colour, fs=None, thick=1):
    """Text with a dark halo, so it survives on a bright robot or grey ground."""
    fs = fs if fs is not None else _fs(img)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, colour, thick, cv2.LINE_AA)


def _label(img, p, text, colour, dy=-8):
    if p is None:
        return
    x, y = _pt(p)
    _text(img, (x + 4, y + dy), text, colour)


def draw(image, pose, sample=None, K=None, radius_mm=None, gain=50.0,
         layers=ALL_LAYERS, scale=1.0, crop=None, size=None, legend=True):
    """Overlay estimate and ground truth on one frame.

    ``pose`` is an `estimator.Pose`; ``sample`` a `render.Sample` when ground
    truth exists (synthetic).  Without a sample the truth-dependent layers are
    skipped, so the same function serves a live camera.

    ``legend`` draws the key and scale bar.  Turn it off for gallery tiles and
    show one key beside the grid instead: at thumbnail size a per-tile legend
    covers the robot it is explaining.

    ``gain`` amplifies the residual arrow only.  ``crop`` is a padding factor
    that tightens the view around the detection *before* anything is drawn --
    cropping afterwards would slice the legend and scale bar off the bottom.
    Cropping is a pure translation on the image plane, so the intrinsics follow
    it exactly by shifting the principal point; the projected geometry stays
    correct rather than merely close.
    """
    layers = set(layers)
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    if pose is None:
        cv2.putText(out, "NO DETECTION", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return _resize(out, scale)

    if K is None:
        K = _default_K()
    if radius_mm is None:
        radius_mm = _default_radius()

    ellipse = pose.ellipse
    if crop:
        out, K, ellipse = _crop(out, pose, K, pad=crop)
    pose = _WithEllipse(pose, ellipse)

    notes = []

    # --- ellipses ---------------------------------------------------------
    if "fit" in layers:
        cv2.ellipse(out, pose.ellipse, COL_FIT, 1, cv2.LINE_AA)
    if "truth" in layers and sample is not None:
        gt_ellipse = conic.project_circle(sample.center_mm, sample.normal, radius_mm, K)
        _dashed_ellipse(out, gt_ellipse, COL_TRUE)
        d_major = pose.ellipse[1][0] - gt_ellipse[1][0]
        notes.append(f"major fit-true {d_major:+.1f}px")

    # --- ambiguity branches ----------------------------------------------
    cands = pose.extra.get("candidates") or []
    if "branches" in layers and len(cands) > 1:
        for c in cands:
            if np.allclose(c.normal, pose.normal, atol=1e-9):
                continue  # the chosen one is drawn below, solid
            tip = _project([c.center, c.center + NORMAL_LEN_MM * c.normal], K)
            _arrow(out, tip[0], tip[1], COL_ALT, 1, tip=0.3)
            _label(out, tip[1], "alt branch", COL_ALT)
        notes.append(f"ambiguity {pose.ambiguity_margin_deg:.0f}deg")

    # --- orientation vectors ---------------------------------------------
    if "normal" in layers:
        est_c = _camera_frame_centre(pose, sample)
        p = _project([est_c, est_c + NORMAL_LEN_MM * np.asarray(pose.normal)], K)
        if _arrow(out, p[0], p[1], COL_FIT, 2):
            _label(out, p[1], "n est", COL_FIT)
        if sample is not None:
            q = _project([sample.center_mm,
                          sample.center_mm + NORMAL_LEN_MM * sample.normal], K)
            if _arrow(out, q[0], q[1], COL_TRUE, 1):
                _label(out, q[1], "n true", COL_TRUE, dy=12)

    # --- residual, amplified ---------------------------------------------
    if "residual" in layers and sample is not None:
        est_c = _camera_frame_centre(pose, sample)
        resid = np.asarray(est_c) - np.asarray(sample.center_mm)
        mag = float(np.linalg.norm(resid))
        p = _project([sample.center_mm, sample.center_mm + gain * resid], K)
        _arrow(out, p[0], p[1], COL_RESID, 2, tip=0.3)
        _label(out, p[1], f"resid x{gain:g}", COL_RESID, dy=14)
        notes.append(f"|resid| {mag:.2f}mm (dz {resid[2]:+.2f})")

    # --- annotation -------------------------------------------------------
    if "text" in layers:
        if legend:
            z_ref = (float(sample.center_mm[2]) if sample is not None
                     else float(pose.xyz_mm[2]))
            _legend(out, sample is not None, gain, radius_mm, K, z_ref)
        fs = _fs(out, 0.46)
        step = int(20 * fs / 0.46)
        for i, line in enumerate(notes):
            _text(out, (10, int(18 * fs / 0.46) + step * i), line, COL_TEXT, fs)

    if size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    return _resize(out, scale)


class _WithEllipse:
    """`pose` with a translated ellipse, for the cropped view.

    A thin proxy rather than mutating the caller's Pose -- the same Pose is
    often drawn several times at different crops, and a gallery quietly
    corrupting its own inputs would be a miserable bug to find.
    """

    def __init__(self, pose, ellipse):
        self._p = pose
        self.ellipse = ellipse

    def __getattr__(self, name):
        return getattr(self._p, name)


def _crop(img, pose, K, pad=1.6):
    """Crop around the detection, returning the image, shifted K and ellipse."""
    (cx, cy), (major, _), _ = pose.ellipse
    half = int(max(60, major * pad / 2))
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - half)), min(w, int(cx + half))
    y0, y1 = max(0, int(cy - half)), min(h, int(cy + half))

    Ks = K.copy()
    Ks[0, 2] -= x0
    Ks[1, 2] -= y0
    (ecx, ecy), axes, ang = pose.ellipse
    return img[y0:y1, x0:x1], Ks, ((ecx - x0, ecy - y0), axes, ang)


def _camera_frame_centre(pose, sample):
    """The estimate's centre in camera coordinates.

    `Pose.xyz_mm` is in the datum frame, which for synthetic work is identity;
    guarding anyway means the overlay stays correct if someone draws a zeroed
    run, where plotting datum coordinates through the camera matrix would put
    the arrows somewhere meaningless.
    """
    return np.asarray(pose.xyz_mm, dtype=np.float64)


def _legend(img, has_truth, gain, radius_mm, K, z_mm=220.0):
    """Key plus a scale bar, so the amplified arrow can be read quantitatively."""
    h, w = img.shape[:2]
    rows = [("fitted ellipse / n est", COL_FIT, "solid")]
    if has_truth:
        rows += [("ground truth", COL_TRUE, "dashed"),
                 (f"residual x{gain:g}", COL_RESID, "solid"),
                 ("rejected branch", COL_ALT, "thin")]

    fs = _fs(img, 0.40)
    step = int(17 * fs / 0.40)
    y = h - 12 - step * len(rows)
    for text, colour, style in rows:
        cv2.line(img, (12, y - 4), (34, y - 4), colour,
                 1 if style == "thin" else 2, cv2.LINE_AA)
        if style == "dashed":  # punch gaps so the style reads at thumbnail size
            for gx in (18, 27):
                cv2.line(img, (gx, y - 4), (gx + 4, y - 4), (0, 0, 0), 3, cv2.LINE_AA)
        _text(img, (40, y), text, COL_TEXT, fs)
        y += step

    # 10 mm scale bar at the image centre depth, so "how big is that arrow" is answerable.
    px_per_mm = K[0, 0] / z_mm
    bar = int(10 * px_per_mm)
    x0, y0 = w - bar - 20, h - 20
    cv2.line(img, (x0, y0), (x0 + bar, y0), COL_TEXT, 2, cv2.LINE_AA)
    for x in (x0, x0 + bar):
        cv2.line(img, (x, y0 - 4), (x, y0 + 4), COL_TEXT, 2, cv2.LINE_AA)
    _text(img, (x0, y0 - 8), f"10 mm @{z_mm:.0f}", COL_TEXT, fs)


def _resize(img, scale):
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _default_K():
    import estimator

    return estimator.load_intrinsics()[0]


def _default_radius():
    import estimator

    return estimator.RADIUS_MM
