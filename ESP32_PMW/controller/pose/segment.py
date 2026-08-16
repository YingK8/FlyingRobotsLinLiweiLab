"""Segment the robot from its background and fit the rim ellipse.

Three rig appearances are supported, selected by `APPEARANCE` / `score_channel`:
a bright robot on a dark ground (the original bench setup, and the one every
calibration constant here was fitted on); a **red** robot on a white ground,
which needs chroma rather than brightness -- see `score_channel` for why
inverting the threshold does not work; and **dark**, a black robot on a white
backdrop under a *monochrome* camera, with the drive coils and the room beyond
the backdrop in frame. Everything below the first paragraph of `segment` is
appearance-agnostic: it operates on whichever single channel makes the robot
bright.

`dark` is the only one that cannot be reduced to a channel and a level. Inverting
the threshold finds the robot and also the coils, the wires and the room, all of
which are darker still; and with a mono sensor there is no chroma to tell them
apart. So it adds two things, and only for this appearance: a **valid region**
(`valid_region`), which is where the robot may be at all, and a **spread limit**
(`silhouette_hull`), which decides which blobs inside that region belong to the
same object. See `score_channel` for what each is measured against.

A fixed threshold is enough and is what `visual_servo/servo.py` already uses.
Deliberately *not* Otsu -- as that file's own comment records, with the robot out
of frame Otsu will happily threshold sensor noise into a confident "detection".
A fixed level simply returns nothing, which is the correct answer.

What this adds over `servo.py`'s `detect()` is that `conic.py` needs the **duct
rim**, and getting it is harder than "largest contour".  The duct is a thin
ring, not a disc: face-on, its outer wall is nearly parallel to the view, so
lighting routinely breaks it into disconnected arcs and the largest connected
blob becomes the four-blade cross in the middle.  Measured on a face-on render,
largest-contour fits a 83 px ellipse where the true rim is 131 px -- a 37%
underestimate, which lands directly on the depth estimate.

So instead: keep every blob big enough to be real, pool their boundary points,
and take the **convex hull**.  The rim is the outermost feature of the robot, so
its hull is the rim, and broken arcs still hull to the same circle.  Same
face-on render, hull fit: 129.9 px against 130.7 px analytic.

The hull is then fitted with **axial weighting**, and that is what makes high
tilt work.  The rod and magnet stick out along the rotor axis, so as the robot
tilts they push the silhouette outward in its *short* direction -- and because
they lie on the axis, they land near the **middle of the major axis**.  Traced at
70 degrees tilt, the two worst hull vertices are rod tips (17.2 px off the true
rim, against the magnet's 7.0 px), both at |projection|/semi-major < 0.04.

So each hull point is weighted by how far along the major axis it sits,
``w = |proj| / a`` (see `AXIAL_WEIGHT_POWER`), which is near zero exactly where
the protrusions appear
and near one at the major-axis ends where the rim is trustworthy.  Two
reweighted refits.  At 70 degrees this pulls the minor axis from 75.6 px to
54.4 px against 45.0 px true.

The weight floor must be **zero**.  A floor of even 0.05 restores most of the
error (68.1 px instead of 54.4), because the rod points are extreme enough to
dominate a fit that gives them any weight at all.

Making the weighting tilt-adaptive was tried and does not work, for a reason
worth recording: the blend has to read tilt from the provisional ellipse, but
that ellipse is the contaminated one, so it under-reads tilt and under-applies
the very correction that would fix it.  Every adaptive variant landed at
12-13 degrees of normal error at high tilt where unconditional weighting gives
0.46.  Apply it always.

Weighting appears to cost accuracy at moderate tilt until the tilt calibration
is refitted for it; the apparent cost is pure systematic bias.  Compared
calibrated-against-calibrated on a held-out split, unconditional axial weighting
is better everywhere: normal error 1.95 -> 0.52 degrees, position 1.17 ->
0.89 mm, relative depth 0.477 -> 0.380%.

That also retires a limit this file used to claim.  The ">45 degrees the
flat-circle model breaks" note was largely a **fitting artefact, not geometry**.
With the protrusions weighted out, calibrated normal error is 0.53 degrees over
tilt 10-55 and 0.46 above 60 -- essentially flat, where before it tripled across
the same span.  The residual minor-axis excess at 70 degrees falls from +69% to
+22%, and what is left is the duct's own wall thickness projecting, which is
genuine geometry.

Plain outlier *trimming* by residual was tried earlier and did fail -- the hull
carries only 20-60 points, so discarding any meaningful fraction destabilises
the fit.  Weighting by *position along the major axis* works where trimming by
residual did not, because it targets where the contamination is known to be
rather than trying to discover it from the residuals it has already corrupted.

Sub-pixel edge refinement was implemented and rejected.  Walking each hull
vertex along its outward normal to the interpolated threshold crossing changed
depth scatter by nothing at all -- 0.661% against 0.662% -- and made lateral
error slightly worse.  The reason is that the hull already averages 30 to 60
points, so pixel quantisation has averaged away before the fit sees it.  The
residual scatter is not measurement precision; it is the 3-D silhouette varying
with blade phase and lighting, which a 2-D ellipse cannot represent at all.
Anything aimed at that floor has to add information, not sharpen the edge.
"""

from __future__ import annotations

import math
import os

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import conic  # noqa: E402
import shape  # noqa: E402  (calib/: owns APPEARANCE)

# Carried over from visual_servo/servo.py so both paths behave the same.
#
# The level trades depth bias against depth scatter, measured over 120 poses:
#   thresh  96 -> bias -0.74%, scatter 0.52%
#   thresh 112 -> bias -0.53%, scatter 0.52%
#   thresh 128 -> bias -0.27%, scatter 0.64%
#   thresh 144 -> bias +0.01%, scatter 0.70%
# A lower level includes more of the dim rim edge and localises it better.  It is
# NOT retuned here: those renders have one particular brightness, and on real
# hardware the useful level depends on exposure, so a number fitted to synthetic
# images would not transfer.  128 is what is already validated on the bench.
#
# If you change this (or the exposure), refit the effective radius -- the bias
# column above moves by 0.75% across this range, which is larger than the entire
# depth residual. See validation/tune.py.
THRESH = 128
MIN_BLOB_AREA_PX = 30

# Opening is deliberately smaller than closing.  The projected rim wall is only
# a few pixels thick, so a 5x5 open erodes it away; 3x3 clears sensor specks
# without doing that.  The 7x7 close then bridges the gaps lighting leaves in
# the ring, which is what lets the hull see one circle instead of several arcs.
#
# Measured over 120 random poses, as scatter in relative depth error:
#   open 3 -> 0.52-0.70%      open 5 -> 0.77-1.65%
#   close 5 / 7 / 9 -> identical to three decimal places
# So the opening size matters a great deal and the closing size not at all; 7 is
# kept as a middle value with no evidence either way.
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# A blob smaller than this fraction of the biggest one is treated as noise.
# Fractional rather than absolute so it holds across resolutions and distances.
_BLOB_KEEP_FRACTION = 0.02

# A contrast-relative threshold (level scaled to the frame's own peak) was tried
# and dropped. It was aimed at the one real segmentation failure the sweep finds
# -- face-on under a hard side light, where the ring's outer wall is barely lit,
# the hull collapses onto the blade cross, and position error hits 63 mm. It
# does not fix it (58 mm instead of 64 mm), because in that geometry the rim is
# not dim, it is genuinely unlit: no threshold recovers a signal that is not
# there. Meanwhile it made every well-lit case slightly worse, by pulling dim
# fringe pixels into the hull and inflating the rim (0.45 -> 0.65 mm face-on,
# 1.59 -> 2.16 mm at 40 degrees). The fix for that failure is lighting, not
# thresholding.

# fitEllipse needs 5 points for 5 ellipse parameters.
_MIN_CONTOUR_PTS = 5

# Axial weighting: w = (|proj on major| / semi-major) ** POWER, floored at zero.
#
# Power 1, and the reason is not obvious from the minor-axis error alone. Judged
# only on how well it recovers the minor axis at 70 degrees tilt, power 2 wins
# (9.4 px against power 1's 13.8). Judged on the pose that comes out, power 1 is
# better, because a harsher power also discards the mid-range points that
# constrain the ellipse's *shape*, and depth is sensitive to that. Measured
# per tilt band, calibrated and on held-out data:
#
#   power   position (10-45 / 45-60 / >60 deg)   normal (same bands)
#   none    0.828 / 1.412 / 2.879 mm             1.109 / 2.384 / 3.283 deg
#   1       0.671 / 0.851 / 1.196                0.602 / 0.737 / 0.636
#   2       0.765 / 0.717 / 1.819                0.538 / 0.529 / 0.456
#
# Power 2 buys about 0.15 deg of orientation and costs 0.1-0.6 mm of position;
# power 1 is the only setting that beats no weighting at all in *every* band on
# *both* metrics, which is what makes it safe to apply unconditionally.
#
# Suppression of the rod does not depend on the power: at |proj|/a = 0.03, where
# the rod tips land, power 1 gives weight 0.03 and power 2 gives 0.0009 -- both
# negligible against points near 1. The power only decides how much legitimate
# mid-range evidence is thrown away with it.
AXIAL_WEIGHT_POWER = 1.0

# Axial weighting is ON. Measured against the unweighted fit on a controlled A/B
# -- same seed, same 400 poses, same gate, same constants, `POSE_AXIAL` the only
# difference -- it costs no gate coverage (59 certified frames against 61) and
# improves both error metrics in every sensor mode (1280x800: position 0.268 ->
# 0.186 mm, angle 0.294 -> 0.178 deg; modes at 100% in spec 5/8 -> 6/8).
#
# Two claims that used to live here are withdrawn; see journal Iterations 12-14
# for the evidence, which is not repeated here:
#   * that the weighting collapses certified detection (it does not -- that was
#     measured while `PoseEstimator.update` ignored this flag entirely);
#   * that the weighted fit needs its own radius, 10.2662 mm (refitting gives
#     10.2418, i.e. 0.03% from the unweighted value -- the same radius serves).
#
# Regenerating the whole constant chain for the weighted fit was tried and
# *rejected on measurement*: it certifies 12 frames where the shipped set
# certifies 59. Refitting the error model properly is open work.
#
# **It costs latency, and that is an open trade rather than a settled one.**
# Measured end to end over a 1000x720 sequence: 2.31 ms/frame unweighted against
# 3.53 ms weighted, i.e. 433 Hz -> 283 Hz. The fit itself goes 0.04 -> 1.02 ms,
# and the breakdown says the axial weighting is the cheap part:
#
#   base fit                     0.039 ms
#   + axial re-weighting (x2)    0.19 ms
#   + one-sided pass (x3)        0.375 ms      POSE_ONE_SIDED=0 disables
#   + trim pass (x3)             0.42 ms       TRIM_FRACTION=0 disables
#
# So two thirds of the cost is the two later robustness stages, whose accuracy
# contribution has never been A/B'd on its own. `test_stereo.py::test_speed`
# currently FAILS because of this: the full two-view solve is 3.40 ms against a
# 4.17 ms budget at 240 Hz, and segmentation needs the rest. That failure is left
# visible on purpose.
#
# Whether it matters depends on the target rate, and the rate the cameras on this
# rig actually deliver is 15-28 fps -- 10x slower than the budget the test
# asserts. At 240 Hz this still fits monocular; at 420 Hz it does not.
#
# ``POSE_AXIAL=1``/``=0`` overrides this for one process, matching the
# ``POSE_ONE_SIDED`` convention below, so an A/B runs as two subprocesses rather
# than by editing this line -- which is how two arms end up differing in more
# than one thing.
AXIAL_DEFAULT = os.environ.get("POSE_AXIAL", "1") not in ("0", "", "false", "False")
AXIAL_WEIGHT_ITERS = 2

# Optional floor on the axial weights. **0.0 -- the re-weighting is not what
# the orientation comes from; see `fit_ellipse`.**
#
# Kept because the investigation behind it is worth not repeating. The weights
# fall to exactly zero over the arc nearest the major axis's centre, and a zero
# weight does not distrust a point, it deletes it. What remains is clustered at
# the two ends of the major axis, and five conic parameters fitted to two
# opposing clusters are ill-conditioned in exactly one direction: rotation. On a
# noise-free ellipse of ratio 0.834 that put the fitted major axis **33.5 deg**
# out, with a Sampson rms of 3.7374 px against 0.0000 px for the plain fit --
# a re-weighted fit that was *worse* against the very points it was fitted to.
#
# A floor looked like the fix and is not. It changes *which* cases fail rather
# than whether they do: at 0.05 the 33.5 deg case became exact, and a different
# pose that had been exact went 14 deg out. The rotation is ill-conditioned under
# these weights, so the error is arbitrary in them. Locking the orientation is
# the fix; this stays at 0 so the axes keep the full benefit of the suppression.
AXIAL_WEIGHT_FLOOR = 0.0

# Weight given to boundary points that fall **outside** the current fit.
#
# The silhouette's contamination is one-sided by construction.
# `silhouette_hull` takes a convex hull, so the extracted boundary is a superset
# of the projected rim: the rod, the magnet mount and any thresholding spill can
# only push the outline *outward*, never inward. A symmetric loss therefore does
# not average the contamination away -- it splits the difference with it, and
# the fit settles outside the true rim by roughly half the excursion.
#
# Weighting outward points below inward ones pulls the fit onto the inner
# envelope, which is where the rim actually is. The value matters less than the
# asymmetry; 0.15 was the best of those measured.
#
# What this buys is **spread**, not bias. `TiltCalibration` already removes the
# mean of the ratio error, so a mean improvement would be absorbed and invisible.
# Measured over 120 poses with tilt and contamination amplitude both varied:
#
#             ratio-error mean     ratio-error std
#   off            +0.01825            0.00479
#   0.15           +0.00935            0.00253      <- 47% less spread
#   0.30           +0.01181            0.00316
#
# Near 45 degrees, where dtheta = d(ratio)/sin(theta), that is 0.388 deg of
# irreducible tilt error falling to 0.205 deg -- and no calibration can recover
# it, because it is scatter rather than offset.
#
# Overridable from the environment so the A/B can be run without editing code:
#   POSE_ONE_SIDED=0 ... disables it, reproducing the previous behaviour.
ONE_SIDED_WEIGHT = float(os.environ.get("POSE_ONE_SIDED", "0.15"))

# Fraction of the outline discarded outright once the rim is well covered, and
# the coverage required before discarding anything.
#
# Down-weighting a contaminated point still lets it pull the fit. Discarding it
# does not, and the mast's contribution is a *localised lobe* rather than a
# spread-out error, so selection suits it better than weighting. Measured on
# silhouettes with a randomly placed, randomly sized lobe -- the case where the
# presented shape genuinely varies -- the ratio-error scatter falls from 0.03586
# to 0.02266, a **37%** reduction.
#
# The guard is not optional. Discarding a quarter of the boundary is safe only
# when there is redundancy left, and with a rim broken into arcs there is not:
# at 70% arc coverage an unguarded trim made the scatter *worse* (0.03623 ->
# 0.04253) and the worst case worse still. Angular coverage of the points about
# the fitted ellipse measures exactly that redundancy, so the trim is applied
# only above `TRIM_MIN_COVERAGE` and skipped entirely below it -- which, on the
# broken-rim cases, leaves the previous behaviour untouched.
TRIM_FRACTION = 0.25
TRIM_MIN_COVERAGE = 0.85
TRIM_ITERS = 3
TRIM_COVERAGE_BINS = 24
ONE_SIDED_ITERS = 3

# Skip the weighting when the provisional ellipse is this circular. Near face-on
# there is nothing to suppress -- the protrusions point at the camera and project
# to a dot -- while the points being down-weighted are the only ones carrying the
# minor axis, so weighting there just adds noise (normal error 3.28 -> 2.84 deg
# below 10 degrees tilt, and identical above it).
#
# This is a binary gate, not the tilt-adaptive *scaling* the docstring rejects,
# and it is safe for a specific reason: contamination can only push the ratio
# UP, and measured over 400 poses the unweighted ratio never exceeds 0.9896
# above 10 degrees of true tilt. The gate cannot fire on a contaminated
# high-tilt frame.
AXIAL_SKIP_RATIO = 0.995


@dataclass
class Segmentation:
    """One frame's segmentation result.

    ``ellipse`` is in OpenCV's ``((cx, cy), (major, minor), angle_deg)`` form,
    major-axis first, and is what `conic.backproject_ellipse` consumes.
    ``contour`` is the convex hull, not a raw contour.  ``fit_rms_px`` is the
    Sampson distance of the hull points to the fitted ellipse -- a direct
    measure of how circular the silhouette really was, and the best single
    number for spotting a frame where segmentation grabbed the wrong thing.
    ``area_px`` is a true pixel count of the kept blobs.  ``valid`` is the region
    the segmenter was allowed to look in (``None`` for appearances that look
    everywhere) -- carried so the overlay can shade what was ignored without
    recomputing it, which for the backdrop finder costs 2.4 ms a frame.
    """

    mask: np.ndarray
    contour: np.ndarray
    ellipse: tuple
    area_px: float
    n_points: int
    fit_rms_px: float
    threshold: int
    t_ms: float
    valid: np.ndarray | None = None


def fit_ellipse_direct(pts):
    """Fitzgibbon direct fit of ``pts``, axes ordered major-first.

    Pairing `cv2.fitEllipseDirect` with `conic.normalise_ellipse` is not
    optional: an unnormalised result reports its angle against whichever axis
    OpenCV happened to call ``width``, so the angle jumps 90 degrees on a nearly
    circular silhouette. That pairing was written out four separate times;
    separating the two calls is the mistake this exists to make impossible.

    It lives here rather than in `conic.py` because that module is deliberately
    OpenCV-free -- pure geometry over numpy -- and this is the OpenCV side.
    """
    return conic.normalise_ellipse(cv2.fitEllipseDirect(np.asarray(pts, dtype=np.float32)))


def sampson_distance_conic(c, pts):
    """First-order geometric distance from points to a conic, in pixels.

    The algebraic residual ``p'Cp`` is scale-dependent and biased toward the
    ellipse's flat sides; dividing by the gradient magnitude (Sampson's
    approximation) turns it into something that reads in pixels and is fair
    around the whole perimeter.  Good enough to rank outliers, and far cheaper
    than a true orthogonal distance.

    Takes the conic directly rather than an ellipse because the stereo solver
    predicts a conic (``K^-T Q K^-1`` for a hypothesised circle pose) and
    converting it to axis form only to convert it straight back would be waste
    inside a least-squares inner loop.  `_sampson_distance` is the ellipse-taking
    wrapper the segmenter uses.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    num = np.abs(np.einsum("ij,jk,ik->i", ph, c, ph))
    grad = 2.0 * (ph @ c.T)[:, :2]
    den = np.linalg.norm(grad, axis=1)
    return num / np.maximum(den, 1e-12)


def _sampson_distance(ellipse, pts):
    """`sampson_distance_conic`, starting from an OpenCV ellipse."""
    return sampson_distance_conic(conic.conic_from_ellipse(ellipse), pts)


def axial_weights(pts, ellipse, power=AXIAL_WEIGHT_POWER,
                  floor=AXIAL_WEIGHT_FLOOR):
    """Weight each point by how far along the major axis it lies.

    ``floor`` at the centre of the major axis, where the rod and magnet project,
    and one at the ends, where the silhouette is the rim and can be trusted.

    The floor is not cosmetic -- see `AXIAL_WEIGHT_FLOOR`. A weight of exactly
    zero removes a contiguous arc rather than distrusting it, and what is left
    cannot pin the ellipse's rotation.
    """
    (cx, cy), (major, _), ang = ellipse
    if major <= 0:
        return None
    t = np.radians(ang)
    u = np.array([np.cos(t), np.sin(t)])
    s = np.abs((np.asarray(pts, dtype=np.float64) - np.array([cx, cy])) @ u) / (major / 2.0)
    return floor + (1.0 - floor) * np.clip(s, 0.0, 1.0) ** power


def angular_coverage(ellipse, pts, bins=TRIM_COVERAGE_BINS):
    """Fraction of the ellipse's perimeter that has points near it.

    Computed in the ellipse's own frame so it measures coverage of the *shape*
    rather than of the image. This is the redundancy test that decides whether
    discarding points is safe -- see `TRIM_FRACTION`.
    """
    (cx, cy), _, ang = ellipse
    t = math.radians(ang)
    d = np.asarray(pts, dtype=np.float64).reshape(-1, 2) - np.array([cx, cy])
    u = d @ np.array([math.cos(t), math.sin(t)])
    v = d @ np.array([-math.sin(t), math.cos(t)])
    h = np.histogram(np.arctan2(v, u), bins=bins, range=(-np.pi, np.pi))[0]
    return float((h > 0).mean())


def _signed_sampson(ellipse, pts):
    """Sampson distance keeping its sign: positive is outside the ellipse."""
    c = conic.conic_from_ellipse(ellipse)
    ph = np.hstack([np.asarray(pts, dtype=np.float64).reshape(-1, 2),
                    np.ones((len(pts), 1))])
    alg = np.einsum("ij,jk,ik->i", ph, c, ph)
    grad = 2.0 * (ph @ c.T)[:, :2]
    return alg / np.maximum(np.linalg.norm(grad, axis=1), 1e-12)


def _outward_weights(pts, ellipse, w_out=ONE_SIDED_WEIGHT,
                     power=AXIAL_WEIGHT_POWER):
    """Axial weights, further reduced for points lying outside ``ellipse``.

    The sign comes from the algebraic conic residual scaled by its gradient --
    the same Sampson quantity `sampson_distance_conic` returns, kept signed
    rather than absolute, so positive means outside.
    """
    (cx, cy), (major, _), ang = ellipse
    if major <= 0:
        return None
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    try:
        c = conic.conic_from_ellipse(ellipse)
    except (ValueError, np.linalg.LinAlgError):
        return None
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    alg = np.einsum("ij,jk,ik->i", ph, c, ph)
    grad = 2.0 * (ph @ c.T)[:, :2]
    signed = alg / np.maximum(np.linalg.norm(grad, axis=1), 1e-12)

    t = np.radians(ang)
    u = np.array([np.cos(t), np.sin(t)])
    s = np.abs((pts - np.array([cx, cy])) @ u) / (major / 2.0)
    axial_w = np.clip(s, 0.0, 1.0) ** power
    return np.where(signed > 0.0, w_out, 1.0) * axial_w


def fit_ellipse(pts, axial=None, power=None, iters=None):
    """Direct ellipse fit, normalised to major-axis-first.

    With ``axial`` (the default) the fit is reweighted twice to suppress the
    axial protrusions -- see the module docstring for why this is unconditional
    rather than tilt-adaptive. ``axial=False`` recovers the plain fit, which is
    what `validation/tune.py` compares against.

    ``power`` and ``iters`` read the module constants at *call* time rather than
    binding them as default arguments. That matters: as defaults they are
    captured when the function is defined, so a test that reassigns
    ``AXIAL_WEIGHT_ITERS`` to compare weighted against unweighted silently
    measures the weighted path twice. Pass ``axial=False`` to disable it.

    ``cv2.fitEllipseDirect`` rather than plain ``fitEllipse``: it stays stable on
    the short, nearly-straight arcs you get when the disc is close to edge-on,
    where the ordinary fit can return a degenerate result.

    Returns ``(ellipse, rms_px)`` or ``None``.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < _MIN_CONTOUR_PTS:
        return None
    try:
        ellipse = fit_ellipse_direct(pts)
    except cv2.error:
        return None

    # The re-weighting corrects the axis *lengths*; the orientation is taken
    # from the plain fit and held.
    #
    # The rod and magnet fatten the silhouette in its short direction -- that is
    # a length error, and suppressing it is what this loop is for. They do not
    # rotate it: the protrusions sit symmetrically about the major axis, so the
    # unweighted fit, which sees every point, reads the orientation correctly.
    # Letting the re-weighted fit set the angle instead hands the one quantity
    # the weights cannot constrain to the one fit that cannot constrain it.
    #
    # Measured, holding the angle costs nothing and buys the failure back:
    # injected-protrusion ratio bias +0.01989 held against +0.01965 free (and
    # +0.05591 with no re-weighting at all), while the worst refinement error on
    # the exact-geometry test goes from 1.60e-01 mm to 3.85e-08 mm.
    plain_angle = ellipse[2]
    axial = AXIAL_DEFAULT if axial is None else bool(axial)
    power = AXIAL_WEIGHT_POWER if power is None else power
    iters = AXIAL_WEIGHT_ITERS if iters is None else iters
    if axial and iters and ellipse[1][1] / ellipse[1][0] <= AXIAL_SKIP_RATIO:
        for _ in range(iters):
            w = axial_weights(pts, ellipse, power)
            if w is None or w.sum() <= 0:
                break
            c = conic.fit_conic_weighted(pts, w)
            if c is None:
                break  # keep the last good fit rather than lose the frame
            try:
                ellipse = conic.normalise_ellipse(conic.ellipse_from_conic(c))
            except (ValueError, np.linalg.LinAlgError):
                break
            ellipse = (ellipse[0], ellipse[1], plain_angle)

    # One-sided pass: the hull can only err outward, so distrust points outside
    # the current fit. Orientation stays with the plain fit throughout, for the
    # reason given above.
    if axial and ONE_SIDED_WEIGHT > 0.0:
        for _ in range(ONE_SIDED_ITERS):
            w = _outward_weights(pts, ellipse)
            if w is None or w.sum() <= 0:
                break
            c = conic.fit_conic_weighted(pts, w)
            if c is None:
                break
            try:
                e = conic.normalise_ellipse(conic.ellipse_from_conic(c))
            except (ValueError, np.linalg.LinAlgError):
                break
            ellipse = (e[0], e[1], plain_angle)

    # Trim, if there is enough of the rim to afford it. Orientation is held at
    # the plain fit's value throughout, for the reason in `fit_ellipse` above.
    if axial and TRIM_FRACTION > 0.0 and len(pts) >= 16:
        try:
            covered = angular_coverage(ellipse, pts) >= TRIM_MIN_COVERAGE
        except (ValueError, np.linalg.LinAlgError):
            covered = False
        if covered:
            k = int(round(TRIM_FRACTION * len(pts)))
            for _ in range(TRIM_ITERS):
                if k < 1 or len(pts) - k < 8:
                    break
                try:
                    d = _signed_sampson(ellipse, pts)
                except (ValueError, np.linalg.LinAlgError):
                    break
                keep = np.argsort(d)[:-k]
                c = conic.fit_conic_weighted(pts[keep], np.ones(len(keep)))
                if c is None:
                    break
                try:
                    e = conic.normalise_ellipse(conic.ellipse_from_conic(c))
                except (ValueError, np.linalg.LinAlgError):
                    break
                ellipse = (e[0], e[1], plain_angle)

    (_, _), (major, minor), _ = ellipse
    if not all(np.isfinite([major, minor])) or major <= 0 or minor <= 0:
        return None

    # rms is over *all* hull points, including the ones the fit down-weights.
    # A weighted rms was tried and is worse for the gate (acceptance 3.5% ->
    # 0.5%): the unweighted value is the signal, not a defect, because it
    # measures how far the silhouette departs from *any* ellipse -- i.e. how much
    # contamination is present, and so how much error survives the weighting.
    return ellipse, float(np.sqrt(np.mean(_sampson_distance(ellipse, pts) ** 2)))


# Half-width, in pixels, of the intensity profile sampled across the boundary
# when locating it to sub-pixel precision.
SUBPIX_SEARCH_PX = 3.0
SUBPIX_SAMPLES = 13


def subpixel_boundary(gray, pts, centre=None, search_px=SUBPIX_SEARCH_PX,
                      n_samples=SUBPIX_SAMPLES):
    """Move each boundary point onto the intensity edge, at sub-pixel precision.

    A threshold-and-hull outline is quantised to the pixel grid, and where it
    falls depends on how the threshold happens to slice a smoothly shaded,
    motion-blurred edge. Measured, that injects a **2.597 deg** standard
    deviation into the per-view tilt at 40-50 deg -- scatter that correlates with
    nothing recorded (pose, lighting, exposure, opacity, spin), so no calibration
    removes it, no gate feature predicts it, and no pose-dependent silhouette
    model corrects it. See lecture notes 12.12.

    Along the outward normal the intensity falls from robot to background through
    a smooth ramp whose **half-height** crossing is where the geometric edge is,
    independently of the threshold that found it approximately. Locating that
    crossing by interpolation is sub-pixel and is a property of the image rather
    than of a constant.

    **It does not fix the problem it was written for, and the gap is the lesson.**
    On a synthetic soft-edged disc of known radius it is decisively better --
    radius bias -0.0858 -> -0.0012 px, scatter 0.0550 -> 0.0030 px, an 18x
    improvement. Carried through to the real pipeline it reduced the per-view
    tilt residual scatter by **1-4%** (2.597 -> 2.537 deg at 40-50 deg tilt).
    The synthetic test measured how precisely a *known* edge can be located; the
    real variability is *which shape presents itself* -- the mast and rod
    projecting differently with pose, rim arcs vanishing under grazing light,
    spin smearing the boundary across the exposure. Locating the wrong shape more
    precisely gains almost nothing.

    So this is off by default: 0.5 ms/frame is not worth 2% against a 2.4 ms
    budget at 420 Hz. It is kept because the code is correct and the comparison
    should be repeatable.

    ``pts`` are the coarse boundary points; ``centre`` fixes the outward
    direction and defaults to their mean. Points whose profile is not a clean
    monotone fall are left where they were rather than moved onto noise.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return pts
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.float32)
    h, w = g.shape

    c = np.asarray(centre, dtype=np.float64) if centre is not None else pts.mean(0)
    d = pts - c
    n = np.linalg.norm(d, axis=1, keepdims=True)
    u = d / np.maximum(n, 1e-9)                      # outward unit normals

    # Sample the profile along each normal in one batched remap.
    t = np.linspace(-search_px, search_px, n_samples, dtype=np.float64)
    xs = pts[:, None, 0] + u[:, None, 0] * t[None, :]
    ys = pts[:, None, 1] + u[:, None, 1] * t[None, :]
    prof = cv2.remap(g, xs.astype(np.float32), ys.astype(np.float32),
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

    inner = prof[:, 0]
    outer = prof[:, -1]
    half = 0.5 * (inner + outer)
    drop = inner - outer

    out = pts.copy()
    # Only move a point when the profile actually falls outward by enough to
    # locate a crossing; a flat or inverted profile means the normal missed the
    # edge, and guessing there would be worse than the quantised original.
    usable = drop > 2.0
    for i in np.nonzero(usable)[0]:
        pr = prof[i]
        below = np.nonzero(pr <= half[i])[0]
        if not len(below) or below[0] == 0:
            continue
        j = below[0]
        y0, y1 = pr[j - 1], pr[j]
        if y0 == y1:
            continue
        # linear interpolation for the exact half-height crossing
        frac = (y0 - half[i]) / (y0 - y1)
        tt = t[j - 1] + frac * (t[j] - t[j - 1])
        out[i] = pts[i] + u[i] * tt
    return out


def silhouette_points(mask, keep_fraction=_BLOB_KEEP_FRACTION):
    """Every boundary pixel of the kept blobs, as an (N, 2) array.

    `silhouette_hull` returns the convex hull's **vertices**, which is 9-31
    points on the real captures in `pose/assets/captures/` -- against
    192-3072 in the contour they came from, a factor of 21-134 discarded before
    anything is fitted. Two reductions do it: `CHAIN_APPROX_SIMPLE` drops
    collinear runs, then `convexHull` keeps only extreme vertices.

    That loss is not free. Robust estimation needs redundancy: measured across
    boundary densities, discarding half the points to suppress the mast improves
    the fit above ~32 points and **degrades** it below, because five conic
    parameters fitted to six survivors are worse than five fitted to twelve
    contaminated ones. At hull density the real captures sit on the wrong side of
    that line; at contour density they do not.

    The one-sidedness that the hull provided is kept, and does not depend on the
    hull. The silhouette is the rim *union* the rod and mount, so its outer
    contour follows the rim except where those protrude, and never falls inside
    it. Contamination therefore remains strictly outward -- the property the
    robust step relies on -- while the count rises by two orders of magnitude.

    Returns ``(points, area_px)`` or ``(None, 0.0)``, matching
    `silhouette_hull`. The hull is still the right answer when a *closed* shape
    is needed; this is for fitting, where it is not.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= max(1.0, keep_fraction * areas.max()))[0] + 1
    if len(keep) == 0:
        return None, 0.0
    if len(keep) == n - 1:
        kept_mask = mask
    else:
        lut = np.zeros(n, dtype=np.uint8)
        lut[keep] = 255
        kept_mask = lut[labels]

    # NONE, not SIMPLE: the collinear runs SIMPLE discards are exactly the
    # redundancy the robust step needs.
    contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0.0
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    if len(pts) < _MIN_CONTOUR_PTS:
        return None, 0.0
    return pts, float(areas[keep - 1].sum())


# How many blobs are tried as the anchor for a grouping. Bounded because each
# candidate costs a hull and a plain ellipse fit, and the robot is never the
# eighth-largest thing in frame -- on both real captures it is the first, and the
# synthetic clutter scene puts it third.
_MAX_ANCHORS = 4

# How elliptical a blob grouping must be to be considered the robot at all,
# as Sampson rms divided by the major axis so it is scale-free.
#
# 0.05 is loose on purpose. It is not trying to measure fit quality -- `fit_rms_px`
# on the result does that, and the gate in `uncertainty.py` acts on it. Its only
# job is to reject a grouping that has spanned two objects, which scores an order
# of magnitude worse. The robot itself sits at 0.006 on both ELP captures, so
# there is most of a decade of margin.
SHAPE_TOL = 0.05


def _group_hull(labels, members, n):
    """Convex hull of a set of labels, as float64 (N, 2), or ``None``."""
    lut = np.zeros(n, dtype=np.uint8)
    lut[members] = 255
    contours, _ = cv2.findContours(lut[labels], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    if len(pts) < _MIN_CONTOUR_PTS:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    return hull if len(hull) >= _MIN_CONTOUR_PTS else None


def _best_group(keep, labels, stats, centroids, n, max_spread):
    """Which blobs belong to the robot, by which grouping is most elliptical.

    Each candidate anchor gathers the blobs within ``max_spread`` of its own
    radius, and the group is scored by the **plain** fit's Sampson rms relative to
    its own size. Relative and not absolute because a large ellipse tolerates more
    pixels of residual than a small one, and an absolute score would systematically
    prefer whichever group is smallest -- which is the clutter.

    The plain fit, not `fit_ellipse`: the weighted refits exist to remove the
    rod and magnet from a silhouette already known to be the robot's, and running
    them here would cost about 1 ms per candidate to answer a question they were
    not built for. Scoring is a shape test, and `cv2.fitEllipseDirect` is the
    cheap honest one.

    **Shape decides which groups are admissible; size decides between them.**
    Ranking on shape alone does not work, and the reason is worth keeping: a
    solid disc is a *perfect* ellipse, so a round piece of clutter scores better
    than the robot ever can. Measured on the synthetic clutter scene, ranking by
    shape returns a 64 px coil in place of the 125 px robot. Every real candidate
    clears the shape tolerance comfortably -- the robot fits at rms/major 0.006 on
    both ELP captures -- so the tolerance is there to reject groups that span two
    objects, and among what survives, the robot is simply the biggest thing in
    frame that is an ellipse at all.
    """
    order = keep[np.argsort(stats[keep, cv2.CC_STAT_AREA])[::-1]][:_MAX_ANCHORS]
    admissible = []
    best, best_score = None, np.inf
    for anchor in order:
        # The anchor's own half-diagonal, so the scale comes from the object
        # rather than a pixel constant that would not survive a resolution change.
        radius = 0.5 * float(np.hypot(stats[anchor, cv2.CC_STAT_WIDTH],
                                      stats[anchor, cv2.CC_STAT_HEIGHT]))
        if radius <= 0:
            continue
        d = np.hypot(*(centroids[keep] - centroids[anchor]).T)
        members = keep[d <= max_spread * radius]
        hull = _group_hull(labels, members, n)
        if hull is None:
            continue
        try:
            ellipse = fit_ellipse_direct(hull)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            continue
        major = ellipse[1][0]
        if not np.isfinite(major) or major <= 0:
            continue
        score = float(np.sqrt(np.mean(_sampson_distance(ellipse, hull) ** 2))) / major
        if score <= SHAPE_TOL:
            admissible.append((major, members))
        if score < best_score:
            best, best_score = members, score
    if admissible:
        return max(admissible, key=lambda t: t[0])[1]
    return best


def silhouette_hull(mask, keep_fraction=_BLOB_KEEP_FRACTION, max_spread=None):
    """Convex hull of every blob worth keeping, as an (N, 2) array.

    Blobs are sized by their true pixel count via `connectedComponentsWithStats`,
    not by `contourArea` -- a broken ring arc is a thin sliver whose polygon area
    is near zero, so an area filter would throw away exactly the pieces we most
    need to keep.

    ``max_spread`` additionally requires a blob to be *near* an anchor blob, in
    multiples of that anchor's own radius, and chooses the anchor by which
    grouping actually fits an ellipse. Area alone cannot separate the robot from
    clutter that survives the region gate: measured on the face-on ELP capture,
    the robot is 17480 px and the largest stray is 4231 px, only 4x smaller, so
    any area fraction that drops the stray is one bad frame away from dropping a
    rim arc.

    Distance separates them with room to spare -- those strays sit 1.76-2.5 robot
    radii from its centroid, while a genuine rim arc cannot be further than one
    radius by definition. **But the anchor cannot be "the largest blob".** The rim
    is a thin ring and therefore hollow, so a solid piece of clutter can outweigh
    it: on the synthetic clutter scene the robot ring is 1783 px against 3225 px
    for a coil, and anchoring on area picks the coil and returns a confident fit
    to the wrong object. Anchoring on shape instead is what the robot uniquely
    has -- it is the thing in frame that *is* an ellipse. Off by default, because
    the bright rig's constants were all fitted without any of this.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= max(1.0, keep_fraction * areas.max()))[0] + 1
    if len(keep) == 0:
        return None, 0.0

    if max_spread and len(keep) > 1:
        keep = _best_group(keep, labels, stats, centroids, n, max_spread)
        if keep is None:
            return None, 0.0

    if len(keep) == n - 1:
        # Nothing was rejected, which is the usual case on a clean frame; reuse
        # the mask rather than rebuild an identical one.
        kept_mask = mask
    else:
        # A lookup table indexed by label, not `np.isin`: one fancy-index pass
        # instead of several full-resolution temporaries. Segmentation is the
        # throughput bottleneck, so full-frame allocations here are worth avoiding.
        lut = np.zeros(n, dtype=np.uint8)
        lut[keep] = 255
        kept_mask = lut[labels]

    contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    if len(pts) < _MIN_CONTOUR_PTS:
        return None, 0.0

    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    return hull, float(areas[keep - 1].sum())


# Which appearance the rig presents, i.e. what makes the robot stand out from its
# background. ``POSE_APPEARANCE`` overrides it per process.
#
#   "bright"  white robot on a dark ground -- the original bench setup, and the
#             only one every calibration constant in this package was fitted on.
#   "red"     red robot on any neutral backdrop, black or white.
#
# **A red robot cannot be segmented reliably by brightness.** Measured over
# plausible patches spanning specular highlight to deep shade, luminance puts the
# robot at 34-172 counts, a white backdrop at 63-252 and a black one at 4-73 --
# overlapping in both cases, so no threshold in either polarity separates them.
# On black a lowered threshold *nearly* works, which is the trap: it detects on
# velvet and matte paper and returns nothing once the backdrop has a sheen.
#
# Chroma does separate them, and is indifferent to the backdrop. ``R - max(G, B)``
# reads 48-135 on the robot and 0-4 on black, grey or white alike, because any
# neutral surface has R = G = B however it is lit. Measured across seven backdrops
# the fitted rim moves by 0.21%, so one threshold and one calibration serve all of
# them.
#
# It also fails the right way. The module docstring rejects Otsu because with the
# robot out of frame it thresholds noise into a confident detection; an absolute
# chroma threshold cannot, since an empty white scene has no red in it at all.
REDNESS_THRESH = 30

# --- "dark": a black body on a white ground, with clutter in frame ---
#
# Inverting the threshold is most of it, but not all of it. The rig has bronze
# drive coils, wires and a dark ambient beyond the backdrop, and against a white
# ground **all of it is dark too**. `silhouette_hull` pools every blob and takes
# one convex hull, so clutter anywhere in frame does not add a stray contour, it
# swallows the rim: the hull spans both and the fitted ellipse is meaningless.
#
# **This appearance used to gate on chroma, and on the ELP that is inoperative.**
# The reasoning was that the body is dark and achromatic, the coils dark and
# coloured, so `max(BGR) - min(BGR)` separates them. It does -- on a colour
# camera. The ELP OV9281 is a mono sensor: measured over both frames in
# `pose/assets/captures/elp/`, chroma is **exactly 0 in every pixel** (max 0,
# mean 0.00), so `compare(chroma, CHROMA_MAX, CMP_LE)` passes the entire frame.
# It is a no-op whether the frame arrives with one channel or three, and it used
# to fail *silently* where the `red` path raises. Recorded rather than deleted
# because the chroma argument is still correct for a colour camera.
#
# Measured on those frames, which is what the constants below are fitted to:
#
#                      p5    median   p95     local sd (15x15)
#     white backdrop   176     181    193       0.8
#     drone rim          4      42-78 182      27
#     grey rod          83      96    162       --
#     coils              8-28   85-104 224     16-34
#     dark ambient       6       7-8   11       1.1
#
# Two separations survive that table, and both are used below:
#
#   * The backdrop is the only thing that is bright **and** smooth -- local sd
#     0.8 against 16-34 for the coils, a 20-40x margin, while the equally smooth
#     ambient is 170 counts darker. That finds the region worth looking in.
#   * The drone is black where the rod is grey, a ~35-count gap. That is what
#     removes the rod, and no morphological rod model was needed: at level 80 the
#     edge-on fit is +1% on the major axis, at level 100 the rod enters and it
#     degrades to +130%.
#
# Luminance alone separates neither: the coils' 85-104 sits on top of the drone's
# 42-78, and the ambient is *darker* than the drone and lives at the frame edge,
# so an ungated threshold hulls the whole image.
BG_DIFF_THRESH = 30

# Where the empty-rig frame lives, next to the module like every other fitted
# artefact in this package.
BACKGROUND_PATH = Path(__file__).resolve().parent / "background_dark.png"

# The backdrop finder, used only when there is no background frame.
#
# Run at a quarter resolution and upsampled, which is not a nicety: at full
# resolution this costs **48.4 ms** (20 Hz) and is disqualified outright, against
# 2.43 ms at 1/4. Most of the full-resolution cost is materialising ~400k
# boundary points, so `cv2.findNonZero` is used rather than `np.nonzero`.
BACKDROP_LUM = 150
BACKDROP_SD = 12
BACKDROP_SCALE = 0.25
BACKDROP_ERODE = 15

# Threshold on 255 - luminance, so the dividing level is 255 - this: at 175 the
# body must read below **80** counts. Note the inversion when changing it -- a
# *larger* number here is a *stricter*, darker cut.
#
# **Raised from 110 on the rod**, i.e. the cut moved from "below 145" to "below
# 80". The rod's p5 is 83, so 145 admits it; the drone's median is 42-78, so 80
# keeps the drone and drops the rod. Measured on the edge-on capture, the major
# axis goes from +130% at the old level to +1% at this one. The
# old value came from a sweep on 80 rendered frames against analytic truth
# (median / p90 error in the fitted major axis):
#
#     DARK_THRESH   80     100    110    120    140    150    180
#     median      0.68%   0.46%   --    0.45%  1.03%  1.36%  5.36%
#     p90         1.55%   1.14%   --    1.93% 14.29% 30.08% 43.57%
#
# That sweep is kept for its shape -- the threshold fails by eating rim arcs, not
# by losing the robot, so detection stayed 80/80 at every level while the p90 ran
# to 30% -- but it was rendered without a rod in frame and so cannot speak to the
# choice actually being made here.
#
# **The margin is the result, not the working point.** Swept against both real
# captures, this passes over **170-215** and fails either side, so 190 is the
# midpoint rather than a value that merely works. 170 lets the rod in; above 215
# the cut eats into the drone's own shading. A value at the edge of a passing
# range is indistinguishable from a correct one until the lighting moves, which
# is why the range is recorded and not just the choice.
#
# **This is the fragile appearance of the three.** Body and ground are both
# neutral, so only brightness separates them, and the drone/rod gap is only ~35
# counts. Keep the backdrop evenly lit, and re-measure the range -- not the
# working point -- after any lighting change.
DARK_THRESH = 190

# How far a blob may sit from the largest one and still count as part of the
# robot, in multiples of that blob's own radius. See `silhouette_hull`.
#
# Above 1.0 because a rim arc's *centroid* can sit slightly outside the radius
# computed from a bounding box, and because the rim is not the only thing that
# belongs to the robot -- the magnet mount and rod tips are real parts of the
# silhouette. Measured on the face-on capture the nearest stray is at 1.76 radii
# and the furthest at 2.5.
#
# Swept against both real captures this passes over **1.0-1.7**; 1.35 is the
# midpoint. The upper bound is where the nearest stray starts to be admitted, and
# it is only 0.4 above the working point -- the tightest margin of any constant
# here, and the first one to re-measure if the coils are moved nearer.
#
# Applied to `dark` only. The bright rig's every fitted constant -- radius, tilt
# calibration, error model -- was measured without it, and a silently different
# silhouette would invalidate all of them.
DARK_MAX_SPREAD = 1.35

# Re-exported from `calib/shape.py`, which owns it: the appearance is the key the
# *calibration files* are named by, and calibration is stage 2 to this stage 3.
# Re-exported rather than referenced through `shape.` because `segment.APPEARANCE`
# is what every caller, test and notebook already reads.
APPEARANCE = shape.APPEARANCE

_BACKGROUND_CACHE = {}


def load_background(path=None):
    """The empty-rig frame, or ``None`` if one has never been captured.

    Cached by path and mtime: `valid_region` runs every frame and re-reading a
    PNG at 120 Hz would cost more than the subtraction it feeds.
    """
    path = Path(path or BACKGROUND_PATH)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    key = (str(path), stamp)
    if key not in _BACKGROUND_CACHE:
        _BACKGROUND_CACHE.clear()
        _BACKGROUND_CACHE[key] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return _BACKGROUND_CACHE[key]


def background_mask(gray, bg=None, thresh=None):
    """Everything that differs from the empty-rig frame.

    The coils, the wires, the dark ambient, the support box and the backdrop are
    all **fixed to the rig**, so one subtraction removes every one of them at
    once -- which is why this is preferred over any amount of cleverness about
    what the clutter looks like. Measured at 0.056 ms on a 1280x800 frame,
    against 2.43 ms for `backdrop_mask` and 48.4 ms for a full-resolution
    version of it, so it is the only option that leaves the loop camera-bound at
    121 fps.

    Returns ``None`` if no background frame exists or it does not match the
    frame size -- a resolution change invalidates it as surely as moving the
    camera does.
    """
    bg = load_background() if bg is None else bg
    if bg is None or bg.shape != gray.shape:
        return None
    t = BG_DIFF_THRESH if thresh is None else thresh
    return cv2.threshold(cv2.absdiff(gray, bg), t, 255, cv2.THRESH_BINARY)[1]


def backdrop_mask(gray, lum=None, sd_max=None, scale=None, erode=None):
    """The white backdrop: the one bright, smooth region, convex-hulled.

    The fallback for when no background frame has been captured. It finds where
    to *look* rather than what to reject, which is the only formulation that
    survives the measurements: the drone and the coils overlap in brightness,
    but nothing else in frame is both bright and smooth.

    **Convex hull, not hole filling.** Filling enclosed holes is the obvious way
    to re-admit the drone, which punches a dark hole in the backdrop, and it does
    not work here: the drone touches the rod, and the rod runs off the bottom of
    the frame, so drone-hole and rod-notch are one region open to the border and
    therefore not enclosed. Both `MORPH_CLOSE` and a border flood-fill return
    nothing on the face-on capture. The hull has no such failure -- and it
    re-admits the rod too, which is why `DARK_THRESH` rather than geometry is
    what removes it.
    """
    lum = BACKDROP_LUM if lum is None else lum
    sd_max = BACKDROP_SD if sd_max is None else sd_max
    scale = BACKDROP_SCALE if scale is None else scale
    erode = BACKDROP_ERODE if erode is None else erode

    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    f = small.astype(np.float32)
    # Local mean and variance as two box filters. E[x^2] - E[x]^2 can go slightly
    # negative on flat regions through rounding, hence the clamp.
    mu = cv2.blur(f, (5, 5))
    sd = cv2.sqrt(cv2.max(cv2.blur(f * f, (5, 5)) - mu * mu, 0.0))
    m = ((mu > lum) & (sd < sd_max)).astype(np.uint8) * 255
    # Open before taking components: a coil's specular highlight is momentarily
    # both bright and locally flat, and would otherwise merge with the backdrop.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # findNonZero, not np.nonzero: materialising the point list is most of the
    # cost, and at full resolution it is what made this 48 ms.
    pts = cv2.findNonZero((labels == k).astype(np.uint8))
    if pts is None or len(pts) < 3:
        return None
    out = np.zeros_like(small)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
    e = max(1, int(round(erode * scale)))
    out = cv2.erode(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (e, e)))
    return cv2.resize(out, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)


def valid_region(gray):
    """Where the robot may be, as a uint8 mask, or ``None`` if it cannot be told.

    Background subtraction when a background frame exists, the backdrop finder
    otherwise. ``None`` is a real answer and callers must honour it: with no
    valid region the alternative is an ungated threshold over the whole frame,
    and the dark ambient beyond the backdrop is *darker than the robot* and
    reaches the frame edge, so the hull spans the image and the fit is confident
    and wrong. The module docstring rejects Otsu for the same reason -- returning
    nothing is the correct answer, not a failure to produce one.
    """
    m = background_mask(gray)
    return m if m is not None else backdrop_mask(gray)


def score_channel(frame, appearance=None, thresh=None, region=None):
    """``(single_channel, level)`` where the robot is bright above ``level``.

    The one place the rig's appearance enters. Everything downstream -- morphology,
    hull, ellipse fit, sub-pixel refinement -- works on the returned channel and
    neither knows nor cares which appearance produced it.

    The channel is ``None`` when the appearance needs a valid region and none can
    be found; callers must treat that as "no detection" rather than thresholding
    anyway. ``region`` supplies one that has already been computed, so `segment`
    can keep it for the overlay without paying for `backdrop_mask` twice.
    """
    appearance = APPEARANCE if appearance is None else appearance

    if appearance == "bright":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return gray, int(THRESH if thresh is None else thresh)

    if appearance == "dark":
        # Mono is the normal case here, not a degraded one: the ELP OV9281 has no
        # colour to give and the chroma gate this used to apply was measured
        # inoperative on it. A three-channel frame is collapsed rather than split.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        region = valid_region(gray) if region is None else region
        if region is None:
            return None, int(DARK_THRESH if thresh is None else thresh)
        dark = cv2.bitwise_and(cv2.bitwise_not(gray), region)
        return dark, int(DARK_THRESH if thresh is None else thresh)

    if appearance == "red":
        if frame.ndim != 3:
            raise ValueError(
                "appearance='red' needs a colour frame; this one is single-channel. "
                "Check the capture path -- sources.CameraSource defaults to "
                "grayscale=True, which throws away the only usable signal here."
            )
        b, g, r = cv2.split(frame)
        # Saturating subtraction, so a bluer-than-red pixel clamps at 0 rather
        # than wrapping to 255 and inventing a detection.
        redness = cv2.subtract(r, cv2.max(g, b))
        return redness, int(REDNESS_THRESH if thresh is None else thresh)

    raise ValueError(
        f"unknown appearance {appearance!r}; use 'bright', 'dark' or 'red'")


def clutter_mask(frame):
    """Everything the segmenter will ignore -- the complement of `valid_region`.

    Exposed on its own because the rejected area is worth having as an object
    rather than only as something to remove. It is what the live overlay shades,
    so a wrong pose and a wrong *rejection* can be told apart at a glance -- in a
    plain ellipse overlay they look identical. It is also a direct check on
    whether the camera has moved, since every source of clutter here is fixed to
    the rig.

    Returns an all-255 mask when no valid region can be found, because in that
    case everything is being ignored, which is exactly what should be displayed.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    region = valid_region(gray)
    if region is None:
        return np.full(gray.shape, 255, np.uint8)
    return cv2.bitwise_not(region)


def segment(frame, thresh=None, min_area=MIN_BLOB_AREA_PX, subpixel=False,
            axial=None, appearance=None):
    """Threshold, clean up, hull the silhouette, fit the rim ellipse.

    ``frame`` may be grayscale or BGR.  Returns a `Segmentation`, or ``None``
    when nothing plausible is present -- callers must handle ``None`` rather
    than assume a detection, since a lost frame is normal in flight.

    ``subpixel`` re-locates the hull onto the intensity edge before fitting; see
    `subpixel_boundary`. Pass ``False`` to recover the previous behaviour, which
    is how the A/B in the journal was run.

    ``axial=False`` disables the weighted refit in `fit_ellipse`. It is a real
    parameter rather than a module flag on purpose: `AXIAL_WEIGHT_ITERS` used to
    be a default argument, so reassigning it in a test changed nothing and the
    A/B silently measured the weighted path twice.

    ``appearance`` selects how the robot is told apart from its background; see
    `score_channel`. ``thresh=None`` takes that appearance's own default level,
    since 128 is meaningful for luminance and meaningless for chroma.
    """
    t0 = time.perf_counter()

    # Computed here rather than inside `score_channel` so it can be kept on the
    # result: the overlay shades it every frame, and re-deriving it would cost
    # another 2.4 ms under the backdrop finder.
    region = None
    max_spread = None
    if (APPEARANCE if appearance is None else appearance) == "dark":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        region = valid_region(g)
        max_spread = DARK_MAX_SPREAD

    gray, level = score_channel(frame, appearance, thresh, region=region)
    if gray is None:
        # No valid region could be established -- see `valid_region`. Refusing is
        # the answer; an ungated threshold here hulls the whole frame.
        return None
    _, mask = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

    hull, area = silhouette_hull(mask, max_spread=max_spread)
    if hull is None or area < min_area:
        return None

    # Move the outline off the pixel grid before fitting anything to it.
    #
    # The hull is where a fixed threshold happened to cut a smoothly shaded
    # edge, and that cut carries scatter which nothing else predicts (lecture
    # notes 12.12). Re-locating each point at the intensity profile's
    # half-height makes the boundary a property of the image instead. Measured
    # on a soft-edged disc of known radius: bias -0.0858 -> -0.0012 px, scatter
    # 0.0550 -> 0.0030 px.
    if subpixel:
        hull = subpixel_boundary(gray, hull)

    fit = fit_ellipse(hull, axial=axial)
    if fit is None:
        return None
    ellipse, rms = fit

    return Segmentation(
        mask=mask,
        contour=hull,
        ellipse=ellipse,
        area_px=area,
        n_points=len(hull),
        fit_rms_px=rms,
        threshold=level,
        t_ms=(time.perf_counter() - t0) * 1e3,
        valid=region,
    )


def undistort_ellipse(ellipse, camera_matrix, dist_coeffs, n_samples=180):
    """Re-fit an ellipse after removing lens distortion.

    `conic.py` assumes an ideal pinhole, but the measured intrinsics carry real
    distortion (k1 = 0.136 on this camera, which moves rim pixels by a couple of
    px near the edges -- small, but it biases the recovered tilt systematically
    rather than randomly, so it is worth removing).

    Distortion does not map an ellipse to an ellipse, so the honest thing is to
    sample the perimeter, undistort the samples, and refit.  Sampling the fitted
    ellipse rather than the raw contour keeps this independent of how many
    contour points there were.
    """
    if dist_coeffs is None or not np.any(dist_coeffs):
        return ellipse

    (cx, cy), (major, minor), ang = ellipse
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    a, b, th = major / 2.0, minor / 2.0, np.radians(ang)
    xs = cx + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
    ys = cy + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)

    src = np.column_stack([xs, ys]).astype(np.float64).reshape(-1, 1, 2)
    dst = cv2.undistortPoints(src, camera_matrix, dist_coeffs, P=camera_matrix)
    return fit_ellipse_direct(dst.reshape(-1, 2))


# How strongly the ignored area is tinted. Faint on purpose: it has to be legible
# as "ignored" while leaving the pixels underneath readable, because the question
# it answers is usually "what is in there that I am throwing away?".
REJECT_ALPHA = 0.35
REJECT_BGR = (0, 0, 255)


def shade_rejected(out, region, alpha=REJECT_ALPHA, colour=REJECT_BGR):
    """Tint everything outside ``region`` in place, and return ``out``.

    A wrong pose and a wrong *rejection* look identical in a plain ellipse
    overlay -- both show an ellipse in the wrong place -- and they have opposite
    fixes. Shading what was ignored separates them at a glance, which is the only
    reason the region is carried on `Segmentation` rather than recomputed.
    """
    if region is None:
        return out
    off = region == 0
    if not off.any():
        return out
    out[off] = (out[off] * (1.0 - alpha) + np.array(colour, np.float32) * alpha).astype(np.uint8)
    return out


def draw(frame, seg, colour=(0, 255, 0), rejected=True, normal_px=None):
    """Overlay the fitted ellipse, its axes and the ignored area on a copy.

    Converts grayscale to BGR first so the overlay survives -- the same trap
    `servo.py:199` documents.  Returns a new image; the input is untouched.

    ``rejected`` shades the area outside the valid region; ``normal_px`` draws the
    rotor axis as an image-space segment ``((x0, y0), (x1, y1))``, which only the
    caller can compute since it needs the camera matrix and the 3-D pose.
    """
    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    if seg is None:
        cv2.putText(out, "no detection", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    if rejected:
        shade_rejected(out, seg.valid)

    cv2.ellipse(out, seg.ellipse, colour, 1)
    (cx, cy), (major, minor), ang = seg.ellipse
    th = np.radians(ang)
    for half, c, col in ((major / 2, (np.cos(th), np.sin(th)), colour),
                         (minor / 2, (-np.sin(th), np.cos(th)), (255, 160, 0))):
        p0 = (int(cx - half * c[0]), int(cy - half * c[1]))
        p1 = (int(cx + half * c[0]), int(cy + half * c[1]))
        cv2.line(out, p0, p1, col, 1)
    if normal_px is not None:
        (nx0, ny0), (nx1, ny1) = normal_px
        cv2.arrowedLine(out, (int(nx0), int(ny0)), (int(nx1), int(ny1)),
                        (255, 255, 0), 2, tipLength=0.15)
    cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    cv2.putText(
        out,
        f"maj={major:.1f} min={minor:.1f} rms={seg.fit_rms_px:.2f}px",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        colour,
        1,
    )
    return out
