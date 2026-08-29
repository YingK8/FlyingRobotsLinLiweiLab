"""
Segment the robot from its background and fit the rim ellipse.

Two rig appearances, selected by `APPEARANCE` / `score_channel`:

    bright  white robot on a dark ground; every calibration constant here was
            fitted on it
    dark    black robot on a white backdrop, coils and room in frame

Both are monochrome, so brightness and smoothness are the only separations
available. Everything after `score_channel` is appearance-agnostic: it works on
whichever single channel makes the robot bright.

`dark` needs more than a channel and a level, because the coils, wires and room
are all darker than the backdrop too. It adds `valid_region` (where the robot may
be at all) and a spread limit in `silhouette_hull` (which blobs inside that region
are one object).

**Hull, not largest contour.** `conic.py` needs the duct *rim*, and the duct is a
thin ring: face-on its outer wall is nearly edge-on, so lighting breaks it into
arcs and the largest blob becomes the blade cross. Measured face-on,
largest-contour fits 83 px where the rim is 131 -- a 37% underestimate landing
straight on depth. Pooling every real blob and hulling gives 129.9 px against
130.7 analytic.

**Axial weighting is what makes high tilt work.** The rod and magnet stick out
along the rotor axis, so tilt pushes them outward in the silhouette's *short*
direction, near the middle of the major axis. Weighting each hull point by
``w = |proj| / a`` (`AXIAL_WEIGHT_POWER`) suppresses them: at 70 deg tilt the
minor axis goes 75.6 -> 54.4 px against 45.0 true. The floor must be **zero** --
at 0.05 the rod points still dominate (68.1 px).

Threshold, not Otsu: with the robot out of frame Otsu thresholds noise into a
confident detection, where a fixed level returns nothing.

Approaches tried and rejected (tilt-adaptive weighting, residual trimming,
sub-pixel refinement), with the numbers: `ai/notes/pose_appearance.md`.
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
# A lower level catches more of the dim rim edge and localises it better. That column
# said so and it was left at 128 anyway, because the renders it came from have one
# brightness and the useful level on hardware follows exposure.
#
# **Refitted on the black backdrop, and it was worth much more than the bias column
# suggested.** With a dark ground there is almost nothing above the level except the
# robot -- a level of 72 keeps 80.5% of rim samples and 1.3% of the backdrop, against
# 78.0% and 1.1% at 128, so two thirds of the range between them buys rim and costs
# nothing. It only matters as a *seed*: the direct fit measures from the evidence map,
# and a seed missing a quarter of the rim starts the solve on a biased ellipse.
#
# Swept over three flights, `never_reject`, with the frames landing more than 30 degrees
# from the flight's own median normal as the headline -- `2026-08-28_131552` is flown
# upright so that column is ground truth:
#
#   THRESH                  128    96     72     56
#   131552  >30 deg out     10.2%  4.1    3.1    5.3
#           under 5 mm      76.1%  72.1   77.6   66.2
#   135533  >30 deg out     2.8%   1.4    0.4    0.6
#           under 5 mm      94.5%  92.7   92.9   94.0
#   092117  >30 deg out     4.3%   4.0    2.9    4.6
#           under 5 mm      77.3%  73.9   81.5   83.2
#           discrepancy p90 37.18  17.84  15.30  15.30
#
# 72 is best or tied on every flight and on both columns, and `ridge` holds or rises
# with it (17.4 -> 18.6 on 092117), which says the extra seed area is rim rather than
# backdrop. Passes over 56-96; below 56 the backdrop starts arriving -- at 32 a level
# keeps 17.5% of it against 2.1% at 40.
#
# Change this (or the exposure) and refit the effective radius: the bias column above
# moves 0.75% across its range, more than the whole depth residual. The renders still
# use 128 by way of `estimator.RADIUS_BY_APPEARANCE["bright"]`, which was fitted with
# it; this constant and that one move together.
THRESH = 72
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

# A contrast-relative level (scaled to the frame's peak) was tried and dropped:
# 58 mm instead of 64 on the face-on hard-side-light failure it targeted, because
# there the rim is not dim but unlit, and it made every well-lit case worse by
# pulling fringe pixels into the hull. That failure is fixed by lighting.

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

# Axial weighting is ON. Controlled A/B (same seed, 400 poses, same gate and
# constants): no gate coverage lost, both error metrics better in every sensor
# mode -- 1280x800 position 0.268 -> 0.186 mm, angle 0.294 -> 0.178 deg.
#
# **It costs latency, and that trade is open.** End to end, 2.31 -> 3.53 ms/frame,
# i.e. 433 -> 283 Hz. Two thirds of that is the two later robustness stages, not
# the weighting:
#
#   base fit                    0.039 ms
#   + axial re-weighting (x2)   0.19 ms
#   + one-sided pass (x3)       0.375 ms     POSE_ONE_SIDED=0 disables
#   + trim pass (x3)            0.42 ms      TRIM_FRACTION=0 disables
#
# `test_stereo::test_speed` fails on this: the two-view solve is 3.40 ms against
# 4.17 ms at 240 Hz. Left visible on purpose. The cameras deliver 15-28 fps, so
# it binds only at the aspirational rate.
#
# ``POSE_AXIAL=1``/``=0`` overrides per process, so an A/B runs as two
# subprocesses rather than by editing this line.
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
# Contamination is one-sided by construction: the hull is a superset of the
# projected rim, so the rod, mount and threshold spill push the outline outward
# only. A symmetric loss splits the difference with it and settles outside the rim
# by about half the excursion; weighting outward points down pulls the fit onto the
# inner envelope, where the rim is. The asymmetry matters more than the value.
#
# This buys **spread**, not bias -- `TiltCalibration` already removes the mean.
# Over 120 poses with tilt and contamination amplitude varied:
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
# A down-weighted point still pulls the fit; a discarded one does not. The mast is
# a *localised lobe* rather than spread-out error, so selection suits it better
# than weighting: on silhouettes with a randomly placed lobe the ratio-error
# scatter falls 0.03586 -> 0.02266, a **37%** reduction.
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
    """
    One frame's segmentation result.

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
        ``valid_from`` names which of the two produced it, because only one of them is
        worth drawing: see `shade_rejected`.

        ``evidence`` and ``ellipse_mask`` appear once `fit_ellipse_image` has been run:
        ``ellipse`` is then the direct fit and ``ellipse_mask`` the threshold's own
        answer, kept so the overlay can show both and the difference between them can
        be measured. See `theory.md` 16.
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
    valid_from: str | None = None
    # Mean rim evidence, set only when `fit_ellipse_image` has refined `ellipse` onto
    # the image. Finite means the ellipse is a direct fit rather than a fit to the
    # mask, so `fit_rms_px` -- which measures the *hull* -- no longer describes it and
    # callers must gate on this instead. NaN means the mask fit stands.
    evidence: float = float("nan")
    coverage: float = float("nan")
    ridge: float = float("nan")
    ellipse_mask: tuple | None = None


@dataclass
class RingFit:
    """
    What `fit_ellipse_image` recovered, and how well the rim supports it.

        ``coverage`` is the fraction of samples carrying at least
        `RING_COVERAGE_FLOOR` of the ring's **own** median evidence, so it says how
        much of the rim the fit is actually resting on without reference to any
        absolute level -- a ring at half the contrast scores the same.

        ``ridge`` is the blunder test, and coverage is **not** -- that was measured
        wrong once and it is worth saying why. Coverage normalises by the ring's own
        median, so a fit resting on nothing still clears "half of nothing" and scores
        ~0.5. Two frames both scored 0.52: one was a well-fitted ring two thirds hidden
        behind the rig, the other had the whole rim in view and an ellipse plainly too
        big for it. Coverage cannot tell those apart, and they need opposite answers.

        `ridge` compares the fitted curve against curves just inside and outside it
        (`RING_SHOULDER`), so it asks the question that actually matters: **is this
        ellipse sitting on a ridge?** On those same two frames it reads 27.3 and 0.60.
        Local, so a shadowed arc is judged against its own surroundings rather than
        the frame's, and free -- two more `sample_map` calls.

    """

    ellipse: tuple
    evidence: float
    coverage: float
    ridge: float = float("nan")


def fit_ellipse_direct(pts):
    """
    Fitzgibbon direct fit of ``pts``, axes ordered major-first.

        Pairing `cv2.fitEllipseDirect` with `conic.normalise_ellipse` is not
        optional: an unnormalised result reports its angle against whichever axis
        OpenCV happened to call ``width``, so the angle jumps 90 degrees on a nearly
        circular silhouette. That pairing was written out four separate times;
        separating the two calls is the mistake this exists to make impossible.

        It lives here rather than in `conic.py` because that module is deliberately
        OpenCV-free -- pure geometry over numpy -- and this is the OpenCV side.
    """

    return conic.normalise_ellipse(
        cv2.fitEllipseDirect(np.asarray(pts, dtype=np.float32))
    )


def sampson_distance_conic(c, pts):
    """
    First-order geometric distance from points to a conic, in pixels.

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
    """
    `sampson_distance_conic`, starting from an OpenCV ellipse.
    """

    return sampson_distance_conic(conic.conic_from_ellipse(ellipse), pts)


def axial_weights(pts, ellipse, power=AXIAL_WEIGHT_POWER, floor=AXIAL_WEIGHT_FLOOR):
    """
    Weight each point by how far along the major axis it lies.

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
    s = np.abs((np.asarray(pts, dtype=np.float64) - np.array([cx, cy])) @ u) / (
        major / 2.0
    )
    return floor + (1.0 - floor) * np.clip(s, 0.0, 1.0) ** power


def angular_coverage(ellipse, pts, bins=TRIM_COVERAGE_BINS):
    """
    Fraction of the ellipse's perimeter that has points near it.

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
    """
    Sampson distance keeping its sign: positive is outside the ellipse.
    """

    c = conic.conic_from_ellipse(ellipse)
    ph = np.hstack(
        [np.asarray(pts, dtype=np.float64).reshape(-1, 2), np.ones((len(pts), 1))]
    )
    alg = np.einsum("ij,jk,ik->i", ph, c, ph)
    grad = 2.0 * (ph @ c.T)[:, :2]
    return alg / np.maximum(np.linalg.norm(grad, axis=1), 1e-12)


def _outward_weights(pts, ellipse, w_out=ONE_SIDED_WEIGHT, power=AXIAL_WEIGHT_POWER):
    """
    Axial weights, further reduced for points lying outside ``ellipse``.

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
    """
    Direct ellipse fit, normalised to major-axis-first.

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


def subpixel_boundary(
    gray, pts, centre=None, search_px=SUBPIX_SEARCH_PX, n_samples=SUBPIX_SAMPLES
):
    """
    Move each boundary point onto the intensity edge, at sub-pixel precision.

        A threshold-and-hull outline is quantised to the pixel grid, and where it
        lands depends on how the threshold happened to slice a shaded, blurred edge.
        Along the outward normal the intensity ramps from robot to background, and its
        half-height crossing is the geometric edge whatever threshold found it
        approximately. Interpolating that crossing is sub-pixel and is a property of
        the image rather than of a constant.

        **Off by default: it does not fix what it was written for.** On a synthetic
        disc it is 18x better (scatter 0.0550 -> 0.0030 px); in the real pipeline it
        buys 1-4% of the tilt residual, for 0.5 ms/frame against a 2.4 ms budget at
        420 Hz. The synthetic test measures how precisely a *known* edge can be
        located; the real variability is *which shape presents itself*. Kept because
        the comparison should stay repeatable. `ai/notes/pose_appearance.md`.

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
    u = d / np.maximum(n, 1e-9)  # outward unit normals

    # Sample the profile along each normal in one batched remap.
    t = np.linspace(-search_px, search_px, n_samples, dtype=np.float64)
    xs = pts[:, None, 0] + u[:, None, 0] * t[None, :]
    ys = pts[:, None, 1] + u[:, None, 1] * t[None, :]
    prof = cv2.remap(
        g,
        xs.astype(np.float32),
        ys.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

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
    """
    Every boundary pixel of the kept blobs, as an (N, 2) array.

        `silhouette_hull` keeps only hull *vertices* -- 9-31 points on the real
        captures against 192-3072 in the contour they came from. That costs the
        redundancy robust estimation needs: measured across densities, trimming half
        the points to suppress the mast helps above ~32 points and **hurts** below,
        and hull density sits on the wrong side of that line.

        One-sidedness survives without the hull. The silhouette is the rim *union* the
        rod and mount, so its outer contour follows the rim except where those
        protrude and never falls inside it -- which is the property the robust step
        relies on.

        @return: ``(points, area_px)``, or ``(None, 0.0)``. Use `silhouette_hull` when
            a closed shape is needed; this is for fitting, where it is not.
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
    contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

# **This gate is close to inert, and tightening it makes things worse.** The score is
# fit error against the convex HULL, and any compact solid blob's hull is an ellipse:
# a rim and a solid disc both measure 0.0004. It cannot tell them apart, so on a normal
# frame nothing is rejected and selection below falls through to size alone.
#
# Left loose deliberately. Swept on both 18:48 flights at threshold 180, gating on
# radial error over the group's mask PIXELS (which does separate ring from blob,
# 0.027 against 0.33) and breaking ties on inlier count, the result improves
# monotonically as the gate is widened -- 184940 solves 23% at tol 0.09, 30% at 0.20,
# 39% at 0.40 and 45% with no gate at all, against 44% for plain size. Every millimetre
# the error criterion is allowed to reject something, the answer gets worse.
#
# The reason is what the candidates actually are. They are not rim-versus-clutter,
# where fit error would win; they are partial-rim-versus-larger-partial-rim, all
# genuinely elliptical and differing only in how much of the ring they hold. Fit error
# cannot rank those and size can, because more of the ring is bigger. Selecting on
# error alone is worse still (8% solved): error has no support term, so its minimum is
# always the smallest candidate -- the roundest blob against hulls, the thinnest arc
# against pixels, since an ellipse follows a short arc almost exactly.
#
# Contamination inflating the major axis was the real objection to size, and `_regrow`
# is what answers it: reassembling the ring makes the correct group the biggest one.
#
# **Anchor-and-grow still cannot reject.** `_regrow` only ever adds blobs, so scenery
# once absorbed is never shed -- on 184830 view A frame 420 this returns 607x502 where
# the rim is 463x278, having eaten the shadow blobs beside it. Backward elimination
# (start from every blob, drop the smallest, refit, keep the set that best explains the
# mask) does not have that failure and is the more accurate extractor measured against a
# leave-one-out consensus ground truth: worst-case centre error 2.7 px across both
# 18:48 flights against 3.6 px here, and 2.0 px against 3.6 on 184940. It is not in this
# file because it scores WORSE on the stereo solve for that same flight (23% against
# 44%), and see the warning below for why that number cannot be trusted right now.
#
# **Cross-view discrepancy is not a valid objective while the extrinsic is stale.**
# Extraction accuracy and stereo agreement came out anti-correlated across the two
# flights: the more accurate extractor loses the stereo solve, because a biased ellipse
# partially cancels a wrong rig and a clean one exposes it. Every millimetre figure used
# to tune this module was measured that way, `RING_BAND` included -- re-derive it after
# `calib/calibrate.py` succeeds. Full study: results/rim_extractor_trials.html.


def _group_hull(labels, members, n, stats=None):
    """
    Convex hull of a set of labels, as float64 (N, 2), or ``None``.

        ``stats`` from `connectedComponentsWithStats` crops the scan to the members'
        own bounding box, which is the whole cost of this function -- `lut[labels]`
        materialises a full-frame array and `findContours` then walks it. Grouped blobs
        are clustered by construction, so the box is a fraction of the frame: 1.9 ms
        to 0.1 ms on a 1280x800 label image, and `_best_group` calls this eight times.
    """

    lut = np.zeros(n, dtype=np.uint8)
    lut[members] = 255

    x0 = y0 = 0
    view = labels
    if stats is not None and len(members):
        m = np.asarray(members)
        x0 = int(stats[m, cv2.CC_STAT_LEFT].min())
        y0 = int(stats[m, cv2.CC_STAT_TOP].min())
        x1 = int((stats[m, cv2.CC_STAT_LEFT] + stats[m, cv2.CC_STAT_WIDTH]).max())
        y1 = int((stats[m, cv2.CC_STAT_TOP] + stats[m, cv2.CC_STAT_HEIGHT]).max())
        view = labels[y0:y1, x0:x1]

    contours, _ = cv2.findContours(
        lut[view], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    if len(pts) < _MIN_CONTOUR_PTS:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    if len(hull) < _MIN_CONTOUR_PTS:
        return None
    return hull + (x0, y0)


# How far off its own fitted ellipse a blob may sit and still be counted part of the
# ring, as a fraction of the local radius: |r - 1| < this, with r the blob centroid's
# radius in the ellipse's own frame.
#
# This exists because `max_spread` measures the wrong thing once the rim breaks up.
# That radius comes from the *anchor blob's* bounding box, which is a whole rim when
# the threshold keeps the rim solid and a single thin arc when it does not -- and
# 1.35 arc-radii cannot reach across the ring to the arcs on the far side. On flight
# 184940 at threshold 180 no candidate group spanned the real 382 px rim; the best
# covered 314 px and the rest were quadrant-sized.
#
# Re-gathering on the fitted ellipse instead asks the question the shape actually
# poses -- is this blob on the ring -- and is scale-free, so it works the same whether
# the rim arrived in one piece or twelve.
#
# Swept on both 18:48 flights, every sixth stereo frame through the two-view solve,
# at threshold 180. Solved-frame % on 184940: 37 at 0.20, 43 at 0.35, **45 at 0.50**,
# 43 at 0.65, 42 at 0.80, 38 at 1.00, and 32 both for the old distance grouping and
# for the no-grouping control. An interior peak, not a range edge, and it beats
# keeping every blob -- which the old rule did not.
#
# 184830 reads 45% at every setting including no grouping at all, so this buys nothing
# there. That flight is limited by its extrinsic, not by segmentation.
#
# **It cannot bootstrap from a single blob.** One arc does not determine the ring, so a
# rim shattered into isolated fragments still fails -- the seed has to span enough of
# the ring to fit an ellipse near the true one first. That is why the seed is still the
# distance rule and this only extends it.
RING_BAND = 0.5
RING_REGROW_ITERS = 2


def _on_ring(ellipse, pts, band=RING_BAND):
    """
    Which of ``pts`` sit on ``ellipse``, as a boolean mask.

        Radius in the ellipse's own frame, so the test is one number regardless of how
        tilted or how large the ellipse is.
    """

    (cx, cy), (major, minor), ang = ellipse
    a, b = major / 2.0, minor / 2.0
    if not (a > 0 and b > 0):
        return np.zeros(len(pts), dtype=bool)
    th = np.radians(ang)
    c, sn = np.cos(th), np.sin(th)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    x, y = dx * c + dy * sn, -dx * sn + dy * c
    return np.abs(np.hypot(x / a, y / b) - 1.0) < band


def _regrow(members, keep, labels, centroids, n, stats=None):
    """
    Extend a group with every kept blob lying on its own fitted ellipse.

        Converges in one or two passes: a group holding half the ring already fits an
        ellipse close enough to find the other half, and the pass after that adds
        nothing. Returns the members unchanged if no ellipse can be fitted.
    """

    for _ in range(RING_REGROW_ITERS):
        hull = _group_hull(labels, members, n, stats)
        if hull is None:
            return members
        try:
            ellipse = fit_ellipse_direct(hull)
        except (cv2.error, ValueError, np.linalg.LinAlgError):
            return members
        if not np.isfinite(ellipse[1][0]) or ellipse[1][0] <= 0:
            return members
        grown = np.union1d(members, keep[_on_ring(ellipse, centroids[keep])])
        if len(grown) == len(members):
            break
        members = grown
    return members


def _best_group(keep, labels, stats, centroids, n, max_spread):
    """
    Which blobs belong to the robot, by which grouping is most elliptical.

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
        Ranking on shape alone fails because a solid disc is a *perfect* ellipse, so
        round clutter outscores the robot: on the synthetic scene it returns a 64 px
        coil over the 125 px robot. Real candidates clear the tolerance easily (the
        robot fits at rms/major 0.006 on both ELP captures), so the tolerance rejects
        groups spanning two objects and size picks among the survivors.
    """

    order = keep[np.argsort(stats[keep, cv2.CC_STAT_AREA])[::-1]][:_MAX_ANCHORS]
    admissible = []
    best, best_score = None, np.inf
    for anchor in order:
        # The anchor's own half-diagonal, so the scale comes from the object
        # rather than a pixel constant that would not survive a resolution change.
        radius = 0.5 * float(
            np.hypot(
                stats[anchor, cv2.CC_STAT_WIDTH], stats[anchor, cv2.CC_STAT_HEIGHT]
            )
        )
        if radius <= 0:
            continue
        d = np.hypot(*(centroids[keep] - centroids[anchor]).T)
        # Distance from the anchor only seeds the group; the ring itself decides the
        # rest. See RING_BAND for why the seed alone is not enough.
        members = _regrow(
            keep[d <= max_spread * radius], keep, labels, centroids, n, stats
        )
        hull = _group_hull(labels, members, n, stats)
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
        # Largest, not best-scoring. Best-scoring is the fallback below and only runs
        # when nothing is admissible -- see SHAPE_TOL for why that ordering is right
        # and what the alternatives measure.
        return max(admissible, key=lambda t: t[0])[1]
    return best


def silhouette_hull(mask, keep_fraction=_BLOB_KEEP_FRACTION, max_spread=None):
    """
    Convex hull of every blob worth keeping, as an (N, 2) array.

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

        Distance separates them with room to spare: strays sit 1.76-2.5 robot radii
        from the centroid, while a rim arc cannot exceed one radius by definition.
        **But the anchor cannot be the largest blob** -- the rim is hollow, so solid
        clutter outweighs it (1783 px ring against 3225 px coil), and anchoring on
        area returns a confident fit to the wrong object. Anchor on shape instead.
        Off by default: the bright rig's constants were fitted without any of this.
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

    contours, _ = cv2.findContours(
        kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0

    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float32)
    if len(pts) < _MIN_CONTOUR_PTS:
        return None, 0.0

    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    return hull, float(areas[keep - 1].sum())


# --- "dark": a black body on a white ground, with clutter in frame ---
#
# Inverting the threshold is most of it, not all. Coils, wires and the room beyond
# the backdrop are all dark against a white ground, and `silhouette_hull` pools
# every blob into one hull -- so clutter does not add a stray contour, it swallows
# the rim.
#
# Measured on the real captures, which is what the constants below are fitted to:
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
# **Raised from 110 on the rod**: the cut moved from "below 145" to "below 80".
# The rod's p5 is 83, so 145 admits it; the drone's median is 42-78, so 80 keeps
# the drone and drops the rod. On the edge-on capture the major axis goes from
# +130% to +1%.
#
# The old value came from a render sweep with no rod in frame, so it cannot speak
# to this choice. Its shape is still worth knowing: the threshold fails by eating
# rim arcs, not by losing the robot, so detection stayed 80/80 at every level while
# p90 ran to 30% (`ai/notes/pose_appearance.md`).
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
# Re-measured on three flights against the empty-rig plate (`background.from_video`),
# every eighth stereo frame through the full estimator: 190 recovers 24-30% of frames,
# 210-220 recovers 43-59%, 230 falls away again. The gap is where the rim's near-black
# stops being separable from the shadowed scene behind it, and 190 was cutting the rim
# itself. 220 wins on two of the three takes and is a point behind on the third.
# Re-measured 2026-08-26 on the two 18:48 flights, every sixth stereo frame through
# the full two-view solve, scored on median cross-view discrepancy. Both flights vote
# 180 independently, over a flat 170-190: 31.8 and 32.7 mm, against 43.6 / 37.2 at 150
# and 38.9 / 120.4 at 220. The lighting moved between sessions -- at 220 camera B's rim
# stops being separable at all and the fit lands on specks (167x61 px where the ring is
# circular), which is what the mask overlay shows at a glance.
#
# **The 32 mm floor was never this constant's fault, and it was not the extrinsic
# either.** That was the reading here, and it was wrong: it blamed the rig because no
# threshold reached the gate. What reaches it is not thresholding at all. With
# `fit_ellipse_image` refitting the ellipse onto the rim evidence, the frames whose fit
# actually covers the rim agree across views to a **median 4.1 mm** -- against a 25.5 mm
# gate -- so the extrinsic was sound the whole time and the floor was the mask.
#
# **This level is now a seed, not a measurement**, and that changes what it is for: it
# has to land the direct fit's starting ellipse within its capture radius, not cut the
# rim in the right place. Re-swept on that basis, end to end through the estimator over
# all three flights, as frames solved:
#
#   100   33.3 / 12.3 / 14.7 %      190   76.0 / 54.0 / 43.7 %
#   150   49.1 / 34.0 / 23.6 %      220   49.7 / 23.4 / 44.2 %
#
# 190 wins on every flight, which is the value the sweeps above and `theory.md` already
# shipped -- the 100 it replaced came from tuning the mask as a measurement, back when
# it was one. Median cross-view discrepancy sat at 3.3-4.0 mm at every level in that
# table (swept before `RADIUS_BY_APPEARANCE` was re-fitted): the level decides how often
# the fit starts close enough to converge, and no longer decides how good the answer is.
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
    """
    The empty-rig frame, or ``None`` if one has never been captured.

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
    """
    Everything that differs from the empty-rig frame.

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


# Rim evidence: how far a pixel stands out from the envelope around it.
#
# A morphological closing with a structuring element of width k fills every dark
# feature narrower than k, so `closing - image` responds **only to dark structures
# thinner than k**. That is the whole shadow/rim discriminator, and it is the one
# separation a level cannot make: the rim is ~8 px thick and scores full, a cast
# shadow is broader than the kernel, survives the closing, and scores ~0.
#
# The `bright` appearance is the mirror image -- `image - opening`, a top-hat -- and it
# is the *rig-side* fix for the same problem: a shadow on a black backdrop has nothing
# to darken, so it never enters the map at all. The kernel argument is unchanged.
#
# Measured over 20 frames per camera on both 18:48 flights, as mask area in pixels:
#   level on 255-luminance   24k -> 154k as shadows come and go
#   this, thresholded at 40  22k -> 36k, and 2.6k on the frames with no robot in view
# The rim itself is ~28k px, so the level was spending most of its mask on shadow.
#
# **MORPH_RECT, not MORPH_ELLIPSE.** Rect is separable and ellipse is not: 2.64 ms
# against **40.2 ms** at 1280x800. Same trap as the full-resolution backdrop finder,
# and the same disqualification -- 40 ms does not fit a 16.7 ms frame.
#
# **What looks like occlusion at this width is mostly this constant, and widening it
# does not help.** The rim is ~8 px of *stroke*, but what the kernel has to exceed is
# the whole bright structure the stroke sits in, and the projected ring is not 8 px wide
# everywhere. At the ends of the projected minor axis the near and far arcs of the duct
# converge into a locally wide bright blob, the opening keeps it, and the top-hat
# returns nothing. Measured on `2026-08-28_131552`, dead-sample rate around the fitted
# ellipse: 9-11% over most of the curve, 27% at the major-axis ends, and **49.7% within
# 20 degrees of the minor-axis ends** -- on rim whose raw luminance reads 139.6 against
# 157.9 where it *is* found. The rim is bright and present and the map does not see it.
#
# A wider kernel fixes that and costs more than it is worth, because it stops
# discriminating. Arc coverage goes to 1.000 and every end metric gets worse:
#
#   k                      41     49     61     81
#   arc alive p50          0.861  --     1.000  1.000
#   at the minor tips      0.567  --     1.000  1.000
#   discrepancy p50, mm    0.77   1.92   2.39   ~3
#   under the 5 mm gate    72.9%  59.8   55.4   ~53
#   more than 30 deg out   8.8%   18.3   13.2   ~14
#   ridge p50              35.8   29.0   23.7   --
#
# `ridge` is the row that explains the rest, and it falls monotonically: a kernel wide
# enough to keep the converged arcs also keeps the body, so the map stops being a ridge
# on the rim and the ellipse is no longer pinned to anything. This is `coverage` failing
# as a blunder test exactly as documented on `RingFit` -- everything is "alive" when the
# whole neighbourhood is bright. 41 stands. Recovering the converged arcs needs a
# growth rule at constant kernel, not a wider kernel.
RING_KSIZE = 41

# How much of the plate's own response to subtract.
#
# **Not 1.0.** The plate removes static thin structures shaped like a rim, and on a
# black backdrop there are few left to remove -- while a `RunningPlate` on a *hover* rig
# converges on a robot that is barely moving and starts subtracting the robot.
# `background.from_video`'s docstring already warns that a take where the robot hovers
# leaves itself in the plate; the running form has the same failure and reaches it
# faster, at a count per frame. It costs real rim: arc alive p10 on `2026-08-28_131552`
# reads 0.603 at full subtraction, 0.667 at half and 0.703 with no plate at all.
#
# It is a trade, not a free win, and the three flights do not fully agree:
#
#                        plate x0.50            plate x1.00
#   131552   poses       639/639                627/639
#            under 5 mm  76.1%                  72.9%
#            p90         34.83 mm               31.38
#   135533   under 5 mm  **94.5%**              90.0%
#            p90         **1.51 mm**            4.94
#   092117   under 5 mm  77.3%                  78.8%
#            p90         37.18 mm               **17.23**
#   ridge p50, all three 17-24                  24-36
#
# 0.5 answers on every frame of all three where full subtraction loses 12, takes the
# best average under the gate, and turns in the single largest improvement anywhere
# here -- 135533's tail by a factor of three. `092117` prefers full subtraction on its
# tail and is the earliest take on this backdrop, which is the case for keeping half the
# correction rather than dropping it. The `ridge` column is the honest cost: the map is
# a slightly weaker ridge with less of the plate taken out of it, though still ten times
# `MIN_RING_RIDGE` at the median.
RING_PLATE_WEIGHT = 0.5

# Gaussian sigma applied to the response. Not cosmetic: `stereo.refine(mode="image")`
# differentiates this map numerically, and an unblurred black-hat is a ridge one
# pixel wide with no gradient to follow more than a pixel away. Roughly the rim
# half-thickness, which is what sets the capture radius of the direct fit.
RING_BLUR_SIGMA = 3.0

# Points sampled around the predicted rim, per view, by the direct fit. Fixed rather
# than scaled to the perimeter -- see `ellipse_points`.
RING_SAMPLES = 180

# Where the fit's reference level sits in the seed's own ring evidence. Every sample
# is scored as a deficit from this, so it decides what "on the rim" means. High
# enough that most of a good ring sits below it and the residual has a gradient
# everywhere; below the maximum, so one specular sample cannot set the bar.
RING_REF_PERCENTILE = 90

# Relative step for the numerical Jacobian, and it is **not** a detail to leave at
# the default. `least_squares` steps by ~1.5e-8 relative, which on a centre near 320
# is 5e-6 px: the objective is a bilinearly sampled image, so that reads gradient off
# one interpolation cell and the direction is noise. At 1e-3 the step is ~0.3 px,
# about a third of the blur, and the fit converges. Left at the default it walked 35
# px off a synthetic ring it had a good seed for.
RING_DIFF_STEP = 1e-3

# Extra blur for the coarse pass, in pixels. Sets the capture radius: roughly this
# far from the rim there is still a gradient pointing at it. Large enough to cover
# the measured seed error, small enough that the coarse answer lands inside the fine
# pass's own basin. Zero disables the two-stage fit.
RING_COARSE_SIGMA = 9.0

# What counts as a sample "on the rim", as a fraction of the ring's own median
# evidence. Half: the rim's response varies by a factor of two around a good ring --
# lighting, and the wall going edge-on -- so a tighter floor measures the lighting
# rather than the fit, and a looser one counts samples that are on nothing.
RING_COVERAGE_FLOOR = 0.5

# How far inside and outside the fitted curve `ridge` looks, as a fraction of the axes.
# Far enough to clear the rim's own width at the sizes seen here (the rim is ~8 px on a
# 400-500 px major, so 0.14 lands ~30 px off it), close enough to stay on the same
# lighting. The statistic is a ratio, so its exact value matters less than that both
# shoulders miss the rim.
RING_SHOULDER = 0.14

# **Bounding how far the fit may move from its seed was tried and is a net loss.** The
# motivation was good: over 2026 views of `2026-08-28_135533` a healthy refinement
# shrinks the major 2.8% (the mask hulls the rim's outer edge, the evidence ridge is its
# centre-line) and moves the centre 4.0 px, p90 6.8 -- while on `2026-08-28_131552` the
# same medians hold and the tails do not, major p10 -11.2%, centre p90 55 px and max
# **466 px**, from frames where the mask seed collapsed to a 14 px major on a ~400 px
# ring. Rejecting those and reporting the seed instead:
#
#   drift / scale bound    off     0.50/0.40  0.30/0.25  0.10/0.15
#   131552  discrepancy p90  26.58   26.58      26.58      11.22
#           more than 30 deg 4.4%    4.2        4.7        8.6
#   135533  discrepancy p90  1.54    5.01       5.01       4.69
#           more than 30 deg 0.6%    3.1        3.1        2.8
#
# Only the tightest bound helps anything, only on one flight, and it costs orientation
# on both. The reason is that a large move is not the same as a wrong move: a seed at
# 252 px on a 415 px ring needs a +65% correction, and that is the fit doing exactly its
# job. Drift does not separate recovery from runaway, and the seed is the worse report
# either way -- its axis ratio is biased by hulling the outer edge, which lands straight
# on the tilt.

# The seed level, as a fraction of the evidence map's own `RING_SEED_PERCENTILE`.
#
# A percentile rather than the maximum, so a single specular pixel cannot set the scale.
# Swept on the black-backdrop flight: 0.25-0.35 all give major 452-456 px at ratio 0.91
# on the same frame, 0.15 lets the cloth folds in (major 527) and 0.50 starts eating the
# rim (398). 0.30 is the midpoint of the range that passes.
RING_SEED_FRACTION = 0.30
RING_SEED_PERCENTILE = 99.9

# How far a blob may sit from the largest and still count as part of the robot, in
# multiples of its radius. The same rule and the same value as `DARK_MAX_SPREAD`, which
# it replaces on this path -- the justification there is about the rim being hollow and
# arriving as arcs, which is a property of the robot and not of the appearance.
RING_MAX_SPREAD = 1.35

# Normalising the response by the local bright envelope -- `(closing - g) / closing`,
# the multiplicative shadow model -- was tried and **rejected**. The intent was to
# even out the evidence where a shadow falls across the rim and lowers its contrast.
# It does not: along-ring p10 as a fraction of the median went 43% -> 51% on one
# frame and was unchanged or worse on the other thirteen. The evidence around a rim
# is intrinsically uneven, and normalising the map does not change that -- what
# handles it is the robust loss in the direct fit, not a better map.


def ring_weight(gray, background=None, ksize=None, sigma=None, roi=None,
                appearance=None, plate_weight=None):
    """
    Rim evidence as a float32 map: bright on the thin rim, ~0 elsewhere.

        The direct fit's whole input. **Never thresholded** -- a weak arc is meant to
        contribute little, not to be decided about, which is the point of dropping the
        binary mask (`theory.md` 16).

        ``background`` subtracts the plate's own response, which removes the static
        thin dark lines between the coil formers. Those are the one thing shaped like
        the rim, so the plate is what tells them apart, not the kernel.

        ``roi`` is an ``(x, y, w, h)`` window to compute in, for when a predicted pose
        says where to look: 0.37 ms on 450x450 against 2.64 ms full-frame. The map is
        returned full-size with zeros outside, so sampling coordinates never shift.

        ``appearance`` sets the polarity, and it is the only appearance-aware thing in
        this path -- everything downstream reads the map and neither knows nor cares.
        ``dark`` (dark rim) takes the black-hat, ``closing - image``; ``bright`` (light
        rim on a dark ground) takes the top-hat, ``image - opening``. Same kernel, same
        argument, mirrored: an opening removes light features narrower than ``k``, so
        subtracting it leaves only those. Defaults to `APPEARANCE`.
    """

    k = RING_KSIZE if ksize is None else ksize
    s = RING_BLUR_SIGMA if sigma is None else sigma
    pw = RING_PLATE_WEIGHT if plate_weight is None else float(plate_weight)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    appearance = APPEARANCE if appearance is None else appearance
    if appearance not in ("dark", "bright"):
        raise ValueError(f"unknown appearance {appearance!r}; use 'bright' or 'dark'")
    op = cv2.MORPH_CLOSE if appearance == "dark" else cv2.MORPH_OPEN

    def response(img):
        # In float so the plate subtraction below can go negative. cv2's MORPH_BLACKHAT
        # and MORPH_TOPHAT are the same two expressions on uint8, but they saturate the
        # difference of two responses at zero, which loses that sign.
        m = cv2.morphologyEx(img, op, kernel).astype(np.float32)
        f = img.astype(np.float32)
        return cv2.subtract(m, f) if appearance == "dark" else cv2.subtract(f, m)

    def plate_response(img, box):
        # The plate does not change, and its closing is 2.6 ms of every frame. Keyed
        # on the array's own buffer, holding a reference so the id cannot be recycled
        # onto a different plate -- the estimator passes the same array every frame.
        key = (id(img), box, k, appearance)
        hit = _PLATE_RESPONSE_CACHE.get(key)
        if hit is None:
            if len(_PLATE_RESPONSE_CACHE) > 4:
                _PLATE_RESPONSE_CACHE.clear()
            x, y, ww, hh = box if box else (0, 0, img.shape[1], img.shape[0])
            hit = (img, response(img[y:y + hh, x:x + ww]))
            _PLATE_RESPONSE_CACHE[key] = hit
        return hit[1]

    usable = background is not None and background.shape == gray.shape

    if roi is None:
        w = response(gray)
        if usable and pw:
            w -= pw * plate_response(background, None)
        return cv2.GaussianBlur(w, (0, 0), s)

    box = _clamp_roi(roi, gray.shape, pad=k)
    x, y, ww, hh = box
    out = np.zeros(gray.shape, np.float32)
    if ww <= 0 or hh <= 0:
        return out
    patch = response(gray[y:y + hh, x:x + ww])
    if usable and pw:
        patch -= pw * plate_response(background, box)
    out[y:y + hh, x:x + ww] = cv2.GaussianBlur(patch, (0, 0), s)
    return out


def ellipse_points(ellipse, n=RING_SAMPLES):
    """
    ``n`` points evenly spaced in parameter angle around an ellipse's perimeter.

        Even in *parameter*, not arc length, so a point count means the same thing at
        every eccentricity. The direct fit relies on that: a fixed count is what makes
        its objective a mean rather than a line integral, and a line integral would
        grow with the perimeter and reward inflating the ellipse.
    """

    (cx, cy), (major, minor), ang = ellipse
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a, b, th = major / 2.0, minor / 2.0, np.radians(ang)
    return np.column_stack([
        cx + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th),
        cy + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th),
    ])


def sample_map(weight, pts):
    """
    Bilinear reads of ``weight`` at ``pts``, zero outside the frame.

        `cv2.remap` on a ``(1, N)`` map rather than a loop or fancy indexing: 2 us for
        360 points, which is what lets the direct fit evaluate its residual a hundred
        times a frame. Off-frame reads return 0 -- the same as "no evidence here",
        which is the right answer for a rim arc that has left the image.
    """

    pts = np.asarray(pts, dtype=np.float32).reshape(1, -1, 2)
    return cv2.remap(
        weight, pts[..., 0].copy(), pts[..., 1].copy(),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    ).ravel()


def fit_ellipse_image(weight, seed, n=None, ref=None, f_scale=None, max_iter=60,
                      coarse=None):
    """
    Refine one ellipse onto the rim evidence in a single view. No mask, no rig.

        The threshold's replacement as a *measurement*. `segment` decides per pixel and
        then fits whatever survived; this fits the ellipse directly to the evidence and
        never decides about a pixel at all -- a shadowed arc contributes little instead
        of being admitted or dropped, which is what the level cannot express (see
        `ring_weight`, and `theory.md` 16).

        Extrinsic-free on purpose. It improves the per-view ellipse that `stereo.match`,
        the gates and `solve_from_major` all read, so it pays off with one camera and
        without a rig; `stereo.refine(mode="image")` is the two-view form, which needs
        both and buys occlusion cover on top.

        ``seed`` is an ``((cx, cy), (major, minor), angle_deg)`` from `segment` or from
        the previous frame. It must be inside the capture radius -- measured at ~5% in
        scale and ~25 px in centre, since the objective falls away outside the rim's
        own width -- the coarse pass is what widens that. Returns a `RingFit`, or
        ``None`` if the solve failed outright.
    """

    from scipy.optimize import least_squares

    n = RING_SAMPLES if n is None else n

    # Coarse pass first, on a heavily blurred copy. The rim is a few pixels thick, so
    # on the map as given the objective is flat more than about five pixels away and a
    # seed further out than that has nothing to descend -- which is not a corner case:
    # the mask seeds this replaces were measured 10-20 px and 5% out on real frames,
    # and one of the three scored 5.5 against a peak of 34. Blurring widens the basin
    # to the seed and costs one extra solve on a map nobody keeps.
    coarse = RING_COARSE_SIGMA if coarse is None else coarse
    if coarse > 0:
        got = fit_ellipse_image(
            _box_blur(weight, coarse), seed,
            n=n, f_scale=f_scale, max_iter=max_iter, coarse=0.0,
        )
        if got is not None:
            seed = got.ellipse

    def evidence(p):
        return sample_map(weight, ellipse_points(((p[0], p[1]), (p[2], p[3]), p[4]), n))

    p0 = np.array(
        [seed[0][0], seed[0][1], seed[1][0], seed[1][1], seed[2]], dtype=np.float64
    )
    if not np.all(np.isfinite(p0)) or min(p0[2], p0[3]) <= 0:
        return None

    e0 = evidence(p0)
    if ref is None:
        ref = float(np.percentile(e0, RING_REF_PERCENTILE))
    ref = max(float(ref), 1e-6)
    # Half the reference: a sample carrying nothing is then a clear outlier rather
    # than merely a large residual, which is the whole job of the loss here.
    f_scale = math.sqrt(ref / 2.0) if f_scale is None else f_scale

    def residual(p):
        if min(p[2], p[3]) <= 1.0:
            return np.full(n, math.sqrt(ref))
        return np.sqrt(np.maximum(ref - evidence(p), 0.0))

    try:
        sol = least_squares(
            residual, p0, method="trf", loss="cauchy", f_scale=f_scale,
            x_scale="jac", diff_step=RING_DIFF_STEP,
            max_nfev=max_iter * 6, xtol=1e-4, ftol=1e-4, gtol=1e-4,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None

    q = sol.x
    if min(q[2], q[3]) <= 1.0 or not np.all(np.isfinite(q)):
        return None
    # Never hand back something the seed already beat. The objective is not convex --
    # a rim arc, a coil edge and the rod are all dark curves -- so a seed far enough
    # out can descend into the wrong one, and there is no cheaper test for that than
    # the score itself. Costs one comparison and makes the contract "at least the seed".
    e1 = evidence(q)
    if float(np.mean(e1)) < float(np.mean(e0)):
        return _ring_fit(weight, conic.normalise_ellipse(seed), e0, n)
    return _ring_fit(
        weight, conic.normalise_ellipse(((q[0], q[1]), (q[2], q[3]), q[4])), e1, n
    )



def ring_seed(weight, frac=None, pct=None, max_spread=None, min_area=MIN_BLOB_AREA_PX):
    """
    A seed ellipse straight from the evidence map. No plate, no level on luminance.

        This is what lets the whole of S15 drop out of the direct path. That machinery
        -- `valid_region`, the plate, `backdrop_mask`, a level on 255-luminance -- exists
        to answer "where may the robot be?" on a scene where the clutter looks like the
        robot. On the evidence map it does not: a top-hat keeps thin bright structures,
        and the bench, the cloth folds and the room are all broad. So the map answers
        that question itself and nothing upstream of it is needed.

        The level is a fraction of the map's **own** high percentile, not of its
        maximum: one specular pixel should not set the scale. Otsu was tried here and is
        too permissive -- the map is 95% near-zero, so Otsu puts the level in the noise
        and the hull spans the frame (72k px on, major 1377 where the ring is ~450).

        `max_spread` is not optional here for the same reason it is not in the dark
        path: the rim is hollow and arrives as arcs, and without a spread limit
        `silhouette_hull` pools every speck in the frame. With it, the same frame gives
        major 452 px at ratio 0.91.

        Returns ``(ellipse, mask, hull, area_px)`` or ``None``.
    """

    frac = RING_SEED_FRACTION if frac is None else frac
    pct = RING_SEED_PERCENTILE if pct is None else pct
    max_spread = RING_MAX_SPREAD if max_spread is None else max_spread

    # Subsampled: a full `np.percentile` over a 1280x800 float map sorts a million
    # values and costs ~30 ms, which was most of a frame. Every fourth pixel in each
    # axis leaves 64k, and the map is smooth at that scale, so the estimate is the same
    # number for a fortieth of the time.
    level = frac * float(np.percentile(weight[::4, ::4], pct))
    mask = (weight >= max(level, 1e-6)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)

    hull, area = silhouette_hull(mask, max_spread=max_spread)
    if hull is None or area < min_area:
        return None
    fit = fit_ellipse(hull)
    if fit is None:
        return None
    return fit[0], mask, hull, area


def segment_ring(gray, background=None, appearance=None, thresh=None, weight=None):
    """
    `Segmentation` from the evidence map alone: `ring_weight` -> `ring_seed`.

        The seed for a rig with **no plate**. Where one exists `segment` is the better
        seed and this is not used: measured over 456 views, seeding the direct fit from
        the plate mask clears the ridge gate on 94% against 92%, and costs 5.4 ms
        against 7.6. The plate is simply more information than one frame carries.

        A separate function rather than another branch inside `segment` because that
        one is the mask pipeline -- three appearances, every constant fitted against it
        -- and the two have nothing in common but their return type.

        ``weight`` reuses a map the caller has already built, which it usually has.
    """

    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if gray.ndim == 3 else gray
    t0 = time.perf_counter()
    if weight is None:
        weight = ring_weight(gray, background=background, appearance=appearance)
    got = ring_seed(weight)
    if got is None:
        return None, weight
    ellipse, mask, hull, area = got
    rms = _sampson_distance(ellipse, hull)
    return Segmentation(
        mask=mask,
        contour=hull,
        ellipse=ellipse,
        area_px=area,
        n_points=len(hull),
        fit_rms_px=float(np.sqrt(np.mean(rms ** 2))),
        threshold=int(thresh or 0),
        t_ms=(time.perf_counter() - t0) * 1e3,
    ), weight


def _ring_fit(weight, ellipse, samples, n):
    """`RingFit` from an ellipse and its sampled evidence."""

    (cx, cy), (a, b), ang = ellipse
    d = RING_SHOULDER
    shoulders = [
        sample_map(weight, ellipse_points(((cx, cy), (a * f, b * f), ang), n))
        for f in (1.0 - d, 1.0 + d)
    ]
    # The *stronger* shoulder, not their mean: an ellipse that has drifted off the rim
    # usually has the rim on one side of it, and averaging in the empty side hides that.
    off = float(np.median(np.maximum(*shoulders)))
    med = float(np.median(samples))
    return RingFit(
        ellipse=ellipse,
        evidence=float(np.mean(samples)),
        coverage=float(np.mean(samples >= RING_COVERAGE_FLOOR * med)) if med > 0 else 0.0,
        ridge=med / max(off, 1e-6),
    )


def _box_blur(img, sigma):
    """
    Two box passes standing in for a Gaussian of ``sigma``.

        Only the coarse pass uses this. There the blur is a *capture radius* rather
        than a shape, so the kernel's exact profile does not matter and the cost does:
        12.3 ms for a sigma-9 Gaussian on a 1280x800 float map against 3.7 ms here.
        The fine pass keeps the Gaussian, since that map is what the reported fit is
        measured against.
    """

    w = max(3, int(round(math.sqrt(3.0 * sigma * sigma + 1.0))) | 1)
    return cv2.blur(cv2.blur(img, (w, w)), (w, w))


_PLATE_RESPONSE_CACHE = {}


def _clamp_roi(roi, shape, pad=0):
    """``(x, y, w, h)`` grown by ``pad`` and clipped to the frame."""

    x, y, w, h = (int(round(v)) for v in roi)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(shape[1], x + w + pad)
    y1 = min(shape[0], y + h + pad)
    return x0, y0, x1 - x0, y1 - y0


def backdrop_mask(gray, lum=None, sd_max=None, scale=None, erode=None):
    """
    The white backdrop: the one bright, smooth region, convex-hulled.

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
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

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
    return cv2.resize(
        out, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
    )


def valid_region(gray, bg=None, with_source=False):
    """
    Where the robot may be, as a uint8 mask, or ``None`` if it cannot be told.

        Background subtraction when a background frame exists, the backdrop finder
        otherwise. ``None`` is a real answer and callers must honour it: with no
        valid region the alternative is an ungated threshold over the whole frame,
        and the dark ambient beyond the backdrop is *darker than the robot* and
        reaches the frame edge, so the hull spans the image and the fit is confident
        and wrong. The module docstring rejects Otsu for the same reason -- returning
        nothing is the correct answer, not a failure to produce one.
    """

    m = background_mask(gray, bg=bg)
    source = "background" if m is not None else "backdrop"
    m = m if m is not None else backdrop_mask(gray)
    return (m, source) if with_source else m


def score_channel(frame, appearance=None, thresh=None, region=None, background=None):
    """
    ``(single_channel, level)`` where the robot is bright above ``level``.

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
        # Gate on the plate when there is one. This branch used to return the frame
        # ungated, which threw away a region that had already been computed -- and on
        # the black-backdrop rig that is the whole failure: the bench at the frame edge
        # reads 252 against the robot's 240, so *no* level separates them, while the
        # plate difference alone segments the robot almost exactly (identical seed to
        # within 3 px of the evidence map's own). The level was never too high; it was
        # being asked to do a job the region does.
        #
        # Only from a plate, never `backdrop_mask`: that finder looks for the bright
        # smooth region, which on a dark backdrop is the robot itself or nothing. And
        # unlike `dark`, no region is not a refusal here -- the renders every `bright`
        # constant was fitted on have no plate and no clutter, and must keep working.
        if region is None and background is not None:
            region = background_mask(gray, bg=background)
        if region is not None:
            gray = cv2.bitwise_and(gray, region)
        return gray, int(THRESH if thresh is None else thresh)

    if appearance == "dark":
        # Mono is the normal case here, not a degraded one: the ELP OV9281 has no
        # colour to give and the chroma gate this used to apply was measured
        # inoperative on it. A three-channel frame is collapsed rather than split.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        region = valid_region(gray, bg=background) if region is None else region
        if region is None:
            return None, int(DARK_THRESH if thresh is None else thresh)
        dark = cv2.bitwise_and(cv2.bitwise_not(gray), region)
        return dark, int(DARK_THRESH if thresh is None else thresh)

    raise ValueError(f"unknown appearance {appearance!r}; use 'bright' or 'dark'")


def clutter_mask(frame, background=None):
    """
    Everything the segmenter will ignore -- the complement of `valid_region`.

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
    region = valid_region(gray, bg=background)
    if region is None:
        return np.full(gray.shape, 255, np.uint8)
    return cv2.bitwise_not(region)


def threshold_mask(frame, thresh=None, appearance=None, background=None,
                   region=None, with_channel=False):
    """
    The binary mask `segment` fits to, without the hull or the fit.

        Exposed on its own because the mask is most worth seeing on the frames that
        yield no pose at all -- and those return ``None`` from `segment`, taking the
        mask with them. A frame that failed with an empty mask and one that failed
        with a mask full of clutter have nothing in common but the ``None``.

        Returns the mask, or ``(channel, mask, level)`` with ``with_channel``. The
        mask is ``None`` when the appearance needs a valid region and none was found.
    """

    gray, level = score_channel(
        frame, appearance, thresh, region=region, background=background
    )
    mask = None
    if gray is not None:
        _, mask = cv2.threshold(gray, level, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return (gray, mask, level) if with_channel else mask


def segment(
    frame,
    thresh=None,
    min_area=MIN_BLOB_AREA_PX,
    subpixel=False,
    axial=None,
    appearance=None,
    background=None,
):
    """
    Threshold, clean up, hull the silhouette, fit the rim ellipse.

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
        since 128 is meaningful for luminance and meaningless for chroma -- pass a
        number only to override it, never to restate it.

        ``background`` is this camera's own empty-rig plate. A stereo pair needs one
        each, so the module-level `BACKGROUND_PATH` cannot serve both, and without it
        `valid_region` falls back to the slower `backdrop_mask`.
    """

    t0 = time.perf_counter()

    # Computed here rather than inside `score_channel` so it can be kept on the
    # result: the overlay shades it every frame, and re-deriving it would cost
    # another 2.4 ms under the backdrop finder.
    region = None
    region_from = None
    max_spread = None
    app = APPEARANCE if appearance is None else appearance
    if app == "dark":
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        region, region_from = valid_region(g, bg=background, with_source=True)
        max_spread = DARK_MAX_SPREAD
    elif app == "bright" and background is not None:
        # Computed here rather than inside `score_channel` for the same reason as the
        # dark branch: the overlay shades it every frame and it is kept on the result.
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        region = background_mask(g, bg=background)
        region_from = "background" if region is not None else None
        max_spread = RING_MAX_SPREAD

    gray, mask, level = threshold_mask(
        frame, thresh=thresh, appearance=appearance, background=background,
        region=region, with_channel=True,
    )
    if mask is None:
        # No valid region could be established -- see `valid_region`. Refusing is
        # the answer; an ungated threshold here hulls the whole frame.
        return None

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
        valid_from=region_from,
    )


def undistort_ellipse(ellipse, camera_matrix, dist_coeffs, n_samples=180):
    """
    Re-fit an ellipse after removing lens distortion.

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

    src = ellipse_points(ellipse, n_samples).reshape(-1, 1, 2)
    dst = cv2.undistortPoints(src, camera_matrix, dist_coeffs, P=camera_matrix)
    return fit_ellipse_direct(dst.reshape(-1, 2))


# How strongly the ignored area is tinted. Faint on purpose: it has to be legible
# as "ignored" while leaving the pixels underneath readable, because the question
# it answers is usually "what is in there that I am throwing away?".
REJECT_ALPHA = 0.35
REJECT_BGR = (0, 0, 255)


def shade_rejected(out, region, alpha=REJECT_ALPHA, colour=REJECT_BGR, source=None):
    """
    Tint everything outside ``region`` in place, and return ``out``.

        A wrong pose and a wrong *rejection* look identical in a plain ellipse
        overlay -- both show an ellipse in the wrong place -- and they have opposite
        fixes. Shading what was ignored separates them at a glance, which is the only
        reason the region is carried on `Segmentation` rather than recomputed.

        Only worth drawing for the **backdrop** region, which answers "where may the
        robot be" and leaves most of the frame admissible. A background-plate region
        answers a different question -- "what is not the empty rig" -- and its
        complement is everything that did not move, some 95% of the frame. Tinting
        that says nothing and hides the image underneath, so it is skipped.
    """

    if region is None or source == "background":
        return out
    off = region == 0
    if not off.any():
        return out
    out[off] = (out[off] * (1.0 - alpha) + np.array(colour, np.float32) * alpha).astype(
        np.uint8
    )
    return out


MASK_ALPHA = 0.45
MASK_BGR = (255, 0, 255)


def draw(frame, seg, colour=(0, 255, 0), rejected=True, normal_px=None, mask=None):
    """
    Overlay the fitted ellipse, its axes and the ignored area on a copy.

        Converts grayscale to BGR first so the overlay survives -- the same trap
        `servo.py:199` documents.  Returns a new image; the input is untouched.

        ``rejected`` shades the area outside the valid region; ``normal_px`` draws the
        rotor axis as an image-space segment ``((x0, y0), (x1, y1))``, which only the
        caller can compute since it needs the camera matrix and the 3-D pose.

        ``mask`` tints the thresholded binary mask -- the thing the ellipse was
        actually fitted to. Drawn under the ellipse and over the image, so a mask that
        misses the rim, or one that has swallowed the rod, is visible against the
        pixels that produced it. This is the first thing to look at when the pose is
        wrong: a clean rim tint with a bad pose means the fault is downstream of the
        segmenter, not in it.

        ``True`` takes `Segmentation.mask`; an **array** is tinted as given, which is
        the only way to see the mask on a frame that produced no `Segmentation` at
        all -- pass `threshold_mask`'s. Those are the frames worth looking at.
    """

    out = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    m = seg.mask if (mask is True and seg is not None) else mask
    if m is not None and m is not False:
        on = m > 0
        out[on] = (
            out[on] * (1.0 - MASK_ALPHA) + np.array(MASK_BGR, np.float32) * MASK_ALPHA
        ).astype(np.uint8)
    if seg is None:
        cv2.putText(
            out, "no detection", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        return out

    if rejected:
        shade_rejected(out, seg.valid, source=seg.valid_from)

    cv2.ellipse(out, seg.ellipse, colour, 1)
    (cx, cy), (major, minor), ang = seg.ellipse
    th = np.radians(ang)
    for half, c, col in (
        (major / 2, (np.cos(th), np.sin(th)), colour),
        (minor / 2, (-np.sin(th), np.cos(th)), (255, 160, 0)),
    ):
        p0 = (int(cx - half * c[0]), int(cy - half * c[1]))
        p1 = (int(cx + half * c[0]), int(cy + half * c[1]))
        cv2.line(out, p0, p1, col, 1)
    if normal_px is not None:
        (nx0, ny0), (nx1, ny1) = normal_px
        cv2.arrowedLine(
            out,
            (int(nx0), int(ny0)),
            (int(nx1), int(ny1)),
            (255, 255, 0),
            2,
            tipLength=0.15,
        )
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


def _self_check():
    """The mask overlay agrees with the fit, and survives a frame with no detection."""

    img = np.full((400, 400), 10, np.uint8)
    cv2.ellipse(img, (200, 200), (120, 70), 20, 0, 360, 240, 14)

    mask = threshold_mask(img, appearance="bright", thresh=128)
    seg = segment(img, appearance="bright", thresh=128)
    assert np.array_equal(seg.mask, mask), "helper and segment must threshold alike"

    # The tint has to reach the image in both cases -- the no-detection one is the
    # whole point, and it is the one that used to return before drawing anything.
    assert not np.array_equal(draw(img, seg, mask=True), draw(img, seg))
    assert not np.array_equal(draw(img, None, mask=mask), draw(img, None))

    # A seed spanning part of a broken rim must pull in the rest of it. This is what
    # the distance rule cannot do: the far arcs are further from the anchor than the
    # anchor's own bounding box, so only the ellipse can reach them.
    broken = np.zeros((400, 400), np.uint8)
    for a0 in (0, 100, 200, 300):
        cv2.ellipse(broken, (200, 200), (150, 110), 0, a0, a0 + 55, 255, 6)
    n, labels, _, centroids = cv2.connectedComponentsWithStats(broken, 8)
    keep = np.arange(1, n)
    seed = keep[:2]                                  # two arcs, roughly half the ring
    grown = _regrow(seed, keep, labels, centroids, n)
    assert len(grown) > len(seed), "the seed did not reach the far arcs"
    major = fit_ellipse_direct(_group_hull(labels, grown, n))[1][0]
    assert major > 250, f"regrouped ring is too small: major {major:.0f} px"
    print("segment self-check ok")


if __name__ == "__main__":
    _self_check()
