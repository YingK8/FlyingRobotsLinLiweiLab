"""
Two views, one pose.

`conic.backproject_ellipse` gives **two** circle poses from one ellipse and cannot
choose; a second camera makes the choice a measurement rather than a bet, and fixes
two other things as a side effect.

    ambiguity  both cameras see the same circle, so of the four cross-view pairings
               one agrees and three do not. `match` reports `margin`, how many sigmas
               better the winner was -- i.e. how much the data actually decided.
    depth      a single view localises the ellipse *centre* far better than its
               *size*, so error is anisotropic ~11:1 (0.078 mm across the optical
               axis, 0.857 along). Each camera's bad axis is the other's good one,
               and `fuse` combines them in information form.
    bias       fusion weights by *direction*, not over time, which is what makes it
               work here: the single-view residual autocorrelates at 0.966 after one
               frame, so temporal averaging cannot touch it. Along camera A's depth
               axis camera B measures laterally and carries ~120x the weight, so a
               3 mm systematic depth error in A enters the fused answer as 0.025 mm.

Three layers, increasing cost, each usable alone:

    match    microseconds. Resolves the ambiguity.
    fuse     microseconds. Adds depth and bias. Enough on its own for a
             sub-millimetre target, and the fallback if refinement is too slow.
    refine   ~0.3 ms. Joint reprojection minimisation on R^3 x S^2 against both
             silhouettes. The only layer that can improve *orientation*: it treats
             the protruding mast as the localised outlier arc it is, where
             `shape.TiltCalibration` must average a scalar correction over it.

**Five unknowns, never six.** Position (3) plus the normal on S^2 (2). Roll is
unobservable -- the ring is rotationally symmetric and the robot spins at 310-350 Hz
against a much slower camera -- so carrying it would leave that direction of ``J'J``
rank-deficient and the normal equations singular.

**Normals are lines here, not vectors.** `conic.backproject` orients each normal
toward its own camera, so a rig with one camera above the rotor plane and one below
legitimately reports opposite normals for one pose. Everything internal compares
with ``abs(dot)``; the sign is applied once at the end against a caller-supplied
reference.
"""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import conic  # noqa: E402
import segment as segmod  # noqa: E402
from shape import CentreCalibration, TiltCalibration  # noqa: E402
from uncertainty import ErrorModel  # noqa: E402
from uncertainty import features as uncertainty_features  # noqa: E402
from uncertainty import GATE_MARGIN as uncertainty_GATE_MARGIN  # noqa: E402
from estimator import RADIUS_MM, _angles_from_normal  # noqa: E402
from zeroing import Zero  # noqa: E402
from filter import ACCEL_MM_S2  # noqa: E402

# Per-view error scales used to weight the fusion, in mm at the calibration
# resolution (1024x768).  Straight from the held-out measurements in
# controller/pose/theory.md S13: lateral 0.078, depth 0.857.  They are weights, not
# predictions -- only their *ratio* matters to the fused answer, which is why a
# rough figure is good enough and why the ratio is the thing to keep current if
# these are ever refitted.
SIGMA_LAT_MM = 0.078
SIGMA_DEPTH_MM = 0.857

# Refinement stops here.  The residual is Sampson distance in pixels and the
# seed is already sub-pixel, so this is about polishing rather than converging;
# more iterations buy nothing and cost frame budget.
MAX_REFINE_ITER = 12

# The robot's rim diameter, from the mesh. Used as the natural length scale for
# deciding when two views cannot be looking at the same thing.
BODY_DIAMETER_MM = 20.409

# Reject a frame when the two views' independent answers disagree by more than
# this. Not a tuned constant: two cameras that disagree about the robot's
# position by more than its own body have not both seen the robot.
#
# It is also the single most valuable thing the second camera provides, and that
# was not obvious in advance. Measured over 900 rendered pairs, 9.1% of frames
# were catastrophic (position error above 5 mm) -- and on every one of them the
# *monocular* estimate was already wrong by ~23 mm, because segmentation had
# grabbed the wrong thing in one view. Stereo cannot repair that. What it can do
# is notice: those frames show a cross-view discrepancy of 349 mm median against
# 5.4 mm for good frames, a 60x separation with no overlap worth speaking of.
#
#   gate     frames kept    catastrophic among kept    p95 |pos|
#   none        100%              9.1%                  5.59 mm
#   25 mm        85%              0.00%                 1.81 mm
#   40 mm        89%              0.87%                 2.11 mm
#
# A controller is far better served by a declared gap than by a confident 50 mm
# error, which is why the default rejects rather than falling back to one view:
# on these frames the surviving view was wrong too.
#
# **Retuned for the direct fit, because the blunder it has to catch has changed.**
# The reasoning above sized it against segmentation failures, which separate by 60x
# (349 mm against 5.4) and need no precision to catch. The direct fit removed those and
# left a subtler one: the *branch flip*, where both views take the mirrored solution
# together, agree with each other, and are jointly wrong. The two conic solutions are
# furthest apart near 45 degrees of tilt -- 84.5 degrees -- so a flip costs almost a
# right angle of orientation while moving position by only ~8.5 mm. A 25.5 mm gate
# never sees it, and the branch *margin* does not either: on flipped frames it reads
# 28.3 sigma against 25.8 on good ones, so the matcher is confidently wrong.
#
# What does separate them is that the good frames got much better: 2.0 mm median
# against the 5.4 the old note records. Swept on `2026-08-28_131552`, a take flown
# deliberately upright so the true normal is constant and its scatter *is* the error:
#
#   gate     kept    normal scatter, median / p90    frames >20 deg out
#   25.5 mm   77%          9.45 / 68.85 deg                 24%
#   12 mm     72%          7.86 / 69.59                     19%
#    8 mm     66%          3.37 / 63.95                     11%
#    6 mm     62%          1.97 /  7.28                      8%
#    5 mm     58%          1.38 /  4.06                      6%
#    4 mm     56%          0.59 /  3.00                      4%
#    3 mm     54%          0.99 /  2.05                      3%
#
# The cliff is between 8 and 6 mm, where p90 falls 64 -> 7 degrees: that is the flips
# being cut. Passes over **4-6 mm**, and 5 is the midpoint; below 4 it buys tenths of a
# degree for whole points of coverage. On `2026-08-28_092117`, which has real attitude
# change and so no constant to measure against, the same move takes frames more than 20
# degrees from the mean from 15% to 3%.
#
# It is now a *precision* gate rather than a plausibility one, so it has to be
# re-measured whenever the fit's precision moves -- unlike the body-diameter reasoning
# above, which needed no numbers.
MAX_DISCREPANCY_MM = 5.0

# Reject a pose whose rotor axis jumps faster than the robot can turn, in degrees per
# second, and only while the previous one is still fresh.
#
# The branch flip of `MAX_DISCREPANCY_MM` is a *discrete* error -- the two conic
# solutions are ~84 degrees apart near 45 degrees of tilt -- so it appears as a jump
# nothing physical can produce and returns the same way. Cross-view discrepancy catches
# it only indirectly, by noticing that a flipped pair agrees slightly less well, which
# is why gating on that alone costs 20 points of coverage to remove the last few.
#
# 600 deg/s is far above the robot and far below a flip. Sampled every third frame at
# 60 fps the window is 50 ms, so this admits 30 degrees between poses where a flip asks
# for 84; the fastest real attitude change measured on any flight here is under 60 deg/s.
#
# **Rate, not a fixed angle**, so it does not tighten when frames are dropped, and
# **only while `dropout_s` has not elapsed**: after a real gap there is no prior worth
# trusting and re-acquisition must be allowed, which is the same rule `_choose` uses
# monocularly. Without that a single bad pose would end the track permanently.
# **Off by default.** It works -- it takes the upright take's normal scatter from
# 1.38 to 0.57 deg p50 -- but it decides the present frame from the previous one, so a
# single wrong pose that slips through can reject the correct poses that follow, and the
# recovery is a dropout rather than a correction. That is unstable in the way a gate must
# not be, and the discrepancy gate above catches the same flips on its own evidence
# (1.38 deg p50, 6% of frames more than 20 deg out, against 1% with this on). Set it to a
# number to turn it back on.
MAX_JUMP_DEG_PER_S = None

# How long a previous normal stays worth comparing against. Matches the monocular
# `estimator.PoseEstimator` default: past this the robot may have done anything and the
# jump gate must stand aside so the track can re-acquire.
DROPOUT_S = 0.25

# Reject a frame whose outline is not elliptical enough for the circle model to
# mean anything, as a **fraction of the major axis** rather than in pixels.
#
# It was 1.5 px, and that was wrong in a way only a resolution sweep exposes:
# detection came out *lower* at 1280x800 (53%) than at 640x480 (68%). Backwards.
# More pixels should mean more information, not fewer accepted frames. The cause
# is that a pixel threshold is dimensional -- at twice the resolution the same
# *relative* outline quality yields twice the pixel residual, so a fixed limit
# silently tightens as the sensor improves, and at 160x120 it silently vanishes,
# passing frames whose outline is nothing like an ellipse.
#
# 0.012 is the old 1.5 px expressed against the ~130 px major axis it was tuned
# at, so at that resolution behaviour is unchanged and only the scaling differs.
#
# It exists because the cross-view gate above cannot catch everything. That gate
# finds views that *disagree*; it is blind to both views failing the same way,
# which is exactly what happens near face-on, where the rim's outer wall goes
# unlit and the outline collapses onto the blade cross in both cameras at once.
# Measured over 112 frames, the surviving 9.8% of catastrophic orientations all
# had this signature -- segmentation IoU 0.32 against 0.58, and both cameras
# agreeing to within 5.6 mm on a wrong answer.
#
#   gate     frames kept   normal >10 deg among kept   normal p50 / p95
#   none        100%             9.8%                   1.87 / 25.66
#   1.5 px       71%             1.3%                   0.96 /  3.15
#   1.25 px      63%             0.0%                   0.85 /  3.01
#
# 1.5 px is roughly five times the sub-pixel precision the boundary is measured
# to, so it fires only when the outline is materially non-elliptical.
MAX_FIT_RMS_REL = 0.012

# Where the direct fit's reference level sits in the seed's own ring evidence.
#
# Every sample is scored as a deficit from this, so it decides what "found the rim"
# means. High enough that most of a good ring sits below it and the residual has a
# gradient everywhere; below the maximum, so one specular sample cannot set the bar
# for the whole ring. p90 of the seed's samples.
REF_PERCENTILE = 90

# How far a direct fit must stand above its own surroundings to be believed.
#
# The blunder gate for the `fit_ellipse_image` path, replacing `MAX_FIT_RMS_REL` for the
# views it refines -- that one measures the hull, and the hull is what the shadow
# contaminated. `segment.RingFit.ridge` is the fitted curve's evidence over that of
# curves just inside and outside it, so it tests whether the ellipse is on a ridge.
#
# **Coverage was tried here first and is the wrong statistic.** It measures how much of
# the rim is present, which is not the same question: a well-fitted ring two thirds
# hidden behind the rig and an ellipse plainly too big for a fully visible one both
# score 0.52, and they need opposite answers. Rejecting the first throws away exactly
# the frames a two-view fit exists to rescue. `ridge` reads 27.3 and 0.60 on that pair.
#
# Swept over 534 stereo fits on three flights, against cross-view agreement:
#
#   ridge   kept   median   under 25.5 mm        coverage   kept   median   under gate
#    1.5     79%   1.10 mm      92%                 0.60      79%   1.10 mm     84%
#    2.0     70%   0.97 mm      94%                 0.70      65%   0.92 mm     91%
#    2.5     63%   0.91 mm      96%                 0.75      59%   0.89 mm     94%
#    3.0     58%   0.89 mm      98%                 0.80      50%   0.88 mm     94%
#    4.0     50%   0.85 mm      99%
#
# It dominates coverage at every operating point -- at 2.0 it keeps 70% of frames for
# the 94% that coverage reaches only by dropping to 59%. Passes over 2.0-4.0; 2.5 is
# near the midpoint and is where the marginal frame stops being worth its error.
MIN_RING_RIDGE = 2.5


def _distort_points(pts, cam):
    """
    Ideal pinhole pixels back to raw sensor pixels, so a prediction can be sampled.

        The reverse of what `StereoPoseEstimator` does to its hulls. `conic` predicts
        in an ideal pinhole, the weight map is built on the raw frame, and distortion
        does not map an ellipse to an ellipse -- so one of the two has to move. Moving
        180 points costs 0.02 ms; undistorting the whole map costs 1-2 ms per view.
    """

    if cam.dist is None or not np.any(cam.dist):
        return pts
    k = cam.K
    xy = (np.asarray(pts, dtype=np.float64) - [k[0, 2], k[1, 2]]) / [k[0, 0], k[1, 1]]
    obj = np.column_stack([xy, np.ones(len(xy))])
    z = np.zeros(3)
    return cv2.projectPoints(obj, z, z, k, cam.dist)[0].reshape(-1, 2)


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def _tangent_basis(n):
    """
    Two unit vectors spanning the plane perpendicular to ``n``.

        The seed for the S^2 half of the refinement parameterisation.  Deterministic,
        so a rerun on the same frame produces the same numbers.
    """

    n = _unit(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = _unit(np.cross(n, seed))
    return t1, np.cross(n, t1)


def viewing_bisector(rig, towards_cameras=False):
    """Unit vector along the mean of the cameras' optical axes, in the world frame.

    ``towards_cameras`` negates it, giving the direction a face must point to be the one
    they can see -- which is what `orient` needs as its reference.
    """

    v = sum(np.asarray(c.optical_axis, dtype=np.float64) for c in rig.cameras)
    n = np.linalg.norm(v)
    v = v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    return -v if towards_cameras else v


def orient(normal, reference=(0.0, 0.0, 1.0)):
    """
    Flip a normal to agree in sign with ``reference``.

        Applied once, at the end.  The default assumes the robot hovers rotor-up,
        which is true for every pose the controller cares about; a rig watching a
        robot lying on its side should pass something else.
    """

    n = _unit(normal)
    return -n if float(n @ np.asarray(reference, dtype=np.float64)) < 0 else n


def line_angle_deg(u, v):
    """
    Angle between two directions treated as undirected lines, in degrees.
    """

    return math.degrees(math.acos(float(np.clip(abs(_unit(u) @ _unit(v)), 0.0, 1.0))))


# --------------------------------------------------------------------------
# Layer 1: which branch


@dataclass
class Match:
    """
    The cross-view branch decision for one frame.

        ``discrepancy_mm`` is how far apart the winning pair's two world poses were;
        ``margin_mm`` is how much worse the runner-up pair was.  Both are logged
        every frame.  A small margin means the two cameras could not tell the
        branches apart -- which happens legitimately when the rotor is near face-on
        to both, where the branches merge and the choice stops mattering -- and it is
        the number to look at before believing an orientation outlier.
    """

    poses: tuple  # one conic.CirclePose per view, in that view's camera frame
    indices: tuple  # which candidate was taken from each view
    discrepancy_mm: float  # the winning pair's disagreement, as a plain distance
    margin: float  # how many sigmas better the winner was than the runner-up

    @property
    def margin_mm(self):
        """
        Deprecated alias kept so existing logs and callers still read.

                The margin stopped being a distance when the agreement test moved to
                Mahalanobis; it is a likelihood ratio now. Reported under both names for
                one release rather than silently changing what a logged column means.
        """

        return self.margin


# Angular scale for the branch-agreement test, radians. Roughly the per-view
# normal error the monocular sweep measures, so an angular disagreement of that
# size counts the same as a positional one at its own noise scale.
SIGMA_NORMAL_RAD = math.radians(3.0)

#: How far a branch pairing may sit from the sliding window's prediction before the
#: prior starts to cost, in degrees, and how many frames the window holds.
#:
#: Swept on `2026-08-28_131552`, which is flown upright so the normal is ground truth,
#: measuring the share of frames landing more than 30 degrees from the flight's own
#: median normal -- the quarter-turn branch flips of `theory.md` 16.17:
#:
#:   prior sigma   off    25     15     10     6      4      3      2
#:   flipped       26.8%  26.6   26.2   25.4   12.4   8.6    8.1    7.0
#:   scatter p90   88.31  88.27  88.17  88.08  41.38  16.94  16.94  16.70
#:   scatter p50   0.84   0.84   0.84   0.84   0.82   0.83   0.83   0.83
#:
#: The last row is the point: the median frame does not move at all, because the prior
#: is asked only about frames already flagged by cross-view discrepancy. It is bought
#: entirely on the frames that were wrong. Passes over 2-4, flattening; 3.0 is the
#: midpoint, and is the same figure as `SIGMA_NORMAL_RAD` -- the window is trusted
#: about as far as one camera is, which is the right order.
#:
#: The window is 15 frames, a quarter second at 60 fps, matching `DROPOUT_S`, and
#: predicts from a **median**: one bad frame cannot move a median, where a
#: frame-to-frame comparison chains, which is why `MAX_JUMP_DEG_PER_S` was unstable
#: enough to be turned off. `MIN_WINDOW_SUPPORT` is the warm-up floor -- a median of one
#: or two samples is that same chaining under another name.
PRIOR_SIGMA_DEG = 3.0
WINDOW_FRAMES = 15
MIN_WINDOW_SUPPORT = 5


def _pair_information(cam_a, cam_b, sigma_lat_mm, sigma_depth_mm):
    """
    ``inv(Sigma_a + Sigma_b)`` for two views' position estimates.

        Depends only on the rig, so `match` computes it once and reuses it across
        the four candidate pairs.
    """

    eye = np.eye(3)
    total = np.zeros((3, 3))
    for cam in (cam_a, cam_b):
        d = cam.optical_axis
        total += (sigma_lat_mm**2) * eye + (
            sigma_depth_mm**2 - sigma_lat_mm**2
        ) * np.outer(d, d)
    return np.linalg.inv(total)


def _agreement(pose_a, pose_b, cam_a, cam_b, info):
    """
    How inconsistent two views' candidate poses are, in sigmas.

        **Mahalanobis, not Euclidean, and the difference decides frames.**  Each
        view's position error is anisotropic by about 11:1 -- loose along its own
        optical axis, tight across it -- so two views that agree perfectly still
        differ by millimetres along their respective depth axes.  A plain distance
        counts that expected disagreement as evidence against the pair, which leaves
        the true pair only marginally ahead of the false ones: measured on rendered
        pairs, a median discrepancy of 5-6 mm against a winning margin of 8 mm, i.e.
        a decision made at 1.6x the noise.  Wrong picks then arrive at a few percent
        and each one is a catastrophic outlier, which is exactly the tail the sweep
        found (normal p95 above 70 degrees while the median sat near 1.2).

        Weighting by ``inv(Sigma_a + Sigma_b)`` asks the right question instead: not
        "how far apart are these two answers" but "how surprised should I be that
        they are this far apart, given how each view is allowed to be wrong".
        Disagreement along a view's blind axis is nearly free; disagreement across
        it is decisive.

        The orientation term stays additive and in sigmas for the same reason -- a
        quantity in degrees and one in millimetres cannot be summed without a scale
        chosen on purpose.
    """

    ca, na = cam_a.to_world(pose_a.center, pose_a.normal)
    cb, nb = cam_b.to_world(pose_b.center, pose_b.normal)
    d = ca - cb
    pos = float(d @ info @ d)
    sin_ang = math.sqrt(
        max(0.0, 1.0 - min(1.0, abs(float(_unit(na) @ _unit(nb)))) ** 2)
    )
    return pos + (sin_ang / SIGMA_NORMAL_RAD) ** 2


def match(
    candidates,
    rig,
    radius_mm=RADIUS_MM,
    sigma_lat_mm=SIGMA_LAT_MM,
    sigma_depth_mm=SIGMA_DEPTH_MM,
    prior_normal=None,
    prior_sigma_deg=PRIOR_SIGMA_DEG,
):
    """
    Pick the one branch pair the two views agree on.

        ``candidates`` is a list per view of `conic.CirclePose` in that view's camera
        frame -- exactly what `conic.backproject_ellipse` returns and what
        `estimator.Pose.extra["candidates"]` already carries.

        Returns a `Match`, or ``None`` if any view has no candidates.  With a single
        view it degenerates gracefully: the first candidate, zero margin, which is
        the honest report that nothing was decided.

        ``prior_normal`` is a world normal the answer is expected near -- in practice
        `StereoPoseEstimator`'s sliding window. **This is where the two-fold ambiguity
        is actually settled.** The reflection the geometry cannot resolve puts the two
        candidates about 84 degrees apart (`theory.md` 16.13), and when both views take
        the mirrored branch together they agree with each other while jointly wrong, so
        no measure of cross-view agreement separates them -- `theory.md` 16.17 measures
        the wrong branch at 28 sigma of confidence. Time does separate them: the true
        branch moves like a robot and the mirror jumps 84 degrees between frames.
        Scored in the same sigmas as the rest, so it is a prior and not an override --
        a genuine manoeuvre still outvotes it if both views insist.
    """

    if any(not c for c in candidates):
        return None
    if len(candidates) == 1:
        return Match(
            poses=(candidates[0][0],), indices=(0,), discrepancy_mm=0.0, margin=0.0
        )

    cam_a, cam_b = rig.cameras[0], rig.cameras[1]
    info = _pair_information(cam_a, cam_b, sigma_lat_mm, sigma_depth_mm)

    prior = None if prior_normal is None else _unit(prior_normal)
    inv_sig = 1.0 / math.radians(max(prior_sigma_deg, 1e-3))
    scored = []
    for i, pa in enumerate(candidates[0]):
        for j, pb in enumerate(candidates[1]):
            score = _agreement(pa, pb, cam_a, cam_b, info)
            if prior is not None:
                # As a line, not a vector: which way the normal points along its own
                # axis is a separate ambiguity (`theory.md` 16.13) and this prior has
                # no opinion about it.
                score += (math.radians(
                    line_angle_deg(cam_a.to_world(pa.center, pa.normal)[1], prior)
                ) * inv_sig) ** 2
            scored.append((score, i, j))
    scored.sort()
    best, i, j = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else float("inf")

    # Report the winner's disagreement as a plain distance too: the Mahalanobis
    # score is the right thing to *decide* on, but a number in millimetres is
    # what a person reading the log can sanity-check against the rig.
    ca, _ = cam_a.to_world(candidates[0][i].center, candidates[0][i].normal)
    cb, _ = cam_b.to_world(candidates[1][j].center, candidates[1][j].normal)

    return Match(
        poses=(candidates[0][i], candidates[1][j]),
        indices=(i, j),
        discrepancy_mm=float(np.linalg.norm(ca - cb)),
        margin=float(runner_up - best),
    )


# --------------------------------------------------------------------------
# Layer 2: combine


def fuse(
    poses,
    rig,
    sigma_lat_mm=SIGMA_LAT_MM,
    sigma_depth_mm=SIGMA_DEPTH_MM,
    reference=(0.0, 0.0, 1.0),
    stamps=None,
    velocity=None,
    vel_cov=None,
    accel_mm_s2=ACCEL_MM_S2,
):
    """
    Information-weighted combination of per-view poses, in world coordinates.

        Position: each view contributes ``inv(Sigma_i)`` with ``Sigma_i``
        anisotropic in its own frame -- tight across the optical axis, loose along
        it.  The sum is dominated, along any given world direction, by whichever
        camera measures that direction *laterally*.  That is the whole mechanism.

        Orientation: weighted by ``sin^2(tilt seen)``, which is the sensitivity of
        the axis-ratio channel and therefore how much each view's normal is worth.
        A view looking straight down the rotor axis contributes almost nothing to
        tilt, correctly, and a view seeing it edge-on dominates.  Normals are summed
        as lines (sign-aligned first) and renormalised.

        **Timing.** The two cameras free-run, so ``poses`` are not simultaneous: view
        *i* saw the robot at its own ``stamps[i]``, up to one sensor frame period apart
        (7.71 ms median on this bench at 119 fps).  Given ``velocity`` and its covariance
        ``vel_cov`` -- `filter.PoseFilter`'s ``rate`` and ``rate_cov`` -- each view is
        moved to the mean instant and *charged for the move*:

            x_i(t_ref) = x_i(t_i) + v (t_ref - t_i) + O(a dt^2)
            Sigma_i'   = Sigma_i + vel_cov dt^2 + (accel dt^2 / 2)^2 I

        measurement noise, plus the uncertainty in the shift, plus the acceleration the
        constant-velocity model does not carry.  ``accel_mm_s2`` is the same number that
        sets `filter.PoseFilter`'s process noise, so the two cannot drift apart.

        This is deliberately *not* what the calibration path does.  There the board is
        static and the rig re-reads until the skew is small, which costs frames; here
        there is no second chance at a frame and no waiting, so the skew is priced rather
        than avoided (`calib/theory.md` section 16, `pose/theory.md` section 17).

        With ``stamps=None`` -- the default, and every existing caller -- the arithmetic
        below is untouched.

        Returns ``(center_world_mm, normal_world, covariance_3x3)``.
    """

    info = np.zeros((3, 3))
    info_c = np.zeros(3)
    n_acc = np.zeros(3)
    ref_n = None
    eye = np.eye(3)

    offsets = np.zeros(len(poses))
    if stamps is not None:
        stamps = np.asarray(stamps, dtype=np.float64)
        offsets = float(np.mean(stamps)) - stamps
    v = None if velocity is None else np.asarray(velocity, dtype=np.float64).ravel()
    p_vv = None if vel_cov is None else np.asarray(vel_cov, dtype=np.float64)

    for pose, cam, dt in zip(poses, rig.cameras, offsets):
        c_w, n_w = cam.to_world(pose.center, pose.normal)
        d = cam.optical_axis
        cov = (sigma_lat_mm**2) * eye + (
            sigma_depth_mm**2 - sigma_lat_mm**2
        ) * np.outer(d, d)
        if dt:
            if v is not None:
                c_w = c_w + v * dt
            if p_vv is not None:
                cov = cov + p_vv * dt**2
            cov = cov + (0.5 * accel_mm_s2 * dt**2) ** 2 * eye
        inv = np.linalg.inv(cov)
        info += inv
        info_c += inv @ c_w

        # sin^2 of the angle between the rotor axis and this camera's axis.
        w = 1.0 - float(n_w @ d) ** 2
        if ref_n is None:
            ref_n = n_w
        n_acc += w * (n_w if float(n_w @ ref_n) >= 0 else -n_w)

    cov = np.linalg.inv(info)
    center = cov @ info_c
    normal = orient(n_acc if np.linalg.norm(n_acc) > 1e-12 else ref_n, reference)
    return center, normal, cov


# --------------------------------------------------------------------------
# Layer 3: refine


def _predict_conic(center_world, normal_world, cam, radius_mm, tilt_cal=None):
    """
    Image conic a hypothesised world circle would produce in one camera.

        ``p = K X`` up to scale, so a cone ``X'QX = 0`` in camera coordinates becomes
        ``p'(K^-T Q K^-1)p = 0`` in pixels.  `conic.project_circle` does the same
        thing and then converts to axis form; the conic is what the residual wants,
        so the conversion is normally skipped.

        ``tilt_cal`` makes the prediction a **silhouette** rather than a rim circle.
        The robot is not a flat circle -- the mast and magnet protrude along the
        rotor axis and widen the short direction of what the camera actually sees --
        so comparing an ideal rim ellipse against a real silhouette biases the fit.
        `calibration.TiltCalibration.unapply` puts the measured widening back in.
        Skipping this was measured: refinement improved position on 75% of frames
        and degraded orientation on 78%, because the centre does not care about the
        mast and the axis ratio is nothing but.
    """

    img = _predict_image_conic(center_world, normal_world, cam, radius_mm)
    if tilt_cal is None or tilt_cal.is_identity:
        return img
    # Round-tripping through axis form is the only way to reach the minor axis
    # the correction is defined on; `conic.ellipse_from_conic` is written in
    # closed form partly to make this affordable here.
    return conic.conic_from_ellipse(
        tilt_cal.unapply(conic.normalise_ellipse(conic.ellipse_from_conic(img)))
    )


def _predict_image_conic(center_world, normal_world, cam, radius_mm):
    """
    The uncorrected image conic: ``K^-T Q K^-1`` for a hypothesised circle.
    """

    c_cam, n_cam = cam.to_camera(center_world, normal_world)
    q = conic.cone_from_circle(c_cam, n_cam, radius_mm)
    kinv = cam.K_inv
    return kinv.T @ q @ kinv


#: Where a sample that cannot be seen is put, in pixels.
#:
#: A trust region walks through poses that put part of the rim at or behind the lens.
#: Those samples must read as no evidence, which `segment.sample_map`'s zero border
#: gives for free once they are off-frame. The bound also has to survive distortion,
#: which raises the radius to the sixth power -- unclipped, the float32 cast in
#: `sample_map` overflows. Ten frames away and cubes harmlessly.
_FAR_PX = 1e4


def _rim_points(center_world, normal_world, radius_mm, phis):
    """
    World points on the rim circle at parameter angles ``phis``.

        The disk has no preferred direction in its own plane, so any orthonormal basis
        of the plane will do -- but it has to be the **same** basis for both cameras, or
        the two views' samples stop describing the same physical points. Taking it from
        `_tangent_basis`, which the solve already uses, makes that structural: there is
        one basis per normal and no second place for it to be derived differently.
    """

    u, v = _tangent_basis(normal_world)
    c = np.asarray(center_world, dtype=np.float64).reshape(3)
    return c + radius_mm * (np.outer(np.cos(phis), u) + np.outer(np.sin(phis), v))


def _project_ideal(pts_world, cam):
    """``(ideal pinhole pixels, in front of the lens)``. Distortion is applied after."""

    R = cam.R
    p = (np.asarray(pts_world, dtype=np.float64) - cam.T_world_cam[:3, 3]) @ R
    # A trust region walks through poses that put part of the rim at or behind the lens.
    # Those points are not visible and must read as no evidence, not as a coordinate the
    # float32 cast in `segment.sample_map` overflows on: parked far off-frame, the
    # constant border gives exactly zero, which is what "not visible" means here.
    z = p[:, 2]
    behind = z < 1e-6
    z = np.where(behind, 1.0, z)
    out = np.column_stack([
        cam.K[0, 0] * p[:, 0] / z + cam.K[0, 2],
        cam.K[1, 1] * p[:, 1] / z + cam.K[1, 2],
    ])
    return out, ~behind


def _predict_ellipse(
    center_world, normal_world, cam, radius_mm, tilt_cal=None, centre_cal=None
):
    """
    Predicted silhouette ellipse, major-axis first.

        The `mode='ellipse'` path's inner loop.  It stops one step short of
        `_predict_conic` -- no conversion back to a conic -- which is most of why
        that mode is cheaper.

        ``centre_cal`` displaces the predicted centre the way a real silhouette's is
        displaced.  This is the coherent place for that correction: here the model
        moves all five parameters together for one hypothesised pose, so it stays a
        physically realisable silhouette.  Applying the same correction to the
        *measured* ellipse instead breaks it -- see `StereoPoseEstimator`.
    """

    e = conic.normalise_ellipse(
        conic.ellipse_from_conic(
            _predict_image_conic(center_world, normal_world, cam, radius_mm)
        )
    )
    if tilt_cal is not None and not tilt_cal.is_identity:
        e = tilt_cal.unapply(e)
    if centre_cal is not None and not centre_cal.is_identity:
        c_cam, n_cam = cam.to_camera(center_world, normal_world)
        d = projected_axis_dir(c_cam, n_cam, cam.K)
        if d is not None:
            tilt = line_angle_deg(n_cam, np.array([0.0, 0.0, 1.0]))
            (cx, cy), axes, ang = e
            shift = centre_cal.offset(tilt) * axes[0]
            e = (cx + shift * d[0], cy + shift * d[1]), axes, ang
    return e


def _axis_endpoints(ellipse, ref_deg=None):
    """
    The four ends of an ellipse's axes, as an ``(8,)`` vector of pixels.

        A compact stand-in for the whole boundary when comparing two ellipses.  Its
        virtue over comparing ``(cx, cy, major, minor, angle)`` directly is that
        every component is already a length in pixels, so no weighting has to be
        invented to trade an angle against a radius -- and the angle's degeneracy on
        a near-circular ellipse takes care of itself, because rotating a circle
        moves its axis ends nowhere.

        ``ref_deg`` fixes the one way this can go wrong.  An ellipse axis is a line,
        not a vector, so an angle of 179 degrees and one of -1 describe the identical
        shape -- but they put the endpoints at opposite ends, and differencing them
        elementwise yields a residual the size of the whole ellipse.  Inside a
        least-squares loop that appears as a discontinuous cliff near the wrap, which
        the optimiser cannot descend and lands on as a spurious minimum.  Passing the
        measured angle rotates the prediction into the same half-turn first.
    """

    (cx, cy), (major, minor), ang = ellipse
    if ref_deg is not None:
        ang -= 180.0 * round((ang - ref_deg) / 180.0)
    t = math.radians(ang)
    ct, stn = math.cos(t), math.sin(t)
    a, b = major / 2.0, minor / 2.0
    return np.array(
        [
            cx + a * ct,
            cy + a * stn,
            cx - a * ct,
            cy - a * stn,
            cx - b * stn,
            cy + b * ct,
            cx + b * stn,
            cy - b * ct,
        ]
    )


@dataclass
class Refinement:
    center: np.ndarray
    normal: np.ndarray
    rms_px: float
    n_iter: int
    n_points: int
    converged: bool
    # Mean rim evidence at the solution, `mode="image"` only. The quantity that
    # says whether the fit landed on the rim, and the one to gate on -- `rms_px`
    # is a deficit in evidence units there, not pixels.
    evidence: float = float("nan")
    #: ``(n_views, sample_n)`` of per-sample evidence at the solution, `mode="image"`
    #: only, indexed by the **shared world angle** so column `k` is the same physical
    #: point of the rim in every view. `union_coverage` is what it is for.
    samples: np.ndarray | None = None


def refine(
    hulls,
    rig,
    seed_center,
    seed_normal,
    radius_mm=RADIUS_MM,
    loss="linear",
    f_scale=None,
    max_iter=MAX_REFINE_ITER,
    reference=(0.0, 0.0, 1.0),
    tilt_cal=None,
    mode="ellipse",
    ellipses=None,
    params="both",
    centre_cal=None,
    weights=None,
    sample_n=None,
    ref=None,
    **ls_kwargs,
):
    """
    Fit one world pose to both views at once.

        At most five parameters: centre in mm, plus a tangent-plane increment to the
        seed normal retracted onto the sphere. Never six -- roll is unobservable and
        the normal equations would be singular.

        @param hulls: one (N,2) array of boundary points per view, in **ideal
            pinhole pixels**. Undistort first, as `StereoPoseEstimator` does: `conic`
            assumes no distortion, and distortion does not map an ellipse to an ellipse.
        @param params: ``"both"`` solves all five; ``"normal"`` holds the centre at
            the fused value and solves the two orientation parameters (2-column
            Jacobian instead of 5). ``"both"`` wins on both channels: 0.348 mm /
            1.503 deg against 0.397 / 1.618.
        @param mode: ``"ellipse"`` residualises the four axis endpoints of predicted
            against measured, 8 numbers per view. ``"hull"`` uses Sampson distance of
            every boundary point. ``"hull"`` looks more principled and measures
            **worse** (1.55 deg against 0.53): `tilt_cal` was fitted against
            `cv2.fitEllipseDirect` output, so it corrects *that statistic*, not
            individual point positions.

            ``"image"`` takes no measured geometry at all. It samples the predicted
            silhouette in `weights` -- one `segment.ring_weight` map per view -- and
            residualises the shortfall in evidence. Both other modes compress a view to
            a fitted ellipse *before* the joint solve, so an occlusion or a shadow has
            already corrupted the measurement the solve is handed; this one lets the
            weak arc contribute little and the other view carry those parameters.
            Needs `weights`, and a seed inside the capture radius: ~5% in scale and
            ~25 px in centre, so the filter's prediction, or the mask path on a cold
            start.
        @param tilt_cal: what makes either mode work. Turns the prediction from a
            bare rim circle into the silhouette a real robot casts, mast included.
            Without it, refinement improved position on 75% of frames and degraded
            orientation on 78%.
        @param loss: plain least squares. A robust loss changes nothing here (0.509
            vs 0.524 deg): with 8 endpoint residuals per view the ellipse fit has
            already pooled the boundary, so there are no outliers left to reject.

            **Not so in `mode="image"`, where a robust loss is the point.** Nothing has
            pooled anything: the residuals are individual samples, and the bottom
            decile of a real ring carries near-zero evidence in every frame measured --
            occlusion, or a shadow across the rim. Pass ``loss="cauchy"`` there so a
            dead sample is bounded rather than dragging all five parameters.
    """

    t1, t2 = _tangent_basis(seed_normal)
    n0 = _unit(seed_normal)
    views = hulls if hulls is not None else (weights if weights is not None else ellipses)
    cams = list(rig.cameras)[: len(views)]
    c0 = np.asarray(seed_center, dtype=np.float64).reshape(3)

    if params == "normal":
        p0 = np.zeros(2)

        def unpack(p):
            return c0, _unit(n0 + p[0] * t1 + p[1] * t2)

    else:
        p0 = np.concatenate([c0, [0.0, 0.0]])

        def unpack(p):
            return p[:3], _unit(n0 + p[3] * t1 + p[4] * t2)

    if mode == "ellipse":
        if ellipses is None:
            fits = [segmod.fit_ellipse(h) for h in hulls]
            if any(f is None for f in fits):
                return None
            ellipses = [f[0] for f in fits]
        ellipses = [conic.normalise_ellipse(e) for e in ellipses]
        measured = np.concatenate([_axis_endpoints(e) for e in ellipses])
        refs = [e[2] for e in ellipses]
        total = len(measured)

        def residual(p):
            centre, normal = unpack(p)
            return (
                np.concatenate(
                    [
                        _axis_endpoints(
                            _predict_ellipse(
                                centre, normal, cam, radius_mm, tilt_cal, centre_cal
                            ),
                            ref,
                        )
                        for cam, ref in zip(cams, refs)
                    ]
                )
                - measured
            )

    elif mode == "image":
        # No mask, no per-view ellipse: the pose is fitted straight to the image
        # evidence in both views at once. An arc a shadow has dimmed or a coil has
        # hidden contributes little here instead of being decided about, and the same
        # arc in the other view constrains the same five parameters. See `theory.md` 16.
        n = segmod.RING_SAMPLES if sample_n is None else sample_n
        total = n * len(weights)
        # **Indexed by world angle, not by each view's own image parameter angle.**
        # Sampling each predicted ellipse independently makes sample `k` a different
        # physical rim point in each view, which is enough for the pose but destroys the
        # one thing two views 83 degrees apart are for: knowing that the arc A has lost
        # is the arc B still has. With a shared `phi` the two rows of the residual
        # vector line up point for point, so an occlusion is a property of an arc rather
        # than of a view. It also samples uniformly on the *disk*, where parameter-angle
        # sampling of a projected ellipse bunches near the minor-axis ends -- exactly
        # where a side-on frame has least to spare.
        phis = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

        def evidence(p):
            centre, normal = unpack(p)
            rim = _rim_points(centre, normal, radius_mm, phis)
            out = np.empty(total)
            k = 0
            for w, cam in zip(weights, cams):
                pts, seen = _project_ideal(rim, cam)
                pts = np.ascontiguousarray(pts)
                if centre_cal is not None and not centre_cal.is_identity:
                    # `_predict_ellipse` displaces the predicted centre and leaves the
                    # axes alone, so carrying a projected point onto the corrected
                    # silhouette is that same displacement. Not the general affine map
                    # between the two ellipses: `mode="image"` needs `direct=True`,
                    # which forces `tilt_cal` to identity, so nothing rescales.
                    #
                    # Guarded because a trust region does reach poses whose image conic
                    # is not a real ellipse -- the circle edge-on, or behind the lens.
                    # The samples there are already parked as unseen, so skipping a
                    # sub-pixel centre correction on them changes nothing except
                    # whether the solve survives to walk back out.
                    try:
                        ideal = conic.normalise_ellipse(conic.ellipse_from_conic(
                            _predict_image_conic(centre, normal, cam, radius_mm)))
                        corr = _predict_ellipse(centre, normal, cam, radius_mm,
                                                tilt_cal, centre_cal)
                        pts += np.subtract(corr[0], ideal[0])
                    except (ValueError, np.linalg.LinAlgError):
                        pass
                # Clipped *before* distortion, which raises the radius to the sixth
                # power: an unclipped coordinate overflows the float32 cast in
                # `sample_map`. A trust region does walk out this far, and so does
                # `_recorrect` when a calibration's axis ratio is extreme. The bound is
                # already ten frames away, and cubes harmlessly.
                np.clip(pts, -_FAR_PX, _FAR_PX, out=pts)
                pts = np.array(_distort_points(pts, cam), dtype=np.float64, copy=True)
                # Parked after distortion, not before: distortion is not monotonic far
                # from the centre, so a point pushed off-frame first can fold back on.
                pts[~seen] = -_FAR_PX
                out[k : k + n] = segmod.sample_map(w, pts)
                k += n
            return out

        # The reference level every sample is scored against a deficit from. Taken
        # from the seed's own evidence rather than fixed, because the response scales
        # with contrast and contrast moves with the lighting -- an absolute level here
        # would be the same mistake as `DARK_THRESH`.
        if ref is None:
            # Outside the solver's own guard, so it needs its own: a seed whose conic
            # is already degenerate raised here and took the caller down with it.
            try:
                ref = float(np.percentile(evidence(p0), REF_PERCENTILE))
            except (ValueError, np.linalg.LinAlgError):
                return None
        ref = max(float(ref), 1e-6)
        if f_scale is None:
            # Half the reference, so a sample carrying nothing is unambiguously an
            # outlier rather than merely a large residual. The same rule and the same
            # number as `segment.fit_ellipse_image`, which is the single-view form of
            # this objective; left at the solver's 1.0 it would be off by roughly the
            # contrast, which moves with the lighting.
            f_scale = math.sqrt(ref / 2.0)

        def residual(p):
            # sqrt of the deficit, so the sum of squares is the deficit itself and
            # minimising it maximises the summed evidence -- linear in the evidence,
            # where a plain (ref - w) residual would penalise a *bright* sample too.
            return np.sqrt(np.maximum(ref - evidence(p), 0.0))

    else:
        counts = [len(h) for h in hulls]
        total = sum(counts)
        if total < 10:
            return None

        def residual(p):
            centre, normal = unpack(p)
            out = np.empty(total)
            k = 0
            for hull, cam in zip(hulls, cams):
                c = _predict_conic(centre, normal, cam, radius_mm, tilt_cal)
                m = len(hull)
                out[k : k + m] = segmod.sampson_distance_conic(c, hull)
                k += m
            return out

    # In `params='both'` the vector mixes millimetres (position, order 10-250)
    # with radians (the tangent increment, order 0.01), so an unscaled trust
    # region is badly conditioned. `x_scale='jac'` takes the scaling off the
    # Jacobian rather than guessing.  Tolerances are 1e-5 because the residual
    # is in pixels and the seed is already sub-pixel: tightening further changed
    # no reported digit and cost iterations.
    opts = dict(
        method="trf",
        loss=loss,
        f_scale=1.0 if f_scale is None else float(f_scale),
        x_scale="jac",
        max_nfev=max_iter * 6,
        xtol=1e-5,
        ftol=1e-5,
        gtol=1e-5,
    )
    opts.update(ls_kwargs)
    try:
        sol = least_squares(residual, p0, **opts)
    except (ValueError, np.linalg.LinAlgError):
        return None

    centre, normal = unpack(sol.x)
    ev = evidence(sol.x).reshape(len(views), -1) if mode == "image" else None
    return Refinement(
        center=centre,
        normal=orient(normal, reference),
        rms_px=float(np.sqrt(np.mean(sol.fun**2))),
        n_iter=int(sol.nfev),
        n_points=total,
        converged=bool(sol.success),
        evidence=float("nan") if ev is None else float(np.mean(ev)),
        samples=ev,
    )


# --------------------------------------------------------------------------
# The estimator


@dataclass
class StereoPose:
    """
    One frame's stereo estimate, in the datum frame set by `zeroing.Zero`.

        Mirrors `estimator.Pose` field for field where the meaning is the same, so
        `recorder.py` and `viz.py` can treat them alike, and adds only what is
        genuinely new: the cross-view agreement numbers and the refinement's report.

        ``discrepancy_mm`` is the residual disagreement between the two views'
        independent answers.  It is the single best health check in the log and it
        needs no ground truth: two cameras that disagree by more than their own
        noise are telling you the extrinsic has drifted, or a view is occluded, or
        the branch pick was wrong.
    """

    t: float
    frame_index: int
    xyz_mm: np.ndarray
    normal: np.ndarray
    theta_deg: float
    phi_deg: float
    n_views: int
    discrepancy_mm: float
    skew_ms: float
    margin: float
    refine_rms_px: float
    refine_iters: int
    jump_deg: float
    t_seg_ms: float
    t_est_ms: float
    # The fields below exist so `recorder.py` and `viz.py` accept a StereoPose
    # wherever they accept an `estimator.Pose`, without either of them growing a
    # type switch. They describe **view A** -- the reference camera -- because a
    # per-view quantity has no single stereo value, and view A is the one the
    # overlay shows. The genuinely stereo numbers are `discrepancy_mm` and
    # `margin` above, which have no monocular analogue.
    # Predicted error bounds for this frame, from `uncertainty.ErrorModel`.
    # Logged whether or not they are used to reject, so the gate's calibration
    # can be audited against outcomes rather than taken on trust.
    pred_pos_mm: float = float("nan")
    pred_ang_deg: float = float("nan")
    psi_deg: float = float("nan")
    ellipse: tuple = (
        (float("nan"), float("nan")),
        (float("nan"), float("nan")),
        float("nan"),
    )
    area_px: float = float("nan")
    fit_rms_px: float = float("nan")
    ambiguity_margin_deg: float = float("nan")
    n_solutions: int = 0
    #: Fraction of the rim at least one view has evidence for, `mode="image"` only.
    #: The honest statement of whether this frame was solvable at all. Per-view
    #: `coverage` cannot say it -- the two cameras sit 83 degrees apart, so each loses a
    #: *different* arc, and it is the union that determines whether the five parameters
    #: are constrained. Measured over 395 frames of `2026-08-28_135533`: per view the
    #: median arc alive is 0.90 and 0.92 with p5 at 0.73 and 0.66, while the union runs
    #: p50 1.00, p25 0.98, p5 0.92. A view is under 80% on 36% of frames; on those the
    #: union median is 0.99. Reported, not gated: below roughly 200 degrees of union the
    #: pose is weakly determined however it is weighted, and that is worth knowing
    #: rather than declining.
    union_coverage: float = float("nan")
    per_view: tuple = ()
    extra: dict = field(default_factory=dict)

    @property
    def t_total_ms(self):
        return self.t_seg_ms + self.t_est_ms

    @property
    def margin_mm(self):
        """
        Deprecated alias for `margin`, which is no longer a distance.

                Kept because logs and readers still use the old name; see `Match.margin_mm`
                for why the quantity changed units when the agreement test became
                Mahalanobis.
        """

        return self.margin


class StereoPoseEstimator:
    """
    Frames in, one 5-DOF world pose out.

        Deliberately the same shape as `estimator.PoseEstimator` -- construct, call
        ``update``, get a pose or ``None`` -- so the two are interchangeable
        downstream and can be A/B'd against each other on the same recording.

        Unlike the monocular estimator this one holds no branch history: the branch
        is decided by the geometry every frame.  ``_prev_normal`` is kept only to
        report ``jump_deg``, which stays useful as a motion sanity check.
    """

    def __init__(
        self,
        rig,
        radius_mm=RADIUS_MM,
        zero=None,
        thresh=None,
        min_area=segmod.MIN_BLOB_AREA_PX,
        backgrounds=None,
        verify=True,
        undistort=True,
        direct=True,
        min_ridge=MIN_RING_RIDGE,
        max_jump_deg_s=MAX_JUMP_DEG_PER_S,
        dropout_s=DROPOUT_S,
        tilt_cal=None,
        do_refine=True,
        loss="linear",
        sigma_lat_mm=SIGMA_LAT_MM,
        sigma_depth_mm=SIGMA_DEPTH_MM,
        reference=None,
        max_discrepancy_mm=MAX_DISCREPANCY_MM,
        use_major_channel=False,
        max_fit_rms_rel=MAX_FIT_RMS_REL,
        centre_cal=None,
        error_model=None,
        require_stereo=True,
        target_pos_mm=None,
        target_angle_deg=None,
        gate_margin=None,
        never_reject=False,
    ):
        if never_reject:
            # **Always answer.** Every gate here exists to decline rather than report a
            # pose it cannot stand behind, and for a controller that is the right
            # default: a declared gap beats a confident 50 mm error. This stands all of
            # them down, so `update` returns the best pose available on every frame and
            # the caller carries the judgement instead.
            #
            # For looking at footage, fitting constants, and any consumer that would
            # rather have a pose plus a quality number than a hole. It is **not** more
            # accurate: on `2026-08-28_135533` it takes 88.1% to 100%, and the frames it
            # adds back are exactly the ones the gates were built to catch.
            # `discrepancy_mm` and the per-view `ridge` ride on every result and are what
            # to sort by -- see `theory.md` 16.18.
            #
            # `require_stereo` goes too, so a frame where one view is lost still answers
            # from the other, and `_view_candidates` falls back to the tracked ellipse
            # when segmentation finds nothing, so a seed always exists.
            max_discrepancy_mm = None
            max_fit_rms_rel = None
            min_ridge = 0.0
            max_jump_deg_s = None
            require_stereo = False
            error_model = False
        self.never_reject = never_reject
        # What a fit must score to be trusted as the *next frame's seed*, as opposed to
        # to be reported. Fixed, so relaxing the reporting gates cannot poison the track.
        # The sliding window. Holds `(t, world normal)` before zeroing, so a datum
        # installed mid-run does not invalidate it.
        self._window = deque(maxlen=WINDOW_FRAMES)
        # The level at which a frame is suspect enough to ask the window about. Fixed at
        # the module constant, never `max_discrepancy_mm`: that one is reporting policy
        # and `never_reject` drives it to None, and "is this worth reporting" is a
        # different question from "is this worth a second opinion" -- the same split as
        # `_track_ridge` against `min_ridge`.
        self._suspect_mm = MAX_DISCREPANCY_MM
        self.n_rearbitrated = 0
        self._track_ridge = MIN_RING_RIDGE
        self.rig = rig
        self.radius_mm = float(radius_mm)
        self.zero = zero if zero is not None else Zero.identity()
        # None, not `segmod.THRESH`: 128 is the *bright* appearance's level, and passing
        # it explicitly overrode the dark path's own on every frame. Measured on a
        # 600-frame flight, 5 poses recovered against 194 with the level left alone.
        self.thresh = thresh
        self.min_area = min_area
        # One empty-rig plate per camera, keyed by `Camera.name`. Without them
        # `segment.valid_region` falls back to `backdrop_mask`, which is 40x slower and
        # cannot tell the robot's rim from a dark gap in the scene behind it.
        self.backgrounds = dict(backgrounds or {})
        self.verify_tol = 1e-6 if verify else None
        self.undistort = undistort
        self.direct = direct

        self.min_ridge = min_ridge
        self.max_jump_deg_s = max_jump_deg_s
        self.dropout_s = dropout_s
        self._prev_t = None
        self.n_rejected_jump = 0
        self._prev_ellipse = {}
        # Applied to the per-view seed only. The refinement works on raw hull
        # points, where the robust loss is the intended mechanism instead.
        self.tilt_cal = TiltCalibration.load() if tilt_cal is None else tilt_cal
        # A tilt calibration describes the gap between a *hull silhouette* and the true
        # rim: the mast and magnet widen the silhouette's short direction, and it widens
        # the model to match. The direct fit produces no silhouette. It settles on the
        # rim's own evidence ridge and never sees the protrusions, so the correction is
        # applied to a widening that is already gone and the minor axis is inflated
        # twice. Tilt is acos(minor/major), so that lands entirely on the normal while
        # leaving the drawn ellipse untouched -- correct ellipse, wrong normal.
        #
        # Measured on 2026-08-28_092117, as the angle between the two views' own
        # normals: **9.58 deg median with the shipped calibration, 2.46 deg with
        # identity**, against a per-view scatter floor of ~2.6 deg (`theory.md` 12.12).
        # Position hardly moves (1.81 vs 1.77 mm), which is why this hid behind the
        # cross-view gate. See `theory.md` 16.11 and 16.14.
        if self.direct and self.tilt_cal is not None and not self.tilt_cal.is_identity:
            self.tilt_cal = TiltCalibration()
        self.do_refine = do_refine
        self.loss = loss
        self.sigma_lat_mm = sigma_lat_mm
        self.sigma_depth_mm = sigma_depth_mm
        # Which way is up, and it is not a preference. A silhouette is identical for a
        # normal and its negative, and the second camera does not help: both see the same
        # outline. Only a prior about the rig settles it, and the prior is that the
        # cameras are looking at the *top* of the disc, so the thrust axis points back
        # towards them -- the opposite of their mean viewing direction.
        #
        # The old default was camera A's own +z, which points from the camera *into* the
        # scene, so every pose came out inverted: measured on four flights, the normal's
        # z ran +0.61 to +0.78 where the geometry requires it negative.
        self.reference = (np.asarray(reference, dtype=np.float64) if reference is not None
                          else viewing_bisector(self.rig, towards_cameras=True))
        self.max_discrepancy_mm = max_discrepancy_mm
        self.use_major_channel = use_major_channel
        self.max_fit_rms_rel = max_fit_rms_rel
        # Refuse a frame in which only one camera produced a usable outline.
        #
        # This is not a tuning choice -- it follows from the noise derivation.
        # Depth from a single view is read off the ellipse's *size*, and since
        # size is a difference of two boundary positions, `dz/z = -dM/M`. At a
        # 0.3 px boundary and 250 mm range that is 0.52 mm at 1280x800 rising to
        # 4.18 mm at 160x120: a monocular solve **cannot** meet +-0.5 mm at any
        # sensor mode this camera offers. Stereo meets it (0.060 mm fused at
        # 1280x800) precisely because the second camera measures laterally the
        # axis the first is blind along.
        #
        # It also closes a hole in the cross-view gate below, which is skipped
        # when `len(usable) == 1` and so passed those frames without ever
        # comparing anything. Measured over 921 samples: the 6 single-view
        # frames were 0% in spec with a 256 mm median error, and refusing them
        # costs 0.7% of frames while taking the worst position error from
        # 755.7 mm to 8.0 mm.
        self.require_stereo = require_stereo
        self.gate_margin = (
            uncertainty_GATE_MARGIN if gate_margin is None else float(gate_margin)
        )
        # The centre correction is loaded but **not applied to the measurement**.
        #
        # It works, in the narrow sense: it removes 22-68x of the ellipse
        # centre's displacement. Applied to the measured ellipse it nonetheless
        # makes 3-D position *worse* -- 0.397 -> 0.533 mm on identical frames --
        # and the reason is worth keeping. `conic.backproject` consumes all five
        # ellipse parameters jointly. `tilt_cal` has already rewritten the minor
        # axis so the ratio implies the true tilt; moving the centre on top of
        # that yields a conic that corresponds to no real circle projection at
        # all, and back-projecting an inconsistent conic is worse than either
        # error alone. One ellipse parameter cannot be corrected in isolation.
        #
        # It belongs in the *forward* model instead, where all five parameters
        # move together -- see `refine`, which predicts a silhouette rather than
        # correcting a measurement.
        self.centre_cal = CentreCalibration.load() if centre_cal is None else centre_cal
        self.apply_centre_to_measurement = False

        # The predicted-error gate. Loaded from disk by default; absent means no
        # gate, which is how a baseline is measured. Targets default to None,
        # meaning "predict but do not reject" -- so the prediction can be logged
        # and its calibration checked before it is trusted to throw frames away.
        # `False` means "no model", distinct from `None` meaning "load the default":
        # `never_reject` needs a way to say the former.
        self.error_model = (
            ErrorModel.load() if error_model is None else (error_model or None)
        )
        self.target_pos_mm = target_pos_mm
        self.target_angle_deg = target_angle_deg
        self.n_rejected_predicted = 0

        self.frame_index = -1
        self.n_detected = 0
        self.n_lost = 0
        self.n_rejected = 0
        self.n_rejected_fit = 0
        self.n_rejected_mono = 0
        self._prev_normal = None

    def reset(self):
        self._prev_normal = None
        self._prev_t = None
        self._prev_ellipse = {}
        self._window.clear()

    def _window_normal(self, now):
        """
        The window's opinion of where the rotor axis is, or ``None``.

            A component-wise median of the recent normals, sign-aligned first because
            they are lines and half of them may point the other way along the same axis.
            A median rather than a fit: this exists to survive outliers, and the frames
            it has to survive are 84 degrees out, so an average of them is useless.

            Silent after a gap longer than `dropout_s` -- past that there is no prior
            worth trusting and re-acquisition has to be free, the same rule `_choose`
            and the old jump gate used.
        """

        if len(self._window) < MIN_WINDOW_SUPPORT:
            return None
        t_last, _ = self._window[-1]
        if not 0.0 <= now - t_last <= self.dropout_s:
            return None
        ns = np.array([n for _, n in self._window])
        ns *= np.where(ns @ ns[-1] < 0.0, -1.0, 1.0)[:, None]
        med = np.median(ns, axis=0)
        norm = float(np.linalg.norm(med))
        return None if norm < 1e-9 else med / norm

    def _view_candidates(self, frame, cam):
        """
        Segment one view and back-project it.

            Returns ``(seg, candidates, ellipse, weight)``. The evidence map comes back
            with the rest because `refine(mode="image")` needs it: it is the one form of
            the solve that sees the occlusion instead of a per-view ellipse fitted before
            it. Building it here and dropping it was why that mode went unused.
        """

        plate = self.backgrounds.get(cam.name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if hasattr(plate, "update"):
            # A `background.RunningPlate`: estimated from the stream itself, so the live
            # loop needs no captured plate and cannot be running on a stale one. Duck
            # typed rather than imported -- `background` pulls in the camera module.
            plate = plate.update(gray)

        if self.direct:
            # The evidence map, for the *joint* two-view solve only. Refining each view
            # separately against it was removed: watching the overlay, the per-view fit
            # comes apart exactly where the seed does, and the seed is the steadier of
            # the two by six times. See `theory.md` 16.23 and 16.24.
            w = segmod.ring_weight(gray, background=plate)
            seg = (segmod.segment(frame, thresh=self.thresh, min_area=self.min_area,
                                  background=plate)
                   if plate is not None else segmod.segment_ring(gray, weight=w)[0])
            if seg is None and self._prev_ellipse.get(cam.name) is not None:
                # Segmentation found nothing this frame, but the rim was here a frame
                # ago and the robot cannot have gone far in 1/60 s. Answering from the
                # previous ellipse is worth the ten frames per flight it recovers, and
                # the joint solve still measures the pose against *this* frame's
                # evidence map -- only the seed is a frame old.
                seg = segmod.Segmentation(
                    mask=None, ellipse=self._prev_ellipse[cam.name],
                    contour=segmod.ellipse_points(self._prev_ellipse[cam.name]),
                    area_px=0.0, n_points=segmod.RING_SAMPLES,
                    fit_rms_px=float("nan"), threshold=0, t_ms=0.0,
                )
            if seg is None:
                self._prev_ellipse[cam.name] = None
                return None, [], None, w
            self._prev_ellipse[cam.name] = seg.ellipse

        else:
            w = None
            seg = segmod.segment(frame, thresh=self.thresh, min_area=self.min_area,
                                 background=plate)
            if seg is None:
                return None, [], None, None

        ellipse = seg.ellipse
        if self.undistort and cam.dist is not None and np.any(cam.dist):
            try:
                ellipse = segmod.undistort_ellipse(ellipse, cam.K, cam.dist)
            except Exception:
                ellipse = (
                    seg.ellipse
                )  # keep the distorted fit rather than lose the view
        ellipse = self.tilt_cal.apply(ellipse)

        return (
            seg,
            conic.backproject_ellipse(
                ellipse, cam.K, self.radius_mm, verify_tol=self.verify_tol
            ),
            ellipse,
            w,
        )

    def _hull_ideal_px(self, seg, cam):
        """
        Hull points in ideal pinhole pixels.

                The monocular path undistorts a *fitted ellipse* by resampling it,
                because that is all it has by then.  Here the raw boundary points are
                still in hand, so they are undistorted directly -- one `cv2` call, no
                resample, and no ellipse-to-ellipse approximation in the middle.
        """

        if not (self.undistort and cam.dist is not None and np.any(cam.dist)):
            return seg.contour
        import cv2

        src = seg.contour.astype(np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(src, cam.K, cam.dist, P=cam.K).reshape(-1, 2)

    def update(self, frames, t=None, frame_index=None, stamps=None, motion=None):
        """
        Estimate one pose from one frame per camera.

                ``None`` is a normal outcome and callers must handle it: the robot
                leaves one camera's field of view, an occluder covers it, the threshold
                finds nothing.

                The frames are **not** simultaneous, and this used to assume they were.
                Two free-running cameras land up to one sensor frame period apart --
                7.71 ms median at 119 fps on this bench.  Pass ``stamps``
                (`sources.StereoCamera.last_stamps`) and ``motion``, a `filter.PoseFilter`
                or any object with ``rate`` and ``rate_cov``, and `fuse` moves each view
                to the mean instant and inflates its covariance to pay for the move.
                Omit them and the behaviour is exactly what it was.
        """

        now = time.monotonic() if t is None else float(t)
        stamps = None if stamps is None else [float(x) for x in stamps]
        vel = None if motion is None else np.asarray(motion.rate, dtype=np.float64)
        vel_cov = None if motion is None else np.asarray(motion.rate_cov, dtype=np.float64)
        skew_ms = (0.0 if not stamps else (max(stamps) - min(stamps)) * 1e3)
        self.frame_index = (
            self.frame_index + 1 if frame_index is None else int(frame_index)
        )

        t_seg0 = time.perf_counter()
        segs, cands, ellipses, maps = [], [], [], []
        for frame, cam in zip(frames, self.rig.cameras):
            s, c, e, w = self._view_candidates(frame, cam)
            segs.append(s)
            cands.append(c)
            ellipses.append(e)
            maps.append(w)
        t_seg_ms = (time.perf_counter() - t_seg0) * 1e3

        t0 = time.perf_counter()
        usable = [i for i, c in enumerate(cands) if c]
        if not usable:
            self.n_lost += 1
            return None

        # A view that dropped out used to be tolerated here, on the reasoning
        # that one camera still yields a monocular answer. Measurement retired
        # that reasoning: see `require_stereo` above. Those frames are not
        # degraded, they are unusable, and they carry no cross-view check.
        if self.require_stereo and len(usable) < 2:
            self.n_rejected_mono += 1
            self.n_lost += 1
            return None

        sub_rig = _subset(self.rig, usable)
        m = match(
            [cands[i] for i in usable],
            sub_rig,
            self.radius_mm,
            self.sigma_lat_mm,
            self.sigma_depth_mm,
        )
        if m is None:
            self.n_lost += 1
            return None

        # **The sliding window arbitrates, but only on frames already known bad.**
        # Measured on `2026-08-28_131552` -- flown upright, so the normal is ground
        # truth -- the frames a quarter turn out are cleanly separable before anything
        # is done about them: cross-view discrepancy reads p50 0.58 mm and p90 2.86 on
        # the frames that are right, against p10 6.20 and p50 23.21 on the frames that
        # are wrong. So the window is asked only about those, and the three quarters
        # that are already self-consistent are left exactly as they were.
        #
        # Asking it about *every* frame was tried and is strictly worse. The prior then
        # drags the good frames too and the window, fed its own output, reinforces
        # whatever it picked: ungated at 15 degrees the quarter-turn frames went 26.8%
        # to 27.4%, at 10 degrees to 37.3%, and at 8 degrees to 45.0%, with the median
        # frame's scatter blowing out from 0.84 to 7.50 degrees. Gated, the same prior
        # at 3 degrees takes 26.8% to 8.1% and leaves the median at 0.83. A temporal
        # prior on a decision that feeds the prior is a positive feedback loop unless
        # something outside it says when to listen. See `theory.md` 16.19.
        prior = self._window_normal(now)
        if (
            prior is not None
            and len(usable) > 1
            and m.discrepancy_mm > self._suspect_mm
        ):
            alt = match(
                [cands[i] for i in usable],
                sub_rig,
                self.radius_mm,
                self.sigma_lat_mm,
                self.sigma_depth_mm,
                prior_normal=prior,
            )
            if alt is not None and alt.indices != m.indices:
                self.n_rearbitrated += 1
                m = alt

        # The cross-view consistency gate. See MAX_DISCREPANCY_MM: this is what
        # turns a 9% catastrophic-outlier rate into zero, and it is the one
        # health check available with no ground truth at all.
        if (
            self.max_discrepancy_mm is not None
            and len(usable) > 1
            and m.discrepancy_mm > self.max_discrepancy_mm
        ):
            self.n_rejected += 1
            self.n_lost += 1
            return None

        centre, normal, _ = fuse(
            m.poses, sub_rig, self.sigma_lat_mm, self.sigma_depth_mm, self.reference,
            stamps=_subset_stamps(stamps, usable), velocity=vel, vel_cov=vel_cov
        )

        # Second pass. The centre correction needs to know which way the robot
        # leans, and only a solved pose supplies that -- so the first pass exists
        # to produce a normal, and this one uses it. One extra back-projection
        # and match, about 100 us.
        if self.apply_centre_to_measurement and not self.centre_cal.is_identity:
            fixed = []
            for k, i in enumerate(usable):
                cam = self.rig.cameras[i]
                # The direction must be the ROTOR AXIS as the robot carries it --
                # the end the mast is on -- not the normal `conic.backproject`
                # returns, which it deliberately flips to face the camera. Using
                # the latter applies the correction backwards on roughly half the
                # frames and doubles the error instead of removing it (measured:
                # dz 0.21 -> 0.47 mm). The world normal out of `fuse` is already
                # oriented against `reference`, i.e. rotor-up, so it carries the
                # sign; mapping it back into this camera preserves it.
                _, n_cam = cam.to_camera(centre, normal)
                c_cam, _ = cam.to_camera(centre, normal)
                d = projected_axis_dir(c_cam, n_cam, cam.K)
                if d is None:
                    fixed.append(None)
                    continue
                tilt_i = line_angle_deg(n_cam, np.array([0.0, 0.0, 1.0]))
                fixed.append(self.centre_cal.apply_to_ellipse(ellipses[i], tilt_i, d))
            if all(f is not None for f in fixed):
                cands2 = [
                    conic.backproject_ellipse(
                        f,
                        self.rig.cameras[i].K,
                        self.radius_mm,
                        verify_tol=self.verify_tol,
                    )
                    for f, i in zip(fixed, usable)
                ]
                if all(cands2):
                    m2 = match(
                        cands2,
                        sub_rig,
                        self.radius_mm,
                        self.sigma_lat_mm,
                        self.sigma_depth_mm,
                    )
                    if m2 is not None:
                        m = m2
                        centre, normal, _ = fuse(
                            m.poses,
                            sub_rig,
                            self.sigma_lat_mm,
                            self.sigma_depth_mm,
                            self.reference,
                            stamps=_subset_stamps(stamps, usable),
                            velocity=vel,
                            vel_cov=vel_cov,
                        )

        # Fit-quality gate, relative to the object's own size. Applied to the
        # per-view outline fits, which exist whether or not refinement runs, so
        # the gate does not depend on it. See MAX_FIT_RMS_REL.
        #
        # This catches *blunders* -- an outline that is not an ellipse at all --
        # which is a different job from the precision prediction below. A
        # blunder has no feature that explains it: the residual looks ordinary
        # and the answer is 500 mm out. Mixing the two lets the blunders set the
        # precision model's error quantile, which is how a fitted k reached 280.
        if self.max_fit_rms_rel is not None:
            # Only for views still carrying a mask fit. Where `fit_ellipse_image` has
            # refined the ellipse onto the image, `fit_rms_px` measures the *hull*'s
            # distance to it -- and the hull is exactly what the shadow contaminated,
            # so the gate would reject the frames the direct fit was added to rescue.
            # Those views are gated by `evidence` below instead.
            # Views the direct fit refined are gated on whether the ellipse sits on
            # a ridge instead; see MIN_RING_RIDGE.
            weak = [
                i for i in usable
                if np.isfinite(segs[i].ridge) and segs[i].ridge < self.min_ridge
            ]
            if weak:
                self.n_rejected_fit += 1
                self.n_lost += 1
                return None
            by_hull = [i for i in usable if not np.isfinite(segs[i].evidence)]
            worst = max(
                (segs[i].fit_rms_px / max(segs[i].ellipse[1][0], 1e-9) for i in by_hull),
                default=0.0,
            )
            if worst > self.max_fit_rms_rel:
                self.n_rejected_fit += 1
                self.n_lost += 1
                return None

        rms, iters = float("nan"), 0
        union_cov, alive = float("nan"), None
        if self.do_refine:
            hulls = [self._hull_ideal_px(segs[i], self.rig.cameras[i]) for i in usable]
            # `mode="ellipse"` refits an ellipse from the hull when it is not given
            # one -- which would throw the direct fit away and put the contaminated
            # hull back in. Hand it the measured silhouettes instead. `unapply` undoes
            # the tilt correction `_view_candidates` applied, because refinement
            # compares against a raw silhouette and corrects its *prediction*.
            measured = None
            if all(np.isfinite(segs[i].coverage) for i in usable):
                measured = [self.tilt_cal.unapply(ellipses[i]) for i in usable]
            # `mode="image"` when the evidence maps exist. Both other modes compress a
            # view to one fitted ellipse *before* the joint solve, so an occluded arc has
            # already corrupted what the solve is handed; this one puts both views' raw
            # samples in one residual vector against one pose, and the arc a view has
            # lost is carried by the same arc in the other. See `theory.md` 16.9.
            weights = [maps[i] for i in usable]
            use_image = all(w is not None for w in weights)
            r = refine(
                hulls,
                sub_rig,
                centre,
                normal,
                self.radius_mm,
                loss="cauchy" if use_image else self.loss,
                reference=self.reference,
                tilt_cal=self.tilt_cal,
                centre_cal=self.centre_cal,
                ellipses=measured,
                mode="image" if use_image else "ellipse",
                weights=weights if use_image else None,
            )
            if r is not None:
                centre, normal, rms, iters = r.center, r.normal, r.rms_px, r.n_iter
                if r.samples is not None:
                    # Alive against each view's own median, so a dim view is judged on
                    # its own scale rather than the brighter one's -- the same rule as
                    # `RingFit.coverage`, applied per view instead of per fit.
                    med = np.median(r.samples, axis=1, keepdims=True)
                    alive = r.samples >= segmod.RING_COVERAGE_FLOOR * np.maximum(
                        med, 1e-6)
                    union_cov = float(alive.any(axis=0).mean())

        # The major-axis channel, blended in by measured precision rather than
        # switched to. It is strongest exactly where the axis-ratio channel is
        # weakest, so a threshold would put a discontinuity in the estimate in
        # the middle of the useful range. See `solve_from_major`.
        sigma_major = float("nan")
        if self.use_major_channel and len(usable) > 1:
            got = solve_from_major(
                [segs[i].ellipse for i in usable], sub_rig, centre, self.reference
            )
            if got is not None:
                n_major, sigma_major = got
                tilt_seen = min(sub_rig.tilt_seen_deg(normal))
                normal, _ = blend_normals(
                    normal,
                    ratio_sigma_deg(tilt_seen),
                    n_major,
                    sigma_major,
                    self.reference,
                )

        jump = float("nan")
        if self._prev_normal is not None:
            jump = line_angle_deg(normal, self._prev_normal)
            dt = t - self._prev_t if t is not None and self._prev_t is not None else None
            fresh = dt is not None and 0.0 < dt <= self.dropout_s
            if fresh and self.max_jump_deg_s is not None:
                if jump > self.max_jump_deg_s * dt:
                    # A branch flip, not a manoeuvre. See MAX_JUMP_DEG_PER_S.
                    self.n_rejected_jump += 1
                    self.n_lost += 1
                    return None
        self._prev_normal = normal
        self._prev_t = t
        # Every frame that gets this far, outliers included. The window defends itself
        # with a median rather than by being fed only good frames -- deciding what to
        # feed it would need the answer it exists to provide.
        self._window.append((now, _unit(normal)))
        self.n_detected += 1

        xyz, n_zeroed = self.zero.apply(centre, normal)
        theta, phi = _angles_from_normal(n_zeroed)

        ref = segs[usable[0]]
        ref_cands = cands[usable[0]]

        pose = StereoPose(
            t=now,
            frame_index=self.frame_index,
            xyz_mm=xyz,
            normal=n_zeroed,
            theta_deg=theta,
            phi_deg=phi,
            n_views=len(usable),
            discrepancy_mm=m.discrepancy_mm,
            skew_ms=skew_ms,
            margin=m.margin,
            refine_rms_px=rms,
            refine_iters=iters,
            jump_deg=jump,
            t_seg_ms=t_seg_ms,
            t_est_ms=(time.perf_counter() - t0) * 1e3,
            psi_deg=self.zero.apply_psi(ref.ellipse[2]),
            ellipse=ref.ellipse,
            area_px=ref.area_px,
            fit_rms_px=ref.fit_rms_px,
            ambiguity_margin_deg=conic.ambiguity_margin_deg(ref_cands),
            n_solutions=len(ref_cands),
            union_coverage=union_cov,
            per_view=tuple(segs),
            # `world` is the refined pose *before* zeroing, which is what anything
            # wanting to reproject the rim needs -- the datum can change mid-run.
            extra={"match": m, "candidates": cands, "views_used": usable,
                   "world": (centre, normal), "alive": alive},
        )

        # Predict, then decide. The prediction is attached either way: a gate
        # that silently drops frames is much harder to debug than one whose
        # reasoning is in the log next to the outcome.
        if self.error_model is not None and not self.error_model.is_identity:
            feat = uncertainty_features(pose, self.radius_mm)
            pose.pred_pos_mm, pose.pred_ang_deg = self.error_model.predict(feat)
            if self.target_pos_mm is not None or self.target_angle_deg is not None:
                # Certify against a fraction of the specification, not the
                # specification itself -- see `uncertainty.GATE_MARGIN`. The
                # margin covers the gate's own generalisation error: its
                # threshold was calibrated on finite data and holds in
                # expectation on new data, not with certainty.
                mg = self.gate_margin
                over_pos = (
                    self.target_pos_mm is not None
                    and pose.pred_pos_mm > mg * self.target_pos_mm
                )
                over_ang = (
                    self.target_angle_deg is not None
                    and pose.pred_ang_deg > mg * self.target_angle_deg
                )
                if over_pos or over_ang:
                    self.n_rejected_predicted += 1
                    self.n_lost += 1
                    self.n_detected -= 1
                    return None
        return pose


def _subset_stamps(stamps, indices):
    """The stamps of the views that survived, or None when there were none."""

    return None if stamps is None else [stamps[i] for i in indices]


def _subset(rig, indices):
    """
    A rig holding only the listed cameras, preserving order.

        Needed because a dropped view must not silently shift camera B's extrinsic
        into camera A's slot -- which is the kind of bug that produces a confident
        answer somewhere near the right place.
    """

    if len(indices) == len(rig.cameras):
        return rig
    from rig import StereoRig

    return StereoRig(
        cameras=tuple(rig.cameras[i] for i in indices),
        meta=dict(rig.meta, subset=list(indices)),
    )


# --------------------------------------------------------------------------
# The channel that does not use the minor axis


# Angular precision of the major-axis direction, degrees, modelled as
# sigma_psi = PSI_SIGMA_NUM / sin(tilt). Fitted to the held-out measurements
# (13.15 deg at 5 deg tilt, 0.68 at 25, 0.20 at 45, 0.48 at 65): multiplying each
# by sin(tilt) gives 1.15, 0.29, 0.14, 0.44 -- flat to a factor of two across a
# 60x change in the raw number, which is what makes the 1/sin form the right one.
# The direction of an ellipse's long axis is undefined on a circle, so this must
# diverge at face-on, and it does.
PSI_SIGMA_NUM = 0.25

# Tilt precision of the axis-ratio channel, degrees. Roughly flat where the
# flat-circle model holds and useless past the mast crossover; see
# `validation/tune_weighting.py` for the measurements behind both numbers.
RATIO_SIGMA_IN_BAND = 0.45
RATIO_SIGMA_HIGH_TILT = 5.0
MAST_CROSSOVER_DEG = 52.5


def major_diameter(ellipse, cam, center_world):
    """
    World direction of the rim diameter that projects onto the major axis.

        The major axis of a projected circle is the diameter that is *not*
        foreshortened, so it lies in the circle's plane and is therefore
        perpendicular to the normal.  That single fact is the whole channel: it does
        not involve the minor axis, which is the quantity the rim wall and the mast
        destroy (see the lecture notes, section 12.5).

        Recovered by back-projecting the two major-axis endpoints to rays and
        cutting them at the depth of the known centre.  Exact under orthography and
        accurate to well under the 0.2 degree measured precision of the major-axis
        direction at the ranges here.
    """

    (cx, cy), (major, _), ang = conic.normalise_ellipse(ellipse)
    th = math.radians(ang)
    half = major / 2.0
    pix = np.array(
        [
            [cx + half * math.cos(th), cy + half * math.sin(th)],
            [cx - half * math.cos(th), cy - half * math.sin(th)],
        ]
    )

    c_cam, _ = cam.to_camera(center_world, np.array([0.0, 0.0, 1.0]))
    if c_cam[2] <= 0:
        return None
    rays = np.hstack([pix, np.ones((2, 1))]) @ cam.K_inv.T
    pts = rays * (c_cam[2] / rays[:, 2:3])  # cut at the centre's depth
    d = pts[0] - pts[1]
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return cam.R @ (d / n)


def projected_axis_dir(center_cam, normal_cam, K, delta_mm=5.0):
    """
    Unit image direction the rotor axis projects to. ``None`` if degenerate.

        Needed by `CentreCalibration`, which must know which way the robot leans --
        information the ellipse alone does not carry, since its angle is defined only
        modulo 180 degrees.
    """

    c = np.asarray(center_cam, dtype=np.float64)
    n = _unit(normal_cam)
    p0 = K @ c
    p1 = K @ (c + delta_mm * n)
    if abs(p0[2]) < 1e-9 or abs(p1[2]) < 1e-9:
        return None
    d = p1[:2] / p1[2] - p0[:2] / p0[2]
    m = float(np.linalg.norm(d))
    return None if m < 1e-9 else d / m


def solve_from_major(ellipses, rig, center_world, reference=(0.0, 0.0, 1.0)):
    """
    Rotor normal from the major axes alone. ``None`` if ill-conditioned.

        Each view supplies a diameter ``d_i`` lying in the rim's plane, so
        ``n . d_i = 0``; two views give ``n = d_1 x d_2`` up to sign.  Five degrees of
        freedom come from major axes and centres only, and the minor axis -- whose
        scatter reaches 26% past 60 degrees of tilt -- never enters.

        The conditioning is the mirror image of the axis-ratio channel's.  This one
        fails near face-on, where an ellipse has no defined long axis; the ratio
        channel fails at high tilt, where the silhouette stops being an ellipse.
        Neither is a fallback for the other -- they are complementary, and `fuse`
        weights them by their measured precision rather than switching between them.

        Returns ``(normal, sigma_deg)``: the estimate and how much to trust it.

        **Off by default, and the reason is the sigma, not the geometry.**  Measured
        A/B on identical frames, blending this channel in improves the normal by
        1.65x overall and 2.25x (8.15 -> 3.62 deg) where the rotor is 55-70 degrees
        from the optical axis -- exactly the regime it was built for.  But in the
        45-55 band it makes things slightly worse (2.39 -> 2.80 deg), and because the
        default rig sits a level robot at 45 degrees, that band is the operating
        point: switching it on regresses the shipped configuration from 1.11 to
        2.19 degrees.

        The cause is `PSI_SIGMA_NUM / sin(tilt)`, which claims 0.35 deg at 45 degrees
        against the ratio channel's 0.45 and therefore outvotes it.  That model came
        from the precision of psi in a *single view*; the normal comes from a cross
        product of two of them and is worse than either.  The fix is to fit the
        channel's sigma from measured normal error rather than derive it from psi,
        which needs a dedicated sweep and is not done.  Until then the geometry is
        here, tested and A/B-able via ``use_major_channel=True``, and not enabled.
    """

    ds = [
        major_diameter(e, cam, center_world)
        for e, cam in zip(ellipses, rig.cameras)
        if e is not None
    ]
    ds = [d for d in ds if d is not None]
    if len(ds) < 2:
        return None

    n = np.cross(ds[0], ds[1])
    mag = float(np.linalg.norm(n))
    if mag < 1e-9:
        return None
    n = orient(n / mag, reference)

    # The cross product degrades as the two diameters become parallel, which is
    # what happens when both views see the rotor the same way up. `mag` is the
    # sine of the angle between them, so it is exactly the conditioning number.
    tilt_a = line_angle_deg(n, rig.cameras[0].optical_axis)
    sin_t = max(math.sin(math.radians(tilt_a)), 1e-3)
    sigma = PSI_SIGMA_NUM / sin_t / max(mag, 1e-3)
    return n, float(sigma)


def ratio_sigma_deg(tilt_seen_deg):
    """
    Precision of the axis-ratio tilt channel at a given viewing tilt.

        Flat where the flat-circle model holds, then degrading past the mast
        crossover. Deliberately a step with a ramp rather than a fitted curve: the
        underlying mechanism *is* a threshold (the mast either clears the rim
        silhouette or it does not), and a smooth fit through it would imply a
        gradual onset the geometry does not have.
    """

    t = float(tilt_seen_deg)
    if t <= MAST_CROSSOVER_DEG:
        return RATIO_SIGMA_IN_BAND
    frac = min(1.0, (t - MAST_CROSSOVER_DEG) / 15.0)
    return RATIO_SIGMA_IN_BAND + frac * (RATIO_SIGMA_HIGH_TILT - RATIO_SIGMA_IN_BAND)


def blend_normals(
    n_ratio, sigma_ratio_deg, n_major, sigma_major_deg, reference=(0.0, 0.0, 1.0)
):
    """
    Inverse-variance combination of the two orientation channels.

        Same information-form logic `fuse` uses for position, one dimension down.
        Normals are summed as lines -- sign-aligned first -- because the two channels
        have no shared sign convention.
    """

    if n_major is None:
        return _unit(n_ratio), sigma_ratio_deg
    wr = 1.0 / max(sigma_ratio_deg, 1e-6) ** 2
    wm = 1.0 / max(sigma_major_deg, 1e-6) ** 2
    a = _unit(n_ratio)
    b = _unit(n_major)
    if float(a @ b) < 0:
        b = -b
    out = wr * a + wm * b
    if np.linalg.norm(out) < 1e-12:
        return a, sigma_ratio_deg
    return orient(out, reference), float(1.0 / math.sqrt(wr + wm))
