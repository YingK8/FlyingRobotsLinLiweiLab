from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.

from controller.pose import conic
from controller.pose import segment as segmod
from controller.calib.shape import CentreCalibration, TiltCalibration
from controller.pose.uncertainty import ErrorModel
from controller.pose.uncertainty import features as uncertainty_features
from controller.pose.uncertainty import GATE_MARGIN as uncertainty_GATE_MARGIN
from controller.pose.estimator import RADIUS_MM, _angles_from_normal
from controller.calib.zeroing import Zero
from controller.pose.filter import ACCEL_MM_S2

SIGMA_LAT_MM = 0.078
SIGMA_DEPTH_MM = 0.857

# Refinement stops here.  The residual is Sampson distance in pixels and the
# seed is already sub-pixel, so this is about polishing rather than converging;
# more iterations buy nothing and cost frame budget.
# Pose quantisation for the centre-cal displacement cache in `refine`. Coarse enough to
# collapse a finite-difference step, fine enough that the solve recomputes as it travels.
CAL_DISP_TOL_MM = 0.01
CAL_DISP_TOL_N = 1e-5

REFINE_TOL = 1e-4

# The stopping tolerance that pairs with the ANALYTIC Jacobian, `mode="image"` only.
# Separate from `REFINE_TOL` because a tolerance is only meaningful against the Jacobian
# that produced the gradient: 19.3 tuned 1e-4 against forward differences whose step is
# noise-dominated (measured: columnwise cosine ~0.1 against the true gradient), and a
# Jacobian that actually points downhill reaches the same optimum in far fewer steps.
# Measured on the 250-frame replay, against the batched-FD baseline at 1e-4:
#
#   tol    est ms   nfev   refine_rms_px   discrepancy_mm   union_coverage
#   FD 1e-4   6.9      7        5.1642           0.4641           0.8611
#   1e-4      7.8     10        4.7890           0.4641           0.8278
#   1e-3      4.1      4        4.9089           0.4641           0.8778
#   3e-3      3.2      3        5.1027           0.4641           0.8833
#   1e-2      2.4      2        5.4185           0.4641           0.9000
#
# 1e-3 is strictly better than the baseline on every quality axis while being 40%
# faster, so it needs no trade argued for it. 3e-3 buys another 0.9 ms and is still
# better than the baseline; 1e-2 is where `refine_rms_px` crosses it and the pose shift
# jumps 0.25 -> 0.34 mm, which is 19.3's signature of stopping early rather than
# converging. Available, not taken.
#: Trust-region stopping tolerance for the analytic-Jacobian solve.
#:
#: **5e-4, not 19.12's 1e-3, and the difference is jitter.** 1e-3 was chosen to buy speed
#: (6.9 -> 4.1 ms) on the argument that `refine_rms_px` and `union_coverage` improved and
#: `discrepancy_mm` was unmoved -- all true, and none of them a measure of *smoothness*.
#: The solve stops early, so where it stops varies frame to frame, and against the prior
#: that real motion is continuous that variation is noise. Scored by the second difference
#: of the trajectory on the bench take (`pose/theory.md` 16.29):
#:
#:      tol      pos d2 p50   ang d2 p50   discrepancy  coverage  est ms  parity
#:      1e-3      0.2502 mm    0.4099 deg    1.46 mm     0.939     0.37    ok
#:      5e-4      0.1851       0.1851        1.46        0.972     0.47    ok      <- here
#:      3e-4      0.1775       0.1798        1.46        0.989     0.55    BREAKS
#:      1e-4      0.0327       0.0488        1.46        1.000     1.01    BREAKS
#:
#: Every guard is flat or better at every row -- the same 246 frames, the same cross-view
#: discrepancy, rising coverage -- so this is the solve converging further, not smoothing.
#: A fit that had stopped listening to the image would have moved `discrepancy_mm`.
#:
#: **Why not 1e-4, which is 7.6x smoother.** Below 5e-4 the solve runs past the noise floor
#: of its own Jacobian: `_rim_shape`'s two normal columns are forward-differenced at
#: sqrt(eps) (21.3), so the trust region's accept/reject decisions start turning on
#: rounding and the two cores take different numbers of steps. `native_parity` then fails
#: -- p95 stays at 1e-4 mm but the worst frame reaches 0.18 mm and `nfev` differs by 8.
#: That is the ill-conditioning 21.3 already named, surfacing. The port's own chapter calls
#: the parity harness its main value, so it is not spent for jitter without the fix it
#: points at: **exact derivatives for those two columns**, which would remove the noise on
#: both sides and unlock 1e-4. Not done -- it changes the reference's descent direction.
REFINE_TOL_ANALYTIC = 5e-4

# Relative finite-difference step for the batched Jacobian, matching what
# `least_squares(jac="2-point")` picks by default -- sqrt of machine epsilon. Measured
# larger steps converge to a BETTER residual but take more iterations, so this is a
# quality knob pointing away from speed; see `control/theory.md` 19.3.
_JAC_REL_STEP = math.sqrt(np.finfo(float).eps)

# Analytic Jacobian (`theory.md` 19.4's "remaining structural item"). Set False to fall
# back to the batched forward differences it replaced -- both are kept because the
# accuracy protocol in 19.2/19.3 is an A/B against each other.
USE_ANALYTIC_JAC = True

# Pixel step for the ONE thing in the analytic Jacobian still differenced in image
# space: d(raw pixel)/d(ideal pixel). Distortion is a smooth polynomial with no image
# data in it, so this is a plain numerical derivative of an analytic function and 1e-3 px
# sits far above float64 cancellation and far below the lens's curvature scale.
_JAC_DISTORT_STEP_PX = 1e-3

# Central-difference step for the evidence map's image gradient, in pixels. The map is
# read bilinearly, so its gradient is piecewise constant within a pixel cell and
# discontinuous across one; a step of half a cell straddles that, and stays far inside
# the RING_BLUR_SIGMA = 3 features it has to resolve.
_JAC_GRAD_STEP_PX = 0.5

# The residual is sqrt(max(ref - E, 0)), whose derivative -dE/(2r) is singular as r -> 0.
# Below this fraction of sqrt(ref) the sample is treated as carrying no gradient. See
# `jac_analytic`.
_JAC_MIN_RESID_FRAC = 1e-3

_RIM_SHAPE_CACHE = {}

MAX_REFINE_ITER = 12

# The robot's rim diameter, from the mesh. Used as the natural length scale for
# deciding when two views cannot be looking at the same thing.
BODY_DIAMETER_MM = 20.409

# Two-view agreement gate: above this the pair is a blunder, not a noisy fix.
MAX_DISCREPANCY_MM = 5.0

MAX_JUMP_DEG_PER_S = None

# How long a previous normal stays worth comparing against. Matches the monocular
# `estimator.PoseEstimator` default: past this the robot may have done anything and the
# jump gate must stand aside so the track can re-acquire.
DROPOUT_S = 0.25

MAX_FIT_RMS_REL = 0.012

# Where the direct fit's reference level sits in the seed's own ring evidence.
#
# Every sample is scored as a deficit from this, so it decides what "found the rim"
# means. High enough that most of a good ring sits below it and the residual has a
# gradient everywhere; below the maximum, so one specular sample cannot set the bar
# for the whole ring. p90 of the seed's samples.
REF_PERCENTILE = 90

# Below this the evidence map has no ridge on the rim and the ellipse is unpinned.
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
    # Explicit components, not `np.cross`. Identical arithmetic, but np.cross spends most
    # of its time in axis bookkeeping (`normalize_axis_tuple`, `moveaxis`) which for a
    # 3-vector dwarfs the six multiplies. `refine`'s residual calls this on every
    # evaluation, ~47 a frame, so it showed up as 124k bookkeeping calls in the profile.
    nx, ny, nz = n
    # seed = +x, or +y when n is too close to +x for the cross to be well conditioned.
    if abs(nx) < 0.9:
        t1 = np.array([0.0, nz, -ny])          # n x (1,0,0)
    else:
        t1 = np.array([-nz, 0.0, nx])          # n x (0,1,0)
    t1 = _unit(t1)
    ax, ay, az = t1
    return t1, np.array([ny * az - nz * ay, nz * ax - nx * az, nx * ay - ny * ax])


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
        ``margin`` is how many sigmas better the winner was than the runner-up.  Both
        are logged every frame.  A small margin means the two cameras could not tell the
        branches apart -- which happens legitimately when the rotor is near face-on
        to both, where the branches merge and the choice stops mattering -- and it is
        the number to look at before believing an orientation outlier.
    """

    poses: tuple  # one conic.CirclePose per view, in that view's camera frame
    indices: tuple  # which candidate was taken from each view
    discrepancy_mm: float  # the winning pair's disagreement, as a plain distance
    margin: float  # how many sigmas better the winner was than the runner-up

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

    c = np.asarray(center_world, dtype=np.float64).reshape(3)
    return c + _rim_shape(np.asarray(normal_world, dtype=np.float64).reshape(3),
                          radius_mm, phis)


def _rim_shape(normal, radius_mm, phis):
    """The rim offsets from the centre: depends on the NORMAL only, so it is cached.

        `least_squares` differentiates 5 parameters, and 3 of them are the centre --
        which shifts the rim rigidly and leaves this untouched. Caching it means the
        tangent basis and the two outer products are computed twice per Jacobian
        instead of six times. Exact: keyed on the normal's own bytes, so a changed
        normal cannot hit a stale entry.
    """

    key = (normal.tobytes(), radius_mm, phis.shape[0])
    hit = _RIM_SHAPE_CACHE.get(key)
    if hit is None:
        u, v = _tangent_basis(normal)
        hit = radius_mm * (np.outer(np.cos(phis), u) + np.outer(np.sin(phis), v))
        if len(_RIM_SHAPE_CACHE) > 64:
            _RIM_SHAPE_CACHE.clear()
        _RIM_SHAPE_CACHE[key] = hit
    return hit


def _project_ideal(pts_world, cam, with_cam_frame=False):
    """``(ideal pinhole pixels, in front of the lens)``. Distortion is applied after.

        `with_cam_frame` also returns the camera-frame points and the guarded depth.
        `refine`'s analytic Jacobian needs both to build d(pixel)/d(world point), and
        recomputing them there would be a second copy of the projection convention --
        the one thing that must not exist twice.
    """

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
    return (out, ~behind, p, z) if with_cam_frame else (out, ~behind)


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

        # The centre-cal displacement, cached on a coarsely quantised pose. It cost two
        # conic decompositions per view per evaluation -- 222 `cone_from_circle` calls a
        # frame, more than the projection it corrects -- and `least_squares` spends 6 of
        # every 7 evaluations on finite differences, so quantising the pose at
        # CAL_DISP_TOL_MM collapses those 6 to one lookup.
        #
        # MEASURED COST, not free: the Jacobian loses the displacement's own derivative,
        # which moves the answer 0.14 mm median (p95 0.34) and raises `refine_rms_px`
        # from 5.112 to 5.126 -- 0.3% on a residual of 5 px, with `discrepancy_mm` and
        # `margin` unchanged, i.e. the optimum is flat over that 0.14 mm. Tightening the
        # quantum to 1e-6 mm does NOT recover it; collapsing the finite differences is
        # the whole effect. Bought 17.0 -> 12.5 ms a pair. Freezing the displacement at
        # the seed instead is 10.1 ms for the same 0.13 mm, but it goes stale when the
        # solve travels, where this recomputes. See `theory.md` 16.9.
        _disp = {}

        def displacement(centre, normal, cam):
            key = (id(cam), *np.round(centre / CAL_DISP_TOL_MM).astype(np.int64),
                   *np.round(normal / CAL_DISP_TOL_N).astype(np.int64))
            if key not in _disp:
                try:
                    ideal = conic.normalise_ellipse(conic.ellipse_from_conic(
                        _predict_image_conic(centre, normal, cam, radius_mm)))
                    corr = _predict_ellipse(centre, normal, cam, radius_mm,
                                            tilt_cal, centre_cal)
                    _disp[key] = np.subtract(corr[0], ideal[0])
                except (ValueError, np.linalg.LinAlgError):
                    # A trust region does reach poses whose image conic is not a real
                    # ellipse -- edge-on, or behind the lens. Those samples are already
                    # parked as unseen, so skipping the correction only decides whether
                    # the solve survives to walk back out.
                    _disp[key] = None
            return _disp[key]

        def evidence_many(ps):
            """Evidence for several poses at once: ``(len(ps), total)``.

                One projection and one `sample_map` per VIEW for the whole batch instead
                of one per view per pose. That matters because the per-call cost is
                mostly fixed: measured on this rig, project+distort is 22.9 us for 45
                points, 40.9 for 180 and 132.6 for 900 -- about 15 us of overhead plus
                ~120 ns a point. Five separate 180-point calls cost 205 us where one
                900-point call costs 133.
            """

            rims, disps = [], []
            for p in ps:
                centre, normal = unpack(p)
                rims.append(_rim_points(centre, normal, radius_mm, phis))
                disps.append(None if centre_cal is None or centre_cal.is_identity
                             else [displacement(centre, normal, c) for c in cams])
            rim = np.concatenate(rims)
            m = len(ps)
            out = np.empty((m, total))
            k = 0
            for vi, (w, cam) in enumerate(zip(weights, cams)):
                pts, seen = _project_ideal(rim, cam)
                pts = np.ascontiguousarray(pts)
                if disps[0] is not None:
                    # Per pose, so each block gets its own sub-pixel centre correction.
                    for j, d in enumerate(disps):
                        if d[vi] is not None:
                            pts[j * n:(j + 1) * n] += d[vi]
                np.clip(pts, -_FAR_PX, _FAR_PX, out=pts)
                pts = np.array(_distort_points(pts, cam), dtype=np.float64, copy=True)
                pts[~seen] = -_FAR_PX
                out[:, k:k + n] = segmod.sample_map(w, pts).reshape(m, n)
                k += n
            return out

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
                    d = displacement(centre, normal, cam)
                    if d is not None:
                        pts += d
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

        def jac_fd(p):
            """Forward differences, all five columns in ONE batched evaluation.

                Arithmetically the same thing `least_squares(jac='2-point')` does -- same
                relative step, same forward difference -- but it costs one pass over
                5*n_points instead of five passes over n_points, and the solve is
                overhead-bound rather than arithmetic-bound (see `evidence_many`).
                Roughly 80% of the solve was this Jacobian.
            """

            f0 = np.sqrt(np.maximum(ref - evidence(p), 0.0))
            # scipy's own rule for 2-point, sign included: `rel * sign(x) * max(1, |x|)`.
            # The sign is not cosmetic -- dropping it steps the negative parameters the
            # other way and the answer moves, which is exactly how this was caught.
            h = (_JAC_REL_STEP * np.where(p >= 0, 1.0, -1.0)
                 * np.maximum(np.abs(p), 1.0))
            ev = evidence_many(p + np.diag(h))
            return ((np.sqrt(np.maximum(ref - ev, 0.0)) - f0) / h[:, None]).T

        n_centre = 3 if params == "both" else 0

        def jac_analytic(p):
            """Image gradient of the evidence map times the pixel-vs-pose derivative.

                `theory.md` 19.4's "remaining structural item". `jac_fd` batches the five
                extra evaluations into one pass; this removes them. Per view per
                iteration it costs ONE projection over n points, ONE distortion call
                over 3n, and three n-point remaps, against `jac_fd`'s projection and
                distortion over 6n.

                The chain, for residual = sqrt(max(ref - E, 0)):

                    d(res)/dp = -(1 / 2 res) * grad_w(pix) . d(pix)/d(rim) . d(rim)/dp

                Only two of those four factors are differenced, and neither touches the
                image: **d(rim)/d(normal)** in 3-space (two cached `_rim_shape` lookups)
                and **d(raw pixel)/d(ideal pixel)**, which is a smooth polynomial. The
                three centre columns of d(rim)/dp are exactly the identity -- the same
                fact `_rim_shape` is cached on -- and the pinhole term is closed form.

                Two derivatives are deliberately dropped, both already accepted upstream:
                the centre-cal `displacement` is held constant (19.5 measured that at
                0.14 mm and took it), and `ref` is fixed before the solve starts.
            """

            centre, normal = unpack(p)
            rim = _rim_points(centre, normal, radius_mm, phis)

            # World-space d(rim point)/d(param), (n, npar, 3).
            D = np.zeros((n, len(p), 3))
            for i in range(n_centre):
                D[:, i, i] = 1.0        # the centre shifts the rim rigidly
            base = _rim_shape(normal, radius_mm, phis)
            for j in range(n_centre, len(p)):
                # scipy's own step rule, sign included, as in `jac_fd`: a normal
                # parameter is an increment on the tangent basis, so this differences
                # pure geometry and samples no pixels.
                h = (_JAC_REL_STEP * (1.0 if p[j] >= 0 else -1.0)
                     * max(abs(p[j]), 1.0))
                q = np.array(p, dtype=np.float64)
                q[j] += h
                D[:, j, :] = (_rim_shape(unpack(q)[1], radius_mm, phis) - base) / h

            out = np.zeros((total, len(p)))
            k = 0
            for w, cam in zip(weights, cams):
                pts, seen, pc, z = _project_ideal(rim, cam, with_cam_frame=True)
                pts = np.ascontiguousarray(pts)
                if centre_cal is not None and not centre_cal.is_identity:
                    d = displacement(centre, normal, cam)
                    if d is not None:
                        pts += d
                np.clip(pts, -_FAR_PX, _FAR_PX, out=pts)

                # d(ideal pixel)/d(param), through the camera frame. `p_cam` is
                # `(rim - T) @ R`, so a world-space delta maps to `delta @ R`, and the
                # pinhole derivative is [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]].
                dc = D @ cam.R
                zi = (1.0 / z)[:, None]
                du = cam.K[0, 0] * (dc[..., 0] - pc[:, 0:1] * dc[..., 2] * zi) * zi
                dv = cam.K[1, 1] * (dc[..., 1] - pc[:, 1:2] * dc[..., 2] * zi) * zi

                # d(raw pixel)/d(ideal pixel), differenced in ONE call over 3n points
                # rather than three over n -- the projection is overhead-bound, so the
                # stacking is what makes this cheaper than the thing it replaces (19.4).
                hp = _JAC_DISTORT_STEP_PX
                stack = np.concatenate([pts, pts + [hp, 0.0], pts + [0.0, hp]])
                sd = np.asarray(_distort_points(stack, cam), dtype=np.float64)
                d_du, d_dv = (sd[n:2 * n] - sd[:n]) / hp, (sd[2 * n:] - sd[:n]) / hp
                # A copy, not a view: with no distortion `_distort_points` hands back
                # its own input, and the park below would write into `stack`.
                pd = np.array(sd[:n], dtype=np.float64, copy=True)
                pd[~seen] = -_FAR_PX    # parked, as in `evidence`: reads zero everywhere

                ddu = d_du[:, 0:1] * du + d_dv[:, 0:1] * dv
                ddv = d_du[:, 1:2] * du + d_dv[:, 1:2] * dv

                # The image gradient, as a central difference OF THE SAMPLED FIELD --
                # not a Sobel of the map. `sample_map` reads bilinearly, so the function
                # the solver is actually descending is piecewise-linear between pixel
                # centres; a Sobel gradient is the derivative of a *different*,
                # 3x3-smoothed function. Measured, that inconsistency cost the trust
                # region its step: columnwise cosine against a true numerical Jacobian
                # was 0.84-0.99 and nfev rose from 7 to 10. All five reads go in ONE
                # remap over 5n points, which is why this is still cheaper than
                # differencing the pose (a remap is ~2 us; a projection is ~15 us fixed).
                hg = _JAC_GRAD_STEP_PX
                g = segmod.sample_map(w, np.concatenate([
                    pd, pd + [hg, 0.0], pd - [hg, 0.0],
                    pd + [0.0, hg], pd - [0.0, hg]])).reshape(5, n)
                dE = (((g[1] - g[2]) / (2.0 * hg))[:, None] * ddu
                      + ((g[3] - g[4]) / (2.0 * hg))[:, None] * ddv)

                r = np.sqrt(np.maximum(ref - g[0], 0.0))
                # d/dp sqrt(max(ref - E, 0)) = -dE / 2r, singular as r -> 0. That is the
                # max()'s own kink: a sample sitting exactly at the reference. Forward
                # differences step across it and get a finite secant; this cannot, so
                # such a sample is given no gradient rather than an enormous one. It
                # keeps no information either way -- its residual is already zero.
                live = r > _JAC_MIN_RESID_FRAC * math.sqrt(ref)
                out[k:k + n] = np.where(
                    live[:, None], -dE / (2.0 * np.where(live, r, 1.0))[:, None], 0.0)
                k += n
            return out

        jac = jac_analytic if USE_ANALYTIC_JAC else jac_fd

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
    # Jacobian rather than guessing.
    opts = dict(
        method="trf",
        loss=loss,
        f_scale=1.0 if f_scale is None else float(f_scale),
        x_scale="jac",
        jac=jac if mode == "image" else "2-point",
        max_nfev=max_iter * 6,
        **dict.fromkeys(
            ("xtol", "ftol", "gtol"),
            REFINE_TOL_ANALYTIC
            if (mode == "image" and USE_ANALYTIC_JAC) else REFINE_TOL),
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
        sigma_lat_mm=None,
        sigma_depth_mm=None,
        noise=None,
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
        # Pixel scale, set on the first frame from the rig's calibration `image_size`.
        # Every pixel constant here -- the ring kernel, the blob floor, the intrinsics --
        # is quoted at that resolution, so running the sensor at a smaller mode has to
        # rescale all three together or segmentation silently stops finding the rim.
        self._px_scale = 1.0
        self._ksize = None
        # Two workers, one per view, alive for the estimator's life: at ~100 Hz the pool
        # would otherwise spend more time spawning threads than segmenting.
        self._pool = ThreadPoolExecutor(max_workers=len(self.rig.cameras))
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
        # Measured scales when a static calibration exists, the S13 floor otherwise.
        # An explicit argument beats both: a caller that passes one means it.
        if noise is None:
            from controller.pose.noise import NoiseModel
            noise = NoiseModel.load()
        self.noise = noise
        self.sigma_lat_mm = (
            noise.sigma_lat_mm if sigma_lat_mm is None else sigma_lat_mm)
        # ponytail: the depth scale is a fraction of range but the fusion wants a
        # fixed pair, so it is frozen at the middle of the measured envelope. The
        # ratio it sets therefore drifts at the ends of the working range. Evaluate
        # per-frame in `fuse` if that ever shows up in the discrepancy statistics.
        self.sigma_depth_mm = (
            noise.sigma_depth_mm(noise.ref_z_mm) if sigma_depth_mm is None
            else sigma_depth_mm)
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

    def _match_scale(self, frame):
        """Rescale the rig and the pixel constants to the frame actually arriving.

            The sensor's 640x400 mode is a true 0.5x of the 1280x800 the rig was
            calibrated at, so `fx, fy, cx, cy` halve and the distortion coefficients do
            not (`rig.Camera.scaled`). Done here rather than at the caller because both
            the live loop and `from_recording` would otherwise each have to remember.
        """

        wh = self.rig.meta.get("image_size")
        h, w = frame.shape[:2]
        if not wh or (w, h) == tuple(wh):
            return
        s = w / float(wh[0])
        if abs(s - h / float(wh[1])) > 1e-3:
            raise ValueError(
                f"frame {w}x{h} is not a uniform rescale of the calibrated {tuple(wh)}; "
                f"a non-square rescale would need separate fx and fy factors")
        if s == self._px_scale:
            return
        self.rig = self.rig.scaled(s / self._px_scale)
        self.min_area *= (s / self._px_scale) ** 2
        # Odd, and never below 3: an even kernel has no centre pixel to sit the
        # top-hat on, and 1 makes the opening the identity.
        self._ksize = max(3, int(round(segmod.RING_KSIZE * s)) | 1)
        self._px_scale = s
        print(f"pose: frames are {w}x{h}, rig calibrated at {wh[0]}x{wh[1]} -- "
              f"intrinsics x{s:.3f}, ring kernel {self._ksize}px, "
              f"min blob {self.min_area:.0f}px", file=sys.stderr)

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
        # A running plate returns a new array each frame, so `ring_weight` cannot key its
        # response cache on the buffer. Version it by the plate's own frame count instead.
        bg_version = (cam.name, getattr(plate, "n", 0) // segmod.PLATE_REFRESH_FRAMES) \
            if hasattr(plate, "update") else None
        if hasattr(plate, "update"):
            # A `background.RunningPlate`: estimated from the stream itself, so the live
            # loop needs no captured plate and cannot be running on a stale one. Duck
            # typed rather than imported -- `background` pulls in the camera module.
            plate = plate.update(gray)

        if self.direct:
            roi = segmod.ellipse_roi(self._prev_ellipse.get(cam.name), gray.shape)
            w = segmod.ring_weight(gray, background=plate, bg_version=bg_version,
                                   roi=roi, ksize=self._ksize)
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

                The frames are **not** simultaneous. Two free-running cameras land up
                to one sensor frame period apart -- 7.71 ms median at 119 fps on this
                bench. Pass ``stamps``
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

        self._match_scale(frames[0])
        t_seg0 = time.perf_counter()
        # The two views share nothing -- separate plates, separate `_prev_ellipse` keys,
        # separate response-cache keys -- and every stage inside is cv2 or numpy, which
        # release the GIL. So this is a real 2x on the segmentation share, not a fake one.
        out = list(self._pool.map(
            lambda a: self._view_candidates(*a), zip(frames, self.rig.cameras)))
        segs, cands, ellipses, maps = ([x[k] for x in out] for k in range(4))
        t_seg_ms = (time.perf_counter() - t_seg0) * 1e3

        t0 = time.perf_counter()
        usable = [i for i, c in enumerate(cands) if c]
        if not usable:
            self.n_lost += 1
            return None

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

        return self._gate_predicted(pose)


    def _gate_predicted(self, pose):
        """The predicted-error gate, last. Shared with `stereo_native`, which builds the
        same `StereoPose` from the compiled core and must be gated identically."""

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
    from controller.calib.rig import StereoRig

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


def _self_check():
    """The claims in `refine`'s fast paths that are supposed to be EXACT."""

    rng = np.random.default_rng(0)
    phis = np.linspace(0.0, 2.0 * np.pi, segmod.RING_SAMPLES, endpoint=False)

    # 1. `_tangent_basis` really is an orthonormal basis of the plane perpendicular to n,
    #    including near the seed-switch at |n_x| = 0.9 where the cross degenerates.
    for n in [rng.normal(size=3) for _ in range(2000)] + [
            np.array([1.0, 0.0, 0.0]), np.array([0.9, 1e-9, 1e-9]),
            np.array([0.89, 0.1, 0.1]), np.array([0.0, 0.0, 1.0])]:
        if np.linalg.norm(n) < 1e-9:
            continue
        u, v = _tangent_basis(n)
        w = _unit(n)
        assert abs(u @ w) < 1e-12 and abs(v @ w) < 1e-12, n
        assert abs(u @ v) < 1e-12, n
        assert abs(np.linalg.norm(u) - 1) < 1e-12, n
        assert abs(np.linalg.norm(v) - 1) < 1e-12, n

    # 2. The `_rim_shape` cache is EXACT, not approximate. It exists because three of
    #    least_squares' five perturbations move only the centre; if it ever returns a
    #    shape for the wrong normal the solve silently fits the wrong circle.
    for _ in range(500):
        c, n = rng.normal(size=3) * 50.0, rng.normal(size=3)
        if np.linalg.norm(n) < 1e-9:
            continue
        u, v = _tangent_basis(n)
        want = (np.asarray(c, dtype=np.float64).reshape(3)
                + 6.0 * (np.outer(np.cos(phis), u) + np.outer(np.sin(phis), v)))
        assert np.array_equal(_rim_points(c, n, 6.0, phis), want), "rim cache is not exact"

    # 3. A cached shape is never handed out for a different normal, and never mutated by
    #    a later call. The cache is keyed on bytes, so this is really a test that nothing
    #    downstream writes into what it was given.
    n0 = np.array([0.0, 0.0, 1.0])
    first = _rim_shape(n0, 6.0, phis).copy()
    for _ in range(100):
        _rim_points(rng.normal(size=3) * 50.0, rng.normal(size=3), 6.0, phis)
    assert np.array_equal(_rim_shape(n0, 6.0, phis), first), "cached shape was mutated"

    # 4. The analytic Jacobian actually points downhill. Plant a pose, render its rim
    #    into two synthetic evidence maps, and seed `refine` well away from it: a wrong
    #    chain rule cannot recover the planted centre, and a right one lands on it.
    #    Synthetic on purpose -- there is no noise, no plate and no occlusion here, so a
    #    failure is the derivative and nothing else.
    #
    #    This is also the measurement that retired `theory.md` 19.4's open item: on this
    #    same scene the batched forward differences it replaced land 2.33 mm out, fifty
    #    times worse, because scipy's default step is far below the evidence map's
    #    float32 pixel resolution and differences rounding rather than signal.
    def _cam_at(az_deg, el_deg, dist, name, size=(640, 400)):
        a, e = math.radians(az_deg), math.radians(el_deg)
        eye = dist * np.array([math.cos(e) * math.cos(a),
                               math.cos(e) * math.sin(a), math.sin(e)])
        z = -eye / np.linalg.norm(eye)
        x = np.cross([0.0, 0.0, 1.0], z)
        x /= np.linalg.norm(x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        T[:3, 3] = eye
        return Camera(K=np.array([[900.0, 0, size[0] / 2],
                                  [0, 900.0, size[1] / 2], [0, 0, 1.0]]),
                      dist=np.zeros(5), T_world_cam=T, name=name)

    def _render(cam, centre, normal, radius, size=(640, 400)):
        rim = _rim_points(centre, normal, radius,
                          np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False))
        pts, seen = _project_ideal(rim, cam)
        m = np.zeros((size[1], size[0]), np.float32)
        for (u, v), ok in zip(pts, seen):
            iu, iv = int(round(u)), int(round(v))
            if ok and 0 <= iv < size[1] and 0 <= iu < size[0]:
                m[iv, iu] = 1.0
        return cv2.GaussianBlur(m, (0, 0), segmod.RING_BLUR_SIGMA)

    from controller.calib.rig import Camera, StereoRig

    rig = StereoRig(cameras=(_cam_at(0, 45, 300, "A"), _cam_at(83, 45, 300, "B")))
    radius, c_true = 10.2, np.array([2.0, -3.0, 1.5])
    n_true = _unit([0.08, -0.05, 1.0])
    maps = [_render(c, c_true, n_true, radius) for c in rig.cameras]
    errs = []
    for _ in range(12):
        r = refine(None, rig, c_true + rng.normal(size=3) * 1.5,
                   _unit(n_true + rng.normal(size=3) * 0.05), radius,
                   loss="cauchy", mode="image", weights=maps)
        errs.append(np.inf if r is None else float(np.linalg.norm(r.center - c_true)))
    errs = np.array(errs)
    assert np.median(errs) < 0.25, f"analytic Jacobian lost the planted pose: {errs}"
    assert errs.max() < 1.0, f"analytic Jacobian has a bad seed: {errs}"

    print(f"stereo: tangent basis orthonormal incl. the |n_x|=0.9 switch, "
          f"rim cache exact over 500 poses, {len(_RIM_SHAPE_CACHE)} entries live, "
          f"analytic jac recovers a planted pose to "
          f"{np.median(errs):.3f} mm median over 12 seeds")


if __name__ == "__main__":
    _self_check()
