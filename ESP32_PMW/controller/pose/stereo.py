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
from dataclasses import dataclass, field
from pathlib import Path

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
MAX_DISCREPANCY_MM = 1.25 * BODY_DIAMETER_MM

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
):
    """
    Pick the one branch pair the two views agree on.

        ``candidates`` is a list per view of `conic.CirclePose` in that view's camera
        frame -- exactly what `conic.backproject_ellipse` returns and what
        `estimator.Pose.extra["candidates"]` already carries.

        Returns a `Match`, or ``None`` if any view has no candidates.  With a single
        view it degenerates gracefully: the first candidate, zero margin, which is
        the honest report that nothing was decided.
    """

    if any(not c for c in candidates):
        return None
    if len(candidates) == 1:
        return Match(
            poses=(candidates[0][0],), indices=(0,), discrepancy_mm=0.0, margin=0.0
        )

    cam_a, cam_b = rig.cameras[0], rig.cameras[1]
    info = _pair_information(cam_a, cam_b, sigma_lat_mm, sigma_depth_mm)

    scored = []
    for i, pa in enumerate(candidates[0]):
        for j, pb in enumerate(candidates[1]):
            scored.append((_agreement(pa, pb, cam_a, cam_b, info), i, j))
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

        Returns ``(center_world_mm, normal_world, covariance_3x3)``.
    """

    info = np.zeros((3, 3))
    info_c = np.zeros(3)
    n_acc = np.zeros(3)
    ref_n = None
    eye = np.eye(3)

    for pose, cam in zip(poses, rig.cameras):
        c_w, n_w = cam.to_world(pose.center, pose.normal)
        d = cam.optical_axis
        cov = (sigma_lat_mm**2) * eye + (
            sigma_depth_mm**2 - sigma_lat_mm**2
        ) * np.outer(d, d)
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


def refine(
    hulls,
    rig,
    seed_center,
    seed_normal,
    radius_mm=RADIUS_MM,
    loss="linear",
    f_scale=1.0,
    max_iter=MAX_REFINE_ITER,
    reference=(0.0, 0.0, 1.0),
    tilt_cal=None,
    mode="ellipse",
    ellipses=None,
    params="both",
    centre_cal=None,
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
        @param tilt_cal: what makes either mode work. Turns the prediction from a
            bare rim circle into the silhouette a real robot casts, mast included.
            Without it, refinement improved position on 75% of frames and degraded
            orientation on 78%.
        @param loss: plain least squares. A robust loss changes nothing here (0.509
            vs 0.524 deg): with 8 endpoint residuals per view the ellipse fit has
            already pooled the boundary, so there are no outliers left to reject.
    """

    t1, t2 = _tangent_basis(seed_normal)
    n0 = _unit(seed_normal)
    cams = list(rig.cameras)[: len(hulls if hulls is not None else ellipses)]
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
        f_scale=f_scale,
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
    return Refinement(
        center=centre,
        normal=orient(normal, reference),
        rms_px=float(np.sqrt(np.mean(sol.fun**2))),
        n_iter=int(sol.nfev),
        n_points=total,
        converged=bool(sol.success),
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
        thresh=segmod.THRESH,
        min_area=segmod.MIN_BLOB_AREA_PX,
        verify=True,
        undistort=True,
        tilt_cal=None,
        do_refine=True,
        loss="linear",
        sigma_lat_mm=SIGMA_LAT_MM,
        sigma_depth_mm=SIGMA_DEPTH_MM,
        reference=(0.0, 0.0, 1.0),
        max_discrepancy_mm=MAX_DISCREPANCY_MM,
        use_major_channel=False,
        max_fit_rms_rel=MAX_FIT_RMS_REL,
        centre_cal=None,
        error_model=None,
        require_stereo=True,
        target_pos_mm=None,
        target_angle_deg=None,
        gate_margin=None,
    ):
        self.rig = rig
        self.radius_mm = float(radius_mm)
        self.zero = zero if zero is not None else Zero.identity()
        self.thresh = thresh
        self.min_area = min_area
        self.verify_tol = 1e-6 if verify else None
        self.undistort = undistort
        # Applied to the per-view seed only. The refinement works on raw hull
        # points, where the robust loss is the intended mechanism instead.
        self.tilt_cal = TiltCalibration.load() if tilt_cal is None else tilt_cal
        self.do_refine = do_refine
        self.loss = loss
        self.sigma_lat_mm = sigma_lat_mm
        self.sigma_depth_mm = sigma_depth_mm
        self.reference = np.asarray(reference, dtype=np.float64)
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
        self.error_model = ErrorModel.load() if error_model is None else error_model
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

    def _view_candidates(self, frame, cam):
        """
        Segment one view and back-project it. Returns ``(seg, candidates)``.
        """

        seg = segmod.segment(frame, thresh=self.thresh, min_area=self.min_area)
        if seg is None:
            return None, [], None

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

    def update(self, frames, t=None, frame_index=None):
        """
        Estimate one pose from a list of simultaneous frames.

                ``None`` is a normal outcome and callers must handle it: the robot
                leaves one camera's field of view, an occluder covers it, the threshold
                finds nothing.
        """

        now = time.monotonic() if t is None else float(t)
        self.frame_index = (
            self.frame_index + 1 if frame_index is None else int(frame_index)
        )

        t_seg0 = time.perf_counter()
        segs, cands, ellipses = [], [], []
        for frame, cam in zip(frames, self.rig.cameras):
            s, c, e = self._view_candidates(frame, cam)
            segs.append(s)
            cands.append(c)
            ellipses.append(e)
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
            m.poses, sub_rig, self.sigma_lat_mm, self.sigma_depth_mm, self.reference
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
            worst = max(
                segs[i].fit_rms_px / max(segs[i].ellipse[1][0], 1e-9) for i in usable
            )
            if worst > self.max_fit_rms_rel:
                self.n_rejected_fit += 1
                self.n_lost += 1
                return None

        rms, iters = float("nan"), 0
        if self.do_refine:
            hulls = [self._hull_ideal_px(segs[i], self.rig.cameras[i]) for i in usable]
            r = refine(
                hulls,
                sub_rig,
                centre,
                normal,
                self.radius_mm,
                loss=self.loss,
                reference=self.reference,
                tilt_cal=self.tilt_cal,
                centre_cal=self.centre_cal,
            )
            if r is not None:
                centre, normal, rms, iters = r.center, r.normal, r.rms_px, r.n_iter

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
        self._prev_normal = normal
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
            per_view=tuple(segs),
            extra={"match": m, "candidates": cands, "views_used": usable},
        )

        # Predict, then decide. The prediction is attached either way: a gate
        # that silently drops frames is much harder to debug than one whose
        # reasoning is in the log next to the outcome.
        if not self.error_model.is_identity:
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
