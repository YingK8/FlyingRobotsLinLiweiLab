"""
Frame in, 5-DOF pose out.

Chains `segment` -> `conic.backproject_ellipse` -> `zeroing`, and owns the one
piece of state the chain needs: which of the two back-projection solutions we
took last frame.

Why five degrees of freedom and not six.  The robot's roll about its own spin
axis is not recoverable: it spins at 310-350 Hz against a camera an order of
magnitude slower, so blade position aliases beyond rescue.  The observable state
is position in R^3 plus the rotor normal on S^2, matching
`docs/pose_localization_project_context.md`.  The sixth reported channel, `psi`,
is the measured in-plane angle of the image ellipse -- not spin.

The two-fold ambiguity.  A tilted circle and its mirror image through the
viewing axis project to the identical ellipse, so every frame yields two poses
and no single frame can choose between them.  We pick the one closest to where
we were last frame, falling back to the zero datum and then to a face-on prior.
`ambiguity_margin_deg` records how far apart the candidates were, so a run can
be audited afterwards for frames where the pick was a coin toss.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import conic  # noqa: E402
import segment as segmod  # noqa: E402
from shape import TiltCalibration  # noqa: E402
from zeroing import Zero  # noqa: E402

# Effective rim radius, in mm.
#
# Starts from the mesh: the *outer* radius of
# ESP32_PMW/controller/flyingrobot_rod2.STL (99.9th percentile of
# vertex radius, 10.2065 mm), not the mean rim radius (9.965 mm), because the
# segmenter returns a convex hull and a hull traces the outermost surface. Using
# the mean instead biases every distance by 2.4% -- a systematic 5 mm at 200 mm
# that no filtering removes.
#
# Then tuned: depth scales exactly linearly with the assumed radius, so a scalar
# error shows up as a constant *relative* depth bias and is directly measurable.
# Fitted on 700 training poses rendered with realistic sensor noise and motion
# blur, where the residual bias was -0.374%.
#
# The same value serves the weighted and unweighted fits, to 0.03%.
#
# The fit uses the *median* bias, not the mean, and the difference is not
# cosmetic: about 1% of noisy frames fail catastrophically (the face-on, dimly
# lit case where the hull collapses onto the blade cross, depth error to 409%).
# On that data the mean reads +1.09% and the median -0.37% -- opposite signs.
# Fitting to the mean moved this constant the wrong way and tripled the median
# position error, from 1.49 to 3.30 mm.
#: One entry per rig appearance: the effective radius depends on where the
#: threshold cuts the boundary, and the two appearances cut different edges.
#:
#: `dark` was carried over from a dataset rendered at a different threshold and was
#: marked provisional, with refitting blocked on the renderer being unable to reproduce
#: this rig. The renderer is not needed: two cameras 83 deg apart disagree about where
#: the robot is by an amount that depends on the radius assumed, and that disagreement
#: has a sharp minimum. Three flights over two days, swept independently
#: (`fit_radius.py`), all land on 10.4 mm -- median cross-view discrepancy 2.2 / 2.4 /
#: 3.2 mm against 5.96 at the old value, and 10.38 by 11.0 mm.
#:
#: **Re-fitted to 10.2 for the direct fit** (`theory.md` S16). It was tied to
#: `segment.DARK_THRESH` because the effective radius is where a *threshold* cuts a
#: shaded edge, and 10.4 belonged to a threshold of 220. `segment.fit_ellipse_image`
#: does not cut anything: it settles on the rim's darkness ridge, which is a different
#: and better-defined edge, so the constant had to move with it. The same three flights
#: swept independently now all land on 10.2, with sharper minima than before -- median
#: cross-view discrepancy **0.83 / 1.17 / 1.04 mm** against 2.4 / 1.5 / 1.9 at 10.4 on
#: the mask fit. That is the number to re-run (`fit_radius.py`) if either the direct
#: fit's constants or the lighting move.
#:
#: Its absolute scale still inherits the rig's, which rests on the board pitch (S14.4).
#: `bright` now names two different things and that is a trap. 10.2446 was fitted on
#: *renders* of a white body against a dark ground, and the render tests still use it.
#: The bench rig as of 2026-08-28 is also `bright` -- a white ring on a black cloth --
#: and the direct fit measures **10.05** on it, a sharp V (1.22 mm at the minimum,
#: 9.19 at 9.5 and 9.96 at 10.5). Pass `radius_mm=` explicitly for the bench rig until
#: the two are given separate appearance keys.
RADIUS_BY_APPEARANCE = {
    "bright": 10.2446,  # white body on dark ground, fitted on renders
    "dark": 10.2,       # direct fit on flight data, see above
}

#: The bench rig's measured radius, for `bright` footage that is not a render.
#:
#: **10.2, not the 9.95 fitted against the direct fit.** This constant tracks whichever
#: ellipse is actually reported, and the effective radius is a property of *which edge
#: the measurement cuts* (S16.7). 9.95 belonged to `segment.fit_ellipse_image`, which
#: settles on the rim's darkness ridge -- its centre-line. That fit has been removed
#: (`theory.md` 16.24) and the mask fit reports instead, which hulls the rim's **outer**
#: edge, so the constant moves outward with it. Swept with the mask fit reporting:
#:
#:   radius mm            9.95   10.10  10.20
#:   131552  under 5 mm   74.6%  98.1   98.1
#:           discrepancy p50  4.50   1.85   1.08
#:   135533  under 5 mm   34.7%  95.9   95.3
#:           discrepancy p50  5.35   2.73   1.21
#:
#: Both flights agree, and the sharpness of it is the point: at 9.95 barely a third of
#: `135533` clears the 5 mm gate and at 10.20 it is 95%. 10.10 is marginally better on
#: orientation (>30 deg out 3.2/1.5% against 3.7/2.0) and 10.20 clearly better on
#: position, so 10.20 is taken. It lands where `RADIUS_BY_APPEARANCE["dark"]` already
#: sits, which is the same measurement on the same kind of edge.
RADIUS_BENCH_MM = 10.2
RADIUS_MM = RADIUS_BY_APPEARANCE[segmod.APPEARANCE]

DEFAULT_INTRINSICS = (
    Path(__file__).resolve().parents[1] / "calib" / "assets" / "camera_intrinsics.npz"
)


def load_intrinsics(path=DEFAULT_INTRINSICS):
    """
    Read ``camera_matrix`` and ``dist_coeffs`` from the calibration npz.
    """

    d = np.load(Path(path))
    return (
        np.asarray(d["camera_matrix"], dtype=np.float64),
        np.asarray(d["dist_coeffs"], dtype=np.float64).ravel(),
    )


@dataclass
class Pose:
    """
    One frame's estimate, in the datum frame set by `zeroing.Zero`.

        ``theta_deg`` is tilt away from the datum axis and ``phi_deg`` the azimuth
        that tilt points along -- together they are the normal on S^2, just in the
        form that plots readably.  ``psi_deg`` is the image ellipse's major-axis
        angle; it is geometrically tied to ``phi_deg``, and is carried separately
        precisely so the two can be compared as a consistency check on the fit.

        ``jump_deg`` is how far the normal moved since the previous frame (NaN on
        the first frame and after a dropout).  Deliberately a raw measurement rather
        than a "did it flip" verdict: read against ``ambiguity_margin_deg`` it tells
        you whether a large step was real motion or a branch flip, and that
        judgement belongs to whoever reads the log rather than to a threshold buried
        in here.
    """

    t: float
    frame_index: int
    xyz_mm: np.ndarray
    normal: np.ndarray
    theta_deg: float
    phi_deg: float
    psi_deg: float
    ellipse: tuple
    area_px: float
    fit_rms_px: float
    ambiguity_margin_deg: float
    n_solutions: int
    jump_deg: float
    t_seg_ms: float
    t_est_ms: float
    extra: dict = field(default_factory=dict)

    @property
    def t_total_ms(self):
        return self.t_seg_ms + self.t_est_ms


def _angles_from_normal(normal):
    """
    Normal -> (tilt from +z, azimuth of that tilt), both in degrees.

        Azimuth is meaningless when the tilt is zero; it returns 0 there rather than
        whatever ``atan2(0, 0)`` yields, so a level hover plots a flat line instead
        of noise.
    """

    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    theta = math.degrees(math.acos(float(np.clip(abs(n[2]), -1.0, 1.0))))
    lateral = math.hypot(n[0], n[1])
    phi = math.degrees(math.atan2(n[1], n[0])) if lateral > 1e-9 else 0.0
    return theta, phi


class PoseEstimator:
    """
    Stateful per-frame estimator.

        State is only what disambiguation needs: the previously chosen normal, and
        how long ago it was seen.  After ``dropout_s`` without a detection the
        history is treated as stale and the pick falls back to the datum, so the
        estimator cannot latch onto a solution branch from before an occlusion.
    """

    def __init__(
        self,
        camera_matrix=None,
        dist_coeffs=None,
        radius_mm=RADIUS_MM,
        zero=None,
        thresh=None,
        background=None,
        min_area=segmod.MIN_BLOB_AREA_PX,
        dropout_s=0.25,
        verify=True,
        undistort=True,
        tilt_cal=None,
    ):
        if camera_matrix is None:
            camera_matrix, loaded_dist = load_intrinsics()
            if dist_coeffs is None:
                dist_coeffs = loaded_dist

        self.K = np.asarray(camera_matrix, dtype=np.float64)
        self.dist = (
            None if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
        )
        self.radius_mm = float(radius_mm)
        self.zero = zero if zero is not None else Zero.identity()
        # None, not `segmod.THRESH`: the level belongs to the appearance, and
        # `segment.score_channel` resolves it at call time.
        self.thresh = thresh
        self.background = background      # this camera's empty-rig plate, see `segment`
        self.min_area = min_area
        self.dropout_s = dropout_s
        self.verify_tol = 1e-6 if verify else None
        self.undistort = undistort
        # Loaded from disk by default so a fresh estimator gets the shipped
        # correction; pass TiltCalibration() explicitly to run uncorrected.
        self.tilt_cal = TiltCalibration.load() if tilt_cal is None else tilt_cal

        self.frame_index = -1
        self.n_detected = 0
        self.n_lost = 0
        self._prev_normal = None
        self._prev_t = None

    def reset(self):
        """
        Forget branch history without touching calibration or the datum.
        """

        self._prev_normal = None
        self._prev_t = None

    def _prior_normal(self):
        """
        Which way we expect the rotor to face, absent recent history.

                The datum's axis if one is set, else straight at the camera.  Normals
                from `conic.backproject` are oriented toward the camera, so the face-on
                prior is -z.
        """

        if not self.zero.is_identity:
            return self.zero.R[:, 2]
        return np.array([0.0, 0.0, -1.0])

    def _choose(self, poses, now):
        """
        Resolve the two-fold ambiguity. Returns ``(pose, jump_deg)``.
        """

        stale = (
            self._prev_normal is None
            or self._prev_t is None
            or (now - self._prev_t) > self.dropout_s
        )
        ref = self._prior_normal() if stale else self._prev_normal

        chosen = (
            poses[0]
            if len(poses) == 1
            else max(poses, key=lambda p: float(p.normal @ ref))
        )
        if stale:
            return chosen, float("nan")

        cos = float(np.clip(chosen.normal @ self._prev_normal, -1.0, 1.0))
        return chosen, math.degrees(math.acos(cos))

    def update(self, frame, t=None, frame_index=None, axial=None):
        """
        Estimate pose for one frame. Returns a `Pose`, or ``None`` if lost.

                ``None`` is a normal outcome -- the robot leaves frame, the threshold
                finds nothing, the contour is too collinear to fit.  Callers must handle
                it rather than assume a detection.

                ``axial`` forwards to `segment.segment`; only the validation code sets
                it, to compare against the unweighted fit. ``None`` means "whatever
                `segment.AXIAL_DEFAULT` says", read at call time -- not a duplicated
                default -- a duplicated default here silently overrides the module flag
                for every caller that goes through the estimator.
        """

        now = time.monotonic() if t is None else float(t)
        self.frame_index = (
            self.frame_index + 1 if frame_index is None else int(frame_index)
        )

        use_axial = segmod.AXIAL_DEFAULT if axial is None else bool(axial)
        seg = segmod.segment(
            frame, thresh=self.thresh, min_area=self.min_area, axial=use_axial,
            background=self.background,
        )
        if seg is None:
            self.n_lost += 1
            return None

        t0 = time.perf_counter()

        ellipse = seg.ellipse
        if self.undistort and self.dist is not None:
            try:
                ellipse = segmod.undistort_ellipse(ellipse, self.K, self.dist)
            except Exception:
                ellipse = (
                    seg.ellipse
                )  # keep the distorted fit rather than lose the frame

        # Shape correction goes after undistortion and before back-projection, so
        # a single ellipse produces mutually consistent position and normal.
        ellipse = self.tilt_cal.apply(ellipse)

        poses = conic.backproject_ellipse(
            ellipse, self.K, self.radius_mm, verify_tol=self.verify_tol
        )
        if not poses:
            self.n_lost += 1
            return None

        margin = conic.ambiguity_margin_deg(poses)
        chosen, jump_deg = self._choose(poses, now)

        self._prev_normal = chosen.normal
        self._prev_t = now
        self.n_detected += 1

        xyz, normal = self.zero.apply(chosen.center, chosen.normal)
        theta, phi = _angles_from_normal(normal)
        psi = self.zero.apply_psi(ellipse[2])

        return Pose(
            t=now,
            frame_index=self.frame_index,
            xyz_mm=xyz,
            normal=normal,
            theta_deg=theta,
            phi_deg=phi,
            psi_deg=psi,
            ellipse=ellipse,
            area_px=seg.area_px,
            fit_rms_px=seg.fit_rms_px,
            ambiguity_margin_deg=margin,
            n_solutions=len(poses),
            jump_deg=jump_deg,
            t_seg_ms=seg.t_ms,
            t_est_ms=(time.perf_counter() - t0) * 1e3,
            # Both branches are kept so the validation sweep can separate
            # geometric error from ambiguity error -- without this you cannot
            # tell a bad fit from a correct fit on the wrong branch.
            extra={"segmentation": seg, "candidates": poses},
        )

    def solve_camera_frame(self, frame):
        """
        Un-zeroed pose, for building a datum. Returns ``(center, normal, psi)``.

                `calibrate_zero.py` needs the raw camera-frame answer -- applying the
                datum you are about to compute would be circular.
        """

        seg = segmod.segment(frame, thresh=self.thresh, min_area=self.min_area)
        if seg is None:
            return None
        ellipse = seg.ellipse
        if self.undistort and self.dist is not None:
            ellipse = segmod.undistort_ellipse(ellipse, self.K, self.dist)
        ellipse = self.tilt_cal.apply(ellipse)
        poses = conic.backproject_ellipse(
            ellipse, self.K, self.radius_mm, verify_tol=self.verify_tol
        )
        if not poses:
            return None
        chosen, _ = self._choose(poses, time.monotonic())
        return chosen.center, chosen.normal, ellipse[2]
