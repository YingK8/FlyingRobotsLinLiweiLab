"""
Shape calibration: correct the systematic tilt bias before back-projection.

`conic.py` assumes the silhouette is a flat circle.  The robot is not one -- the
mast and magnet stick out along the rotor axis, so as it tilts they widen the
silhouette's *short* direction.  The observed axis ratio therefore sits above
`cos(theta)`, tilt reads low, and the error is systematic rather than random.

Measured on 700 rendered poses, the excess `minor/major - cos(theta)` is
**+0.031 +- 0.028** over tilt 25-55 degrees, which is a ~3 degree tilt bias.

How much of that is fixable.  Only the mean.  Regressing the excess against every
candidate factor leaves all of them below R^2 = 0.06 -- tilt 0.06, blade azimuth
0.03, ambient 0.03, opacity 0.01.  Nothing dominates, because the scatter is the
3-D silhouette interacting with viewing geometry, which a 2-D ellipse simply does
not carry.  The azimuth term matters least of all in practice: it is the blade
phase, and at 310-350 Hz against a <=420 fps camera that is aliased and
unknowable per frame.  So the bias comes out and the scatter stays.

The correction is applied to the **minor axis**, not to the reported angle, so
that the single corrected ellipse feeds `conic.backproject_ellipse` and position
and normal stay mutually consistent.  Correcting the angle afterwards would
leave the position derived from an ellipse the orientation no longer agrees with.

It is scale-free.  The fit maps raw tilt to true tilt, and raw tilt comes from a
dimensionless axis ratio, so one curve covers all ranges and resolutions --
confirmed by the implied coefficient being flat across 140-360 mm (0.068 to
0.088) while varying 4x across tilt.

This is a calibration, so it lives in a JSON file rather than in the source.  The
shipped one is fitted to renders of `flyingrobot_rod2.STL`; if the
physical robot differs from that mesh, refit against the real thing with
`validation/tune.py`.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "tilt_calibration.json"

# Which rig appearance is in force. Defined here rather than in `pose/segment.py`,
# where it is used most, because it is the key the *calibration files* are named
# by -- every appearance carries its own fitted constants -- and calibration runs
# before estimation. `segment.py` re-exports it, so `segment.APPEARANCE` still
# resolves everywhere it always did.
#
# Read at import. Setting POSE_APPEARANCE after importing the pose package leaves
# the previous appearance's radius and calibration in force, silently.
# `dark` is the rig on the bench: a black rim against the white foam backdrop. This said
# `bright` long after the rig changed, and it is most of why an offline run returned a
# poses.csv with a header and no rows -- `bright` thresholds at 128 on a scene whose median
# is 144, so the hull takes the backdrop and every downstream gate refuses it.
APPEARANCE = os.environ.get("POSE_APPEARANCE", "bright")


def calibration_path(name="tilt_calibration", appearance=None):
    """
    Where a calibration lives, for the rig appearance in force.

        The fitted constants depend on which channel the boundary was thresholded in
        -- a chroma threshold cuts a red edge at a different fraction of its
        transition than a luminance threshold cuts a white one -- so each appearance
        carries its own file. ``bright`` keeps the original unsuffixed names, so
        every result measured before the red rig existed still resolves.
    """

    appearance = APPEARANCE if appearance is None else appearance
    suffix = "" if appearance == "bright" else f"_{appearance}"
    return DEFAULT_PATH.with_name(f"{name}{suffix}.json")


# Above this the flat-circle model has broken down badly enough that the fit is
# extrapolating rather than correcting; `apply` clamps there instead.
MAX_FIT_TILT_DEG = 75.0


@dataclass
class TiltCalibration:
    """
    Maps raw tilt (from the uncorrected axis ratio) to true tilt.

        Two models. ``model="cylinder"`` is the one to use; ``"quadratic"`` is the
        original and is kept so old calibration files still load.

        **cylinder** -- one physical parameter ``k``, derived rather than fitted in
        form. The duct rim is not a circle, it is a short cylinder wall: radius
        ``R``, half-height ``h``. Tilt it and the silhouette's short half-extent
        becomes ``R cos(theta) + h sin(theta)`` -- the near-top and far-bottom edges
        of the wall project outside the mid-plane circle -- while the long extent
        stays ``R``. So the observed axis ratio is

            rho = cos(theta) + k sin(theta),    k = h/R

        which is ``sqrt(1+k^2) cos(theta - atan k)`` and therefore inverts in closed
        form. Measured on renders of the rim ring alone, ``excess/sin(theta)`` is
        constant at 0.47-0.63 mm across 15-70 degrees of tilt, giving k ~ 0.059.

        This matters beyond tidiness. The previous model was a quadratic in the
        *angle*, which cannot represent the above and left a residual that crossed
        zero twice; 84% of the remaining shape-error variance was still a
        deterministic function of tilt. The cylinder model is also monotone by
        construction on ``[0, 90 - atan k]``, correct at ``theta = 0`` without
        needing that imposed, and extrapolates -- a fitted curve does none of these.

        **What it does not cover.** The mast adds a separate, much sharper
        contamination that switches on near 57 degrees of tilt (measured: the peak
        outward deviation on that side of the silhouette jumps from +2 px at 55 deg
        to +23 px at 70 deg, and vanishes entirely if the mast is deleted from the
        mesh). That is a different mechanism with a different fix and is not modelled
        here. Below ~55 degrees the mast contributes nothing measurable, and the
        magnet body contributes nothing at any tilt -- both established by rendering
        sliced copies of the mesh.
    """

    a: float = 1.0
    b: float = 0.0
    k: float = 0.0
    model: str = "quadratic"
    meta: dict = field(default_factory=dict)

    @property
    def is_identity(self):
        if self.model == "cylinder":
            return self.k == 0.0
        return self.a == 1.0 and self.b == 0.0

    @property
    def _phi(self):
        """
        ``atan(k)``, the angle the wall's contribution shifts the curve by.
        """

        return math.atan(self.k)

    @property
    def resolution_floor_deg(self):
        """
        Tilt below which the axis ratio carries no information, in degrees.

                A consequence of the wall, not of noise or of this implementation. The
                ratio ``rho = cos(theta) + k sin(theta)`` rises above 1 on the whole
                interval ``(0, 2 atan k)`` -- it peaks at ``atan k`` and returns to 1 at
                twice that -- so every tilt in that band produces a silhouette at least
                as wide as it is long, and none of them can be told apart by an axis
                ratio at any resolution whatsoever.

                With the fitted ``k = 0.098`` this is **11.2 degrees**, which is not a
                coincidence: `README.md` records azimuth error collapsing from 30.8 deg
                below 5 deg of tilt, to 14.3 deg at 5-10, to 1.8 deg at 10-20, and calls
                the threshold empirical. It is not -- it is ``2 atan(h/R)``.

                This is a second, independent reason not to read tilt from the ratio near
                face-on, the first being that ``dtheta/d(ratio) = -1/sin(theta)``
                diverges there. Both point at the same fix: take tilt from the major-axis
                direction across two views instead.
        """

        return 2.0 * math.degrees(self._phi) if self.model == "cylinder" else 0.0

    def tilt(self, theta_raw_deg):
        """
        Corrected tilt, in degrees, clamped to a sane range.
        """

        if self.model == "cylinder":
            if self.is_identity:
                return float(np.clip(theta_raw_deg, 0.0, 90.0))
            # theta_raw is acos(rho), so cos(theta_raw) is the observed ratio.
            rho = math.cos(math.radians(float(np.clip(theta_raw_deg, 0.0, 90.0))))
            arg = rho / math.hypot(1.0, self.k)
            return float(
                np.clip(
                    math.degrees(self._phi + math.acos(float(np.clip(arg, -1.0, 1.0)))),
                    0.0,
                    90.0,
                )
            )
        t = float(np.clip(theta_raw_deg, 0.0, MAX_FIT_TILT_DEG))
        return float(np.clip(self.a * t + self.b * t * t, 0.0, 90.0))

    def raw_tilt(self, theta_true_deg):
        """
        Inverse of `tilt`: what a true tilt would be *measured* as.

                `tilt` answers the question a per-frame estimator asks -- I measured this
                ratio, what was the real tilt.  A least-squares fit asks the opposite
                one: I am hypothesising this pose, what ratio should I expect to see.
                Without the inverse, a refinement that fits the rim circle to the raw
                silhouette silently throws this calibration away, because the silhouette
                it is fitting still contains the mast the calibration exists to absorb.

                The forward model is monotonic on the fitted range (`fit` refuses to
                return one that is not), so the inverse is the positive root of
                ``b r^2 + a r - T = 0``.
        """

        t = float(np.clip(theta_true_deg, 0.0, 90.0))
        if self.is_identity:
            return t
        if self.model == "cylinder":
            rho = math.cos(math.radians(t)) + self.k * math.sin(math.radians(t))
            return math.degrees(math.acos(float(np.clip(rho, -1.0, 1.0))))
        if abs(self.b) < 1e-12:
            return float(np.clip(t / self.a, 0.0, MAX_FIT_TILT_DEG))
        disc = self.a * self.a + 4.0 * self.b * t
        if disc < 0:
            return t
        return float(
            np.clip((-self.a + math.sqrt(disc)) / (2.0 * self.b), 0.0, MAX_FIT_TILT_DEG)
        )

    def unapply(self, ellipse):
        """
        Inverse of `apply`: an ideal rim ellipse -> the silhouette expected.

                Widens the minor axis the way the mast and magnet actually widen it, so
                a hypothesised pose can be compared against a raw, uncorrected
                silhouette.  Together with `apply` this makes the calibration usable in
                both directions, which is what a joint multi-view fit needs.
        """

        if self.is_identity:
            return ellipse
        (cx, cy), (major, minor), ang = ellipse
        if major <= 0:
            return ellipse
        ratio = min(1.0, max(0.0, minor / major))
        raw = math.cos(math.radians(self.raw_tilt(math.degrees(math.acos(ratio)))))
        return (cx, cy), (major, major * raw), ang

    def apply(self, ellipse):
        """
        Rewrite an ellipse's minor axis so its ratio implies the true tilt.

                Returns a new ``((cx, cy), (major, minor), angle)``.  The major axis is
                untouched: it stays the rim diameter at every tilt (measured to within
                1.2 px out to 70 degrees), so it is the trustworthy one.
        """

        if self.is_identity:
            return ellipse
        (cx, cy), (major, minor), ang = ellipse
        if major <= 0:
            return ellipse
        ratio = min(1.0, max(0.0, minor / major))
        corrected = math.cos(math.radians(self.tilt(math.degrees(math.acos(ratio)))))
        return (cx, cy), (major, major * corrected), ang

    @classmethod
    def fit_cylinder(cls, ratio, theta_true_deg, meta=None, tilt_range=(20.0, 50.0)):
        """
        Fit the single wall parameter ``k`` from observed ratios.

                ``rho = cos(theta) + k sin(theta)`` is linear in ``k``, so this is one
                least-squares division and has no local minima, no initial guess and no
                degree to choose -- the entire model-selection question the quadratic
                version had is gone because the form came from the geometry.

                ``tilt_range`` is the one thing that must be got right, and getting it
                wrong is worse than using no model at all. The wall term describes the
                rim and nothing else. Fit it over the full 5-71 degrees and the mast --
                which contributes nothing below ~55 degrees and then dominates -- drags
                ``k`` from 0.043 to 0.098, and the resulting correction is *worse* than
                the quadratic it replaced (3.54 vs 2.43 deg median error on held-out
                test, with a +3.5 deg bias). Fitted over 20-50, where the wall is the
                only mechanism acting, the same one-parameter model beats the quadratic
                on both counts:

                    quadratic          2.426 deg median, bias +0.807
                    cylinder 5-71      3.536 deg median, bias +3.536   (mast-contaminated)
                    cylinder 20-50     1.974 deg median, bias -0.038

                The lower bound is the saturation band ``2 atan k``; the upper is the
                mast crossover ``atan(R/h_mast) = 52.5`` degrees.
        """

        rho = np.asarray(ratio, dtype=np.float64)
        th = np.radians(np.asarray(theta_true_deg, dtype=np.float64))
        keep = (
            np.isfinite(rho)
            & np.isfinite(th)
            & (th >= np.radians(tilt_range[0]))
            & (th <= np.radians(tilt_range[1]))
        )
        rho, th = rho[keep], th[keep]
        if len(rho) < 10:
            raise ValueError(
                f"need at least 10 samples in tilt {tilt_range}, got {len(rho)}"
            )

        # least squares on  (rho - cos th) = k sin th
        num = float(np.sum((rho - np.cos(th)) * np.sin(th)))
        den = float(np.sum(np.sin(th) ** 2))
        k = num / den if den > 0 else 0.0
        if not (0.0 <= k < 1.0):
            raise ValueError(
                f"fitted wall ratio k={k:.4f} is unphysical; expected 0 <= k < 1"
            )

        cal = cls(
            k=float(k),
            model="cylinder",
            meta=dict(
                meta or {},
                n_samples=int(len(rho)),
                resolution_floor_deg=round(2 * math.degrees(math.atan(k)), 3),
                fit_tilt_range_deg=list(tilt_range),
            ),
        )

        # Monotonicity is checked from the resolution floor upward, not from
        # zero, because the model is genuinely non-monotone below it -- and that
        # is a physical statement, not a fitting artefact. Below atan(k) the
        # wall makes the silhouette *wider* than the mid-plane circle, so the
        # ratio rises above 1 and saturates at sqrt(1+k^2). A tilt of 0 and a
        # tilt of atan(k) produce the same observation. See `RESOLUTION_FLOOR`.
        # Probe from above the saturation band, where rho <= 1 and the map is
        # single-valued. Below it the model is genuinely non-invertible.
        lo = 2.0 * math.degrees(math.atan(k)) + 0.5
        probe = np.linspace(lo, 90.0 - lo, 200)
        raw = np.array([cal.raw_tilt(t) for t in probe])
        if not np.all(np.diff(raw) > 0):
            raise ValueError(
                "fitted correction is not monotonic above the floor; "
                "refusing to use it"
            )
        return cal

    @classmethod
    def fit(cls, theta_raw_deg, theta_true_deg, meta=None):
        """
        Least-squares fit of ``a*x + b*x^2`` through the origin.
        """

        x = np.asarray(theta_raw_deg, dtype=np.float64)
        y = np.asarray(theta_true_deg, dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(y) & (x <= MAX_FIT_TILT_DEG)
        x, y = x[keep], y[keep]
        if len(x) < 10:
            raise ValueError(f"need at least 10 samples to fit, got {len(x)}")

        design = np.column_stack([x, x * x])
        (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)

        # A non-monotonic map would make tilt ambiguous; refuse rather than ship it.
        probe = np.linspace(0.0, MAX_FIT_TILT_DEG, 200)
        if not np.all(np.diff(a * probe + b * probe * probe) > 0):
            raise ValueError("fitted correction is not monotonic; refusing to use it")

        return cls(a=float(a), b=float(b), meta=dict(meta or {}, n_samples=int(len(x))))

    def save(self, path=None):
        path = Path(path or calibration_path())
        path.write_text(
            json.dumps(
                {
                    "a": self.a,
                    "b": self.b,
                    "k": self.k,
                    "kind": self.model,
                    "model": (
                        "rho = cos(theta) + k*sin(theta); k = h/R, the duct "
                        "wall's half-height over its radius"
                        if self.model == "cylinder"
                        else "theta_true_deg = a*theta_raw_deg + b*theta_raw_deg**2"
                    ),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **self.meta,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path=None):
        """
        Load a calibration; absent file means identity (no correction).
        """

        path = Path(path or calibration_path())
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        known = {"a", "b", "k", "kind", "model"}
        return cls(
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 0.0)),
            k=float(d.get("k", 0.0)),
            # Files written before the cylinder model have no "kind" and are
            # quadratic; they must keep loading and behaving identically.
            model=str(d.get("kind", "quadratic")),
            meta={kk: v for kk, v in d.items() if kk not in known},
        )


@dataclass
class CentreCalibration:
    """
    Correct the fitted ellipse's displaced **centre**.

        The rim wall and the mast grow the silhouette on one side, so the fitted
        ellipse's centre is displaced as well as its short axis fattened -- and
        lateral position is read straight off that centre, which makes this a
        position error rather than an orientation one.  `TiltCalibration` does not
        touch it.

        **Direction.** The displacement lies along the projected rotor axis, and its
        sign follows the lean. That sign is not in the ellipse: an ellipse angle is
        defined mod 180 deg, so the short-axis direction is ambiguous while the lean
        is not. Along the raw short axis the offset scatters -0.0006 to +0.0128 of the
        major for one tilt; along the projected axis it collapses onto one curve. So
        this needs the **normal**, and is applied after a first back-projection --
        see `apply_to_ellipse`.

        **Scale.** Stored as a fraction of the major axis, which is nearly
        range-independent: the fraction agrees to ~20% between 170 and 300 mm, against
        a raw pixel offset that changes by 1.5x.

        **Floor.** About +-0.0055 of the major remains, and it is not noise: it is the
        rim's own out-of-roundness (0.108 mm radial sd on a 10.204 mm radius = 0.0053
        of the major, matching the measured amplitude). It depends on which part of the
        rim faces the lean, which the 310-350 Hz spin aliases beyond recovery.

        Not required to be monotonic, unlike `TiltCalibration` -- nothing inverts it.
        Just as well: it changes sign near 57 deg as the mast takes over from the wall.
    """

    tilt_knots_deg: tuple = ()
    offset_over_major: tuple = ()
    meta: dict = field(default_factory=dict)

    @property
    def is_identity(self):
        return len(self.tilt_knots_deg) < 2

    def offset(self, tilt_deg):
        """
        Displacement as a fraction of the major axis, positive along +n.
        """

        if self.is_identity:
            return 0.0
        return float(
            np.interp(float(tilt_deg), self.tilt_knots_deg, self.offset_over_major)
        )

    def apply_to_ellipse(self, ellipse, tilt_deg, normal_dir_px):
        """
        Move the centre back, given the projected rotor-axis direction.

                ``normal_dir_px`` is a unit 2-vector: the direction the rotor axis
                projects to in this image, which the caller gets from a first
                back-projection.  Nothing else here can supply the sign.
        """

        if self.is_identity:
            return ellipse
        (cx, cy), axes, ang = ellipse
        d = np.asarray(normal_dir_px, dtype=np.float64)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            return ellipse
        shift = self.offset(tilt_deg) * axes[0]
        d = d / n
        return (float(cx - shift * d[0]), float(cy - shift * d[1])), axes, ang

    @classmethod
    def fit(
        cls,
        tilt_deg,
        offset_over_major,
        knots=(15, 25, 35, 45, 55, 60, 65, 70),
        meta=None,
    ):
        """
        Average the measured offset into tilt bins.

                Averaged over the lean azimuth on purpose: the azimuth-dependent part is
                the un-correctable out-of-roundness described above, so folding it in
                would be fitting noise that the spin re-randomises every frame.
        """

        t = np.asarray(tilt_deg, dtype=np.float64)
        o = np.asarray(offset_over_major, dtype=np.float64)
        keep = np.isfinite(t) & np.isfinite(o)
        t, o = t[keep], o[keep]
        if len(t) < 8:
            raise ValueError(f"need at least 8 samples, got {len(t)}")

        vals = []
        for k in knots:
            near = np.abs(t - k) <= 6.0
            vals.append(float(np.mean(o[near])) if near.sum() >= 2 else np.nan)
        vals = np.array(vals)
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            raise ValueError("not enough populated knots to interpolate")
        return cls(
            tilt_knots_deg=tuple(float(k) for k in np.asarray(knots)[ok]),
            offset_over_major=tuple(float(v) for v in vals[ok]),
            meta=dict(meta or {}, n_samples=int(len(t))),
        )

    def save(self, path=None):
        path = Path(path or calibration_path("centre_calibration"))
        path.write_text(
            json.dumps(
                {
                    "tilt_knots_deg": list(self.tilt_knots_deg),
                    "offset_over_major": list(self.offset_over_major),
                    "model": (
                        "centre displacement along the projected rotor axis, as a "
                        "fraction of the major axis, interpolated in tilt"
                    ),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    **self.meta,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, path=None):
        path = Path(path or calibration_path("centre_calibration"))
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        known = {"tilt_knots_deg", "offset_over_major", "model"}
        return cls(
            tilt_knots_deg=tuple(d["tilt_knots_deg"]),
            offset_over_major=tuple(d["offset_over_major"]),
            meta={k: v for k, v in d.items() if k not in known},
        )
