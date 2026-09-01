"""
Predict this frame's error, so the estimator can decline when it cannot meet spec.

The target is +-1 deg and +-0.5 mm on **100% of reported frames**. Some frames
carry no usable information -- a rotor near face-on has no measurable tilt, a rim
broken into two arcs has no measurable size -- so the only way to be right every
time you answer is to sometimes not answer.

Two stages, because they answer different questions:

1. **Scale.** Least squares of log error against log features gives the typical
   error at ``x``. Log space is required, not convenient: the derivations below
   make error a *product* of factors, so the relationship is a power law. A fit on
   raw error is dominated by the frames that failed outright and returns negative
   coefficients.
2. **Quantile.** The typical error would admit every frame predicted just under
   the limit whose actual lands above -- about half. The scale is multiplied by a
   high quantile of ``actual / predicted`` measured on the training split.

Features are derived, not tried. ``M`` major axis (px), ``e`` boundary residual
(px), ``z`` range, ``f`` focal length, ``R`` rim radius:

    position  e / M^2          from z = 2fR/M, so dz = -2fR dM/M^2
    angle     e / (M sin θ)    from θ = arccos(b/a), so dθ = -d(b/a)/sin θ
    stereo    d                two views disagreeing by d mm bound the error at
                               roughly d, whatever the resolution

The ``1/M^2`` is the part that matters: halving resolution quadruples position
error at fixed boundary quality. The angle term diverges face-on, correctly -- that
is a real degeneracy. A fitted exponent far from the derived one says the model is
missing something.

Assumes the training conditions cover the deployment ones, that observables are
computed identically at fit and run time (both go through `features`), and that
the relationship is monotone in each feature -- which the derivations give, and
which is what lets a linear fit work instead of a general regressor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parent / "error_model.json"

# The specification the gate enforces.
TARGET_POS_MM = 0.5
TARGET_ANGLE_DEG = 1.0

RATIO_QUANTILE = 0.999

# Fraction of the specification the gate enforces.
#
# The operating point is fitted on finite data, so it holds in expectation, not
# with certainty: on a fresh seed 640x480 admitted a frame at 0.519 mm. Derating
# absorbs a prediction that is off.
#
# Set from a measured cost curve on a held-out sweep:
#
#   margin   core frames kept   coverage
#   1.00        183 / 228        98.91%
#   0.90        138 / 228        98.55%
#   0.80        107 / 228        98.13%
#   0.75         61 / 228       100.00%
#
# Coverage is flat from 1.00 to 0.80, so those margins refuse frames that were
# fine; the last failure clears between 0.80 and 0.75 and takes two thirds of the
# accepted population with it. Frames near the limit are not the ones the model is
# worst about, so the last 1.1% of coverage costs more than the first 98.9%.
#
# 0.75 because the specification is 100% of *reported* frames. Anything looser
# reports a number known to be out of specification. To trade coverage for three
# times the frames, this is the one constant to change.
GATE_MARGIN = 0.75

# Features enter as logs; `one` is the intercept. Kept deliberately few: with a
# few hundred samples per fit, every extra term is a chance to fit noise, and
# each of these is one of the derived quantities above rather than a guess.
FEATURES_POS = ("log_e_over_m2", "log1p_disc", "log1p_refine_rms", "one")
# The angle model separates `e/M` from `1/sin(theta)` rather than carrying their
# product as one term. Bundled, a single exponent has to serve both, and the fit
# chose +0.592 -- while the derivation (dtheta = -d(ratio)/sin theta) says the
# tilt term's exponent is 1. Under-weighting it is not academic: of the eight
# out-of-spec frames measured at 800 poses, five sat at ~25 deg apparent tilt,
# below the 10th percentile of the accepted population, exactly where the
# 1/sin(theta) amplification bites. Split, each can take the weight the data and
# the derivation agree on.
FEATURES_ANG = (
    "log_e_over_m",
    "log_inv_sin",
    "log1p_disc",
    "log_inv_margin",
    "log1p_refine_rms",
    "one",
)

_EPS = 1e-9


def features(pose, radius_mm, reference=(0.0, 0.0, 1.0)):
    """
    Observable features for one `stereo.StereoPose`. Returns a dict.

        Everything here is available at run time with no ground truth and no extra
        computation -- these are quantities the estimator already produced on its way
        to the answer.
    """

    segs = [s for s in pose.per_view if s is not None]
    if not segs:
        return None

    majors = [max(s.ellipse[1][0], _EPS) for s in segs]
    ratios = [min(1.0, s.ellipse[1][1] / max(s.ellipse[1][0], _EPS)) for s in segs]
    fits = [s.fit_rms_px for s in segs]

    # The worst view governs: a pair is only as good as its weaker half, and
    # averaging would let one clean view hide one that failed.
    m = min(majors)
    e = max(fits)
    # sin of the tilt each view sees, from its own axis ratio -- available
    # without the solved pose, so the feature survives a failed solve.
    sin_t = max(min(math.sqrt(max(0.0, 1.0 - r * r)) for r in ratios), 1e-3)

    disc = float(pose.discrepancy_mm) if np.isfinite(pose.discrepancy_mm) else 0.0
    margin = float(pose.margin) if np.isfinite(pose.margin) else 0.0

    # Both were already computed and neither was used. At 1280x800 that cost a
    # 7.1x gap: 42.5% of frames are within specification, 6.0% could be certified.
    #
    # `refine_rms_px` is how far the outline sits from the best circle-projection
    # the solver found, so it measures the silhouette model's fit on *this* frame --
    # the systematic error that dominates the residual, which `fit_rms_px` (an
    # ellipse-fit residual) cannot see.
    #
    # `ambiguity_margin_deg` is how nearly the two branches coincided. When they
    # do, the branch choice is a coin toss and the normal can be badly wrong while
    # every other observable looks ordinary.
    rrms = getattr(pose, "refine_rms_px", None)
    rrms = float(rrms) if rrms is not None and np.isfinite(rrms) else 0.0
    ambig = getattr(pose, "ambiguity_margin_deg", None)
    ambig = float(ambig) if ambig is not None and np.isfinite(ambig) else 90.0

    return {
        "log_e_over_m2": math.log(max(e, 1e-3) / (m * m)),
        "log_e_over_m_sin": math.log(max(e, 1e-3) / (m * sin_t)),
        "log_e_over_m": math.log(max(e, 1e-3) / m),
        # the conditioning of the ratio->tilt inversion, on its own
        "log_inv_sin": math.log(1.0 / sin_t),
        # log1p, not log: the disagreement is legitimately zero on a clean frame,
        # and log(0) is not a feature value.
        "log1p_disc": math.log1p(max(disc, 0.0)),
        "log_inv_margin": math.log(1.0 / math.sqrt(max(margin, 1.0))),
        "log1p_refine_rms": math.log1p(max(rrms, 0.0)),
        # inverted: a *small* ambiguity margin is the dangerous case
        "log1p_ambig": math.log1p(1.0 / (1.0 + max(ambig, 0.0))),
        "one": 1.0,
        # kept for diagnostics and for anything that wants the raw quantities
        "e_over_m2": e / (m * m),
        "e_over_m_sin": e / (m * sin_t),
        "disc": disc,
        "major_px": m,
        "sin_tilt": sin_t,
        "refine_rms": rrms,
        "ambig_deg": ambig,
    }


def _design(rows, names):
    return np.array([[r[n] for n in names] for r in rows], dtype=np.float64)


@dataclass
class ErrorModel:
    """
    Fitted scale plus quantile inflation, for position and angle.

        ``coef_*`` are the least-squares coefficients on the named features;
        ``k_*`` is the ratio quantile that turns the fitted scale into a bound.
    """

    coef_pos: tuple = ()
    coef_ang: tuple = ()
    k_pos: float = 1.0
    k_ang: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def is_identity(self):
        return not self.coef_pos or not self.coef_ang

    def predict(self, feat):
        """
        ``(pos_mm, angle_deg)`` bounds for one feature dict.
        """

        if self.is_identity or feat is None:
            return float("inf"), float("inf")
        x_p = np.array([feat[n] for n in FEATURES_POS])
        x_a = np.array([feat[n] for n in FEATURES_ANG])
        # exp of a log-space fit: positive by construction, no clamp needed.
        pos = math.exp(float(np.dot(self.coef_pos, x_p))) * self.k_pos
        ang = math.exp(float(np.dot(self.coef_ang, x_a))) * self.k_ang
        return pos, ang

    def accepts(
        self, feat, target_pos=TARGET_POS_MM, target_ang=TARGET_ANGLE_DEG, margin=None
    ):
        """
        Accept only if the predicted error clears ``margin`` x the target.

                See `GATE_MARGIN`: the shortfall being covered is the gate's own
                generalisation error, not the estimator's.
        """

        margin = GATE_MARGIN if margin is None else margin
        pos, ang = self.predict(feat)
        return pos <= margin * target_pos and ang <= margin * target_ang

    @classmethod
    def fit(cls, rows, pos_err, ang_err, quantile=RATIO_QUANTILE, meta=None):
        """
        Fit on ``rows`` of feature dicts against measured errors.

                Both stages run in log space. The scale fit is ordinary least squares on
                ``log(error)``, and the quantile is taken on ``log(actual) -
                log(predicted)`` -- equivalently the quantile of the *ratio*, which is
                the scale-free way to state "how much worse than typical does this get".
        """

        pos_err = np.asarray(pos_err, dtype=np.float64)
        ang_err = np.asarray(ang_err, dtype=np.float64)
        keep = np.isfinite(pos_err) & np.isfinite(ang_err)
        rows = [r for r, k in zip(rows, keep) if k]
        pos_err, ang_err = pos_err[keep], ang_err[keep]
        if len(rows) < 40:
            raise ValueError(f"need at least 40 samples to fit, got {len(rows)}")

        # Floor the targets before taking logs: an error of exactly zero is a
        # rounding artefact, not information, and log(0) would remove the sample.
        Xp, Xa = _design(rows, FEATURES_POS), _design(rows, FEATURES_ANG)
        coef_pos, *_ = np.linalg.lstsq(
            Xp, np.log(np.maximum(pos_err, 1e-4)), rcond=None
        )
        coef_ang, *_ = np.linalg.lstsq(
            Xa, np.log(np.maximum(ang_err, 1e-4)), rcond=None
        )

        pred_p = np.exp(Xp @ coef_pos)
        pred_a = np.exp(Xa @ coef_ang)
        k_pos = float(np.quantile(pos_err / pred_p, quantile))
        k_ang = float(np.quantile(ang_err / pred_a, quantile))

        return cls(
            coef_pos=tuple(float(c) for c in coef_pos),
            coef_ang=tuple(float(c) for c in coef_ang),
            k_pos=k_pos,
            k_ang=k_ang,
            meta=dict(meta or {}, n_samples=int(len(rows)), quantile=float(quantile)),
        )

    def save(self, path=DEFAULT_PATH):
        path = Path(path)
        path.write_text(
            json.dumps(
                {
                    "coef_pos": list(self.coef_pos),
                    "features_pos": list(FEATURES_POS),
                    "coef_ang": list(self.coef_ang),
                    "features_ang": list(FEATURES_ANG),
                    "k_pos": self.k_pos,
                    "k_ang": self.k_ang,
                    "model": (
                        "least-squares scale on derived features, inflated by the "
                        "high quantile of actual/predicted"
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
    def load(cls, path=DEFAULT_PATH):
        """
        Load a model; a missing file means no gate, not an error.

                Running without one is legitimate -- it is how the baseline is measured
                -- and forcing a fit before the first run would make the baseline
                impossible to obtain.
        """

        path = Path(path)
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())

        # A model saved against a different feature set is not usable, and the
        # failure it causes is a shape mismatch deep inside `predict` on the
        # first frame -- long after the point where the cause is visible. The
        # file records the features it was fitted with, so check them: a stale
        # model degrades to no gate, which is the same safe state as no model at
        # all, rather than crashing the estimator.
        saved_pos = tuple(d.get("features_pos", ()))
        saved_ang = tuple(d.get("features_ang", ()))
        if (saved_pos and saved_pos != FEATURES_POS) or (
            saved_ang and saved_ang != FEATURES_ANG
        ):
            import warnings

            warnings.warn(
                f"ignoring {path.name}: it was fitted with features "
                f"{saved_pos} / {saved_ang}, but this build uses "
                f"{FEATURES_POS} / {FEATURES_ANG}. Refit with "
                f"validation/fit_error_model.py --write.",
                stacklevel=2,
            )
            return cls()
        known = {
            "coef_pos",
            "coef_ang",
            "k_pos",
            "k_ang",
            "features_pos",
            "features_ang",
            "model",
        }
        return cls(
            coef_pos=tuple(d["coef_pos"]),
            coef_ang=tuple(d["coef_ang"]),
            k_pos=float(d["k_pos"]),
            k_ang=float(d["k_ang"]),
            meta={k: v for k, v in d.items() if k not in known},
        )
