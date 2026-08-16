"""Predict this frame's error, so the estimator can decline when it cannot meet spec.

The target is ±1° and ±0.5 mm on **100%** of reported frames.  No amount of
accuracy work reaches that on its own, because some frames carry no usable
information: a rotor near face-on has no measurable tilt, and a rim broken into
two short arcs has no measurable size.  The only way to be right every time you
answer is to sometimes not answer.

So the question stops being "how large is the error on average" and becomes
"how large is it *on this frame*, before I know the truth".

## What is being estimated

Not a probability, and not the error itself -- a **conditional scale**.  For a
frame with observables `x`, we want a number `s(x)` such that the actual error is
almost always below it, then reject when `s(x)` exceeds the specification.  Two
stages, because they answer different questions:

1. **Scale.**  A least-squares fit of **log error against log features** gives
   the typical error at `x`.  Log space is not a convenience: the derivations
   below produce error as a *product* of scaling factors, so the relationship is
   a power law and taking logs is what makes it linear.  It also fixes two
   things that break a fit on raw error, both observed rather than anticipated:
   residuals span four decades, so squared loss on raw values is dominated
   entirely by the frames that failed outright; and an unconstrained linear fit
   returns negative coefficients, predicts a negative error for easy frames, and
   -- once clamped at zero -- sends the actual/predicted ratio to 2.4e9.
   Exponentiating a log-space fit is positive by construction.

   The fitted exponents are also a check on the derivation. Physics says error
   scales as the first power of `e/M²`, so a fitted exponent near 1 is
   corroboration and one far from it says the model is missing something.
2. **Quantile.**  Rejecting at the typical error would let through every frame
   whose predicted error sits just under the limit but whose actual error lands
   above it -- about half of them.  So the fitted scale is multiplied by the
   high quantile of the observed ratio `actual / predicted`, measured on the
   training split.  Fitting the quantile directly (quantile regression) would
   estimate a tail from far fewer effective samples; separating the two lets the
   plentiful data set the shape and the tail set only one number.

## The features, and why these

Derived rather than tried.  Let `M` be the major axis in pixels, `e` the fitted
boundary residual in pixels, `z` the range, `f` the focal length and `R` the rim
radius.

**Position.**  Depth comes from the ellipse's size, `z = 2fR/M`, so
`dz = -2fR·dM/M²`.  A boundary error `e` propagates into `dM` roughly in
proportion, giving

    position error  ∝  e / M²     (times the constant 2fR)

That `1/M²` is the important part: halving the resolution quadruples the
position error at fixed boundary quality, which is why the feature is `e/M²`
and not `e/M`.

**Angle.**  Tilt is `θ = arccos(b/a)`, so `dθ = -d(b/a)/sin θ`.  A boundary
error perturbs both axes, `d(b/a) ≈ (e/M)(1 + b/a)`, hence

    angle error  ∝  e / (M · sin θ)

which diverges as the rotor turns face-on -- correctly, because that is a real
degeneracy and not a modelling artefact.

**Cross-view disagreement** enters both linearly and with no scaling: two views
that disagree by `d` millimetres about the same object bound the error at
roughly `d`, whatever the resolution.

## What it assumes

- That the training conditions cover the deployment ones.  A frame unlike
  anything in the fit gets an unreliable prediction, and the gate is then no
  safer than the observables it is built from.
- That the observables are computed identically at fit and at run time -- they
  are, because both go through `features()` here.
- That the relationship is monotone in each feature.  It is, by the derivations
  above, which is what lets a linear fit in the derived quantities work rather
  than needing a general regressor.
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

# Quantile of the actual/predicted ratio used to turn a scale into a bound.
#
# Not 1.0: the maximum of a training sample is an outlier, not a bound, and
# calibrating to it rejects most of a fresh dataset for no gain. Not 0.99
# either -- the target is 100% of reported frames, and on a controlled synthetic
# check the quantile buys coverage at a steep and very visible price:
#
#   quantile   accepted   coverage
#   0.99         70.7%      99.53%
#   0.995        64.3%      99.48%
#   0.999        46.7%     100.00%
#
# That trade is structural, not a tuning artefact: a bound that is right every
# time has to sit above the worst case, and most frames are nowhere near it.
# Whether 0.999 suffices on real data is verified on a held-out split, never
# assumed.
RATIO_QUANTILE = 0.999

# Fraction of the specification the gate enforces.
#
# The operating point is calibrated on finite data, so it holds *in expectation*
# on new data rather than with certainty: on the sample set the threshold was
# chosen from every mode reports 100% coverage, yet on a fresh seed 640x480
# admitted a frame at 0.519 mm. Certifying against a fraction of the
# specification absorbs a prediction that is off, the way a rated load is derated.
#
# The value is set by a measured cost curve, not by taste. On a held-out sweep,
# with the offending frame predicted at 0.3826 mm against an actual 0.5190 mm --
# a 1.36x under-prediction, poor but not a blunder:
#
#   margin   core frames kept   coverage
#   1.00        183 / 228        98.91%
#   0.90        138 / 228        98.55%
#   0.80        107 / 228        98.13%
#   0.75         61 / 228       100.00%
#
# Coverage is nearly flat from 1.00 to 0.80 -- those margins refuse frames that
# were fine -- and the last failure clears between 0.80 and 0.75, taking two
# thirds of the accepted population with it. The frames near the specification
# limit are not the frames the model is worst about, so the last 1.1% of coverage
# costs more than the first 98.9%.
#
# 0.75 is set because the specification is 100% of reported frames and coverage
# is the requirement detection is spent on. Anything looser reports a number
# known to be out of specification. If the deployment would rather have three
# times the frames and 98.9% coverage, this is the one constant to change.
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
FEATURES_ANG = ("log_e_over_m", "log_inv_sin", "log1p_disc", "log_inv_margin",
                "log1p_refine_rms", "one")

_EPS = 1e-9


def features(pose, radius_mm, reference=(0.0, 0.0, 1.0)):
    """Observable features for one `stereo.StereoPose`. Returns a dict.

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

    # The refinement's own residual, and how close the two conic branches were.
    #
    # Both were computed on the way to the answer and neither was being used. The
    # omission mattered: measured at 1280x800, 42.5% of frames are genuinely
    # within specification but only 6.0% could be certified -- a 7.1x gap that is
    # the predictor's ignorance rather than the estimator's error.
    #
    # `refine_rms_px` is the most direct evidence available about *this* frame:
    # it is how far the observed outline sits from the best circle-projection the
    # solver could find, so it measures the silhouette model's fit on this frame
    # specifically -- exactly the systematic error that dominates the residual
    # (lecture notes 12.9) and that `fit_rms_px`, an ellipse-fit residual, cannot
    # see. A frame whose outline is well described by a projected circle is one
    # whose pose can be trusted; one where the mast or a broken rim pushes the
    # refinement's residual up is not, however cleanly an ellipse fitted it.
    #
    # `ambiguity_margin_deg` says how nearly the two back-projection branches
    # coincided. When they do, the branch choice is a coin toss and the normal
    # can be badly wrong while every other observable looks ordinary.
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
    """Fitted scale plus quantile inflation, for position and angle.

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
        """``(pos_mm, angle_deg)`` bounds for one feature dict."""
        if self.is_identity or feat is None:
            return float("inf"), float("inf")
        x_p = np.array([feat[n] for n in FEATURES_POS])
        x_a = np.array([feat[n] for n in FEATURES_ANG])
        # exp of a log-space fit: positive by construction, no clamp needed.
        pos = math.exp(float(np.dot(self.coef_pos, x_p))) * self.k_pos
        ang = math.exp(float(np.dot(self.coef_ang, x_a))) * self.k_ang
        return pos, ang

    def accepts(self, feat, target_pos=TARGET_POS_MM, target_ang=TARGET_ANGLE_DEG,
                margin=None):
        """Accept only if the predicted error clears ``margin`` x the target.

        See `GATE_MARGIN`: the shortfall being covered is the gate's own
        generalisation error, not the estimator's.
        """
        margin = GATE_MARGIN if margin is None else margin
        pos, ang = self.predict(feat)
        return pos <= margin * target_pos and ang <= margin * target_ang

    @classmethod
    def fit(cls, rows, pos_err, ang_err, quantile=RATIO_QUANTILE, meta=None):
        """Fit on ``rows`` of feature dicts against measured errors.

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
        coef_pos, *_ = np.linalg.lstsq(Xp, np.log(np.maximum(pos_err, 1e-4)),
                                       rcond=None)
        coef_ang, *_ = np.linalg.lstsq(Xa, np.log(np.maximum(ang_err, 1e-4)),
                                       rcond=None)

        pred_p = np.exp(Xp @ coef_pos)
        pred_a = np.exp(Xa @ coef_ang)
        k_pos = float(np.quantile(pos_err / pred_p, quantile))
        k_ang = float(np.quantile(ang_err / pred_a, quantile))

        return cls(
            coef_pos=tuple(float(c) for c in coef_pos),
            coef_ang=tuple(float(c) for c in coef_ang),
            k_pos=k_pos, k_ang=k_ang,
            meta=dict(meta or {}, n_samples=int(len(rows)), quantile=float(quantile)),
        )

    def save(self, path=DEFAULT_PATH):
        path = Path(path)
        path.write_text(json.dumps({
            "coef_pos": list(self.coef_pos), "features_pos": list(FEATURES_POS),
            "coef_ang": list(self.coef_ang), "features_ang": list(FEATURES_ANG),
            "k_pos": self.k_pos, "k_ang": self.k_ang,
            "model": ("least-squares scale on derived features, inflated by the "
                      "high quantile of actual/predicted"),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **self.meta,
        }, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        """Load a model; a missing file means no gate, not an error.

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
        if (saved_pos and saved_pos != FEATURES_POS) or \
           (saved_ang and saved_ang != FEATURES_ANG):
            import warnings
            warnings.warn(
                f"ignoring {path.name}: it was fitted with features "
                f"{saved_pos} / {saved_ang}, but this build uses "
                f"{FEATURES_POS} / {FEATURES_ANG}. Refit with "
                f"validation/fit_error_model.py --write.", stacklevel=2)
            return cls()
        known = {"coef_pos", "coef_ang", "k_pos", "k_ang", "features_pos",
                 "features_ang", "model"}
        return cls(
            coef_pos=tuple(d["coef_pos"]), coef_ang=tuple(d["coef_ang"]),
            k_pos=float(d["k_pos"]), k_ang=float(d["k_ang"]),
            meta={k: v for k, v in d.items() if k not in known},
        )
