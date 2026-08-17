"""
Does re-weighting the hull beat fitting it plainly?

Three ellipse fits on identical hulls, scored against render ground truth:

    A  plain      cv2.fitEllipseDirect, what ships today
    B  symmetric  IRLS weighted by position along the major axis
    C  one-sided  IRLS tolerating points OUTSIDE the current ellipse

**Why B exists.** The contamination is localised: the rim wall and the mast push
the silhouette outward near the ellipse parameter t = +-90 (the minor-axis
extremes, equivalently the middle of the major axis), and leave the major-axis
tips clean.  Down-weighting by position along the major axis targets exactly
that region without needing to detect anything.

**Why B is expected to lose anyway.**  A boundary point's sensitivity to the
semi-minor axis is d(b sin t)/db = sin t, so the points at t = +-90 carry *all*
the information about b.  Down-weighting them symmetrically removes the quantity
being recovered: less bias, much more noise.  It is measured rather than argued
away -- the prediction is cheap to falsify and this project's record on such
predictions is poor.

**Why C should work.**  The rim is part of the object and the projection of a
circle is convex, so the hull always *contains* the true rim ellipse and every
hull point lies on or outside it.  Contamination is strictly one-sided.  C
penalises points inside the current estimate normally and saturates on points
outside, which keeps the minor-extreme points that are genuinely on the rim and
rejects only those pushed out.  A symmetric robust loss cannot express this,
which is why `soft_l1` measured as doing nothing.

Run: uv run python controller/pose/validation/tune_weighting.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [
    str(HERE),
    str(HERE.parent / "validation"),
    str(_C / "pose"),
    str(_C / "calib"),
    str(_C / "camera"),
]

import conic  # noqa: E402
import render as rendermod  # noqa: E402
import segment as segmod  # noqa: E402
from shape import TiltCalibration  # noqa: E402

N_IRLS = 4


def _fit_weighted(pts, w):
    """
    Weighted direct ellipse fit, by replicating points in proportion to w.

        `cv2.fitEllipseDirect` takes no weights.  Rather than reimplement Fitzgibbon
        -- and inherit its conditioning problems -- weights are applied by
        resampling: a point with twice the weight appears twice.  Quantised, but the
        quantisation is far below the effect being measured and it keeps the
        numerics identical to the shipped path, which is the point of the comparison.
    """

    w = np.clip(np.asarray(w, dtype=np.float64), 0.0, None)
    if w.sum() <= 0:
        return None
    reps = np.maximum(1, np.round(w / max(w.max(), 1e-12) * 8.0).astype(int))
    rep = np.repeat(pts, reps, axis=0)
    if len(rep) < 5:
        return None
    try:
        return segment.fit_ellipse_direct(rep)
    except cv2.error:
        return None


def _params(pts, ellipse):
    """
    Ellipse parameter t (deg from the major tip) and signed radial deviation.
    """

    (cx, cy), (major, minor), ang = ellipse
    a, b, th = major / 2.0, minor / 2.0, math.radians(ang)
    c, s = math.cos(th), math.sin(th)
    x, y = pts[:, 0] - cx, pts[:, 1] - cy
    u, v = c * x + s * y, -s * x + c * y
    t = np.arctan2(v / max(b, 1e-9), u / max(a, 1e-9))
    k = np.sqrt((u / max(a, 1e-9)) ** 2 + (v / max(b, 1e-9)) ** 2)
    r = np.hypot(a * np.cos(t), b * np.sin(t))
    return t, (k - 1.0) * r


def fit_plain(pts):
    f = segmod.fit_ellipse(pts)
    return None if f is None else f[0]


def fit_symmetric(pts, power=2.0):
    """
    B: weight by |cos t| -- full weight at the major tips, zero at the minor.
    """

    ell = fit_plain(pts)
    for _ in range(N_IRLS):
        if ell is None:
            return None
        t, _ = _params(pts, ell)
        w = np.abs(np.cos(t)) ** power + 0.05
        nxt = _fit_weighted(pts, w)
        if nxt is None:
            return ell
        ell = nxt
    return ell


def fit_one_sided(pts, scale_px=0.6):
    """
    C: points inside the current estimate keep full weight; outside saturate.

        ``scale_px`` is the deviation at which an outward point is half-weighted.
        Set from the measured sub-pixel boundary precision, so genuine rim points
        (which scatter by a few tenths of a pixel) are untouched and contamination
        (which reaches tens of pixels) is suppressed.
    """

    ell = fit_plain(pts)
    for _ in range(N_IRLS):
        if ell is None:
            return None
        _, dev = _params(pts, ell)
        w = np.where(dev <= 0.0, 1.0, 1.0 / (1.0 + (dev / scale_px) ** 2))
        nxt = _fit_weighted(pts, w)
        if nxt is None:
            return ell
        ell = nxt
    return ell


SCHEMES = {
    "A plain": fit_plain,
    "B symmetric": fit_symmetric,
    "C one-sided": fit_one_sided,
}


def build(n, seed, width=1024, height=768):
    """
    Render poses and keep the hull plus the analytic truth for each.
    """

    rng = np.random.default_rng(seed)
    cos_lo = math.cos(math.radians(70.0))
    tilt = np.degrees(np.arccos(rng.uniform(cos_lo, 1.0, n)))
    az = rng.uniform(0.0, 360.0, n)
    z = rng.uniform(160.0, 320.0, n)
    out = []
    with rendermod.Renderer(width, height) as r:
        for i in range(n):
            light = rendermod.LightRig(
                dome=((rng.uniform(35, 80), rng.uniform(0, 360)),),
                ambient=rng.uniform(0.3, 0.6),
                intensity=rng.uniform(8, 18),
            )
            s = r.render(
                float(tilt[i]),
                float(az[i]),
                [0.0, 0.0, float(z[i])],
                light=light,
                bg_level=0.0,
            )
            seg = segmod.segment(s.image)
            if seg is None:
                continue
            truth = conic.normalise_ellipse(
                conic.project_circle(
                    s.center_mm, s.normal, rendermod.RIM_RADIUS_MM, r.K
                )
            )
            out.append(
                {
                    "hull": seg.contour,
                    "truth": truth,
                    "tilt": float(tilt[i]),
                    "K": r.K,
                    "centre": s.center_mm,
                    "normal": s.normal,
                }
            )
            if (i + 1) % 40 == 0:
                print(f"    {i + 1}/{n}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260809)
    args = ap.parse_args(argv)

    print(f"rendering {args.poses} poses ...", flush=True)
    data = build(args.poses, args.seed)
    print(f"  {len(data)} usable\n")

    cal = TiltCalibration.load()
    bands = [(0, 20), (20, 40), (40, 55), (55, 71)]
    rows = {k: {b: [] for b in bands} for k in SCHEMES}

    for d in data:
        band = next((b for b in bands if b[0] <= d["tilt"] < b[1]), None)
        if band is None:
            continue
        for name, fn in SCHEMES.items():
            ell = fn(d["hull"])
            if ell is None:
                continue
            # tilt through the shipped correction, so the comparison is against
            # what actually ships rather than against a raw ratio nobody uses
            ratio = min(1.0, max(0.0, ell[1][1] / max(ell[1][0], 1e-9)))
            est = cal.tilt(math.degrees(math.acos(ratio)))
            rows[name][band].append(
                (
                    est - d["tilt"],
                    ell[1][0] / d["truth"][1][0] - 1.0,
                )
            )

    print(f"{'scheme':<12}" + "".join(f"{f'{a}-{b} deg':>22}" for a, b in bands))
    print(f"{'':12}" + "".join(f"{'|tilt err| / bias':>22}" for _ in bands))
    for name in SCHEMES:
        line = f"{name:<12}"
        for b in bands:
            v = np.array(rows[name][b])
            line += (
                f"{np.median(np.abs(v[:, 0])):9.2f} /{np.median(v[:, 0]):+7.2f}   "
                if len(v)
                else f"{'--':>22}"
            )
        print(line)

    print()
    print(f"{'scheme':<12}" + "".join(f"{f'{a}-{b} deg':>22}" for a, b in bands))
    print(f"{'':12}" + "".join(f"{'major axis err %':>22}" for _ in bands))
    for name in SCHEMES:
        line = f"{name:<12}"
        for b in bands:
            v = np.array(rows[name][b])
            line += (
                f"{np.median(v[:, 1]) * 100:9.2f} +-{np.std(v[:, 1]) * 100:6.2f}   "
                if len(v)
                else f"{'--':>22}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
