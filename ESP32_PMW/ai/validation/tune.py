"""
Fit the estimator's calibration on train, then score it on held-out test.

    uv run python controller/pose/validation/make_dataset.py   # once
    uv run python controller/pose/validation/tune.py            # fit + report
    uv run python controller/pose/validation/tune.py --write    # ...and save it

Two free parameters, both physically motivated rather than fudge factors:

**Effective radius.**  The segmenter returns a convex hull, which rides the
outermost surface, so the radius fed to `conic.py` should be the outer rim.  A
scalar error here is a *pure scale* error, showing up as a constant relative
depth bias -- which makes it directly measurable and directly correctable.

**Tilt calibration.**  The mast and magnet widen the silhouette's short axis as
the robot tilts, so tilt reads low.  See `calibration.py` for why only the mean
of that is recoverable.

Both are fitted on `split == 0` and every number reported is from `split == 1`,
which the fit never saw.  Residuals are given per degree of freedom, as absolute
values and as percentages, because "1 mm" means something different at 140 mm
range than at 360 mm.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

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
from shape import TiltCalibration  # noqa: E402

DEFAULT_DATA = HERE.parents[2] / "results" / "pose_validation" / "dataset.npz"

# The robot's own diameter -- the natural yardstick for a position error on an
# object this size, and resolution/range independent.
BODY_MM = 20.409


def load(path):
    d = np.load(path)
    return d, d["split"] == 0, d["split"] == 1


def solve_all(d, mask, radius, tilt_cal):
    """
    Run the back-projection over a split. Returns a dict of per-sample arrays.

        Position and orientation are scored against the branch that is best *for
        that quantity*, which needs care: the two ambiguity branches share almost
        the same centre but differ wildly in normal, so picking the branch by
        position and then reading its normal measures the coin toss rather than the
        geometry.  Both are therefore oracle-per-quantity, isolating what tuning can
        actually move.  The cost of the branch pick is real and is reported
        separately, from `margin_deg`.
    """

    K = d["K"]
    idx = np.nonzero(mask)[0]
    dxyz, ang, tilt_err, az_err, margins, kept = [], [], [], [], [], []

    for i in idx:
        ellipse = tilt_cal.apply(
            (
                (d["e_cx"][i], d["e_cy"][i]),
                (d["major"][i], d["minor"][i]),
                d["e_deg"][i],
            )
        )
        poses = conic.backproject_ellipse(ellipse, K, radius, verify_tol=None)
        if not poses:
            continue

        gt_c = np.array([d["cx_mm"][i], d["cy_mm"][i], d["cz_mm"][i]])
        gt_n = np.array([d["nx"][i], d["ny"][i], d["nz"][i]])

        pos_best = min(poses, key=lambda p: np.linalg.norm(p.center - gt_c))
        norm_best = min(poses, key=lambda p: -abs(float(p.normal @ gt_n)))

        dxyz.append(pos_best.center - gt_c)
        ang.append(
            math.degrees(math.acos(min(1.0, abs(float(norm_best.normal @ gt_n)))))
        )
        tilt_err.append(
            math.degrees(math.acos(min(1.0, abs(float(norm_best.normal[2])))))
            - d["tilt_deg"][i]
        )
        # Azimuth of the tilt: the fifth DOF. Sign-normalise the normal first --
        # it is only defined up to sign, and a flipped one rotates phi by 180.
        n_up = norm_best.normal if norm_best.normal[2] >= 0 else -norm_best.normal
        gt_up = gt_n if gt_n[2] >= 0 else -gt_n
        az_err.append(
            (
                math.degrees(math.atan2(n_up[1], n_up[0]))
                - math.degrees(math.atan2(gt_up[1], gt_up[0]))
                + 180.0
            )
            % 360.0
            - 180.0
        )
        margins.append(conic.ambiguity_margin_deg(poses))
        kept.append(i)

    kept = np.array(kept)
    return {
        "dxyz": np.array(dxyz),
        "normal_deg": np.array(ang),
        "tilt_err_deg": np.array(tilt_err),
        "az_err_deg": np.array(az_err),
        "margin_deg": np.array(margins),
        "idx": kept,
        "z": d["cz_mm"][kept],
        "tilt": d["tilt_deg"][kept],
    }


def fit_radius(d, train, radius0, tilt_cal):
    """
    Radius that removes the *median* relative depth bias.

        Depth scales exactly linearly with the assumed radius, so one pass measures
        it: if the estimate reads 0.2% deep, the radius is 0.2% small.

        The median, not the mean, and this matters a great deal.  About 1% of frames
        fail catastrophically under realistic noise -- the face-on, dimly-lit case
        where the hull collapses onto the blade cross -- with depth errors up to
        409%.  On that data the mean relative bias reads **+1.09%** while the median
        reads **-0.37%**: opposite signs.  Fitting to the mean moved the radius the
        wrong way and tripled the median position error (1.49 -> 3.30 mm).  A
        handful of broken frames must not be allowed to set a calibration constant
        for every good one.
    """

    r = solve_all(d, train, radius0, tilt_cal)
    rel = r["dxyz"][:, 2] / r["z"]
    bias = float(np.median(rel))
    return radius0 * (1.0 - bias), bias


def pct(x):
    return f"{x:.2f}%"


def report(name, r, extra=""):
    d = r["dxyz"]
    lateral = np.hypot(d[:, 0], d[:, 1])
    total = np.linalg.norm(d, axis=1)
    rel_z = np.abs(d[:, 2]) / r["z"]
    rel_tot = total / r["z"]

    print(f"\n--- {name} {extra}  (n = {len(total)})")
    print(
        f"  {'dof':<16s} {'median':>10s} {'p95':>10s} {'max':>10s}   {'% of range':>22s}"
    )
    for label, v, rel in (
        ("x  (mm)", np.abs(d[:, 0]), np.abs(d[:, 0]) / r["z"]),
        ("y  (mm)", np.abs(d[:, 1]), np.abs(d[:, 1]) / r["z"]),
        ("z  (mm)", np.abs(d[:, 2]), rel_z),
        ("lateral (mm)", lateral, lateral / r["z"]),
        ("|position| (mm)", total, rel_tot),
    ):
        print(
            f"  {label:<16s} {np.median(v):10.3f} {np.percentile(v, 95):10.3f} "
            f"{v.max():10.3f}   {pct(np.median(rel) * 100):>10s} med"
            f" {pct(np.percentile(rel, 95) * 100):>10s} p95"
        )

    print(
        f"  {'|position|/body':<16s} {np.median(total)/BODY_MM*100:9.2f}% "
        f"{np.percentile(total, 95)/BODY_MM*100:9.2f}% "
        f"{total.max()/BODY_MM*100:9.2f}%   (as % of the 20.4 mm robot)"
    )
    for label, v in (
        ("tilt θ (deg)", np.abs(r["tilt_err_deg"])),
        ("azimuth φ (deg)", np.abs(r["az_err_deg"])),
        ("normal (deg)", r["normal_deg"]),
    ):
        print(
            f"  {label:<16s} {np.median(v):10.3f} {np.percentile(v, 95):10.3f} {v.max():10.3f}"
        )
    print(
        f"  {'tilt bias (deg)':<16s} {r['tilt_err_deg'].mean():+10.3f}  (mean, i.e. the "
        f"systematic part)"
    )
    return {"pos": float(np.median(total)), "normal": float(np.median(r["normal_deg"]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--write", action="store_true", help="save the fitted calibration")
    args = ap.parse_args(argv)

    d, train, test = load(args.data)
    radius0 = float(d["rim_radius_mm"])
    identity = TiltCalibration()

    print("=" * 78)
    print("estimator tuning -- fit on train, all reported numbers from held-out test")
    print("=" * 78)
    print(
        f"data {Path(args.data).name} | train {train.sum()} | test {test.sum()} "
        f"| {int(d['resolution'][0])}x{int(d['resolution'][1])}"
    )

    base = report(
        "BASELINE (test)",
        solve_all(d, test, radius0, identity),
        f"radius {radius0:.4f} mm, no tilt correction",
    )

    # --- fit on train only -------------------------------------------------
    radius, bias = fit_radius(d, train, radius0, identity)
    print(
        f"\n[fit on train] relative depth bias {bias*100:+.4f}% "
        f"-> radius {radius0:.4f} -> {radius:.4f} mm"
    )

    tr = solve_all(d, train, radius, identity)
    theta_raw = tr["tilt"] + tr["tilt_err_deg"]  # what the uncorrected solve reported
    cal = TiltCalibration.fit(
        theta_raw,
        tr["tilt"],
        meta={
            "source": Path(args.data).name,
            "radius_mm": radius,
            "split": "train",
            "body_mm": BODY_MM,
        },
    )
    print(
        f"[fit on train] tilt correction  theta_true = {cal.a:.5f}*t + {cal.b:.6f}*t^2  "
        f"(n={cal.meta['n_samples']})"
    )
    print(
        f"               maps 10->{cal.tilt(10):.1f}  30->{cal.tilt(30):.1f}  "
        f"50->{cal.tilt(50):.1f}  70->{cal.tilt(70):.1f} deg"
    )

    # --- score on test only ------------------------------------------------
    r_rad = report(
        "+ radius only (test)",
        solve_all(d, test, radius, identity),
        f"radius {radius:.4f} mm",
    )
    tuned = solve_all(d, test, radius, cal)
    r_all = report(
        "+ radius + tilt calibration (test)", tuned, f"radius {radius:.4f} mm"
    )

    print("\n" + "=" * 78)
    print("improvement on held-out test, median")
    print("=" * 78)
    for label, before, after in (
        ("|position| (mm)", base["pos"], r_all["pos"]),
        ("normal (deg)", base["normal"], r_all["normal"]),
    ):
        print(
            f"  {label:<18s} {before:7.3f} -> {after:7.3f}   "
            f"({(after/before - 1) * 100:+.1f}%)"
        )

    print("\n-- ambiguity, unaffected by any of this (test)")
    m = tuned["margin_deg"]
    print(
        f"   candidate branches differ by a median of {np.median(m):.1f} deg; one frame"
    )
    print(f"   cannot choose, so a wrong pick costs that much. Needs a 2nd camera.")

    print("\n-- accuracy within the well-conditioned band (tilt 10-45 deg, test)")
    band = (tuned["tilt"] >= 10) & (tuned["tilt"] <= 45)
    sub = {
        k: (v[band] if isinstance(v, np.ndarray) and len(v) == len(band) else v)
        for k, v in tuned.items()
    }
    report("tuned, tilt 10-45 (test)", sub)

    if args.write:
        p = cal.save()
        print(f"\nwrote {p}")
        print(
            f"NOTE radius: set RADIUS_MM = {radius:.4f} in estimator.py "
            f"(was {radius0:.4f})"
        )
    else:
        print("\n(dry run -- pass --write to save the calibration)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
