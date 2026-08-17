"""
Fine-grained noise and background curves.

    uv run python controller/pose/validation/sensitivity.py

`sweep.py` crosses many conditions at once, which leaves only 3 noise levels and
4 background levels -- enough to say "no strong effect", not enough to draw a
curve or to find the level where things break.  This sweeps each axis on its own,
finely.

**The same fixed pose and lighting set is used at every level.**  That is the
whole design: with a few hundred samples, redrawing poses per level would let
sampling noise move the curve more than the condition does, and the result would
be an artefact you could mistake for a threshold.  Holding poses fixed makes each
level a paired comparison, so a difference between levels is the condition.

Motion blur is deliberately *not* crossed in here.  It costs `subframes` times as
much to render and was already measured to leave pose accuracy unchanged (0.61 ->
0.67 mm through 475 degrees of blade sweep), because the convex hull is bounded
by the rotationally symmetric rim that spinning cannot smear.  Paying 7x to
re-establish a null result would buy resolution on the axes that do matter.

On the statistic: about 1% of frames fail catastrophically, with depth errors to
409%.  A mean squared error over that is decided by whether a failure landed in
the bin, so every level reports RMSE, the median, **and the failure rate** -- the
last being the honest version of what MSE is reaching for.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
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

import render as rendermod  # noqa: E402
from estimator import PoseEstimator  # noqa: E402
from recorder import write_metadata  # noqa: E402

DEFAULT_OUT = HERE.parents[2] / "results" / "pose_validation" / "sensitivity.csv"

NOISE_LEVELS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0)
BG_LEVELS = (0.0, 0.083, 0.167, 0.25, 0.333, 0.417, 0.5)

# A frame is "failed" when depth is off by more than this fraction of range.
# Well clear of the ~1% normal spread, so it selects genuine breakdowns rather
# than the tail of the healthy distribution.
FAIL_REL_DEPTH = 0.05

COLUMNS = [
    "axis",
    "level",
    "sigma",
    "bg_level",
    "tilt_deg",
    "azimuth_deg",
    "z_mm",
    "ambient",
    "detected",
    "pos_err_mm",
    "dx_mm",
    "dy_mm",
    "dz_mm",
    "rel_depth",
    "normal_err_deg",
    "normal_err_best_deg",
    "tilt_err_deg",
    "seg_iou",
    "fit_rms_px",
    "major_px",
    "failed",
]


def fixed_poses(n, seed=99):
    """
    One pose/lighting set, reused at every level of every axis.
    """

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        tilt = float(
            np.degrees(
                np.arccos(
                    rng.uniform(np.cos(np.radians(55.0)), np.cos(np.radians(8.0)))
                )
            )
        )
        out.append(
            (
                tilt,
                float(rng.uniform(0, 360)),
                np.array(
                    [rng.uniform(-14, 14), rng.uniform(-14, 14), rng.uniform(160, 320)]
                ),
                rendermod.LightRig(
                    dome=((float(rng.uniform(40, 80)), float(rng.uniform(0, 360))),),
                    ambient=float(rng.uniform(0.30, 0.55)),
                    intensity=float(rng.uniform(8, 16)),
                ),
                float(rng.choice([0.7, 0.85, 1.0])),
            )
        )
    return out


def score(sample, pose, seg, axis, level):
    row = {
        "axis": axis,
        "level": level,
        "sigma": sample.exposure.sigma if sample.exposure else 0.0,
        "bg_level": sample.bg_level,
        "tilt_deg": sample.tilt_deg,
        "azimuth_deg": sample.azimuth_deg,
        "z_mm": sample.center_mm[2],
        "ambient": sample.light.ambient,
        "detected": int(pose is not None),
    }
    if pose is None:
        row.update({k: "" for k in COLUMNS if k not in row})
        row["detected"], row["failed"] = 0, 1
        return row

    d = np.asarray(pose.xyz_mm) - sample.center_mm
    cands = pose.extra.get("candidates") or []
    errs = [
        math.degrees(
            math.acos(float(np.clip(abs(c.normal @ sample.normal), -1.0, 1.0)))
        )
        for c in cands
    ]
    chosen = math.degrees(
        math.acos(float(np.clip(abs(pose.normal @ sample.normal), -1.0, 1.0)))
    )
    rel = float(d[2] / sample.center_mm[2])
    inter = np.logical_and(seg.mask > 0, sample.mask).sum()
    union = np.logical_or(seg.mask > 0, sample.mask).sum()

    row.update(
        {
            "pos_err_mm": float(np.linalg.norm(d)),
            "dx_mm": float(d[0]),
            "dy_mm": float(d[1]),
            "dz_mm": float(d[2]),
            "rel_depth": rel,
            "normal_err_deg": chosen,
            "normal_err_best_deg": min(errs) if errs else chosen,
            "tilt_err_deg": math.degrees(
                math.acos(float(np.clip(abs(pose.normal[2]), -1.0, 1.0)))
            )
            - sample.tilt_deg,
            "seg_iou": float(inter / union) if union else 0.0,
            "fit_rms_px": seg.fit_rms_px,
            "major_px": pose.ellipse[1][0],
            "failed": int(abs(rel) > FAIL_REL_DEPTH),
        }
    )
    return row


def run(out_path, n_poses=300, width=1024, height=768, quick=False):
    if quick:
        n_poses = 20
    poses = fixed_poses(n_poses)
    noise = NOISE_LEVELS[::3] if quick else NOISE_LEVELS
    bgs = BG_LEVELS[::3] if quick else BG_LEVELS
    total = len(poses) * (len(noise) + len(bgs))

    print(
        f"sensitivity: {n_poses} fixed poses x ({len(noise)} noise + {len(bgs)} background) "
        f"= {total} renders"
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    done = 0

    with rendermod.Renderer(width, height) as r, out_path.open("w", newline="") as fh:
        est = PoseEstimator(camera_matrix=r.K, dist_coeffs=None)
        write_metadata(
            fh,
            {
                "n_poses": n_poses,
                "noise_levels": " ".join(f"{s:g}" for s in noise),
                "bg_levels": " ".join(f"{b:g}" for b in bgs),
                "estimator_radius_mm": f"{est.radius_mm:.4f}",
                "fail_threshold": f"|dz|/z > {FAIL_REL_DEPTH:.0%}",
                "note": "poses held fixed across levels, so differences are the condition",
            },
        )
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()

        for axis, levels in (("noise", noise), ("background", bgs)):
            for level in levels:
                for tilt, az, ctr, light, alpha in poses:
                    # The other axis is held at its realistic baseline, so each
                    # curve is a clean one-factor-at-a-time cut.
                    sigma = level if axis == "noise" else 8.0
                    bg = level if axis == "background" else 0.05
                    exp = rendermod.Exposure(sigma=sigma) if sigma > 0 else None
                    s = r.render(
                        tilt,
                        az,
                        ctr,
                        alpha=alpha,
                        light=light,
                        bg_level=bg,
                        exposure=exp,
                    )
                    est.reset()
                    pose = est.update(s.image)
                    seg = pose.extra.get("segmentation") if pose is not None else None
                    w.writerow(score(s, pose, seg, axis, float(level)))

                    done += 1
                    if done % 250 == 0:
                        el = time.monotonic() - t0
                        print(
                            f"  {done:5d}/{total}  {el:5.1f}s, {el/done*(total-done):5.1f}s left",
                            flush=True,
                        )

    print(f"wrote {out_path}")
    return out_path


def summarise(path):
    import pandas as pd

    d = pd.read_csv(path, comment="#")
    for axis, unit in (("noise", "sigma (DN)"), ("background", "grey level")):
        sub = d[d.axis == axis]
        if sub.empty:
            continue
        print(f"\n-- {axis}: {unit}")
        print(
            f"  {'level':>8s} {'n':>5s} {'median':>9s} {'RMSE':>9s} {'p95':>9s} "
            f"{'fail %':>7s} {'IoU':>6s} {'normal':>8s}"
        )
        for level, g in sub.groupby("level"):
            det = g[g.detected == 1]
            if det.empty:
                print(f"  {level:8g} {len(g):5d}   (nothing detected)")
                continue
            e = det.pos_err_mm
            print(
                f"  {level:8g} {len(g):5d} {e.median():8.3f}mm "
                f"{math.sqrt((e**2).mean()):8.3f}mm {e.quantile(.95):8.3f}mm "
                f"{g.failed.mean()*100:6.1f}% {det.seg_iou.median():6.3f} "
                f"{det.normal_err_best_deg.median():7.3f}d"
            )
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses", type=int, default=300)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    args = ap.parse_args(argv)

    path = run(
        args.out,
        n_poses=args.poses,
        width=args.width,
        height=args.height,
        quick=args.quick,
    )
    try:
        summarise(path)
    except Exception as e:
        print(f"(summary unavailable: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
