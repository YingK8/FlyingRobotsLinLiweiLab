"""Render the robot across appearance and pose conditions, score the estimator.

    uv run python controller/pose/validation/sweep.py --quick
    uv run python controller/pose/validation/sweep.py

Four appearance axes, crossed with a pose grid:

  material alpha   0.7 .. 1.0     how translucent the robot is
  lighting         lateral ring   directional light from the side, by bearing
                   azimuthal dome overhead light, by elevation
                   ambient        the flat component
  background       0.0 .. 0.3     black through to mid grey
  pose             tilt, azimuth, depth

Each cell renders one frame, runs the real `segment` -> `estimator` path over
it, and scores against the pose that produced it.  Results go to a CSV that
`report.py` turns into figures.

Two things this measures that a single accuracy number would hide:

*Ambiguity versus geometry.*  Back-projecting a conic always yields two poses
and one frame cannot choose between them, so `normal_err_deg` (what the
estimator picked) and `normal_err_best_deg` (the better of the two) are reported
separately.  When they diverge the fit was fine and the branch was wrong -- a
completely different problem, with a completely different fix.

*The timing columns here are not a throughput measure.* Each cell renders a
frame through pyrender before estimating it, so `t_seg_ms` and `t_total_ms` are
measured with GL work interleaved and read 30-50% high; running a sweep
alongside another render job inflates them further (one stretch of this grid
recorded a 10.1 ms median against a 3.0 ms tail). They are kept because relative
comparisons *within* a run are still meaningful. For an actual rate, use
`validation/latency.py`, which estimates from a decoded video with nothing else
running.

*Where the model stops holding.*  Past roughly 45 degrees of tilt the mast and
magnet dominate the silhouette's short axis and the flat-circle assumption
breaks down.  The tilt axis runs well past that on purpose, so the envelope
appears in the data instead of in a footnote.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import render as rendermod  # noqa: E402
import segment as segmod  # noqa: E402
from estimator import PoseEstimator  # noqa: E402
from recorder import write_metadata  # noqa: E402

COLUMNS = [
    "alpha", "bg_level", "light", "ambient", "intensity",
    "exposure", "exposure_s", "sigma",
    "tilt_deg", "azimuth_deg", "x_mm", "y_mm", "z_mm",
    "detected",
    "pos_err_mm", "dx_mm", "dy_mm", "dz_mm",
    "normal_err_deg", "normal_err_best_deg", "tilt_err_deg",
    "ambiguity_margin_deg", "branch_wrong",
    "seg_iou", "area_px", "fit_rms_px", "major_px", "minor_px",
    "t_seg_ms", "t_est_ms", "t_total_ms",
]


def light_rigs(quick=False):
    """The lighting conditions to sweep.

    Ambient is swept alongside the directional terms because it is the knob that
    decides whether the silhouette exists at all -- the duct's outer wall is
    nearly edge-on face-on, so at ambient 0.05 only ~21% of the true silhouette
    clears the threshold, against ~89% at 0.30. Directional light sets contrast;
    ambient sets visibility.
    """
    if quick:
        return [
            rendermod.LightRig(dome=((60.0, 0.0),), ambient=0.35, intensity=10.0),
            rendermod.LightRig(lateral_deg=(0.0,), ambient=0.15, intensity=20.0),
        ]

    rigs = []
    # Lateral ring: a single hard side light, swept round the bearing circle.
    for bearing in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        rigs.append(
            rendermod.LightRig(
                lateral_deg=(bearing,), ambient=0.15, intensity=20.0, key_from_camera=False
            )
        )
    # Azimuthal dome: overhead diffuse at three elevations.
    for elev in (20.0, 50.0, 80.0):
        rigs.append(
            rendermod.LightRig(
                dome=((elev, 0.0), (elev, 120.0), (elev, 240.0)),
                ambient=0.25,
                intensity=8.0,
                key_from_camera=False,
            )
        )
    # Ambient-dominant and ambient-starved extremes, with a lens-side key.
    for amb in (0.05, 0.35, 0.60):
        rigs.append(rendermod.LightRig(dome=((60.0, 0.0),), ambient=amb, intensity=10.0))
    return rigs


def exposure_grid(quick=False, sensor=False, subframes=7):
    """Sensor conditions: clean, noisy, blurred, and both.

    Kept small on purpose.  Measured separately, read noise is negligible below
    about sigma 20 and rotor blur does not move the pose estimate at all (the
    hull is bounded by the rotationally symmetric rim, which spinning cannot
    smear).  A fine grid over either would spend a lot of rendering to
    re-establish that, so this samples the corners and lets the poses and
    lighting carry the resolution.

    Blurred cells cost `subframes` times as much to render, which is the other
    reason to keep this coarse.
    """
    spin = 330.0
    if not sensor:
        # Default: read noise, but no motion blur. Not a shortcut -- blur is
        # measured (here and in --sensor) to leave pose accuracy unchanged,
        # because the hull is bounded by the rotationally symmetric rim that
        # spinning cannot smear. Rendering every cell blurred would cost
        # `subframes` times as much to re-establish a null result, so the main
        # grid spends its budget on lighting and pose instead, and `--sensor`
        # carries the blur evidence.
        return [("noise s12", rendermod.Exposure(sigma=12.0))]
    if quick:
        return [
            ("clean", None),
            ("noisy+blurred", rendermod.Exposure(
                exposure_s=1 / 2000, subframes=5, spin_hz=spin, sigma=12.0,
                velocity_mm_s=(60.0, 0.0, 0.0))),
        ]
    return [
        ("clean", None),
        ("noise s10", rendermod.Exposure(sigma=10.0)),
        ("noise s25", rendermod.Exposure(sigma=25.0)),
        ("blur 1/2000", rendermod.Exposure(
            exposure_s=1 / 2000, subframes=subframes, spin_hz=spin,
            velocity_mm_s=(60.0, 0.0, 0.0), tilt_rate_deg_s=40.0)),
        ("blur 1/1000 + noise", rendermod.Exposure(
            exposure_s=1 / 1000, subframes=subframes, spin_hz=spin,
            velocity_mm_s=(60.0, 0.0, 0.0), tilt_rate_deg_s=40.0, sigma=8.0)),
    ]


def pose_grid(quick=False, reduced=False):
    """(tilt, azimuth, centre) triples.

    Depth is varied along with tilt because the conic's conditioning goes as
    (r/z)^2 -- accuracy at 150 mm and at 350 mm are different questions.

    ``reduced`` trims azimuth and depth when the sensor axis is being swept, to
    keep the total render count sane: blurred cells cost `subframes` times a
    clean one, so the full product would run for hours.
    """
    if quick:
        tilts = (0.0, 20.0, 40.0)
        azis = (0.0, 180.0)
        centres = ([0.0, 0.0, 220.0],)
    elif reduced:
        tilts = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
        azis = (0.0, 180.0)
        centres = ([0.0, 0.0, 160.0], [-18.0, 14.0, 320.0])
    else:
        tilts = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0)
        azis = (0.0, 90.0, 180.0, 270.0)
        centres = ([0.0, 0.0, 160.0], [12.0, -9.0, 220.0], [-18.0, 14.0, 320.0])
    return [(t, a, c) for t in tilts for a in azis for c in centres]


def score(sample, pose, seg, exposure_label=""):
    """Metrics for one cell. ``pose``/``seg`` may be ``None`` if detection failed."""
    row = {
        "alpha": sample.alpha,
        "bg_level": sample.bg_level,
        "light": sample.light.label(),
        "ambient": sample.light.ambient,
        "intensity": sample.light.intensity,
        "exposure": exposure_label,
        "exposure_s": sample.exposure.exposure_s if sample.exposure else 0.0,
        "sigma": sample.exposure.sigma if sample.exposure else 0.0,
        "tilt_deg": sample.tilt_deg,
        "azimuth_deg": sample.azimuth_deg,
        "x_mm": sample.center_mm[0],
        "y_mm": sample.center_mm[1],
        "z_mm": sample.center_mm[2],
        "detected": int(pose is not None),
    }
    if pose is None:
        row.update({k: "" for k in COLUMNS if k not in row})
        row["detected"] = 0
        return row

    d = pose.xyz_mm - sample.center_mm
    cands = pose.extra.get("candidates") or []
    errs = [
        math.degrees(math.acos(float(np.clip(abs(c.normal @ sample.normal), -1.0, 1.0))))
        for c in cands
    ]
    chosen_err = math.degrees(
        math.acos(float(np.clip(abs(pose.normal @ sample.normal), -1.0, 1.0)))
    )
    best_err = min(errs) if errs else chosen_err

    # Tilt is what the controller would actually consume, so score it directly
    # as well as the full normal angle.
    tilt_est = math.degrees(math.acos(float(np.clip(abs(pose.normal[2]), -1.0, 1.0))))

    inter = np.logical_and(seg.mask > 0, sample.mask).sum()
    union = np.logical_or(seg.mask > 0, sample.mask).sum()

    row.update(
        {
            "pos_err_mm": float(np.linalg.norm(d)),
            "dx_mm": float(d[0]), "dy_mm": float(d[1]), "dz_mm": float(d[2]),
            "normal_err_deg": chosen_err,
            "normal_err_best_deg": best_err,
            "tilt_err_deg": tilt_est - sample.tilt_deg,
            "ambiguity_margin_deg": pose.ambiguity_margin_deg,
            "branch_wrong": int(chosen_err > best_err + 1.0),
            "seg_iou": float(inter / union) if union else 0.0,
            "area_px": seg.area_px,
            "fit_rms_px": seg.fit_rms_px,
            "major_px": pose.ellipse[1][0],
            "minor_px": pose.ellipse[1][1],
            "t_seg_ms": pose.t_seg_ms,
            "t_est_ms": pose.t_est_ms,
            "t_total_ms": pose.t_total_ms,
        }
    )
    return row


def run(out_path, quick=False, sensor=False, width=1024, height=768, progress_every=50):
    alphas = (0.7, 1.0) if quick else (0.7, 0.8, 0.9, 1.0)
    bgs = (0.0, 0.2) if quick else (0.0, 0.1, 0.2, 0.3)
    rigs = light_rigs(quick)
    poses = pose_grid(quick, reduced=sensor)
    exps = exposure_grid(quick, sensor=sensor)
    if sensor and not quick:
        rigs = rigs[:1] + rigs[8:11] + rigs[-2:]  # one lateral, the domes, two ambients

    total = len(alphas) * len(bgs) * len(rigs) * len(poses) * len(exps)
    print(f"sweep: {len(alphas)} alpha x {len(bgs)} background x {len(rigs)} lighting "
          f"x {len(poses)} poses x {len(exps)} sensor = {total} renders")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    done = 0

    with rendermod.Renderer(width, height) as r, out_path.open("w", newline="") as fh:
        # Deliberately the estimator's own defaults for radius and tilt
        # calibration, not the mesh's raw geometry: this sweep should measure the
        # configuration that actually ships, not an idealised one.
        est = PoseEstimator(camera_matrix=r.K, dist_coeffs=None)
        write_metadata(
            fh,
            {
                "mesh": rendermod.MESH_PATH.name,
                "mesh_rim_radius_mm": f"{rendermod.RIM_RADIUS_MM:.4f}",
                "estimator_radius_mm": f"{est.radius_mm:.4f}",
                "tilt_calibration": "identity" if est.tilt_cal.is_identity
                else f"a={est.tilt_cal.a:.5f} b={est.tilt_cal.b:.6f}",
                "resolution": f"{width}x{height}",
                "intrinsics": rendermod.INTRINSICS_PATH.name,
                "cells": total,
                "mode": "quick" if quick else "full",
                "sensor_conditions": "|".join(lbl for lbl, _ in exps),
                "note": "normal_err_deg is the chosen branch; normal_err_best_deg is the better of the two",
            },
        )
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()

        for alpha, bg, rig, (exp_label, exp), (tilt, az, ctr) in itertools.product(
                alphas, bgs, rigs, exps, poses):
            sample = r.render(tilt, az, ctr, alpha=alpha, light=rig, bg_level=bg,
                              exposure=exp)

            # Reset per cell: cells are independent conditions, not a trajectory,
            # so carrying branch history between them would let an earlier cell
            # silently fix a later one's ambiguity.
            est.reset()
            pose = est.update(sample.image)
            seg = pose.extra.get("segmentation") if pose is not None else None

            w.writerow(score(sample, pose, seg, exp_label))

            done += 1
            if done % progress_every == 0 or done == total:
                el = time.monotonic() - t_start
                eta = el / done * (total - done)
                print(f"  {done:5d}/{total}  {el:6.1f}s elapsed, {eta:6.1f}s left", flush=True)

    print(f"wrote {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small grid, for a smoke test")
    ap.add_argument("--sensor", action="store_true",
                    help="sweep noise and motion-blur conditions (reduces the pose and "
                         "lighting grids to keep the render count sane)")
    ap.add_argument("--out", default=None, help="output CSV")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    args = ap.parse_args(argv)

    # Results land in ESP32_PMW/results/, alongside the rest of the project's run
    # artifacts, rather than next to the code -- a full sweep CSV is ~8 MB.
    results = HERE.parents[2] / "results" / "pose_validation"
    name = ("validation_results_quick.csv" if args.quick
            else "validation_results_sensor.csv" if args.sensor
            else "validation_results.csv")
    out = args.out or str(results / name)
    path = run(out, quick=args.quick, sensor=args.sensor,
               width=args.width, height=args.height)

    try:
        import report

        report.summarise(path)
    except Exception as e:  # a failed summary must not lose the sweep data
        print(f"(summary unavailable: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
