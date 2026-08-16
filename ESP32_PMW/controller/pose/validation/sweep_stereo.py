"""Score the stereo estimator against render ground truth, across rig geometry.

The question this exists to answer is narrow and quantitative: **is every axis of
position inside 0.5 mm at the operating resolution, and which rig geometry gets
there.**  Everything else it reports is in service of reading that number
correctly.

Three design choices worth knowing before reading the output.

**Monocular gets the oracle branch.**  A single view cannot resolve the two-fold
ambiguity, so scoring its *actual* pick would mostly measure how often a coin
came up heads, and would flatter stereo for a reason that has nothing to do with
geometry.  So the mono column is scored on whichever of its two branches is
closer to truth -- an estimator that cannot exist -- and the price of the
ambiguity is reported separately as ``mono_branch_wrong``.  Stereo has to beat
the oracle, not the coin.

**Position error is reported per world axis, not just as a norm.**  "0.5 mm in
all dimensions" is a statement about axes.  A norm can sit comfortably under the
gate while one axis quietly does not, and for a rig whose whole argument is
about *anisotropy* that is exactly the failure worth catching.

**Rig geometry is a swept axis, not a constant.**  Elevation trades triangulation
against the flat-circle model in opposite directions -- lower elevation separates
the optical axes further but shows each camera a more tilted rotor -- and which
wins is an empirical question the analysis cannot settle.

Usage:
    uv run python controller/pose/validation/sweep_stereo.py --quick
    uv run python controller/pose/validation/sweep_stereo.py
    uv run python controller/pose/validation/sweep_stereo.py --elev 50 --poses 400
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import conic  # noqa: E402
import render as rendermod  # noqa: E402
import render_stereo as rs  # noqa: E402
import segment as segmod  # noqa: E402
import stereo as stereomod  # noqa: E402
from estimator import RADIUS_MM  # noqa: E402
from rig import StereoRig  # noqa: E402

DEFAULT_OUT = HERE.parents[2] / "results" / "pose_validation" / "stereo_results.csv"

# The intrinsics were calibrated on a ~1024 px wide image, so rendering smaller
# means scaling K to match. Without this the principal point (cx = 497.6) lands
# at the edge of a 500 px frame and the robot renders off-centre.
CALIB_WIDTH = 1024.0

# The gate.
TARGET_AXIS_MM = 0.5
# Reported against, not gated on -- see the plan: this is past what the
# monocular residual's own floor allows and the point is to measure the gap.
TARGET_NORMAL_DEG = 0.5

COLUMNS = [
    "elev_a_deg", "elev_b_deg", "azim_a_deg", "azim_b_deg", "range_mm",
    "axis_sep_deg", "width", "height", "occluders",
    "tilt_deg", "azimuth_deg", "cx_mm", "cy_mm", "cz_mm",
    "n_views", "detected",
    "stereo_dx_mm", "stereo_dy_mm", "stereo_dz_mm", "stereo_pos_mm", "stereo_normal_deg",
    "mono_dx_mm", "mono_dy_mm", "mono_dz_mm", "mono_pos_mm", "mono_normal_deg",
    "mono_branch_wrong", "mono_margin_deg",
    "discrepancy_mm", "margin_mm", "refine_rms_px", "refine_iters",
    "seed_pos_mm", "seed_normal_deg",
    "tilt_seen_a_deg", "tilt_seen_b_deg",
    "seg_iou_a", "seg_iou_b", "t_seg_ms", "t_est_ms", "rejected", "rejected_fit",
]


def sample_poses(rng, n, tilt_max=25.0, z_range=(-40.0, 40.0), lateral_mm=22.0):
    """Random world poses inside the working volume.

    Two departures from `make_dataset.sample_poses`, both because the world frame
    here is the rig's rather than a camera's:

    * ``tilt`` is lean from **vertical**, and is capped near the robot's real
      attitude envelope (1.1 deg RMS, 5.2 deg peak-to-peak per the platform
      notes, so 25 deg is already generous).  The camera-relative tilt that
      actually stresses the flat-circle model is set by the *rig*, not by this.
    * ``z`` is height about the hover point rather than range from a lens; range
      is a property of where the cameras were put.
    """
    cos_lo = math.cos(math.radians(tilt_max))
    tilt = np.degrees(np.arccos(rng.uniform(cos_lo, 1.0, n)))
    az = rng.uniform(0.0, 360.0, n)
    x = rng.uniform(-lateral_mm, lateral_mm, n)
    y = rng.uniform(-lateral_mm, lateral_mm, n)
    z = rng.uniform(*z_range, n)
    return tilt, az, np.column_stack([x, y, z])


def lighting(rng, n):
    """Adequately-lit rigs only, matching `make_dataset.lighting`.

    Ambient below ~0.25 is a segmentation failure, already characterised by the
    monocular sweep, and no amount of stereo geometry repairs an unlit rim.
    Including those cases here would add outliers that swamp the signal being
    measured, which is the geometry.
    """
    out = []
    for _ in range(n):
        amb = rng.uniform(0.25, 0.6)
        if rng.random() < 0.5:
            out.append(rendermod.LightRig(dome=((rng.uniform(30, 80), rng.uniform(0, 360)),),
                                          ambient=amb, intensity=rng.uniform(6, 20)))
        else:
            out.append(rendermod.LightRig(lateral_deg=(rng.uniform(0, 360),),
                                          ambient=amb, intensity=rng.uniform(6, 20)))
    return out


def exposures(rng, n, subframes=5):
    """Realistic high-frame-rate sensor conditions, per `make_dataset.exposures`.

    Exposure and read noise are drawn anti-correlated because that is the real
    trade: a shorter exposure collects fewer photons and reads out noisier.
    Velocity is a world quantity here, so one motion blurs both views
    consistently.
    """
    out = []
    for i in range(n):
        exp_s = float(rng.uniform(1 / 4000, 1 / 1000))
        sigma = float(np.clip(4.5 * (1 / 1000) / exp_s, 3.0, 25.0))
        out.append(rendermod.Exposure(
            exposure_s=exp_s, subframes=subframes,
            spin_hz=float(rng.uniform(310.0, 350.0)),
            velocity_mm_s=tuple(rng.uniform(-40.0, 40.0, 3)),
            tilt_rate_deg_s=float(rng.uniform(-40.0, 40.0)),
            sigma=sigma, seed=i,
        ))
    return out


def _mono_oracle(seg, cam, truth_c, truth_n, radius_mm, tilt_cal):
    """Best-case single-view answer for this view, in world coordinates.

    Returns ``(dxyz, normal_deg, branch_wrong, margin_deg)``.  ``branch_wrong``
    is whether a face-on prior -- what `estimator.PoseEstimator` falls back to
    with no history -- would have taken the worse branch.  That is the cost the
    second camera removes, kept separate from the geometric error it also
    reduces, because they are different claims.
    """
    if seg is None:
        return None
    ellipse = seg.ellipse
    if cam.dist is not None and np.any(cam.dist):
        try:
            ellipse = segmod.undistort_ellipse(ellipse, cam.K, cam.dist)
        except Exception:
            pass
    ellipse = tilt_cal.apply(ellipse)
    cands = conic.backproject_ellipse(ellipse, cam.K, radius_mm)
    if not cands:
        return None

    scored = []
    for c in cands:
        cw, nw = cam.to_world(c.center, c.normal)
        scored.append((float(np.linalg.norm(cw - truth_c)),
                       stereomod.line_angle_deg(nw, truth_n), cw))
    best = min(scored, key=lambda s: s[0])

    # What a prior-based pick would have chosen: the candidate whose normal is
    # closest to facing the camera, which is `PoseEstimator._prior_normal`.
    prior = np.array([0.0, 0.0, -1.0])
    picked = max(range(len(cands)), key=lambda i: float(cands[i].normal @ prior))
    branch_wrong = int(scored[picked][0] > best[0] + 1e-6)

    return (best[2] - truth_c, best[1], branch_wrong,
            conic.ambiguity_margin_deg(cands))


def run(renderer, rig, n_poses, seed, occluders=(), subframes=5, loss="linear",
        radius_mm=RADIUS_MM, progress=True):
    """Render and score ``n_poses`` through one rig configuration.

    Takes an existing `render_stereo.StereoRenderer` rather than making one:
    there is a single GL context per process, so a sweep across geometries must
    re-point one renderer instead of constructing several.
    """
    width, height = renderer.width, renderer.height
    renderer.set_rig(rig)
    renderer.set_occluders(occluders)
    rng = np.random.default_rng(seed)
    tilt, az, centres = sample_poses(rng, n_poses)
    rigs = lighting(rng, n_poses)
    exps = exposures(rng, n_poses, subframes=subframes)
    alphas = rng.choice([0.8, 0.9, 1.0], n_poses)
    bgs = rng.choice([0.0, 0.1, 0.2], n_poses)

    est = stereomod.StereoPoseEstimator(rig, radius_mm=radius_mm, loss=loss)
    rows = []
    prev_rejected = prev_fit = 0
    occ_label = "+".join(o[3] if len(o) > 3 else "occ" for o in occluders) or "none"
    t0 = time.monotonic()
    r = renderer

    if True:
        for i in range(n_poses):
            s = r.render_pair(float(tilt[i]), float(az[i]), centres[i],
                              alpha=float(alphas[i]), light=rigs[i],
                              bg_level=float(bgs[i]), exposure=exps[i])

            row = {
                "elev_a_deg": rig.meta.get("elev_deg", [np.nan, np.nan])[0],
                "elev_b_deg": rig.meta.get("elev_deg", [np.nan, np.nan])[1],
                "azim_a_deg": rig.meta.get("azim_deg", [np.nan, np.nan])[0],
                "azim_b_deg": rig.meta.get("azim_deg", [np.nan, np.nan])[1],
                "range_mm": rig.meta.get("range_mm", np.nan),
                "axis_sep_deg": rig.axis_separation_deg(),
                "width": width, "height": height, "occluders": occ_label,
                "tilt_deg": s.tilt_deg, "azimuth_deg": s.azimuth_deg,
                "cx_mm": s.center_world[0], "cy_mm": s.center_world[1],
                "cz_mm": s.center_world[2],
                "tilt_seen_a_deg": rig.tilt_seen_deg(s.normal_world)[0],
                "tilt_seen_b_deg": rig.tilt_seen_deg(s.normal_world)[1],
            }

            pose = est.update([v.image for v in s.views], t=float(i))
            row["detected"] = int(pose is not None)
            if pose is not None:
                d = pose.xyz_mm - s.center_world
                row.update({
                    "n_views": pose.n_views,
                    "stereo_dx_mm": d[0], "stereo_dy_mm": d[1], "stereo_dz_mm": d[2],
                    "stereo_pos_mm": float(np.linalg.norm(d)),
                    "stereo_normal_deg": stereomod.line_angle_deg(pose.normal, s.normal_world),
                    "discrepancy_mm": pose.discrepancy_mm, "margin_mm": pose.margin_mm,
                    "refine_rms_px": pose.refine_rms_px, "refine_iters": pose.refine_iters,
                    "t_seg_ms": pose.t_seg_ms, "t_est_ms": pose.t_est_ms,
                })
                # The closed-form seed, so the refinement's contribution is
                # separable from the fusion's rather than bundled together.
                m = pose.extra["match"]
                used = pose.extra["views_used"]
                seed_c, seed_n, _ = stereomod.fuse(
                    m.poses, stereomod._subset(rig, used))
                row["seed_pos_mm"] = float(np.linalg.norm(seed_c - s.center_world))
                row["seed_normal_deg"] = stereomod.line_angle_deg(seed_n, s.normal_world)

                segs = pose.per_view
                for k, (tag, view) in enumerate(zip("ab", s.views)):
                    sg = segs[k] if k < len(segs) else None
                    if sg is not None:
                        inter = np.logical_and(sg.mask > 0, view.mask).sum()
                        union = np.logical_or(sg.mask > 0, view.mask).sum()
                        row[f"seg_iou_{tag}"] = inter / union if union else 0.0

                mono = _mono_oracle(segs[0], rig.cameras[0], s.center_world,
                                    s.normal_world, radius_mm, est.tilt_cal)
                if mono is not None:
                    md, mn, bw, mm_deg = mono
                    row.update({
                        "mono_dx_mm": md[0], "mono_dy_mm": md[1], "mono_dz_mm": md[2],
                        "mono_pos_mm": float(np.linalg.norm(md)),
                        "mono_normal_deg": mn, "mono_branch_wrong": bw,
                        "mono_margin_deg": mm_deg,
                    })

            row["rejected"] = int(pose is None and est.n_rejected > prev_rejected)
            row["rejected_fit"] = int(pose is None and est.n_rejected_fit > prev_fit)
            prev_rejected, prev_fit = est.n_rejected, est.n_rejected_fit
            rows.append(row)
            if progress and (i + 1) % 50 == 0:
                el = time.monotonic() - t0
                print(f"    {i + 1}/{n_poses}  {el:5.1f}s elapsed, "
                      f"{el / (i + 1) * (n_poses - i - 1):5.1f}s left", flush=True)
    return rows


def summarise(rows, label=""):
    """Print the gate verdict and the numbers behind it."""
    det = [r for r in rows if r.get("detected")]
    if not det:
        print(f"  {label}: no detections")
        return None

    def col(name):
        return np.array([r[name] for r in det if r.get(name) is not None and
                         np.isfinite(r.get(name, np.nan))])

    out = {"label": label, "n": len(det), "detect_rate": len(det) / len(rows)}
    # A consistency gate that silently drops frames would flatter every number
    # below it, so say how many it took and why.
    n_rej = sum(1 for r in rows if r.get("rejected"))
    n_fit = sum(1 for r in rows if r.get("rejected_fit"))
    n_seg = len(rows) - len(det) - n_rej - n_fit
    print(f"\n  {label}   n={len(det)}/{len(rows)} detected"
          + (f"  ({n_rej} cross-view, {n_fit} fit-quality, {n_seg} not segmented)"
             if len(det) < len(rows) else ""))

    print(f"    {'axis':<10} {'median':>9} {'p95':>9} {'max':>9}   gate {TARGET_AXIS_MM} mm")
    worst_median = 0.0
    for ax in ("dx", "dy", "dz"):
        v = np.abs(col(f"stereo_{ax}_mm"))
        if not len(v):
            continue
        worst_median = max(worst_median, float(np.median(v)))
        flag = "OK" if np.median(v) < TARGET_AXIS_MM else "OVER"
        print(f"    {ax:<10} {np.median(v):9.4f} {np.percentile(v, 95):9.4f} "
              f"{v.max():9.4f}   {flag}")
        out[f"stereo_{ax}_median"] = float(np.median(v))

    for name, key in (("|pos|", "stereo_pos_mm"), ("normal deg", "stereo_normal_deg")):
        v = col(key)
        if len(v):
            print(f"    {name:<10} {np.median(v):9.4f} {np.percentile(v, 95):9.4f} "
                  f"{v.max():9.4f}")
            out[key] = float(np.median(v))

    mono_p, mono_n = col("mono_pos_mm"), col("mono_normal_deg")
    if len(mono_p):
        sp, sn = col("stereo_pos_mm"), col("stereo_normal_deg")
        bw = col("mono_branch_wrong")
        print(f"    vs mono (oracle branch):  |pos| {np.median(mono_p):.4f} -> "
              f"{np.median(sp):.4f} mm ({np.median(mono_p) / max(np.median(sp), 1e-9):.2f}x)"
              f"   normal {np.median(mono_n):.3f} -> {np.median(sn):.3f} deg "
              f"({np.median(mono_n) / max(np.median(sn), 1e-9):.2f}x)")
        if len(bw):
            print(f"    mono would have picked the wrong branch on "
                  f"{bw.mean():.1%} of frames; stereo picks geometrically")
        out["mono_pos_median"] = float(np.median(mono_p))
        out["mono_normal_median"] = float(np.median(mono_n))

    sd, sn2 = col("seed_pos_mm"), col("seed_normal_deg")
    if len(sd):
        print(f"    closed-form seed -> refined:  |pos| {np.median(sd):.4f} -> "
              f"{np.median(col('stereo_pos_mm')):.4f} mm,  normal {np.median(sn2):.3f} -> "
              f"{np.median(col('stereo_normal_deg')):.3f} deg")

    disc, marg = col("discrepancy_mm"), col("margin_mm")
    if len(disc):
        print(f"    cross-view discrepancy {np.median(disc):.4f} mm median, "
              f"branch margin {np.median(marg):.2f} mm median")
    ts, te = col("t_seg_ms"), col("t_est_ms")
    if len(ts):
        budget = 1000.0 / 420
        tot = np.median(ts) + np.median(te)
        print(f"    compute: seg {np.median(ts):.3f} + solve {np.median(te):.3f} = "
              f"{tot:.3f} ms  ({1000 / tot:.0f} Hz sustained, budget {budget:.3f} ms at 420 fps)")
    out["worst_axis_median"] = worst_median
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses", type=int, default=250, help="poses per rig configuration")
    ap.add_argument("--quick", action="store_true", help="40 poses, one rig, for a smoke test")
    ap.add_argument("--elev", type=float, nargs="*", default=None,
                    help="elevations to sweep (both cameras); default 35 45 55")
    ap.add_argument("--mixed", action="store_true",
                    help="also run the mixed-hemisphere rig at each elevation")
    ap.add_argument("--occluders", action="store_true",
                    help="also run with the takeoff stand present")
    ap.add_argument("--width", type=int, default=500)
    ap.add_argument("--height", type=int, default=375)
    ap.add_argument("--range-mm", type=float, default=250.0)
    ap.add_argument("--loss", default="linear", choices=["soft_l1", "linear", "huber", "cauchy"])
    ap.add_argument("--radius-mm", type=float, default=RADIUS_MM)
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    n = 40 if args.quick else args.poses
    elevs = [45.0] if args.quick else (args.elev if args.elev else [35.0, 45.0, 55.0])
    scale = args.width / CALIB_WIDTH

    configs = []
    for e in elevs:
        configs.append(((e, e), (0.0, 90.0), ()))
        if args.mixed:
            configs.append(((e, -e), (0.0, 90.0), ()))
    if args.occluders:
        configs.append(((elevs[0], elevs[0]), (0.0, 90.0), (rs.takeoff_stand(),)))
        configs.append(((elevs[0], -elevs[0]), (0.0, 90.0), (rs.takeoff_stand(),)))

    print(f"stereo sweep: {len(configs)} configuration(s) x {n} poses at "
          f"{args.width}x{args.height} (K scaled {scale:.4f}), loss={args.loss}")

    all_rows, summaries, renderer = [], [], None
    for elev, azim, occ in configs:
        rig = StereoRig.from_spherical(
            elev_deg=elev, azim_deg=azim, range_mm=args.range_mm).scaled(scale)
        label = (f"elev {elev[0]:+.0f}/{elev[1]:+.0f}  sep {rig.axis_separation_deg():.0f}deg"
                 f"  tilt seen {rig.tilt_seen_deg()[0]:.0f}deg"
                 + ("  +stand" if occ else ""))
        print(f"\n  rendering [{label}] ...", flush=True)
        if renderer is None:
            renderer = rs.StereoRenderer(rig, args.width, args.height)
        rows = run(renderer, rig, n, args.seed, occluders=occ,
                   loss=args.loss, radius_mm=args.radius_mm)
        all_rows.extend(rows)
        s = summarise(rows, label)
        if s:
            summaries.append(s)

    if renderer is not None:
        renderer.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        fh.write(f"# generated, {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        fh.write(f"# radius_mm, {args.radius_mm}\n")
        fh.write(f"# mesh_rim_radius_mm, {rendermod.RIM_RADIUS_MM}\n")
        fh.write(f"# resolution, {args.width}x{args.height}\n")
        fh.write(f"# loss, {args.loss}\n")
        fh.write(f"# target_axis_mm, {TARGET_AXIS_MM}\n")
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    if len(summaries) > 1:
        print(f"\n  {'configuration':<44} {'worst axis':>11} {'|pos|':>9} {'normal':>9}")
        for s in sorted(summaries, key=lambda s: s["worst_axis_median"]):
            print(f"  {s['label']:<44} {s['worst_axis_median']:9.4f}mm "
                  f"{s.get('stereo_pos_mm', float('nan')):8.4f}mm "
                  f"{s.get('stereo_normal_deg', float('nan')):8.3f}deg")

    print(f"\nwrote {out}  ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
