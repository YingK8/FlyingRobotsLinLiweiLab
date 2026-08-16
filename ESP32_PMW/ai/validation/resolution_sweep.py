"""Accuracy against sensor resolution, for the iteration journal.

The target is **±1° and ±0.5 mm on 100% of reported frames**, at every resolution
the camera can run.  Not the median, not the 95th percentile: the worst case.
That changes what the estimator has to do — it is no longer enough to be
accurate on average, it has to *know when it cannot be* and say so.

**One render, many resolutions.**  pyglet allows one GL context per process, so
a renderer cannot be rebuilt per resolution.  `render.py` documents the way
round this and it is the one used here: render once at the largest size, then
resize.  Resizing is not identical to rendering natively — the antialiasing
differs — but it is identical *across* the sweep, which is what a comparison
needs.

Aspect ratios differ (1280×800 is 1.6, 640×480 is 1.333), so a plain resize
would distort. Each target is produced by scaling on **width** and then
centre-cropping or padding the height, with the principal point moved to match.
That keeps pixels square and the projection exact.

Run: uv run python controller/pose/validation/resolution_sweep.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import conditions as cond  # noqa: E402
import render_stereo as rs  # noqa: E402
import stereo as stereomod  # noqa: E402
import uncertainty  # noqa: E402
from rig import StereoRig  # noqa: E402

RESULTS = HERE.parents[2] / "results" / "pose_validation"

# The camera's actual modes, with the frame rate each allows. Accuracy is only
# half the trade -- 160x120 is useless if it cannot see the robot, and 1280x800
# is useless if the loop needs 420 Hz.
MODES = [
    (1280, 800, 120),
    (1280, 720, 120),
    (1024, 768, 120),
    (800, 600, 120),
    (640, 480, 210),
    (640, 400, 210),
    (320, 240, 420),
    (160, 120, 640),
]

RENDER_W, RENDER_H = 1280, 800
CALIB_WIDTH = 1024.0

TARGET_POS_MM = 0.5
TARGET_ANGLE_DEG = 1.0

ELEV = (45.0, 45.0)
AZIM = (0.0, 90.0)
RANGE_MM = 250.0


def _resample(image, K, width, height):
    """Scale on width, then centre-crop or pad the height. Returns (image, K).

    Scaling both axes independently to hit an arbitrary aspect would make pixels
    non-square, which the pinhole model this whole package rests on does not
    allow. Width-scale-then-crop keeps fx == fy and moves only the principal
    point, which is exactly what physically cropping a sensor does.
    """
    s = width / image.shape[1]
    scaled = cv2.resize(image, (width, int(round(image.shape[0] * s))),
                        interpolation=cv2.INTER_AREA)
    K = K.copy()
    K[:2, :] *= s

    h = scaled.shape[0]
    if h > height:                       # crop
        top = (h - height) // 2
        scaled = scaled[top:top + height]
        K[1, 2] -= top
    elif h < height:                     # pad
        pad = (height - h) // 2
        out = np.zeros((height, width), dtype=scaled.dtype)
        out[pad:pad + h] = scaled
        scaled, K[1, 2] = out, K[1, 2] + pad
    return scaled, K


def render_frames(n_poses, seed, subframes=5):
    """Render both tiers once, at the largest resolution, and keep the frames.

    The pose set is **shared between tiers** on purpose: any difference between
    core and edge results is then a difference in conditions, not in what was
    asked of the estimator.
    """
    rng = np.random.default_rng(seed)
    tilt, az, centres = cond.sample_poses(rng, n_poses)
    shape = (RENDER_H, RENDER_W)
    tiers = {"core": cond.core(rng, n_poses, shape, subframes),
             "edge": cond.edge(rng, n_poses, shape, subframes)}

    base_rig = StereoRig.from_spherical(elev_deg=ELEV, azim_deg=AZIM, range_mm=RANGE_MM)
    render_rig = base_rig.scaled(RENDER_W / CALIB_WIDTH)

    frames = {"core": [], "edge": []}
    total = n_poses * len(tiers)
    done = 0
    print(f"rendering {total} pairs at {RENDER_W}x{RENDER_H} "
          f"({n_poses} poses x {len(tiers)} tiers) ...", flush=True)
    with rs.StereoRenderer(render_rig, RENDER_W, RENDER_H) as r:
        for tier, conds in tiers.items():
            for i in range(n_poses):
                c = conds[i]
                s = r.render_pair(float(tilt[i]), float(az[i]), centres[i],
                                  alpha=c.alpha, light=c.light,
                                  exposure=c.exposure, background=c.background)
                frames[tier].append(([v.image for v in s.views], s.center_world,
                                     s.normal_world, float(tilt[i]), c.label))
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{total}", flush=True)
    return frames, render_rig


def _bootstrap(rows, n_boot=400, seed=12345):
    """Sampling spread of each reported metric, by resampling the frames.

    This is the tolerance the journal uses to separate a real change from noise,
    and it is measured rather than chosen. It is also exact for this harness
    rather than an approximation: the renderer and the estimator are both
    deterministic given a seed, so re-running with the same seed reproduces
    every number bit for bit, and **all** run-to-run variation comes from which
    poses and conditions were drawn. Resampling the frames reproduces precisely
    that, with no extra rendering.

    Returned as the half-width of the 5th-95th percentile band, so a change
    smaller than this is indistinguishable from having drawn a different sample.
    """
    if len(rows) < 5:
        return {}
    rng = np.random.default_rng(seed)
    pos = np.array([r["pos"] for r in rows])
    ang = np.array([r["ang"] for r in rows])
    idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
    out = {}
    for name, v in (("pos", pos), ("ang", ang)):
        draws = v[idx]
        out[f"{name}_mean_tol"] = float(
            0.5 * np.subtract(*np.percentile(draws.mean(axis=1), [95, 5])))
        out[f"{name}_max_tol"] = float(
            0.5 * np.subtract(*np.percentile(draws.max(axis=1), [95, 5])))
    return out


def score(frames, render_rig, estimator_kw=None, keep_features=False):
    """Score one tier's frames at every camera mode.

    ``keep_features`` also returns the per-frame observables paired with the
    measured errors, which is the training data `fit_error_model.py` needs. Off
    by default because it is a lot of rows and only the fitting path wants them.
    """
    samples = []

    out = []
    for w, h, fps in MODES:
        rig = StereoRig(
            cameras=tuple(
                type(c)(K=_resample(np.zeros((RENDER_H, RENDER_W), np.uint8),
                                    c.K, w, h)[1],
                        dist=c.dist, T_world_cam=c.T_world_cam, name=c.name)
                for c in render_rig.cameras),
            meta=dict(render_rig.meta, resolution=f"{w}x{h}"),
        )
        est = stereomod.StereoPoseEstimator(rig, **(estimator_kw or {}))
        rows = []
        t0 = time.perf_counter()
        for k, (images, ctr, nrm, tilt_deg, label) in enumerate(frames):
            small = [_resample(im, render_rig.cameras[j].K, w, h)[0]
                     for j, im in enumerate(images)]
            pose = est.update(small, t=float(k))
            if pose is None:
                continue
            d = pose.xyz_mm - ctr
            rows.append({
                "dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]),
                "pos": float(np.linalg.norm(d)),
                "ang": stereomod.line_angle_deg(pose.normal, nrm),
                "tilt": tilt_deg,
                "seen": float(min(rig.tilt_seen_deg(nrm))),
                "label": label,
            })
            if keep_features:
                feat = uncertainty.features(pose, stereomod.RADIUS_MM)
                if feat is not None:
                    samples.append({
                        "features": feat,
                        "pos_err": rows[-1]["pos"],
                        "ang_err": rows[-1]["ang"],
                        "mode": f"{w}x{h}",
                    })
        dt = (time.perf_counter() - t0) / max(len(frames), 1) * 1e3

        det = len(rows) / len(frames)
        if rows:
            pos = np.array([r["pos"] for r in rows])
            ang = np.array([r["ang"] for r in rows])
            axes = np.abs(np.array([[r["dx"], r["dy"], r["dz"]] for r in rows]))
            in_spec = float(np.mean((axes.max(axis=1) <= TARGET_POS_MM)
                                    & (ang <= TARGET_ANGLE_DEG)))
            rec = {
                "width": w, "height": h, "fps": fps, "n": len(rows),
                "detect_rate": det,
                "pos_mean": float(pos.mean()), "pos_p95": float(np.percentile(pos, 95)),
                "pos_max": float(pos.max()),
                "axis_max": float(axes.max()),
                "ang_mean": float(ang.mean()), "ang_p95": float(np.percentile(ang, 95)),
                "ang_max": float(ang.max()),
                "in_spec": in_spec,
                "ms_per_frame": dt,
                **_bootstrap(rows),
            }
        else:
            rec = {"width": w, "height": h, "fps": fps, "n": 0, "detect_rate": 0.0,
                   "in_spec": 0.0, "ms_per_frame": dt}
        out.append(rec)
        print(f"    {w}x{h} @{fps}: n={len(rows)}/{len(frames)} "
              f"({det:.0%})  pos avg {rec.get('pos_mean', float('nan')):.3f} "
              f"max {rec.get('pos_max', float('nan')):.3f} mm  "
              f"ang avg {rec.get('ang_mean', float('nan')):.3f} "
              f"max {rec.get('ang_max', float('nan')):.3f} deg  "
              f"IN SPEC {rec['in_spec']:.1%}", flush=True)
    return (out, samples) if keep_features else out


def chart(tiers, path, title=""):
    """Angle residual (left axis) and position residual (right) against mode.

    Both tiers on one pair of axes, core solid and edge faint, with the two
    targets drawn as dotted lines. Mean is plotted heavy and worst-case light,
    because the target is on the worst case and a mean-only chart would look
    like success while the specification is being missed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    any_rec = next(iter(tiers.values()))
    x = np.arange(len(any_rec))
    labels = [f"{r['width']}x{r['height']}\n{r['fps']} fps" for r in any_rec]

    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=150, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax2 = ax.twinx()
    handles = []
    for tier, records in tiers.items():
        solid = tier == "core"
        alpha = 1.0 if solid else 0.45
        ang = [r.get("ang_mean", np.nan) for r in records]
        pos = [r.get("pos_mean", np.nan) for r in records]
        angm = [r.get("ang_max", np.nan) for r in records]
        posm = [r.get("pos_max", np.nan) for r in records]
        h1, = ax.plot(x, ang, "o-", color="#2A8FC4", lw=2.2 if solid else 1.4,
                      ms=5 if solid else 3.5, alpha=alpha, label=f"angle mean ({tier})")
        h2, = ax.plot(x, angm, "o:", color="#2A8FC4", lw=1.2, ms=3, alpha=alpha * 0.75,
                      label=f"angle worst ({tier})")
        h3, = ax2.plot(x, pos, "s-", color="#CE7412", lw=2.2 if solid else 1.4,
                       ms=5 if solid else 3.5, alpha=alpha,
                       label=f"position mean ({tier})")
        h4, = ax2.plot(x, posm, "s:", color="#CE7412", lw=1.2, ms=3, alpha=alpha * 0.75,
                       label=f"position worst ({tier})")
        handles += [h1, h2, h3, h4]

    ax.axhline(TARGET_ANGLE_DEG, color="#2A8FC4", lw=1.4, ls="--", alpha=0.9)
    ax2.axhline(TARGET_POS_MM, color="#CE7412", lw=1.4, ls="--", alpha=0.9)
    ax.text(0.01, TARGET_ANGLE_DEG, f" target {TARGET_ANGLE_DEG:g}\u00b0",
            color="#2A8FC4", fontsize=8, va="bottom", transform=ax.get_yaxis_transform())
    ax2.text(0.99, TARGET_POS_MM, f"target {TARGET_POS_MM:g} mm ", color="#CE7412",
             fontsize=8, va="bottom", ha="right",
             transform=ax2.get_yaxis_transform())

    ax.set_ylabel("angle residual (deg)", color="#2A8FC4")
    ax2.set_ylabel("position residual (mm)", color="#CE7412")
    ax.tick_params(axis="y", colors="#2A8FC4")
    ax2.tick_params(axis="y", colors="#CE7412")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("sensor mode (resolution and the frame rate it allows)")
    # Log scales, because the residuals span three and a half decades: the
    # useful modes sit near 0.2 mm and the ones that fail outright reach several
    # hundred, and on a linear axis the failures flatten everything worth reading
    # into the bottom pixel row.
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_ylim(bottom=max(1e-2, min(
        [v for r in tiers.values() for k in ("ang_mean", "ang_max")
         for v in [r_i.get(k) for r_i in r] if v] + [1.0]) * 0.5))
    ax2.set_ylim(bottom=max(1e-3, min(
        [v for r in tiers.values() for k in ("pos_mean", "pos_max")
         for v in [r_i.get(k) for r_i in r] if v] + [1.0]) * 0.5))
    ax.grid(True, which="both", color="#D6DCE4", lw=0.6, alpha=0.7)
    if title:
        ax.set_title(title, fontsize=11)
    ax.legend(handles=handles, fontsize=7.5, loc="upper center", ncol=4, frameon=True)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="#FFFFFF")
    plt.close(fig)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--tag", default="run", help="name for the output files")
    ap.add_argument("--save-samples", action="store_true",
                    help="also dump per-frame observables, so a frame that "
                         "defeats the gate can be characterised afterwards "
                         "rather than only counted")
    ap.add_argument("--gate", action="store_true",
                    help="apply the predicted-error gate, so the estimator "
                         "declines frames it cannot certify to specification")
    args = ap.parse_args(argv)

    # The gate is opt-in rather than always-on, because the ungated numbers are
    # what show whether the *estimator* improved. With the gate active a rise in
    # in-spec % can come either from better estimation or from stricter
    # refusal, and the two need to stay separable across iterations.
    kw = {}
    if args.gate:
        kw = {"target_pos_mm": TARGET_POS_MM, "target_angle_deg": TARGET_ANGLE_DEG}

    frames, render_rig = render_frames(args.poses, args.seed)
    tiers = {}
    samples = []
    for tier in ("core", "edge"):
        print(f"\n  scoring {tier} tier:", flush=True)
        if args.save_samples:
            tiers[tier], s_t = score(frames[tier], render_rig, estimator_kw=kw,
                                     keep_features=True)
            for row in s_t:
                row["tier"] = tier
            samples += s_t
        else:
            tiers[tier] = score(frames[tier], render_rig, estimator_kw=kw)
    if args.save_samples:
        sp = RESULTS / f"samples_{args.tag}.json"
        sp.write_text(json.dumps(samples))
        print(f"wrote {sp}  ({len(samples)} frames)")

    out = RESULTS / f"resolution_{args.tag}.json"
    out.write_text(json.dumps({"tiers": tiers, "poses": args.poses,
                               "seed": args.seed, "tag": args.tag,
                               "gated": bool(args.gate),
                               "target_pos_mm": TARGET_POS_MM,
                               "target_angle_deg": TARGET_ANGLE_DEG}, indent=2))
    chart(tiers, RESULTS / f"resolution_{args.tag}.png",
          title=f"accuracy vs sensor mode - {args.tag} "
                f"({args.poses} poses/tier, seed {args.seed})")
    print(f"\nwrote {out}")
    ok = sum(1 for r in tiers["core"] if r.get("in_spec", 0) >= 1.0)
    print(f"CORE modes at 100% in spec: {ok}/{len(tiers['core'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
