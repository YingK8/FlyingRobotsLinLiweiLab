"""
Render a randomised pose dataset once, so tuning does not re-render.

    uv run python controller/pose/validation/make_dataset.py

Rendering costs ~30 ms a frame; fitting a correction costs microseconds.  Doing
both in one loop means every candidate correction pays the render cost again, so
instead this dumps the raw per-frame measurements -- the fitted ellipse and the
ground-truth pose that produced it -- to an npz.  `tune.py` then fits against
that in milliseconds.

Split into `train` and `test` by construction, from independent random draws.
Corrections are fitted on `train` only and reported on `test` only.  A
correction validated on the data that produced it is just a restatement of that
data, and the whole point here is to produce numbers that mean something.
"""

from __future__ import annotations

import argparse
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
import segment as segmod  # noqa: E402

DEFAULT_OUT = HERE.parents[2] / "results" / "pose_validation" / "dataset.npz"


def sample_poses(rng, n, tilt_max=70.0, z_range=(140.0, 360.0), lateral_mm=28.0):
    """
    Random poses spanning the working volume.

        Tilt is drawn uniformly in the *cosine* rather than in the angle, so samples
        are spread evenly over the sphere of orientations instead of piling up near
        face-on where the geometry is degenerate anyway.
    """

    cos_lo = np.cos(np.radians(tilt_max))
    tilt = np.degrees(np.arccos(rng.uniform(cos_lo, 1.0, n)))
    az = rng.uniform(0.0, 360.0, n)
    z = rng.uniform(*z_range, n)
    # Keep lateral offset proportional to range so the robot stays in frame.
    scale = z / z_range[1]
    x = rng.uniform(-lateral_mm, lateral_mm, n) * scale
    y = rng.uniform(-lateral_mm, lateral_mm, n) * scale
    return tilt, az, np.column_stack([x, y, z])


def exposures(rng, n, realistic=True, subframes=7):
    """
    Per-sample sensor conditions for a high-frame-rate camera.

        Exposure has to fit inside the frame period, so a 420 fps camera is capped
        near 2.4 ms and in practice runs shorter; 1/4000 to 1/1000 s is the realistic
        band.  Read noise is drawn *anti-correlated* with exposure, because that is
        the real trade -- a shorter exposure collects fewer photons and reads out
        noisier.  Sampling them independently would let the fit see a bright, short,
        quiet frame that no sensor can actually produce.

        Spin is drawn across the 310-350 Hz drive band from
        `docs/pose_localization_project_context.md`.
    """

    if not realistic:
        return [None] * n

    out = []
    for i in range(n):
        exp_s = float(rng.uniform(1 / 4000, 1 / 1000))
        # Short exposure -> less light -> more noise. Scaled so 1/4000 s gives
        # about sigma 18 and 1/1000 s about sigma 5.
        sigma = float(np.clip(4.5 * (1 / 1000) / exp_s, 3.0, 25.0))
        out.append(
            rendermod.Exposure(
                exposure_s=exp_s,
                subframes=subframes,
                spin_hz=float(rng.uniform(310.0, 350.0)),
                velocity_mm_s=tuple(rng.uniform(-80.0, 80.0, 3)),
                tilt_rate_deg_s=float(rng.uniform(-60.0, 60.0)),
                sigma=sigma,
                seed=i,
            )
        )
    return out


def lighting(rng, n):
    """
    Adequately-lit rigs only.

        Lighting is a separate, already-characterised axis: the sweep showed ambient
        below ~0.25 is what breaks segmentation, and no geometric correction can
        repair an unlit rim.  Mixing those failures in here would just add outliers
        that swamp the signal being fitted.
    """

    rigs = []
    for i in range(n):
        amb = rng.uniform(0.25, 0.6)
        if rng.random() < 0.5:
            rigs.append(
                rendermod.LightRig(
                    dome=((rng.uniform(30, 80), rng.uniform(0, 360)),),
                    ambient=amb,
                    intensity=rng.uniform(6, 20),
                )
            )
        else:
            rigs.append(
                rendermod.LightRig(
                    lateral_deg=(rng.uniform(0, 360),),
                    ambient=amb,
                    intensity=rng.uniform(6, 20),
                )
            )
    return rigs


def build(
    n_train,
    n_test,
    width,
    height,
    seed=20260806,
    realistic=True,
    subframes=7,
    appearance="bright",
):
    rng = np.random.default_rng(seed)
    n = n_train + n_test

    tilt, az, centres = sample_poses(rng, n)
    rigs = lighting(rng, n)
    exps = exposures(rng, n, realistic=realistic, subframes=subframes)
    alphas = rng.choice([0.7, 0.8, 0.9, 1.0], n)
    # The backdrop sweep runs *away from* whatever the robot is, since the whole
    # question a backdrop level asks is how close it gets to the robot's own
    # level before segmentation gives up.
    if appearance == "dark":
        body = rendermod.BLACK_BODY
        # Not down to 0.7: `segment.DARK_THRESH` needs the ground above ~145
        # counts, and a dim backdrop is a lighting fault rather than a condition
        # the estimator should be scored against.
        bgs = rng.choice([1.0, 0.95, 0.9, 0.85], n)
    else:
        body = None
        bgs = rng.choice([0.0, 0.1, 0.2, 0.3], n)

    rows = {
        k: []
        for k in (
            "tilt_deg",
            "az_deg",
            "cx_mm",
            "cy_mm",
            "cz_mm",
            "nx",
            "ny",
            "nz",
            "e_cx",
            "e_cy",
            "major",
            "minor",
            "e_deg",
            "area_px",
            "fit_rms_px",
            "alpha",
            "bg",
            "ambient",
            "seg_ms",
            "split",
            "iou",
            "exposure_s",
            "sigma",
            "spin_hz",
        )
    }

    t0 = time.monotonic()
    with rendermod.Renderer(width, height) as r:
        K = r.K
        for i in range(n):
            s = r.render(
                float(tilt[i]),
                float(az[i]),
                centres[i],
                alpha=float(alphas[i]),
                light=rigs[i],
                bg_level=float(bgs[i]),
                exposure=exps[i],
                body_colour=body,
            )
            seg = segmod.segment(s.image, appearance=appearance)
            if seg is None:
                continue

            inter = np.logical_and(seg.mask > 0, s.mask).sum()
            union = np.logical_or(seg.mask > 0, s.mask).sum()
            (ecx, ecy), (major, minor), edeg = seg.ellipse

            for k, v in (
                ("tilt_deg", s.tilt_deg),
                ("az_deg", s.azimuth_deg),
                ("cx_mm", s.center_mm[0]),
                ("cy_mm", s.center_mm[1]),
                ("cz_mm", s.center_mm[2]),
                ("nx", s.normal[0]),
                ("ny", s.normal[1]),
                ("nz", s.normal[2]),
                ("e_cx", ecx),
                ("e_cy", ecy),
                ("major", major),
                ("minor", minor),
                ("e_deg", edeg),
                ("area_px", seg.area_px),
                ("fit_rms_px", seg.fit_rms_px),
                ("alpha", alphas[i]),
                ("bg", bgs[i]),
                ("ambient", rigs[i].ambient),
                ("seg_ms", seg.t_ms),
                ("split", 0 if i < n_train else 1),
                ("exposure_s", exps[i].exposure_s if exps[i] else 0.0),
                ("sigma", exps[i].sigma if exps[i] else 0.0),
                ("spin_hz", exps[i].spin_hz if exps[i] else 0.0),
                ("iou", inter / union if union else 0.0),
            ):
                rows[k].append(float(v))

            if (i + 1) % 200 == 0:
                el = time.monotonic() - t0
                print(
                    f"  {i+1}/{n}  {el:5.1f}s elapsed, {el/(i+1)*(n-i-1):5.1f}s left",
                    flush=True,
                )

    data = {k: np.asarray(v) for k, v in rows.items()}
    data["K"] = K
    data["rim_radius_mm"] = np.array(rendermod.RIM_RADIUS_MM)
    data["resolution"] = np.array([width, height])
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", type=int, default=700)
    ap.add_argument("--test", type=int, default=700)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--clean",
        action="store_true",
        help="no noise or motion blur (the original idealised renders)",
    )
    ap.add_argument(
        "--subframes",
        type=int,
        default=7,
        help="sub-frames averaged across the exposure; cost scales with this",
    )
    ap.add_argument(
        "--appearance",
        default="bright",
        choices=("bright", "dark"),
        help="rig appearance; 'dark' renders a black body on a light "
        "ground (see segment.score_channel)",
    )
    args = ap.parse_args(argv)

    print(
        f"rendering {args.train} train + {args.test} test at {args.width}x{args.height}"
        f" ({'clean' if args.clean else f'noise+blur, {args.subframes} subframes'})"
    )
    data = build(
        args.train,
        args.test,
        args.width,
        args.height,
        realistic=not args.clean,
        subframes=args.subframes,
        appearance=args.appearance,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    n_tr = int((data["split"] == 0).sum())
    n_te = int((data["split"] == 1).sum())
    print(
        f"wrote {out}  ({n_tr} train, {n_te} test, "
        f"{n_tr + n_te}/{args.train + args.test} detected)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
