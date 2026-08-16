"""Measure the real system against the Cramer-Rao floors in `pose/bounds.py`.

`bounds.py` says what is possible, `sweep.py` says what happens; the gap between
them is what says *what to work on next*.

Every rendered frame has an analytic ground-truth ellipse, so each segmented hull
point has a signed distance to where the rim actually projects. Decomposing that
separates failure modes a single residual fuses together: ``mean(d)`` is radial
**bias** (silhouette fatter than the rim -- maps into depth, does not shrink with
resolution or exposure), ``1.4826*MAD(d)`` is the Gaussian-core **scatter** a CRLB
can consume, ``std/MAD`` is **contamination**, and the along-contour correlation
length deflates the honest point count.

The pose error is then placed against four levels, which answer four different
questions and must not be conflated -- conflating them is how a "bound" ends up
being beaten by the thing it bounds:

    photon      sigma_r from the sensor's own edge CRLB, one sample per pixel of
                perimeter. The absolute limit.
    quantised   sigma_r = 1/sqrt(12), same sampling: the limit for a
                threshold-and-contour method with no sub-pixel refinement.
    hulled      same, but only the points the convex hull keeps. The gap from
                `quantised` is the price of hulling, which is a design choice.
    noise-equiv the MEASURED scatter at the MEASURED effective point count. NOT a
                bound -- a prediction of what iid Gaussian error of that size
                would give. The estimator beats it, which is the finding: real
                boundary errors are correlated, and their common mode is what the
                radius calibration absorbs.

Observed error far above the bounds means the residual is not noise, and every
lever that acts on noise is inert.

Run: uv run python controller/pose/validation/limits.py [--poses 200] [--quick]
Writes: results/pose_validation/limits.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import bounds  # noqa: E402
import conic  # noqa: E402
import estimator as est  # noqa: E402
import segment  # noqa: E402
import render as R  # noqa: E402

OUT = HERE.parents[2] / "results" / "pose_validation" / "limits.json"

#: Edge-localisation CRLB for the render's own exposure model: a white rotor on
#: a near-black ground is ~200 counts of contrast, `conditions._exposure` draws
#: read noise around 5 counts, and the mesh edge lands on roughly a 1 px PSF.
#: This is the sensor's floor, and it is two orders of magnitude below anything
#: the pipeline actually achieves -- which is the point.
PHOTON_SIGMA_PX = bounds.edge_crlb_discrete(200.0, 5.0, 1.0)


# ---------------------------------------------------------------------------
# boundary statistics
# ---------------------------------------------------------------------------


def signed_radial_distance(pts, ellipse):
    """Signed distance from each point to the ellipse, positive outward, in px.

    Uses the Sampson (first-order geometric) distance, which is what the rest of
    the package uses for the same job, with the sign taken from the algebraic
    form so "outward" means "outside the ellipse". Exact geometric distance would
    need a quartic root per point and changes nothing at these magnitudes.
    """
    c = conic.conic_from_ellipse(ellipse)
    h = np.column_stack([np.asarray(pts, dtype=np.float64).reshape(-1, 2),
                         np.ones(len(pts))])
    alg = np.einsum("ni,ij,nj->n", h, c, h)
    grad = 2.0 * (c @ h.T).T[:, :2]
    gn = np.linalg.norm(grad, axis=1)
    return np.where(gn > 1e-12, alg / np.maximum(gn, 1e-12), 0.0)


def correlation_length_px(pts, resid):
    """Along-contour correlation length of the boundary residual, in pixels.

    Contour points arrive in order, so the residual is a 1-D signal indexed by
    arc length. The correlation length is taken as the lag at which the
    normalised autocorrelation first falls below 1/e -- a robust definition that
    does not assume the correlation is Gaussian, exponential, or anything else.

    This is the number that turns a raw point count into an honest one. A hull
    with 400 vertices whose residual decorrelates only after 6 px is carrying
    about 65 independent measurements, not 400, and the CRLB differs by 2.5x.
    """
    r = np.asarray(resid, dtype=np.float64)
    r = r - r.mean()
    if len(r) < 8 or r.std() < 1e-12:
        return 1.0
    step = np.linalg.norm(np.diff(np.asarray(pts, dtype=np.float64).reshape(-1, 2),
                                  axis=0), axis=1)
    spacing = float(np.median(step)) if len(step) else 1.0

    n = len(r)
    ac = np.correlate(r, r, mode="full")[n - 1:]
    ac /= ac[0]
    below = np.nonzero(ac < math.exp(-1.0))[0]
    lag = float(below[0]) if len(below) else float(n)
    return max(1.0, lag * spacing)


# ---------------------------------------------------------------------------
# per-frame comparison
# ---------------------------------------------------------------------------


def analyse(sample, seg, pose, K, radius_mm):
    """One frame: boundary decomposition, CRLB, observed error.

    Two methodological points, each worth more than the effect being measured:

    **The CRLB is a local bound**, describing the curvature of the likelihood at
    a maximum -- and this likelihood has two equal maxima
    (`bounds.ambiguity_is_exact`). A frame where the wrong branch was picked is
    not an imprecise estimator, it is one answering a question the data cannot
    answer, so pose error is scored against the branch nearest truth and branch
    failures are counted separately.

    **std(d) is not a Gaussian sigma.** The departures are one-sided and
    heavy-tailed, so std is set by a few excursions; feeding it to a Gaussian
    CRLB inflates the floor until the observed error sits *below* it. The floor
    uses 1.4826*MAD, with std carried alongside -- the gap between them is the
    contamination.
    """
    gt = sample.ellipse_gt
    pts = np.asarray(seg.contour, dtype=np.float64).reshape(-1, 2)
    d = signed_radial_distance(pts, gt)

    bias = float(np.mean(d))
    scatter = float(np.std(d))
    robust = float(1.4826 * np.median(np.abs(d - np.median(d))))
    lcorr = correlation_length_px(pts, d)

    (_, _), (ma, mi), ang = gt
    a_px, b_px = 0.5 * ma, 0.5 * mi
    perim = math.pi * (3.0 * (a_px + b_px)
                       - math.sqrt(max(0.0, (3 * a_px + b_px) * (a_px + 3 * b_px))))
    n_eff = bounds.effective_point_count(len(pts), lcorr,
                                         spacing_px=perim / max(1, len(pts)))

    row = {
        "tilt_deg": float(sample.tilt_deg),
        "azimuth_deg": float(sample.azimuth_deg),
        "z_mm": float(sample.center_mm[2]),
        "semi_major_px": a_px,
        "n_points": int(len(pts)),
        "corr_len_px": lcorr,
        "n_effective": n_eff,
        "boundary_bias_px": bias,
        "boundary_scatter_px": scatter,
        "boundary_scatter_robust_px": robust,
        "contamination_ratio": scatter / robust if robust > 1e-9 else float("nan"),
    }

    # Four reference levels, which are four different questions. Conflating them
    # is how a "bound" ends up being beaten by the thing it bounds.
    #
    #   photon  -- sigma_r from the sensor's own CRLB, one sample per pixel of
    #              perimeter, independent. THE bound: no estimator operating on
    #              this image can do better, whatever it does with the pixels.
    #   quant   -- sigma_r = 1/sqrt(12), same sampling. The bound for any method
    #              that thresholds and takes pixel-grid boundary points without
    #              sub-pixel refinement.
    #   hull    -- same sigma, but only the `n_points` the convex hull actually
    #              keeps. The gap between this and `quant` is the price of
    #              hulling: a convex hull retains only extreme vertices, so a
    #              350 px rim is fitted from ~30 points, and precision goes as
    #              1/sqrt(N). This is a *choice*, not a limit -- the hull is there
    #              to reject the blade cross -- so its cost belongs on the record.
    #   noise-  -- sigma_r from the MEASURED robust scatter at the MEASURED
    #   equiv      effective point count. Not a bound at all: a prediction of
    #              what iid Gaussian boundary error of the observed size would
    #              produce. The estimator is allowed to beat it, and does,
    #              because the real errors are correlated and the radius
    #              calibration absorbs their common mode.
    def crlb(prefix, sigma, n):
        cov, _ = bounds.pose_crlb((gt[0][0], gt[0][1]), (a_px, b_px),
                                  math.radians(ang), K, radius_mm, n,
                                  max(float(sigma), 1e-6))
        if cov is None:
            return
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        row[f"{prefix}_lateral_mm"] = float(math.hypot(sd[0], sd[1]))
        row[f"{prefix}_depth_mm"] = float(sd[2])
        row[f"{prefix}_pos_mm"] = float(np.linalg.norm(sd[:3]))
        row[f"{prefix}_angle_deg"] = float(math.degrees(np.linalg.norm(sd[3:])))

    crlb("photon", PHOTON_SIGMA_PX, perim)
    crlb("quant", bounds.QUANTISATION_SIGMA_PX, perim)
    crlb("hull", bounds.QUANTISATION_SIGMA_PX, len(pts))
    crlb("noiseq", max(robust, 1e-4), n_eff)

    # What actually happened, with the ambiguity branch chosen by an oracle.
    truth_n = np.asarray(sample.normal, dtype=np.float64)
    if pose is not None:
        cands = conic.backproject_ellipse(pose.ellipse, K, radius_mm)
        if cands:
            best = max(cands, key=lambda p: float(np.asarray(p.normal) @ truth_n))
            err = np.asarray(best.center) - np.asarray(sample.center_mm)
            row["obs_lateral_mm"] = float(math.hypot(err[0], err[1]))
            row["obs_depth_mm"] = float(abs(err[2]))
            row["obs_pos_mm"] = float(np.linalg.norm(err))
            dot = float(np.clip(np.asarray(best.normal) @ truth_n, -1.0, 1.0))
            row["obs_angle_deg"] = float(math.degrees(math.acos(dot)))
            # Did the shipped branch choice agree with the oracle?
            shipped = float(np.clip(np.asarray(pose.normal) @ truth_n, -1.0, 1.0))
            row["shipped_angle_deg"] = float(math.degrees(math.acos(shipped)))
            row["branch_wrong"] = bool(row["shipped_angle_deg"]
                                       > row["obs_angle_deg"] + 1e-6)
            row["ambiguity_margin_deg"] = float(pose.ambiguity_margin_deg)

    # The bias, converted to the depth error it alone would cause. A boundary
    # uniformly fat by `bias` px inflates the apparent semi-major axis by the
    # same amount, and z = f R / a, so dz/z = -bias/a.
    #
    # Reported against BOTH radii, because the calibrated `estimator.RADIUS_MM`
    # is larger than the true rim precisely to absorb this bias -- so the naive
    # figure double-counts a correction the pipeline already applies.
    row["bias_implied_depth_mm"] = float(sample.center_mm[2] * bias / a_px)
    residual_bias = bias - a_px * (est.RADIUS_MM - radius_mm) / radius_mm
    row["residual_bias_px"] = float(residual_bias)
    row["residual_bias_depth_mm"] = float(sample.center_mm[2] * residual_bias / a_px)
    return row


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run(n_poses=200, seed=7, width=1280, height=800, subframes=5, tiers=("core", "edge")):
    """Render both condition tiers and analyse every frame.

    Conditions (lighting, exposure, alpha, background) come from `conditions.py`
    -- the same ``core``/``edge`` vocabulary the resolution sweep and the error
    model use -- so the numbers here sit on the same axis as every other result
    in the package. The pose set is shared between tiers, so a tier difference is
    a difference in conditions and not in what was asked.

    Poses are drawn here rather than by ``conditions.sample_poses``, and the
    difference is not stylistic: that function returns **world** offsets for the
    stereo rig, with z in +-40 mm about the rig origin. Feeding those to the
    monocular `Renderer`, whose centres are in **camera** coordinates, puts the
    robot on top of the lens. It renders without complaint and produces 35-point
    hulls, 9 px boundary scatter and 47 deg of angular error -- numbers that look
    like a catastrophic estimator rather than a units mistake.

    Tilt is drawn uniformly in the cosine, matching `sample_poses`, so the
    orientation distribution is comparable.
    """
    import conditions as cond

    rng = np.random.default_rng(seed)
    cos_lo = math.cos(math.radians(70.0))
    tilt = np.degrees(np.arccos(rng.uniform(cos_lo, 1.0, n_poses)))
    az = rng.uniform(0.0, 360.0, n_poses)
    centres = np.column_stack([
        rng.uniform(-22.0, 22.0, n_poses),
        rng.uniform(-22.0, 22.0, n_poses),
        rng.uniform(170.0, 340.0, n_poses),
    ])
    shape = (height, width)
    plans = {t: getattr(cond, t)(rng, n_poses, shape, subframes) for t in tiers}

    rows = []
    with R.Renderer(width=width, height=height) as ren:
        K = ren.K
        radius = R.RIM_RADIUS_MM
        e = est.PoseEstimator(camera_matrix=K, dist_coeffs=np.zeros(5),
                              radius_mm=est.RADIUS_MM)
        total, done = n_poses * len(plans), 0
        print(f"rendering {total} frames at {width}x{height} "
              f"({n_poses} poses x {len(plans)} tiers) ...", flush=True)
        for tier, conds in plans.items():
            for i in range(n_poses):
                c = conds[i]
                s = ren.render(float(tilt[i]), float(az[i]), centres[i],
                               alpha=c.alpha, light=c.light, exposure=c.exposure,
                               background=c.background)
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{total}", flush=True)
                seg = segment.segment(s.image)
                if seg is None:
                    continue
                e.reset()
                pose = e.update(s.image, t=0.0, frame_index=i)
                row = analyse(s, seg, pose, K, radius)
                row["tier"] = tier
                rows.append(row)
    return rows


def summarise(rows):
    def col(k):
        return np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])

    out = {"n": len(rows)}
    for k in ("boundary_bias_px", "boundary_scatter_px",
              "boundary_scatter_robust_px", "contamination_ratio", "corr_len_px",
              "n_points", "n_effective",
              "photon_pos_mm", "photon_depth_mm", "photon_lateral_mm",
              "photon_angle_deg", "quant_pos_mm", "quant_depth_mm",
              "quant_lateral_mm", "quant_angle_deg",
              "hull_pos_mm", "hull_depth_mm", "hull_lateral_mm", "hull_angle_deg",
              "noiseq_pos_mm", "noiseq_depth_mm", "noiseq_lateral_mm",
              "noiseq_angle_deg", "obs_pos_mm", "obs_depth_mm",
              "obs_lateral_mm", "obs_angle_deg", "shipped_angle_deg",
              "bias_implied_depth_mm", "residual_bias_px",
              "residual_bias_depth_mm"):
        v = col(k)
        if len(v):
            out[k] = {"median": float(np.median(v)),
                      "mean": float(np.mean(v)),
                      "p95": float(np.percentile(v, 95))}

    # The headline: how many times the floor is the observed error.
    pairs = [(f"obs_{q}", f"{lvl}_{q}")
             for lvl in ("photon", "quant", "hull", "noiseq")
             for q in ("pos_mm", "depth_mm", "lateral_mm", "angle_deg")]
    for a, b in pairs:
        ratio = [r[a] / r[b] for r in rows
                 if a in r and b in r and r[b] > 0 and np.isfinite(r[a])]
        if ratio:
            out[f"ratio_{a}_over_{b}"] = {"median": float(np.median(ratio)),
                                 "p95": float(np.percentile(ratio, 95))}

    # And how well the bias alone accounts for the depth error, which is the
    # positive form of the same claim.
    pair = [(abs(r["bias_implied_depth_mm"]), r["obs_depth_mm"]) for r in rows
            if "obs_depth_mm" in r]
    if pair:
        p = np.array(pair)
        out["bias_explains_depth"] = {
            "median_predicted_mm": float(np.median(p[:, 0])),
            "median_observed_mm": float(np.median(p[:, 1])),
            "correlation": float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])
            if len(p) > 2 else float("nan"),
        }

    # Branch selection: how often the single-view ambiguity is resolved wrongly.
    # Structural, not a defect -- see `bounds.ambiguity_is_exact`.
    flags = [r["branch_wrong"] for r in rows if "branch_wrong" in r]
    if flags:
        out["branch_wrong_fraction"] = float(np.mean(flags))
        marg = [r["ambiguity_margin_deg"] for r in rows
                if r.get("branch_wrong") and "ambiguity_margin_deg" in r]
        out["branch_wrong_median_margin_deg"] = (
            float(np.median(marg)) if marg else float("nan"))

    # The photon floor, for scale: what the scatter WOULD be if the sensor were
    # the limit. Contrast and read noise taken from the render's exposure model.
    out["photon_sigma_px"] = bounds.edge_crlb_discrete(200.0, 5.0, 1.0)
    out["quantisation_sigma_px"] = bounds.QUANTISATION_SIGMA_PX
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="30 poses, for a smoke test")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    n = 30 if a.quick else a.poses

    print(f"rendering and analysing {n} poses at 1280x800 ...")
    rows = run(n, seed=a.seed)
    summary = summarise(rows)

    print("\nboundary decomposition (px)")
    for k in ("boundary_bias_px", "residual_bias_px", "boundary_scatter_px",
              "boundary_scatter_robust_px", "contamination_ratio", "corr_len_px"):
        s = summary[k]
        print(f"  {k:<28} median {s['median']:+.4f}   mean {s['mean']:+.4f}"
              f"   p95 {s['p95']:+.4f}")
    print(f"  {'photon CRLB':<28} {summary['photon_sigma_px']:.4f} px"
          f"   (pixel quantisation {summary['quantisation_sigma_px']:.4f} px)")
    print(f"  robust scatter is "
          f"{summary['boundary_scatter_robust_px']['median'] / summary['photon_sigma_px']:.0f}x"
          f" the photon floor and "
          f"{summary['boundary_scatter_robust_px']['median'] / summary['quantisation_sigma_px']:.1f}x"
          f" the pixel-quantisation floor")

    print("\npoints: raw vs effective")
    print(f"  n_points   median {summary['n_points']['median']:.0f}")
    print(f"  n_effective median {summary['n_effective']['median']:.0f}"
          f"  (correlation length {summary['corr_len_px']['median']:.2f} px)")

    print("\nthree reference levels vs observed  (medians)")
    print(f"  {'quantity':<14}{'photon':<11}{'quantised':<11}{'hulled':<11}"
          f"{'noise-eq':<11}{'observed':<11}{'obs/photon':<12}{'obs/hull'}")
    for nm, q in (("position mm", "pos_mm"), ("depth mm", "depth_mm"),
                  ("lateral mm", "lateral_mm"), ("angle deg", "angle_deg")):
        keys = [f"{lvl}_{q}" for lvl in ("photon", "quant", "hull", "noiseq")] + [f"obs_{q}"]
        if not all(k in summary for k in keys):
            continue
        vals = [summary[k]["median"] for k in keys]
        rp = summary.get(f"ratio_obs_{q}_over_photon_{q}", {}).get("median", float("nan"))
        rq = summary.get(f"ratio_obs_{q}_over_hull_{q}", {}).get("median", float("nan"))
        print(f"  {nm:<14}" + "".join(f"{v:<11.4f}" for v in vals[:4])
              + f"{vals[4]:<11.4f}{rp:<12.0f}{rq:.0f}")
    print("  (photon/quantised/hulled are BOUNDS -- nothing may beat them."
          " noise-equivalent is a\n   prediction, and the estimator beating it is"
          " itself a result: see the module docstring)")
    if "hull_pos_mm" in summary and "quant_pos_mm" in summary:
        print(f"  hulling alone costs "
              f"{summary['hull_pos_mm']['median'] / summary['quant_pos_mm']['median']:.1f}x "
              f"(it keeps {summary['n_points']['median']:.0f} of ~"
              f"{summary['n_effective']['median'] * 0 + 0:.0f}".replace(' of ~0','')
              + " boundary points)")

    print("\n(pose error is against the ORACLE ambiguity branch; the local CRLB "
          "does not\n describe a bimodal likelihood -- branch failures are "
          "counted separately)")
    if "branch_wrong_fraction" in summary:
        print(f"  branch chosen wrongly on {summary['branch_wrong_fraction'] * 100:.1f}% "
              f"of frames (median margin "
              f"{summary['branch_wrong_median_margin_deg']:.1f} deg)")
        print("  NB: these are independent poses with the estimator reset each"
              " frame, so this is\n      the PRIOR-FREE rate -- the worst case."
              " In a video stream the live loop carries a\n      temporal prior"
              " and the rate is far lower; this number is the size of the"
              " problem\n      that prior has to solve, not the shipped failure"
              " rate.")
        print(f"  shipped angular error median "
              f"{summary['shipped_angle_deg']['median']:.3f} deg vs oracle "
              f"{summary['obs_angle_deg']['median']:.3f} deg")

    b = summary.get("bias_explains_depth")
    if b:
        print("\nresidual bias (after the radius calibration absorbs the mean), "
              "propagated to depth")
        print(f"  predicted {summary['residual_bias_depth_mm']['median']:+.3f} mm "
              f"vs observed depth error {b['median_observed_mm']:.3f} mm"
              f"  (r = {b['correlation']:+.3f})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
