"""
Monte-Carlo verification of every bound in `bounds.py`.

Run: uv run python ai/tests/test_bounds.py

A derivation that is never checked against a simulation is a conjecture. Each
test here generates data from the *same* model the bound assumes, runs the real
estimator on it, and compares the empirical covariance against the predicted
one. The figure of merit is the **statistical efficiency**

    eta = CRLB variance / observed variance

which must satisfy ``eta <= 1`` for any unbiased estimator. ``eta`` close to 1
says the estimator extracts essentially all the information in the pixels and no
amount of algorithm work will help; ``eta`` well below 1 says there is headroom.
An ``eta`` meaningfully above 1 is not a triumph, it is a bug -- either the bound
is wrong or the estimator is biased -- so it is asserted against in both
directions.

Dependency-free, self-contained, and seeded, matching `test_conic.py`.
"""

from __future__ import annotations

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

import bounds  # noqa: E402
import conic  # noqa: E402
import segment  # noqa: E402

K = np.array(
    [[1408.78, 0.0, 497.55], [0.0, 1407.69, 355.70], [0.0, 0.0, 1.0]], dtype=np.float64
)
RADIUS_MM = 10.204

# Monte-Carlo sizing. 4000 trials puts the standard error on an estimated
# standard deviation at 1/sqrt(2N) = 1.1%, so an efficiency computed from it is
# good to about 2% -- which is why the tolerances below are 10-25% rather than
# tighter. Loosening them further would stop the test from detecting anything.
TRIALS = 4000
RNG = np.random.default_rng(20260813)

_failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def band(name, value, lo, hi, fmt="{:.4g}"):
    ok = lo <= value <= hi
    check(name, ok, f"{fmt.format(value)} in [{fmt.format(lo)}, {fmt.format(hi)}]")
    return ok


# ---------------------------------------------------------------------------
# (A) edge localisation
# ---------------------------------------------------------------------------


def test_edge_crlb():
    """
    ML edge fitting on synthetic 1-D profiles must approach `edge_crlb_*`.

        The estimator here is exact maximum likelihood for the model (grid search
        refined by parabolic interpolation on the likelihood), so it should be
        efficient to within the sampling error, and any discrepancy indicts the
        Fisher-information algebra rather than the fitter.
    """

    print("\n(A) edge localisation CRLB")
    contrast, sigma_n, psf = 200.0, 6.0, 1.0
    half = 8
    idx = np.arange(-half, half + 1, dtype=np.float64)

    def profile(x0):
        up = (idx + 0.5 - x0) / psf
        lo = (idx - 0.5 - x0) / psf
        return contrast * (
            bounds._Phi(up) - bounds._Phi(lo)
        ).cumsum() * 0.0 + contrast * bounds._Phi((idx - x0) / psf)

    # Box-integrated step, matching bounds.edge_crlb_discrete(box_pixel=True).
    def box_profile(x0):
        # INT_{i-1/2}^{i+1/2} Phi((x-x0)/s) dx, in closed form:
        #   INT Phi(u) du = u Phi(u) + phi(u), scaled by s.
        def anti(x):
            u = (x - x0) / psf
            return psf * (u * bounds._Phi(u) + bounds._phi(u))

        return contrast * (anti(idx + 0.5) - anti(idx - 0.5))

    grid = np.linspace(-1.5, 1.5, 3001)
    templates = np.array([box_profile(g) for g in grid])

    errs = []
    for _ in range(2000):
        x0 = RNG.uniform(-0.5, 0.5)
        obs = box_profile(x0) + RNG.normal(0.0, sigma_n, size=idx.shape)
        k = int(np.argmin(((templates - obs) ** 2).sum(axis=1)))
        errs.append(grid[k] - x0)
    emp = float(np.std(errs))
    crlb = bounds.edge_crlb_discrete(contrast, sigma_n, psf, box_pixel=True)
    eta = (crlb / emp) ** 2
    band("ML edge fit efficiency vs CRLB", eta, 0.80, 1.12)
    print(f"        CRLB {crlb:.5f} px, empirical {emp:.5f} px")

    # The two closed forms must agree where they overlap: for s >> 1 the box
    # integration is a negligible perturbation on point sampling.
    a = bounds.edge_crlb_continuum(contrast, sigma_n, 4.0)
    b = bounds.edge_crlb_discrete(contrast, sigma_n, 4.0, box_pixel=False)
    band("continuum vs discrete point-sampled at s=4", b / a, 0.97, 1.03)

    # And they must DISAGREE in the sharp limit -- that disagreement is the
    # physical content of the box-pixel result. Compare worst sub-pixel phase,
    # which is where point sampling actually fails: an edge midway between two
    # samples moves no sample at all.
    sharp_point = bounds.edge_crlb_discrete(
        contrast, sigma_n, 0.15, box_pixel=False, reduce="worst"
    )
    sharp_box = bounds.edge_crlb_discrete(
        contrast, sigma_n, 0.15, box_pixel=True, reduce="worst"
    )
    check(
        "point sampling loses a sharp edge, box pixels do not",
        sharp_point > 5.0 * sharp_box,
        f"point {sharp_point:.4g} px vs box {sharp_box:.4g} px",
    )
    # The sharp-edge box bound is bracketed by [1, sqrt(2)] * sigma_n/C.
    unit = sigma_n / contrast
    band(
        "sharp box bound, best phase -> sigma_n/C",
        bounds.edge_crlb_discrete(contrast, sigma_n, 0.05, reduce="best") / unit,
        0.98,
        1.02,
    )
    band(
        "sharp box bound, worst phase -> sqrt(2) sigma_n/C",
        bounds.edge_crlb_discrete(contrast, sigma_n, 0.05, reduce="worst") / unit,
        1.39,
        1.43,
    )
    # Point sampling has an interior optimum in blur; box pixels do not.
    grid = np.linspace(0.05, 3.0, 60)
    pt = [
        bounds.edge_crlb_discrete(contrast, sigma_n, s, box_pixel=False, reduce="worst")
        for s in grid
    ]
    bx = [
        bounds.edge_crlb_discrete(contrast, sigma_n, s, box_pixel=True, reduce="worst")
        for s in grid
    ]
    check(
        "point sampling has an interior optimum in blur",
        0 < int(np.argmin(pt)) < len(grid) - 1,
        f"best at s = {grid[int(np.argmin(pt))]:.2f} px",
    )
    check(
        "box pixels are monotone in blur (sharper is better)",
        int(np.argmin(bx)) == 0,
        f"best at s = {grid[int(np.argmin(bx))]:.2f} px",
    )


def test_shipped_subpixel_against_the_bound():
    """
    `segment.subpixel_boundary` on a synthetic disc, against the edge CRLB.

        This is the test that decides whether sub-pixel refinement is worth more
        effort. The disc has a known radius and a smoothly shaded edge, so every
        boundary point has an exact answer; the scatter of the refined points about
        the true circle is compared against `edge_crlb_discrete` for the same
        contrast, noise and blur.

        If the shipped refinement is near the bound, then better interpolation
        cannot help and §12.12's negative result (18x better edge localisation
        bought 2% of the outcome) is explained rather than merely observed: the
        boundary being located is not the rim.
    """

    print("\n(A'') shipped sub-pixel refinement vs the edge CRLB")
    import cv2

    h = w = 256
    cx, cy, rad = 128.3, 127.6, 60.0
    contrast, sigma_n, psf = 200.0, 5.0, 1.2

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sdf = rad - np.hypot(xx - cx, yy - cy)  # >0 inside
    disc = contrast * bounds._Phi(sdf / psf)  # soft edge of width psf

    quant, refined = [], []
    for _ in range(24):
        img = np.clip(disc + RNG.normal(0.0, sigma_n, disc.shape), 0, 255).astype(
            np.uint8
        )
        _, mask = cv2.threshold(img, int(contrast / 2), 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        pts = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
        quant.append(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - rad)
        sp = np.asarray(segment.subpixel_boundary(img, pts)).reshape(-1, 2)
        refined.append(np.hypot(sp[:, 0] - cx, sp[:, 1] - cy) - rad)

    q = np.concatenate(quant)
    r = np.concatenate(refined)
    crlb = bounds.edge_crlb_discrete(contrast, sigma_n, psf)
    print(f"        thresholded   bias {q.mean():+.4f} px   scatter {q.std():.4f} px")
    print(f"        refined       bias {r.mean():+.4f} px   scatter {r.std():.4f} px")
    print(f"        edge CRLB                          {crlb:.4f} px")

    check(
        "sub-pixel refinement reduces the scatter",
        r.std() < q.std(),
        f"{q.std():.4f} -> {r.std():.4f} px",
    )
    check(
        "sub-pixel refinement reduces the bias",
        abs(r.mean()) < abs(q.mean()),
        f"{q.mean():+.4f} -> {r.mean():+.4f} px",
    )
    check(
        "refined scatter does not beat the CRLB",
        r.std() >= crlb * 0.9,
        f"{r.std() / crlb:.2f}x the bound",
    )
    eff = (crlb / r.std()) ** 2
    print(
        f"        efficiency {eff:.2f} -- headroom in edge localisation is "
        f"{'small' if eff > 0.25 else 'real'}"
    )
    check("scatter is within an order of magnitude of the bound", r.std() < 10 * crlb)


def test_quantisation():
    """
    A pixel-quantised boundary point has sigma = 1/sqrt(12), by construction.
    """

    print("\n(A') quantisation floor")
    x = RNG.uniform(-50, 50, size=200000)
    err = np.round(x) - x
    band(
        "thresholded boundary sigma",
        float(np.std(err)),
        0.985 * bounds.QUANTISATION_SIGMA_PX,
        1.015 * bounds.QUANTISATION_SIGMA_PX,
    )
    snr = bounds.subpixel_breakeven_snr(psf_px=1.0)
    print(f"        sub-pixel refinement beats rounding above C/sigma_n = {snr:.2f}")
    check("break-even SNR is modest (sub-pixel is worth doing)", snr < 20.0)


# ---------------------------------------------------------------------------
# (C) ellipse parameters
# ---------------------------------------------------------------------------


def test_circle_closed_form():
    """
    `circle_crlb_closed_form` must equal the numerical Fisher inverse.
    """

    print("\n(C) circle CRLB: closed form vs numerical Fisher")
    r, n, e = 70.0, 400, 0.1
    j = bounds.ellipse_fisher((300.0, 250.0), (r, r), 0.0, n, e)
    # theta is unobservable for a circle: the Fisher matrix must be rank 4.
    sv = np.linalg.svd(j, compute_uv=False)
    check(
        "orientation of a circle is unobservable (Fisher rank 4)",
        sv[-1] / sv[0] < 1e-12,
        f"cond^-1 = {sv[-1] / sv[0]:.2e}",
    )

    sub = np.linalg.inv(j[np.ix_([0, 1, 2], [0, 1, 2])])
    sc, sr = bounds.circle_crlb_closed_form(r, n, e)
    band("sigma_centre closed form vs numerical", math.sqrt(sub[0, 0]) / sc, 0.98, 1.02)
    # The (a, b) block of an a==b ellipse splits the radius information in two,
    # so compare the derived radius bound against a 3-parameter (cx,cy,r) fit.
    j3 = np.array(
        [
            [j[0, 0], j[0, 1], j[0, 2] + j[0, 3]],
            [j[1, 0], j[1, 1], j[1, 2] + j[1, 3]],
            [
                j[2, 0] + j[3, 0],
                j[2, 1] + j[3, 1],
                j[2, 2] + j[2, 3] + j[3, 2] + j[3, 3],
            ],
        ]
    )
    s3 = np.linalg.inv(j3)
    band("sigma_radius closed form vs numerical", math.sqrt(s3[2, 2]) / sr, 0.98, 1.02)
    print(
        f"        sigma_centre {sc:.5f} px, sigma_radius {sr:.5f} px "
        f"(ratio {sc / sr:.4f}, predicted sqrt(2) = 1.4142)"
    )


def test_ellipse_fit_efficiency():
    """
    The shipped `segment.fit_ellipse` against the ellipse CRLB.

        Points are generated on an exact ellipse with Gaussian noise along the
        normal only -- the model the Fisher matrix assumes -- so the only thing under
        test is the fitter. ``axial=False`` because that is the shipped default and
        because the weighted path deliberately discards points, which would make an
        efficiency comparison against an all-points bound meaningless.
    """

    print("\n(C') ellipse-fit efficiency vs CRLB")
    centre, axes, ang = (497.0, 355.0), (70.0, 52.0), math.radians(23.0)
    n, e = 380, 0.08
    phis = bounds.arclength_phis(axes, n)
    truth, nrm = bounds.ellipse_points(centre, axes, ang, phis)

    cov = bounds.ellipse_crlb(centre, axes, ang, n, e, phis=phis)
    pred = np.sqrt(np.diag(cov))

    got = []
    for _ in range(TRIALS):
        pts = truth + nrm * RNG.normal(0.0, e, size=(len(truth), 1))
        out = segment.fit_ellipse(pts, axial=False)
        if out is None:
            continue
        (cx, cy), (ma, mi), deg = out[0]
        got.append([cx, cy, 0.5 * ma, 0.5 * mi, math.radians(deg)])
    got = np.asarray(got)
    check("fit converged on every trial", len(got) == TRIALS, f"{len(got)}/{TRIALS}")

    # Unwrap the orientation onto the branch nearest truth (an ellipse axis has
    # no direction, so the angle lives on a pi-periodic circle).
    got[:, 4] = ang + np.angle(np.exp(1j * 2.0 * (got[:, 4] - ang))) / 2.0

    emp = got.std(axis=0)
    bias = got.mean(axis=0) - np.array([centre[0], centre[1], axes[0], axes[1], ang])
    names = ["cx", "cy", "a", "b", "theta"]
    print("        param    CRLB        empirical   efficiency   bias/sigma")
    for i, nm in enumerate(names):
        eta = (pred[i] / emp[i]) ** 2
        print(
            f"        {nm:<7}{pred[i]:<12.5g}{emp[i]:<12.5g}"
            f"{eta:<13.3f}{bias[i] / emp[i]:+.3f}"
        )
        check(f"{nm}: efficiency in (0.5, 1.05]", 0.5 < eta <= 1.05, f"eta={eta:.3f}")

    # The direct algebraic fit is known to be biased toward smaller ellipses;
    # quantify it rather than assume it away. At this noise level the bias must
    # stay well inside the statistical scatter, or the CRLB comparison above is
    # not measuring what it claims to.
    for i, nm in enumerate(names):
        check(f"{nm}: bias below 0.3 sigma", abs(bias[i]) < 0.3 * emp[i])

    # Short arcs: the same bound, evaluated on 40% of the perimeter, must blow up
    # -- this is the quantitative version of "a partly occluded rim is unusable".
    short = bounds.ellipse_crlb(centre, axes, ang, n, e, arc_fraction=0.4)
    ratio = math.sqrt(short[2, 2] / cov[2, 2])
    check("40% arc inflates the semi-major bound", ratio > 3.0, f"{ratio:.2f}x")


# ---------------------------------------------------------------------------
# (D) pose
# ---------------------------------------------------------------------------


def test_pose_efficiency():
    """
    Full chain: noisy boundary -> `fit_ellipse` -> `backproject_ellipse`.
    """

    print("\n(D) pose efficiency vs CRLB")
    z, tilt = 250.0, math.radians(35.0)
    f = 0.5 * (K[0, 0] + K[1, 1])
    a_px = f * RADIUS_MM / z
    b_px = a_px * math.cos(tilt)
    centre = (K[0, 2], K[1, 2])
    axes, ang = (a_px, b_px), math.radians(11.0)
    n, e = (
        int(
            round(
                math.pi
                * (3 * (a_px + b_px) - math.sqrt((3 * a_px + b_px) * (a_px + 3 * b_px)))
            )
        ),
        0.05,
    )

    phis = bounds.arclength_phis(axes, n)
    truth, nrm = bounds.ellipse_points(centre, axes, ang, phis)

    cov, ref = bounds.pose_crlb(centre, axes, ang, K, RADIUS_MM, n, e, phis=phis)
    pred = np.sqrt(np.diag(cov))

    got = []
    for _ in range(600):
        pts = truth + nrm * RNG.normal(0.0, e, size=(len(truth), 1))
        out = segment.fit_ellipse(pts, axial=False)
        if out is None:
            continue
        poses = conic.backproject_ellipse(out[0], K, RADIUS_MM)
        if not poses:
            continue
        got.append(bounds._pick_branch(poses, ref))
    got = np.asarray(got)
    emp = got.std(axis=0)

    print("        axis     CRLB        empirical   efficiency")
    for i, nm in enumerate(["X mm", "Y mm", "Z mm", "nx", "ny", "nz"]):
        if emp[i] < 1e-12:
            continue
        eta = (pred[i] / emp[i]) ** 2
        print(f"        {nm:<9}{pred[i]:<12.5g}{emp[i]:<12.5g}{eta:<13.3f}")
        check(
            f"pose {nm}: efficiency in (0.5, 1.10]", 0.5 < eta <= 1.10, f"eta={eta:.3f}"
        )

    got = emp[2] / math.hypot(emp[0], emp[1])
    want = bounds.depth_lateral_ratio(z, RADIUS_MM, math.degrees(tilt))
    band("Monte-Carlo depth/lateral vs predicted", got / want, 0.85, 1.15)
    print(
        f"        empirical {got:.2f} vs sqrt(3)-corrected law {want:.2f} "
        f"(naive z/2R would say {z / (2 * RADIUS_MM):.2f})"
    )


def test_depth_lateral_law():
    """
    sigma_z/|sigma_lat| = g(tilt) * z/(2R), with g(0) = sqrt(3) analytically.
    """

    print("\n(D') the depth/lateral law")
    f = 0.5 * (K[0, 0] + K[1, 1])
    centre = (K[0, 2], K[1, 2])

    # The sqrt(3) is the price of not knowing the tilt. Derive it directly from
    # the (a,b) block of the Fisher matrix for a near-circular ellipse, which the
    # docstring claims is (N/8 sigma^2) [[3,1],[1,3]].
    r, n, e = 60.0, 4000, 0.01
    j = bounds.ellipse_fisher(centre, (r, r), 0.0, n, e)
    blk = j[np.ix_([2, 3], [2, 3])] * (8.0 * e**2 / n)
    band("Fisher (a,b) block diag == 3", blk[0, 0], 2.94, 3.06)
    band("Fisher (a,b) block off-diag == 1", blk[0, 1], 0.94, 1.06)
    sigma_a = math.sqrt(np.linalg.inv(j[np.ix_([2, 3], [2, 3])])[0, 0])
    band(
        "sigma_a == sqrt(3) e / sqrt(N)",
        sigma_a / (math.sqrt(3.0) * e / math.sqrt(n)),
        0.97,
        1.03,
    )

    # The law is exact in the WEAK-PERSPECTIVE limit. Assert it tightly there.
    rows = []
    for z in (1200.0, 3000.0):
        for tilt in (10.0, 30.0, 60.0, 80.0):
            for e in (0.02, 0.10):
                got = bounds.depth_lateral_ratio_numeric(
                    z, RADIUS_MM, K, tilt, n_points=2000, sigma_r_px=e
                )
                rows.append(
                    (z, tilt, e, got, bounds.depth_lateral_ratio(z, RADIUS_MM, tilt))
                )
    worst = max(abs(r[3] / r[4] - 1.0) for r in rows)
    check(
        f"weak-perspective law exact over {len(rows)} (z, tilt, noise) points",
        worst < 0.01,
        f"worst deviation {worst * 100:.2f}%",
    )

    # The sharp claim: the ratio depends on NOTHING but z and tilt -- not the
    # noise level, not the point count, not the resolution.
    bad = [
        (r[0], r[1])
        for r in rows
        if abs(r[3] / [q[3] for q in rows if q[0] == r[0] and q[1] == r[1]][0] - 1.0)
        > 1e-3
    ]
    check("ratio is independent of the noise level", not bad, f"{len(bad)} outliers")
    n_a = bounds.depth_lateral_ratio_numeric(3000.0, RADIUS_MM, K, 30.0, n_points=500)
    n_b = bounds.depth_lateral_ratio_numeric(3000.0, RADIUS_MM, K, 30.0, n_points=8000)
    check(
        "ratio is independent of the point count",
        abs(n_a / n_b - 1.0) < 1e-3,
        f"N=500 -> {n_a:.3f}, N=8000 -> {n_b:.3f}",
    )

    # Finite range is a real correction, not a rounding error. It scales roughly
    # as (R/z)^2 / sin^2(tilt): negligible for a tilted rim, severe for a
    # near-face-on one seen from close up. Report it rather than hide it.
    print("        finite-range correction  (measured / weak-perspective law)")
    print("        z mm     R/z      tilt 10    tilt 30    tilt 60")
    env = []
    for z in (100.0, 150.0, 250.0, 400.0, 700.0):
        vals = [
            bounds.depth_lateral_ratio_numeric(
                z, RADIUS_MM, K, t, n_points=2000, sigma_r_px=0.02
            )
            / bounds.depth_lateral_ratio(z, RADIUS_MM, t)
            for t in (10.0, 30.0, 60.0)
        ]
        print(
            f"        {z:<9.0f}{RADIUS_MM / z:<9.4f}"
            + "".join(f"{v:<11.4f}" for v in vals)
        )
        if 150.0 <= z <= 400.0:
            env.extend(vals[1:])  # operating envelope: tilt >= 30 deg
    check(
        "law holds to 2% inside the operating envelope (z 150-400, tilt >= 30)",
        all(abs(v - 1.0) < 0.02 for v in env),
        f"worst {max(abs(v - 1.0) for v in env) * 100:.2f}%",
    )

    # At EXACTLY face-on there is no finite bound on the 5-parameter fit: theta
    # is unobservable, the Fisher matrix drops rank, and the CRLB is undefined
    # rather than large. The ratio still has a limit, approached from above.
    check(
        "no finite bound at exactly face-on (Fisher drops rank)",
        bounds.depth_lateral_ratio_numeric(3000.0, RADIUS_MM, K, 0.0) is None,
    )

    print("        tilt   g(tilt) table   g measured at z=3000mm")
    for tilt in (0.5, 10.0, 30.0, 60.0, 80.0):
        got = bounds.depth_lateral_ratio_numeric(3000.0, RADIUS_MM, K, tilt)
        print(
            f"        {tilt:<7.1f}{bounds.depth_lateral_ratio(1.0, 0.5, tilt):<16.4f}"
            f"{got / (3000.0 / (2 * RADIUS_MM)):.4f}"
        )
    g0 = bounds.depth_lateral_ratio_numeric(3000.0, RADIUS_MM, K, 0.5) / (
        3000.0 / (2 * RADIUS_MM)
    )
    band("g -> sqrt(3) as tilt -> 0", g0 / math.sqrt(3.0), 0.98, 1.02)


def test_face_on_lateral_collapse():
    """
    Near face-on, the tilt degeneracy contaminates LATERAL position too.

        This is not in the textbook statement of the problem and it is worth
        isolating: at 2 deg of tilt the recovered centre is seven times worse than at
        10 deg, at identical range, resolution and noise. The mechanism is that an
        eccentricity error indistinguishable from noise gets interpreted as a tilt of
        the circle's *plane*, and tilting the plane slides the recovered centre. So
        "face-on is bad for tilt" understates it -- face-on is bad for everything.
    """

    print("\n(D'') face-on contaminates lateral position")
    f = 0.5 * (K[0, 0] + K[1, 1])
    centre = (K[0, 2], K[1, 2])
    z, e = 250.0, 0.05
    a_px = f * RADIUS_MM / z

    def lat_at(tilt):
        b_px = a_px * math.cos(math.radians(tilt))
        n = max(24, int(round(2.0 * math.pi * a_px)))
        cov, _ = bounds.pose_crlb(centre, (a_px, b_px), 0.0, K, RADIUS_MM, n, e)
        return math.hypot(*np.sqrt(np.diag(cov))[:2])

    ref = lat_at(40.0)
    print("        tilt   sigma_lat mm   inflation vs 40 deg")
    for tilt in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        lat = lat_at(tilt)
        print(f"        {tilt:<7.0f}{lat:<15.5f}{lat / ref:.2f}x")

    check(
        "lateral error inflates below ~5 deg of tilt",
        lat_at(2.0) / lat_at(20.0) > 4.0,
        f"{lat_at(2.0) / lat_at(20.0):.1f}x worse at 2 deg than 20 deg",
    )
    check(
        "lateral error is flat above ~20 deg",
        0.8 < lat_at(20.0) / lat_at(40.0) < 1.25,
        f"{lat_at(20.0) / lat_at(40.0):.2f}x",
    )


def test_tilt_laws():
    """
    Both tilt regimes: 1/sin(theta) away from face-on, sqrt() at face-on.
    """

    print("\n(D''') tilt conditioning")
    a = 70.0
    sigma_ratio = 0.004

    for deg in (10.0, 30.0, 60.0):
        th = math.radians(deg)
        ratios = math.cos(th) + RNG.normal(0.0, sigma_ratio, size=40000)
        emp = float(np.std(np.degrees(np.arccos(np.clip(ratios, -1.0, 1.0)))))
        pred = math.degrees(bounds.tilt_from_ratio_sigma(th, sigma_ratio))
        band(f"tilt {deg:.0f} deg: 1/sin law", emp / pred, 0.95, 1.05)

    ratios = 1.0 - np.abs(RNG.normal(0.0, sigma_ratio, size=200000))
    emp = float(np.mean(np.degrees(np.arccos(np.clip(ratios, -1.0, 1.0)))))
    pred = math.degrees(bounds.tilt_bias_at_face_on(sigma_ratio))
    band("face-on apparent tilt (sqrt law)", emp / pred, 0.95, 1.05)
    print(
        f"        a genuinely face-on rim reads {emp:.2f} deg of tilt "
        f"at sigma_ratio={sigma_ratio}"
    )


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------


def test_structural():
    print("\n(S) structural results")
    c = np.array([12.0, -7.0, 260.0])
    nrm = np.array([0.32, -0.18, -0.93])
    nrm /= np.linalg.norm(nrm)

    roll = bounds.roll_null_space(c, nrm, RADIUS_MM, K)
    check("roll is exactly unobservable", roll < 1e-6, f"max image shift {roll:.3e} px")

    gap, margin, nb = bounds.ambiguity_is_exact(c, nrm, RADIUS_MM, K)
    check("two branches exist", nb == 2, f"{nb} branches")
    check("branches are physically distinct", margin > 5.0, f"margin {margin:.2f} deg")
    check(
        "branches are photometrically identical", gap < 1e-6, f"image gap {gap:.3e} px"
    )
    print(
        "        -> the single-view likelihood has two exactly equal maxima;"
        " no data from this view can choose"
    )


def test_budget_table():
    """
    Print the composed budget; not an assertion, the evidence for the report.
    """

    print("\n(E) composed error budget, 1280x800, C=200 sigma_n=6 s=1.0 px")
    print("        z mm   a px   N_eff   sigma_r    sigma_lat   sigma_z    z/2R")
    for z in (150.0, 250.0, 400.0):
        b = bounds.budget(z, RADIUS_MM, K, tilt_deg=30.0, correlation_px=2.0)
        print(
            f"        {z:<7.0f}{b['semi_major_px']:<7.1f}{b['n_effective']:<8.0f}"
            f"{b['sigma_r_px']:<11.4f}{b['sigma_lateral_mm']:<12.5f}"
            f"{b['sigma_z_mm']:<11.5f}{b['depth_lateral_ratio']:.1f}"
        )


def main():
    print("=" * 72)
    print("Cramer-Rao bounds for the single-view conic pose estimator")
    print("=" * 72)
    test_edge_crlb()
    test_shipped_subpixel_against_the_bound()
    test_quantisation()
    test_circle_closed_form()
    test_ellipse_fit_efficiency()
    test_pose_efficiency()
    test_depth_lateral_law()
    test_face_on_lateral_collapse()
    test_tilt_laws()
    test_structural()
    test_budget_table()
    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("all bounds verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
