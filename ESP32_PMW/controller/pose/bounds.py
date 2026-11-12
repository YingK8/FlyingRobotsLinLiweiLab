"""
Fundamental precision limits for the single-view conic pose estimator.

How well could *any* estimator do, given the same pixels? Everything else in
`controller/pose/` measures what this one does; without a floor to compare
against, a residual of a millimetre could mean the solver is sloppy or it could
mean the information is not in the image, and those point at opposite work.

Four stages, each separately checkable:

    photons  ->  edge position  ->  ellipse parameters  ->  pose
      (A)           (B)                  (C)                (D)

(A) and (B) are Cramer-Rao bounds on a scalar; (C) is a 5x5 Fisher matrix; (D) is
a change of variables through the back-projection, whose Jacobian comes from
finite differences of the real `conic.backproject_ellipse` rather than a second
hand-linearised model.

Three results here are *structural* -- no sensor, lens or algorithm removes them:
`roll_null_space` (roll is exactly unobservable, so the Jacobian is rank 5),
`ambiguity_is_exact` (two distinct poses produce the identical ellipse, so the
likelihood is genuinely bimodal), and `depth_lateral_ratio` (depth costs
``g(tilt) * z/2R``, independent of noise, focal length, resolution and point
count).

The point of a floor is to find out how far above it we are. Measured boundary
scatter is ~23x the photon bound (`validation/limits.py`), which is what proves
the residual is silhouette *bias* rather than sensor noise -- and therefore that
better sensors, longer exposures and finer sub-pixel interpolation cannot help.

Full derivations: `controller/pose/theory.md` S13.
Monte-Carlo verification of every formula here: `test_bounds.py`.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# (A) Edge localisation: how well a boundary point can be placed
# ---------------------------------------------------------------------------
#
# Model a boundary as a step of contrast C blurred by a Gaussian PSF of width s
# pixels, sitting on a background B, sampled by pixels that *integrate* over
# their own area:
#
#     I_i = B + C * [ Phi((i + 1/2 - x0)/s) - Phi((-inf)) ]      (ideal, no box)
#     I_i = B + C * (1/1) * INT_{i-1/2}^{i+1/2} Phi((x - x0)/s) dx   (box pixel)
#
# with additive Gaussian read noise of standard deviation sigma_n per pixel.
# Fisher information about the edge position x0 is
#
#     J(x0) = (1/sigma_n^2) * SUM_i (dI_i/dx0)^2
#
# and the Cramer-Rao bound is sigma_x0 >= 1/sqrt(J).
#
# Two limits are worth having in closed form, because they bracket every real
# case and they disagree in an instructive way.

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(u):
    return np.exp(-0.5 * np.asarray(u, dtype=np.float64) ** 2) / _SQRT_2PI


def _Phi(u):
    from math import erf

    u = np.asarray(u, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf)(u / math.sqrt(2.0)))


def edge_crlb_discrete(
    contrast, sigma_noise, psf_px, box_pixel=True, half_width=12, reduce="rms"
):
    """
    CRLB on edge position for a sampled sensor. Lecture notes S13.1.

        ``box_pixel`` integrates the blurred step over the pixel's own footprint,
        which is what a real sensor does, and it changes the limit rather than
        refining it:

          point sampling  a sharp edge between samples moves no sample, so the bound
                          *diverges*. Interior optimum near s ~ 0.5 px.
          box pixels      the derivatives form a partition of unity over two pixels,
                          so ``sigma_n/C <= sigma_x0 <= sqrt(2) sigma_n/C``: finite at
                          every blur and monotone, sharper always better. The sensor's
                          own area integration is what makes sub-pixel phase
                          observable at all.

        @param reduce: how the unknown sub-pixel phase is handled -- ``"rms"`` (to
            compare against a Monte Carlo over uniform phase), ``"worst"`` (the figure
            to quote), or ``"best"``.
    """

    if contrast <= 0.0 or sigma_noise <= 0.0 or psf_px <= 0.0:
        raise ValueError("contrast, sigma_noise and psf_px must be positive")
    offsets = np.linspace(0.0, 0.5, 21)
    idx = np.arange(-half_width, half_width + 1, dtype=np.float64)
    sigmas = []
    for x0 in offsets:
        if box_pixel:
            # d/dx0 INT_{i-1/2}^{i+1/2} Phi((x-x0)/s) dx = -[Phi(u+) - Phi(u-)]
            up = (idx + 0.5 - x0) / psf_px
            lo = (idx - 0.5 - x0) / psf_px
            deriv = -contrast * (_Phi(up) - _Phi(lo))
        else:
            deriv = -contrast * _phi((idx - x0) / psf_px) / psf_px
        j = float(np.sum(deriv**2)) / (sigma_noise**2)
        sigmas.append(math.inf if j <= 0.0 else 1.0 / math.sqrt(j))
    sigmas = np.asarray(sigmas)
    if reduce == "best":
        return float(sigmas.min())
    if reduce == "worst":
        return float(sigmas.max())
    return float(np.sqrt(np.mean(sigmas**2)))


#: Standard deviation of a boundary point quantised to the pixel grid with no
#: sub-pixel refinement at all. A threshold-and-contour outline reports the
#: *centre* of the last inside pixel, so the true edge is uniform over one pixel
#: about it: Var = 1/12.
QUANTISATION_SIGMA_PX = 1.0 / math.sqrt(12.0)  # 0.2887 px


# ---------------------------------------------------------------------------
# (B) Correlated boundary noise: the effective number of independent points
# ---------------------------------------------------------------------------


def effective_point_count(n_points, correlation_px, spacing_px=1.0):
    """
    Independent-sample count for boundary points whose errors are correlated.

        Adjacent contour points are *not* independent: they are produced by the same
        PSF, so their position errors share a correlation length of roughly the blur
        width. For noise with correlation length ``L`` sampled every ``d``, the
        variance of an average over N samples is inflated by about ``L/d``, so the
        honest count to put in a 1/sqrt(N) is

            N_eff = N * d / max(d, L).

        Measured on real renders, ``L`` is 17.5 px against a hull-point spacing of
        about 11 px, so 31 hull points carry roughly 15 independent measurements --
        a factor of 1.4 in the bound. That is smaller than the factor of 3.4 lost by
        hulling a ~350 px perimeter down to 31 points in the first place, so it is
        the *second* largest source of optimism in a naive contour CRLB, not the
        first; both are reported separately by `validation/limits.py`, which is also
        what measures ``L``.
    """

    n_points = float(n_points)
    lag = max(float(spacing_px), float(correlation_px))
    return max(1.0, n_points * float(spacing_px) / lag)


# ---------------------------------------------------------------------------
# (C) Ellipse-parameter Fisher information
# ---------------------------------------------------------------------------
#
# Parametrise the ellipse as p = (cx, cy, a, b, theta) with a >= b the SEMI-axes
# and theta the rotation of the major axis. A boundary point at eccentric angle
# phi is
#
#     x(phi) = c + R(theta) [a cos phi, b sin phi]^T
#
# with outward unit normal proportional to R(theta) [b cos phi, a sin phi]^T.
#
# Only the component of a boundary point's error ALONG THE NORMAL is observable.
# The tangential component slides the point along the curve and changes nothing
# -- the aperture problem, in its cleanest form. So the measurement for point i
# is the signed normal offset, with variance sigma_r^2, and
#
#     g_i = n_i^T dx(phi_i)/dp,     J = (1/sigma_r^2) SUM_i g_i g_i^T.


def ellipse_points(centre, semi_axes, angle_rad, phis):
    """
    Boundary points and outward unit normals at eccentric angles ``phis``.
    """

    a, b = float(semi_axes[0]), float(semi_axes[1])
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[ca, -sa], [sa, ca]])
    phis = np.asarray(phis, dtype=np.float64)
    local = np.column_stack([a * np.cos(phis), b * np.sin(phis)])
    pts = np.asarray(centre, dtype=np.float64) + local @ rot.T
    nrm = np.column_stack([b * np.cos(phis), a * np.sin(phis)]) @ rot.T
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    return pts, nrm


def arclength_phis(semi_axes, n_points):
    """
    Eccentric angles spaced uniformly in *arc length*.

        Contour pixels are spaced along the boundary, not in the eccentric angle, and
        for an elongated ellipse the two differ enormously -- uniform-in-phi puts far
        too few points near the ends of the major axis, which is exactly where the
        curvature information about ``a`` lives. Getting this wrong makes the bound
        on ``a`` too pessimistic and the bound on tilt too optimistic.
    """

    a, b = float(semi_axes[0]), float(semi_axes[1])
    dense = np.linspace(0.0, 2.0 * math.pi, 4096)
    speed = np.hypot(a * np.sin(dense), b * np.cos(dense))
    s = np.concatenate(
        [[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(dense))]
    )
    targets = np.linspace(0.0, s[-1], int(n_points), endpoint=False)
    return np.interp(targets, s, dense)


def ellipse_fisher(
    centre, semi_axes, angle_rad, n_points, sigma_r_px, phis=None, arc_fraction=1.0
):
    """
    5x5 Fisher information for ``(cx, cy, a, b, theta)`` in pixel units.

        ``arc_fraction`` < 1 keeps only that fraction of the perimeter, which is how
        an occluded or partly-thresholded rim behaves; the conditioning collapse from
        a short arc is one of the things this makes quantitative.

        The matrix is singular when a == b: a circle has no defined orientation, so
        the theta row and column vanish identically. That is a genuine rank
        deficiency of the measurement, not a numerical artefact, and callers that
        need a bound near face-on must marginalise rather than invert.
    """

    a, b = float(semi_axes[0]), float(semi_axes[1])
    if phis is None:
        phis = arclength_phis((a, b), n_points)
        if arc_fraction < 1.0:
            phis = phis[: max(5, int(round(arc_fraction * len(phis))))]
    phis = np.asarray(phis, dtype=np.float64)

    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[ca, -sa], [sa, ca]])
    drot = np.array([[-sa, -ca], [ca, -sa]])  # dR/dtheta
    c, s = np.cos(phis), np.sin(phis)

    _, nrm = ellipse_points(centre, (a, b), angle_rad, phis)

    d = np.empty((len(phis), 5, 2), dtype=np.float64)
    d[:, 0] = np.array([1.0, 0.0])
    d[:, 1] = np.array([0.0, 1.0])
    d[:, 2] = np.column_stack([c, np.zeros_like(c)]) @ rot.T
    d[:, 3] = np.column_stack([np.zeros_like(s), s]) @ rot.T
    d[:, 4] = np.column_stack([a * c, b * s]) @ drot.T

    g = np.einsum("nkd,nd->nk", d, nrm)
    return (g.T @ g) / (float(sigma_r_px) ** 2)


def ellipse_crlb(centre, semi_axes, angle_rad, n_points, sigma_r_px, **kw):
    """
    Cramer-Rao covariance for ``(cx, cy, a, b, theta)``; ``None`` if singular.
    """

    j = ellipse_fisher(centre, semi_axes, angle_rad, n_points, sigma_r_px, **kw)
    if np.linalg.cond(j) > 1e12:
        return None
    return np.linalg.inv(j)


# ---------------------------------------------------------------------------
# (D) Ellipse -> pose, and the depth/lateral law
# ---------------------------------------------------------------------------


def _pose_vector(pose):
    return np.concatenate(
        [
            np.asarray(pose.center, dtype=np.float64),
            np.asarray(pose.normal, dtype=np.float64),
        ]
    )


def _pick_branch(poses, reference):
    """
    Back-projection returns two branches; follow the one nearest ``reference``.
    """

    best, best_d = None, math.inf
    for p in poses:
        v = _pose_vector(p)
        d = float(np.linalg.norm(v[:3] - reference[:3])) + 50.0 * float(
            np.linalg.norm(v[3:] - reference[3:])
        )
        if d < best_d:
            best, best_d = v, d
    return best


def pose_jacobian(centre, semi_axes, angle_rad, camera_matrix, radius_mm, step=1e-4):
    """
    ``d(pose)/d(cx, cy, a, b, theta)``: 6x5, by central differences.

        Differentiating the *real* `conic.backproject_ellipse` rather than a
        hand-linearised model. That is deliberate: an analytic Jacobian of the
        eigen-decomposition would be a second implementation to keep in sync, and the
        thing we want bounded is the pipeline that ships.

        The branch is re-selected at every perturbed point by proximity to the
        nominal pose, so the derivative does not jump between the two ambiguity
        solutions and report a spurious infinity.
    """

    from controller.pose import conic as _conic

    def solve(p):
        e = ((p[0], p[1]), (2.0 * p[2], 2.0 * p[3]), math.degrees(p[4]))
        poses = _conic.backproject_ellipse(e, camera_matrix, radius_mm)
        if not poses:
            return None
        return poses

    p0 = np.array(
        [centre[0], centre[1], semi_axes[0], semi_axes[1], angle_rad], dtype=np.float64
    )
    base = solve(p0)
    if not base:
        return None, None
    ref = _pose_vector(base[0])

    jac = np.zeros((6, 5), dtype=np.float64)
    scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * step
    scale[:4] *= max(1.0, float(semi_axes[0]))  # px-scaled steps for px params
    for k in range(5):
        dp = np.zeros(5)
        dp[k] = scale[k]
        hi, lo = solve(p0 + dp), solve(p0 - dp)
        if hi is None or lo is None:
            return None, None
        jac[:, k] = (_pick_branch(hi, ref) - _pick_branch(lo, ref)) / (2.0 * scale[k])
    return jac, ref


def pose_crlb(
    centre, semi_axes, angle_rad, camera_matrix, radius_mm, n_points, sigma_r_px, **kw
):
    """
    Covariance of ``(Cx, Cy, Cz, nx, ny, nz)`` in mm and unit-vector units.

        Composition of (C) and (D): ``Sigma_pose = G Sigma_ellipse G^T``. This is the
        delta-method push-forward of the Cramer-Rao bound, which is itself the CRLB
        for the transformed parameters whenever G has full column rank -- and G
        cannot have rank more than 5, which is the formal statement that roll is
        unobservable.
    """

    sig = ellipse_crlb(centre, semi_axes, angle_rad, n_points, sigma_r_px, **kw)
    if sig is None:
        return None, None
    g, ref = pose_jacobian(centre, semi_axes, angle_rad, camera_matrix, radius_mm)
    if g is None:
        return None, None
    return g @ sig @ g.T, ref


#: Asymptotic tilt factor ``g(theta)`` in the depth/lateral law, tabulated from
#: the Fisher matrix in the weak-perspective limit (see `depth_lateral_ratio`).
#: ``g(0) = sqrt(3)`` analytically; the rest is a slow, bounded, monotone rise.
_TILT_FACTOR_DEG = np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90], float)
_TILT_FACTOR = np.array(
    [
        1.7321,
        1.7278,
        1.7395,
        1.7514,
        1.7671,
        1.8114,
        1.8730,
        1.9503,
        2.0384,
        2.1250,
        2.1800,
        2.2000,
    ],
    float,
)


def depth_lateral_ratio(z_mm, radius_mm, tilt_deg=0.0, per_axis=False):
    """
    CRLB ratio ``sigma_depth / sigma_lateral`` = ``g(tilt) * z / (2R)``.

        Derivation in lecture notes S13.4. In short: with the tilt *known* the ratio
        is exactly ``z/2R``, range over diameter. With it estimated from the same
        ellipse -- which is what ships -- the semi-axis must be disentangled from
        ``b`` and ``theta``, the ``(a,b)`` Fisher block is
        ``(N/8 sigma^2)[[3,1],[1,3]]``, and ``sigma_a`` gains a factor ``sqrt(3)``:

            sigma_depth / |sigma_lat| = g(theta) * z / (2R),   g(0) = sqrt(3)

        ``g`` rises slowly and stays in [1.73, 2.2] (``_TILT_FACTOR``), so a working
        form is ``sigma_depth ~ (z/R) sigma_lat``, range over *radius*, good to 15%
        at any tilt.

        Noise level, point count, focal length and resolution all cancel -- this is
        perspective projection, not the sensor or the algorithm. Exact in the
        weak-perspective limit and good to 2% inside the operating envelope
        (z 150-400 mm, tilt >= 30 deg); ``per_axis`` divides by sqrt(2).
    """

    g = float(np.interp(abs(float(tilt_deg)), _TILT_FACTOR_DEG, _TILT_FACTOR))
    denom = 2.0 * radius_mm
    if per_axis:
        denom /= math.sqrt(2.0)
    return g * float(z_mm) / denom


# ---------------------------------------------------------------------------
# Structural results: rank and ambiguity
# ---------------------------------------------------------------------------


def roll_null_space(centre_mm, normal, radius_mm, camera_matrix, n_probe=64):
    """
    Largest image displacement produced by spinning the circle about its axis.

        Rotating the rim about its own normal maps the circle onto itself, so the
        projected conic is *identical* and the derivative of the image with respect
        to roll is exactly zero. Returned in pixels; anything above float noise would
        mean the parametrisation had smuggled in an orientation the geometry does not
        have. This is the formal version of "the estimator is 5-DOF, not 6".
    """

    from controller.pose import conic as _conic

    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e0 = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, e0)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    worst = 0.0
    # The projected conic is pose-dependent only, not roll-dependent, so it is
    # built once and every rolled rim is tested against that same conic.
    e = _conic.project_circle(centre_mm, n, radius_mm, camera_matrix)
    c = _conic.conic_from_ellipse(e)
    for k in range(n_probe):
        psi = 2.0 * math.pi * k / n_probe
        # A roll by psi permutes the rim's points; the circle -- hence the conic
        # -- is unchanged. Rebuild it from a rolled basis to prove that.
        uu = math.cos(psi) * u + math.sin(psi) * v
        vv = -math.sin(psi) * u + math.cos(psi) * v
        pts = []
        for t in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            p = np.asarray(centre_mm) + radius_mm * (
                math.cos(t) * uu + math.sin(t) * vv
            )
            q = camera_matrix @ p
            pts.append(q[:2] / q[2])
        pts = np.asarray(pts)
        # Every rolled point must still lie on that same conic.
        h = np.column_stack([pts, np.ones(len(pts))])
        alg = np.abs(np.einsum("ni,ij,nj->n", h, c, h))
        # Convert the algebraic residual to an approximate pixel distance.
        grad = np.linalg.norm((c @ h.T).T[:, :2], axis=1)
        worst = max(worst, float(np.max(alg / np.maximum(grad, 1e-30))) / 2.0)
    return worst


def ambiguity_is_exact(centre_mm, normal, radius_mm, camera_matrix):
    """
    How far apart the two ambiguity branches' *images* are, in pixels.

        The two back-projected poses are physically distinct -- their normals differ
        by the ambiguity margin, tens of degrees -- yet they must reproject to the
        same ellipse to machine precision. If they do, no single-view estimator can
        prefer one on evidence: the likelihood has two exactly equal maxima, and the
        choice is made by prior or by a second view, never by data from this one.

        Returns ``(image_gap_px, margin_deg, n_branches)``.
    """

    from controller.pose import conic as _conic

    e = _conic.project_circle(centre_mm, normal, radius_mm, camera_matrix)
    poses = _conic.backproject_ellipse(e, camera_matrix, radius_mm)
    gap = 0.0
    for p in poses:
        e2 = _conic.project_circle(p.center, p.normal, radius_mm, camera_matrix)
        gap = max(gap, _ellipse_gap_px(e, e2))
    return gap, _conic.ambiguity_margin_deg(poses), len(poses)


def _ellipse_gap_px(e1, e2, n=180):
    """
    Max distance between two ellipses' boundaries, sampled.
    """

    def pts(e):
        (cx, cy), (ma, mi), ang = e
        t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        r = math.radians(ang)
        loc = np.column_stack([0.5 * ma * np.cos(t), 0.5 * mi * np.sin(t)])
        rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
        return np.array([cx, cy]) + loc @ rot.T

    p1, p2 = pts(e1), pts(e2)
    d = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


# ---------------------------------------------------------------------------
# Convenience: the whole chain, for one operating point
# ---------------------------------------------------------------------------


def budget(
    z_mm,
    radius_mm,
    camera_matrix,
    tilt_deg=30.0,
    sigma_r_px=None,
    contrast=200.0,
    sigma_noise=6.0,
    psf_px=1.0,
    correlation_px=1.0,
):
    """
    End-to-end error budget at one operating point.

        Runs (A) -> (B) -> (C) -> (D) and returns a dict of every intermediate, so a
        report can show where each factor of ten comes from rather than presenting a
        single number. ``sigma_r_px`` overrides (A) with a *measured* boundary
        scatter, which is how `validation/limits.py` demonstrates that the real error
        is nowhere near the photon bound.
    """

    photon = edge_crlb_discrete(contrast, sigma_noise, psf_px)
    sigma_r = float(sigma_r_px) if sigma_r_px is not None else photon

    f = 0.5 * (camera_matrix[0, 0] + camera_matrix[1, 1])
    theta = math.radians(tilt_deg)
    a_px = f * radius_mm / z_mm
    b_px = a_px * math.cos(theta)
    perim = math.pi * (
        3.0 * (a_px + b_px) - math.sqrt((3.0 * a_px + b_px) * (a_px + 3.0 * b_px))
    )
    n_eff = effective_point_count(perim, correlation_px)

    centre = (float(camera_matrix[0, 2]), float(camera_matrix[1, 2]))
    cov, ref = pose_crlb(
        centre, (a_px, b_px), 0.0, camera_matrix, radius_mm, n_eff, sigma_r
    )
    out = {
        "z_mm": z_mm,
        "tilt_deg": tilt_deg,
        "photon_sigma_px": photon,
        "quantisation_sigma_px": QUANTISATION_SIGMA_PX,
        "sigma_r_px": sigma_r,
        "semi_major_px": a_px,
        "semi_minor_px": b_px,
        "perimeter_px": perim,
        "n_effective": n_eff,
        "depth_lateral_ratio": depth_lateral_ratio(z_mm, radius_mm, tilt_deg),
    }
    if cov is not None:
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        out.update(
            {
                "sigma_x_mm": float(sd[0]),
                "sigma_y_mm": float(sd[1]),
                "sigma_z_mm": float(sd[2]),
                "sigma_lateral_mm": float(math.hypot(sd[0], sd[1])),
                # The normal is a unit vector, so the standard deviations of its
                # three components combine to a small-angle error directly.
                "sigma_normal_deg": float(math.degrees(np.linalg.norm(sd[3:]))),
            }
        )
    return out
