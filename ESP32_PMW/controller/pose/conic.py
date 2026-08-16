"""Pose of a circle of known radius from its image ellipse.

The robot's duct ring is a circle -- measured off ``flyingrobot_rod2.STL``
it is 20.409 mm across with a radial standard deviation of 0.108 mm, so treating
the silhouette rim as a perfect circle is good to about half a percent.

That makes the pose problem analytic.  A circle in space and the camera centre
define a cone; the image ellipse is that cone cut by the image plane.  Given the
cone and the true radius, the circle's 3-D centre and normal fall out of an
eigendecomposition.  There are always exactly two solutions -- the cone is
symmetric about its axis, so a circle tilted one way and its mirror image
produce the identical ellipse.  Picking between them is the caller's job
(``estimator.py`` uses temporal continuity).

Reference: Safaee-Rad, Tchoukanov, Smith & Benhabib, "Three-dimensional location
estimation of circular features for machine vision", IEEE T-RA 8(5), 1992.

Everything here is pure numpy on 3x3 matrices, so a solve costs a few
microseconds -- nowhere near the frame budget even at 420 fps.

Sign conventions are never assumed.  ``backproject`` enumerates all four sign
combinations and keeps the ones that both sit in front of the camera and
reproject onto the observed cone.  That is a handful of extra 3x3 products and
it makes the module immune to the convention mismatches that make this
algorithm notoriously fiddly to port.
"""

from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

# A recovered circle pose.  `normal` is a unit 3-vector in camera coordinates,
# `center` is the circle centre in the same units as the radius passed in (mm).
CirclePose = namedtuple("CirclePose", "center normal")


def cone_from_circle(center, normal, radius):
    """Analytic cone matrix ``Q`` for a circle seen from the origin.

    A point ``X`` lies on the cone iff the ray ``sX`` pierces the circle.  With
    the circle's plane written ``n . X = d`` (``d = n . C``) that ray meets the
    plane at ``P = d X / (n . X)``, and demanding ``|P - C| = r`` then clearing
    the denominator gives a quadratic form:

        d^2 |X|^2 - 2d (n.X)(C.X) + (|C|^2 - r^2)(n.X)^2 = 0

    so ``Q = d^2 I - d (n C^T + C n^T) + (|C|^2 - r^2) n n^T``.

    This is the exact inverse of `backproject` and is what the round-trip test
    checks against.  It is also how `backproject` validates its own candidates.
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / np.linalg.norm(n)
    d = float(n @ c)
    return (
        d * d * np.eye(3)
        - d * (np.outer(n, c) + np.outer(c, n))
        + (float(c @ c) - radius * radius) * np.outer(n, n)
    )


def normalise_ellipse(ellipse):
    """Reorder an OpenCV ellipse so the first axis is the major one.

    ``cv2.fitEllipse`` returns a RotatedRect whose ``width`` and ``height`` are
    in no particular order, and whose angle refers to ``width``.  So on an image
    that is a hair taller than it is wide, the reported angle jumps by 90
    degrees for no physical reason.  That is harmless inside
    `conic_from_ellipse` (which only needs "axis along the angle" and "axis
    across it"), but it wrecks anything that reads the angle as a heading --
    which is exactly what the estimator's ``psi`` channel does.

    Swapping the axes means rotating the angle by 90 degrees; the result is
    wrapped into [0, 180) since an ellipse axis has no direction.
    """
    (cx, cy), (a, b), ang = ellipse
    if b > a:
        a, b, ang = b, a, ang + 90.0
    return (float(cx), float(cy)), (float(a), float(b)), float(ang % 180.0)



def conic_from_ellipse(ellipse):
    """3x3 image conic ``C`` from an OpenCV ``fitEllipse`` result.

    Takes ``((cx, cy), (major, minor), angle_deg)`` exactly as
    ``cv2.fitEllipse`` returns it -- note those are full axis *lengths*, not
    semi-axes -- and returns ``C`` with ``p^T C p = 0`` for homogeneous image
    points ``p = (u, v, 1)`` on the ellipse.

    The rotation is applied as written, without trying to reason about OpenCV's
    y-down image axis.  Whichever way the angle turns, the resulting conic fits
    the same point set, because a sign flip on the angle is absorbed by the
    symmetric form.  `residual` exists to confirm that empirically.
    """
    (cx, cy), (major, minor), angle_deg = ellipse
    a = major / 2.0
    b = minor / 2.0
    if a <= 0 or b <= 0:
        raise ValueError(f"degenerate ellipse axes: {major}x{minor}")

    t = math.radians(angle_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    # M is the 2x2 form of the centred ellipse in image axes.
    m = rot @ np.diag([1.0 / (a * a), 1.0 / (b * b)]) @ rot.T

    ctr = np.array([cx, cy], dtype=np.float64)
    conic = np.eye(3)
    conic[:2, :2] = m
    conic[:2, 2] = -m @ ctr
    conic[2, :2] = -(m @ ctr)
    conic[2, 2] = float(ctr @ m @ ctr) - 1.0
    return conic


def ellipse_from_conic(conic):
    """Inverse of `conic_from_ellipse`: ``C`` -> ``((cx,cy),(major,minor),deg)``.

    Used to draw the predicted ellipse for a hypothesised pose, to compare a
    rendered silhouette against the analytic ground-truth projection, and -- the
    reason it is written out longhand rather than calling numpy -- inside the
    stereo refinement's residual, which evaluates it a few hundred times a frame
    at 420 Hz.

    Everything here is 2x2, where `np.linalg.solve` and `np.linalg.eigh` spend
    almost all their time in dispatch rather than arithmetic.  The closed forms
    are the standard ones: the centre solves ``M c = -b``, and a symmetric
    ``[[a, b], [b, c]]`` has eigenvalues ``(a+c)/2 +- sqrt(((a-c)/2)^2 + b^2)``
    with eigenvector ``(b, l - a)``.  `test_conic.test_ellipse_conic_roundtrip`
    is what keeps this honest against the algebra it replaced.
    """
    a, b, c = float(conic[0, 0]), float(conic[0, 1]), float(conic[1, 1])
    bx, by = float(conic[0, 2]), float(conic[1, 2])

    det = a * c - b * b
    if abs(det) < 1e-300:
        raise ValueError("degenerate conic (singular quadratic part)")
    # M c = -[bx, by], by Cramer's rule.
    cx = (-bx * c + by * b) / det
    cy = (-by * a + bx * b) / det

    # Value of the quadratic form at the centre; the ellipse is the level set
    # where the centred form equals -k.
    k = (a * cx * cx + 2.0 * b * cx * cy + c * cy * cy) + 2.0 * (bx * cx + by * cy) + conic[2, 2]
    if abs(k) < 1e-15:
        raise ValueError("degenerate conic (zero scale)")

    s = -1.0 / k
    qa, qb, qc = a * s, b * s, c * s
    half = 0.5 * (qa + qc)
    disc = math.sqrt(max(0.0, (0.5 * (qa - qc)) ** 2 + qb * qb))
    l_hi, l_lo = half + disc, half - disc
    if l_lo <= 0.0:
        raise ValueError("conic is not a real ellipse")

    # Smaller eigenvalue -> longer semi-axis, so the major axis is along the
    # eigenvector of `l_lo`.
    if abs(qb) > 1e-300:
        vx, vy = qb, l_lo - qa
    else:
        # Already axis-aligned; `(b, l-a)` degenerates, so read the axis off
        # which diagonal entry the eigenvalue came from.
        vx, vy = (1.0, 0.0) if qa <= qc else (0.0, 1.0)

    return (
        (cx, cy),
        (2.0 / math.sqrt(l_lo), 2.0 / math.sqrt(l_hi)),
        math.degrees(math.atan2(vy, vx)),
    )


def residual(conic, points):
    """RMS of ``p^T C p`` over image points, normalised by the conic scale.

    A sanity probe, not part of the solve: it is how the tests confirm that a
    conic built from `conic_from_ellipse` really does pass through the contour
    it came from, without anyone having to be right about angle conventions.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    vals = np.einsum("ij,jk,ik->i", ph, conic, ph)
    scale = np.abs(conic).max()
    return float(np.sqrt(np.mean(vals**2)) / scale)


def _normalise_cone(cone):
    """Scale/flip ``Q`` to the canonical (+, +, -) eigenvalue signature.

    Returns ``(eigenvalues, eigenvectors)`` ordered ``l1 >= l2 > 0 > l3``, or
    ``None`` if the quadratic form is not a real elliptic cone (which happens on
    a degenerate or near-line contour).
    """
    q = 0.5 * (cone + cone.T)  # enforce symmetry against accumulated round-off
    scale = np.abs(q).max()
    if not np.isfinite(scale) or scale < 1e-300:
        return None
    q = q / scale

    w, v = np.linalg.eigh(q)
    if np.count_nonzero(w > 0) == 1:  # signature is (-, -, +): flip it
        q, w, v = -q, -w[::-1], v[:, ::-1]
    if np.count_nonzero(w > 0) != 2:
        return None

    pos = np.where(w > 0)[0]
    neg = np.where(w < 0)[0]
    if len(neg) != 1:
        return None
    # l1 >= l2 among the positive pair, l3 the lone negative one.
    pos = pos[np.argsort(w[pos])[::-1]]
    idx = np.array([pos[0], pos[1], neg[0]])
    return w[idx], v[:, idx]


def backproject(cone, radius, verify_tol=1e-6):
    """Recover the two circle poses consistent with a cone.

    ``cone`` is ``Q`` in normalised camera coordinates -- either straight from
    `cone_from_circle`, or ``K^T C K`` for an image conic ``C`` (see
    `backproject_ellipse`).  ``radius`` sets the scale, and its units are the
    units of the returned centre.

    Returns a list of `CirclePose`, normally length 2.  Both are geometrically
    valid and produce the identical image; disambiguation needs outside
    information.  A shorter list means the conic was degenerate or the circle
    straddles the camera plane, and the caller should treat the frame as lost.

    Rather than committing to one published sign convention, all four
    ``(s1, s2)`` branches are generated and then filtered on two objective
    tests: the centre must be in front of the camera, and rebuilding the cone
    from the candidate must reproduce ``Q``.  ``verify_tol`` is that relative
    reprojection tolerance; set it to ``None`` to skip the check once you trust
    the inputs (it is roughly a third of the cost, and irrelevant next to the
    segmentation that precedes it).
    """
    decomposed = _normalise_cone(cone)
    if decomposed is None:
        return []
    (l1, l2, l3), v = decomposed

    # A quadric cone has exactly two families of circular cross-section, and
    # their plane normals lie in the span of the extreme eigenvectors (l1 and
    # l3) -- the middle eigenvalue l2 fixes where between them.  In the
    # eigenframe those normals are (+-h, 0, +-g).  Guard the roots against
    # ordering round-off when l1 and l2 collide on a near-circular image.
    denom = l1 - l3
    if denom <= 0:
        return []
    h = math.sqrt(max(0.0, (l1 - l2) / denom))
    g = math.sqrt(max(0.0, (l2 - l3) / denom))
    lam = np.array([l1, l2, l3])

    out = []
    # Two branches, not four: (h, 0, g) and (-h, 0, -g) describe the same plane
    # with the normal flipped, so they yield the identical circle.
    for sz in (+1.0, -1.0):
        n_e = np.array([h, 0.0, sz * g])
        nn = np.linalg.norm(n_e)
        if nn < 1e-12:
            continue
        n_e = n_e / nn

        c_e = _circle_on_cone(lam, n_e, radius)
        if c_e is None:
            continue

        n = v @ n_e
        c = v @ c_e
        if c[2] < 0:  # mirrored copy behind the camera; take the near one
            c = -c
        if c[2] <= 0:
            continue
        if n @ c > 0:  # orient normals toward the camera so the two differ in tilt
            n = -n

        if verify_tol is not None:
            rebuilt = cone_from_circle(c, n, radius)
            a = cone / np.abs(cone).max()
            b = rebuilt / np.abs(rebuilt).max()
            if min(np.abs(a - b).max(), np.abs(a + b).max()) > verify_tol:
                continue
        out.append(CirclePose(c, n))

    return _dedupe(out)


def _circle_on_cone(lam, n_e, radius, isotropy_tol=1e-6):
    """Centre of the radius-``radius`` circle cut from a cone by normal ``n_e``.

    Works in the cone's eigenframe, where the quadric is ``diag(lam)``.  Slice
    with the plane ``n.X = d`` and write points as ``X = d n + a u + b v`` for an
    orthonormal ``u, v`` spanning the plane.  The quadratic form becomes

        (u'Lu) a^2 + 2(u'Lv) ab + (v'Lv) b^2 + 2d[(n'Lu) a + (n'Lv) b] + d^2 n'Ln = 0

    which is a circle exactly when ``u'Lu == v'Lv`` and ``u'Lv == 0`` -- checked
    here, and the check is what confirms ``n_e`` really is a circular-section
    normal rather than an artefact of eigenvalue ordering.  Completing the
    square then gives centre and radius, both linear in ``d``, so ``d`` follows
    directly from the wanted radius.

    Deriving the centre this way rather than quoting a published closed form
    keeps the module free of sign conventions that are easy to transcribe wrong.
    """
    # The quadric is diagonal here, so `diag(lam) @ x` is just `lam * x`.
    # Any two vectors completing n_e to an orthonormal frame.
    seed = np.array([0.0, 1.0, 0.0]) if abs(n_e[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n_e, seed)
    u /= np.linalg.norm(u)
    w = np.cross(n_e, u)

    lu, lw, ln = lam * u, lam * w, lam * n_e
    a_uu = float(u @ lu)
    a_ww = float(w @ lw)
    a_uw = float(u @ lw)
    scale = max(abs(a_uu), abs(a_ww), 1e-300)
    if abs(a_uu - a_ww) / scale > isotropy_tol or abs(a_uw) / scale > isotropy_tol:
        return None  # slice is a genuine ellipse, not a circle
    if abs(a_uu) < 1e-300:
        return None

    p_u = float(ln @ u)
    p_w = float(ln @ w)
    p_n = float(ln @ n_e)

    # Radius at d = 1, from completing the square.
    r1_sq = (p_u * p_u + p_w * p_w) / (a_uu * a_uu) - p_n / a_uu
    if r1_sq <= 0:
        return None
    d = radius / math.sqrt(r1_sq)

    return d * n_e - (d * p_u / a_uu) * u - (d * p_w / a_uu) * w


def _dedupe(poses, tol=1e-9):
    """Drop duplicate branches (they collide when the circle is head-on)."""
    kept = []
    for p in poses:
        if not any(
            np.allclose(p.center, q.center, atol=tol) and np.allclose(p.normal, q.normal, atol=tol)
            for q in kept
        ):
            kept.append(p)
    return kept


def backproject_ellipse(ellipse, camera_matrix, radius, verify_tol=1e-6):
    """`backproject` starting from an image ellipse instead of a cone.

    ``p = K X`` up to scale, so ``p^T C p = 0`` becomes ``X^T (K^T C K) X = 0``
    and the cone in normalised coordinates is simply ``K^T C K``.

    The ellipse must already be in ideal-pinhole pixels -- undistort the contour
    before fitting, which is what `estimator.py` does.
    """
    conic = conic_from_ellipse(ellipse)
    cone = camera_matrix.T @ conic @ camera_matrix
    return backproject(cone, radius, verify_tol=verify_tol)


def ambiguity_margin_deg(poses):
    """Angle between the two candidate normals, in degrees.

    Near zero the two solutions have merged (the circle is close to head-on) and
    the choice barely matters; when it is large a wrong pick is a large error.
    Logged every frame so ambiguity failures are visible in the CSV rather than
    hidden inside an averaged residual.
    """
    if len(poses) < 2:
        return 0.0
    d = float(np.clip(poses[0].normal @ poses[1].normal, -1.0, 1.0))
    return math.degrees(math.acos(d))


def project_circle(center, normal, radius, camera_matrix):
    """Forward model: circle -> image ellipse. Handy for overlays and tests."""
    cone = cone_from_circle(center, normal, radius)
    kinv = np.linalg.inv(camera_matrix)
    return ellipse_from_conic(kinv.T @ cone @ kinv)


def fit_conic_weighted(pts, weights=None):
    """Direct least-squares ellipse fit with per-point weights.

    Halir & Flusser's numerically stable form of Fitzgibbon's direct fit: the
    design matrix is split into its quadratic and linear halves so the
    generalised eigenproblem stays well conditioned, and the ``4ac - b^2 = 1``
    constraint guarantees the result is an ellipse rather than a hyperbola --
    which matters here, because a weighted fit on a short arc can otherwise run
    away to an unbounded conic.

    ``weights`` enters as ``D' W D`` on the scatter matrices, which is the whole
    reason this exists: `cv2.fitEllipseDirect` has no weighted form, and the
    alternative -- duplicating points in proportion to weight -- quantises the
    weighting and inflates the point count.

    Returns the 3x3 conic matrix, or ``None`` if the fit degenerates.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 5:
        return None

    # Work in a centred, unit-scaled frame. Raw pixel coordinates put x^2 terms
    # in the 10^5 range against 1s in the constant column, and the eigenproblem
    # loses most of its precision to that spread.
    mu = pts.mean(axis=0)
    scale = np.sqrt((np.sum((pts - mu) ** 2, axis=1)).mean())
    if not np.isfinite(scale) or scale < 1e-12:
        return None
    q = (pts - mu) / scale
    x, y = q[:, 0], q[:, 1]

    w = np.ones(len(q)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(q),) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        return None
    w = w / w.max()

    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones_like(x)])
    wd1, wd2 = d1 * w[:, None], d2 * w[:, None]

    s1, s2, s3 = d1.T @ wd1, d1.T @ wd2, d2.T @ wd2
    try:
        t = -np.linalg.solve(s3, s2.T)
    except np.linalg.LinAlgError:
        return None
    m = s1 + s2 @ t
    # Multiply by inv(C1) for the 4ac - b^2 constraint, written out.
    m = np.array([m[2] / 2.0, -m[1], m[0] / 2.0])

    evals, evecs = np.linalg.eig(m)
    cond = 4.0 * evecs[0] * evecs[2] - evecs[1] ** 2
    valid = np.nonzero(np.isfinite(cond) & (cond > 0))[0]
    if len(valid) == 0:
        return None
    a1 = np.real(evecs[:, valid[0]])
    a, b, c = a1
    d, e, f = np.real(t @ a1)

    # Undo the normalising similarity: x = (X - mu)/s.
    s, mx, my = scale, mu[0], mu[1]
    conic = np.array([
        [a / s**2, b / (2 * s**2), (d / s - (2 * a * mx + b * my) / s**2) / 2.0],
        [b / (2 * s**2), c / s**2, (e / s - (b * mx + 2 * c * my) / s**2) / 2.0],
        [0.0, 0.0, 0.0],
    ])
    conic[2, 0], conic[2, 1] = conic[0, 2], conic[1, 2]
    conic[2, 2] = (a * mx**2 + b * mx * my + c * my**2) / s**2 \
        - (d * mx + e * my) / s + f
    return conic if np.all(np.isfinite(conic)) else None
