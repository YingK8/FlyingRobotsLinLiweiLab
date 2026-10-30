"""
5-DOF pose from two silhouettes' moments: centroid + minor axis per view, no model.

    The Python reference for `native/src/axis_pose.c`; `native_parity.py --stage axis`
    holds the two together. `pose/theory.md` 23 has the argument.

    Per view: the mask is ``|gray - plate| > thresh`` with the rod and other thin
    clutter removed at the disc's own scale (`mask`), and its first and second moments
    give the centroid and the minor axis -- the covariance eigenvectors are exactly the
    SVD of the centred pixel coordinates. A ring and a filled disc share both by
    symmetry, so rim and no-rim robots run the same code. No radius, no thickness, no cone.

    Across views: the minor-axis line and the camera centre span a plane holding the spin
    axis (the body's mirror plane through the lens), so the axis is the two planes'
    intersection. Which eigenvector is the axis is decided by continuity (`pose`'s
    ``ref``), not by which eigenvalue is smaller. Position is the closest approach of the
    two centroid rays, and a gap between them past `MAX_GAP_MM` refuses the pair: the two
    views did not find the same object.
"""

from __future__ import annotations

import numpy as np

THRESH = 90             # `disc_pose.DIFF_THRESH`, the plate-subtracted disc threshold
MIN_PX = 200            # guess: fewer pixels than this is not the rotor at ~140 mm
MAX_ROUND = 0.95        # guess: lambda_min/lambda_max above this has no minor axis to read
MIN_SIN = np.sin(np.radians(5.0))   # guess: planes closer than 5 deg do not intersect well
UNDISTORT_ITERS = 20    # fixed-point steps; converges to <1e-9 px on this rig's dist
UP = (0.0, -1.0, 0.0)   # the rod direction at rest (`disc_pose.MAST_UP`): world = camera A, image-up is -y
MAX_GAP_MM = 1.0        # guess: good synthetic poses close to <=0.22 mm, a view on the hub ~10 mm
ALPHA = 0.5             # core depth as a fraction of the thickest half-width; theory.md 23.3 alpha sweep
CFG = dict(min_px=MIN_PX, max_round=MAX_ROUND, min_sin=float(MIN_SIN), iters=UNDISTORT_ITERS,
           max_gap_mm=MAX_GAP_MM)


def mask(gray, plate=None, thresh=THRESH, alpha=ALPHA):
    """The disc's silhouette with the rod, hub, legs and wires removed, or None.

        The body sets the scale (`theory.md` 23.3): the distance transform's maximum ``b``
        is the half-width of its thickest part. The core ``dt > alpha*b`` loses everything
        thinner than ``alpha*b``; of its components the LARGEST is kept -- the hub can be
        thicker than an edge-on disc, never bigger -- and it is grown back by ``alpha*b``
        inside the mask, so the disc's edge stays at full resolution.

        Holes are NOT filled: a ring's moments already equal the filled disc's, and the
        fill closed rod + legs + disc into one fake blob bigger than the disc.
    """

    import cv2

    d = gray if plate is None else cv2.absdiff(gray, plate)
    m = (d > thresh).astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    r = alpha * float(dt.max())
    n, lab, st, _ = cv2.connectedComponentsWithStats((dt > r).astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    back = cv2.distanceTransform((lab != keep).astype(np.uint8), cv2.DIST_L2, 3)
    return ((back <= r) & (m > 0)).astype(np.uint8)


def moments(m):
    """``(n, cx, cy, ux, uy)`` of binary mask ``m`` -- count, centroid, unit minor axis."""

    if m is None:
        return None
    ys, xs = np.nonzero(m)
    n = len(xs)
    if n < MIN_PX:
        return None
    xs, ys = xs.astype(np.int64), ys.astype(np.int64)
    sx, sy = xs.sum(), ys.sum()
    cxx = (xs @ xs - sx * sx / n) / n
    cyy = (ys @ ys - sy * sy / n) / n
    cxy = (xs @ ys - sx * sy / n) / n
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
    if tr / 2 - disc > MAX_ROUND * (tr / 2 + disc):
        return None
    t = 0.5 * np.arctan2(2 * cxy, cxx - cyy) + np.pi / 2
    return n, sx / n, sy / n, np.cos(t), np.sin(t)


def _ray(cam, u, v):
    """World unit ray through distorted pixel ``(u, v)`` (OpenCV 5-term model)."""

    K, (k1, k2, p1, p2, k3) = cam.K, (cam.dist if cam.dist is not None else np.zeros(5))[:5]
    x0, y0 = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
    x, y = x0, y0
    for _ in range(UNDISTORT_ITERS):
        r2 = x * x + y * y
        ic = 1.0 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
        dx, dy = 2 * p1 * x * y + p2 * (r2 + 2 * x * x), p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = (x0 - dx) * ic, (y0 - dy) * ic
    d = cam.R @ np.array([x, y, 1.0])
    return d / np.linalg.norm(d)


def pose(cams, ms, ref=UP):
    """``(xyz_mm, axis, gap_mm)`` from two views' `moments`, or None.

        ``ref`` is the last frame's axis (`UP` on the first frame or after a loss). Each
        view offers both eigenvectors; of the four plane pairs, the axis nearest ``ref``
        wins and takes its sign. A rod or hub that survives `mask` can swap which
        eigenvalue is smaller, but not the eigenvectors themselves (`theory.md` 23.3).
    """

    if any(m is None for m in ms):
        return None
    ref = np.asarray(ref, float)
    rays = [_ray(c, m[1], m[2]) for c, m in zip(cams, ms)]
    planes = [[np.cross(r, _ray(c, m[1] + ux, m[2] + uy)) for ux, uy in ((m[3], m[4]), (-m[4], m[3]))]
              for c, m, r in zip(cams, ms, rays)]
    best = None
    for pa in planes[0]:
        for pb in planes[1]:
            a = np.cross(pa, pb)
            s = np.linalg.norm(a)
            if s / (np.linalg.norm(pa) * np.linalg.norm(pb)) < MIN_SIN:
                continue
            a = a / s
            if best is None or abs(a @ ref) > abs(best @ ref):
                best = a
    if best is None:
        return None
    a = -best if best @ ref < 0 else best
    (o1, o2), (d1, d2) = (cams[0].position, cams[1].position), rays
    w, b = o1 - o2, d1 @ d2
    den = 1 - b * b
    if den < 1e-12:
        return None
    s1, s2 = (b * (d2 @ w) - d1 @ w) / den, (d2 @ w - b * (d1 @ w)) / den
    p1, p2 = o1 + s1 * d1, o2 + s2 * d2
    gap = float(np.linalg.norm(p1 - p2))
    return None if gap > MAX_GAP_MM else (0.5 * (p1 + p2), a, gap)


def _render(cam, centre, axis, thick=2.0, hole=0.0, rod=(0.0, 0.0), hub=False, wire=False,
            salt=0.0, radius=5.0, size=(400, 640)):
    """Binary silhouette of a (hollow if ``hole``) disc, plus clutter like `rod_demo.png`.

        ``rod = (width, length)`` mm along the axis; ``hub`` a 2 mm blob at its end with two
        0.5 mm legs; ``wire`` a 0.3 mm wire from the rim, 30 deg off the axis -- the one
        thing here that is not symmetric about the axis. Point-splatted, then closed.
    """

    import cv2

    rng = np.random.default_rng(1)
    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    def cyl(base, d, w, L, n):
        a = np.cross(d, e1 if abs(d @ e1) < 0.9 else e2)
        a /= np.linalg.norm(a)
        q, f = np.sqrt(rng.uniform(0, 1, n)) * w / 2, rng.uniform(0, 2 * np.pi, n)
        return (base + np.outer(rng.uniform(0, L, n), d) + np.outer(q * np.cos(f), a)
                + np.outer(q * np.sin(f), np.cross(d, a)))

    N = 300_000
    r = np.sqrt(rng.uniform((hole / radius) ** 2, 1, N)) * radius
    t = rng.uniform(0, 2 * np.pi, N)
    # disc: a zero-width `cyl` gives points along the thickness, spread over the radius here
    P = [cyl(centre - thick / 2 * axis, axis, 0, thick, N)
         + np.outer(r * np.cos(t), e1) + np.outer(r * np.sin(t), e2)]
    if rod[0]:
        P.append(cyl(centre, axis, *rod, 80_000))
    if hub:
        top = centre + rod[1] * axis
        P.append(top + rng.normal(0, 0.7, (60_000, 3)))
        P += [cyl(top, np.cos(1.2) * axis + k * np.sin(1.2) * e2, 0.5, 10.0, 40_000) for k in (1, -1)]
    if wire:
        P.append(cyl(centre + radius * e1, np.cos(0.5) * axis + np.sin(0.5) * e1, 0.3, 15.0, 40_000))
    P = np.vstack(P)
    rvec, _ = cv2.Rodrigues(cam.R.T)
    uv, _ = cv2.projectPoints(P, rvec, -cam.R.T @ cam.position, cam.K, cam.dist)
    uv = np.round(uv.reshape(-1, 2)).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < size[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < size[0])
    img = np.zeros(size, np.uint8)
    img[uv[ok, 1], uv[ok, 0]] = 200
    if salt:
        img[rng.uniform(size=size) < salt] = 200
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


CASES = (   # (name, body, frames of 20 that must solve, max axis error deg; None = known failure)
    ("thin disc 0.3 mm", dict(thick=0.3), 20, 1.0),
    ("disc 2 mm", dict(), 20, 1.0),
    ("ring", dict(hole=4.0), 20, 1.0),
    ("rod 0.6x12", dict(rod=(0.6, 12.0)), 20, 1.0),
    ("rod 1.2x20", dict(rod=(1.2, 20.0)), 20, 1.0),
    ("rod + wire", dict(rod=(0.6, 12.0), wire=True), 20, 1.0),
    ("rod + wire + salt", dict(rod=(0.6, 12.0), wire=True, salt=0.002), 20, 1.0),
    # KNOWN FAILURE (theory.md 23.3): an upright robot is edge-on to both level cameras,
    # and this guessed 4 mm hub is thicker than the 2 mm disc, so both views can keep the
    # hub, agree with each other, and pass the gap gate. Printed, not asserted, until the
    # real hub's size is measured on a take.
    ("rod 1.2x12 + hub", dict(rod=(1.2, 12.0), hub=True), None, None),
)


def _poses(n=20, seed=0):
    """``(centre, axis)`` around the views' crossing (~138 mm), tilt 0-40 deg from `UP`."""

    rng, up = np.random.default_rng(seed), np.asarray(UP)
    e = np.cross(up, [1.0, 0, 0])
    out = []
    for _ in range(n):
        tilt, az = np.radians(rng.uniform(0, 40)), rng.uniform(0, 2 * np.pi)
        axis = np.cos(tilt) * up + np.sin(tilt) * (np.cos(az) * e + np.sin(az) * np.cross(up, e))
        out.append((np.array([0.0, 0.0, 138.0]) + rng.uniform(-8, 8, 3), axis))
    return out


def _self_check():
    import cv2
    from controller.calib import rig as rigmod

    cams = rigmod.StereoRig.load().cameras
    # the undistortion matches OpenCV's
    for u, v in ((10.0, 10.0), (320.0, 200.0), (630.0, 390.0)):
        ref = cv2.undistortPoints(np.array([[[u, v]]]), cams[0].K, cams[0].dist,
                                  criteria=(cv2.TERM_CRITERIA_COUNT, 100, 0)).reshape(2)
        assert np.allclose(_ray(cams[0], u, v), np.r_[ref, 1] / np.linalg.norm(np.r_[ref, 1]), atol=1e-9)
    bad = []
    for name, kw, need, worst in CASES:
        ea, ep = [], []
        for centre, axis in _poses():
            out = pose(cams, [moments(mask(_render(c, centre, axis, **kw))) for c in cams])
            if out is not None:
                ea.append(np.degrees(np.arccos(min(1.0, out[1] @ axis))))
                ep.append(np.linalg.norm(out[0] - centre))
        ea, ep = np.array(ea), np.array(ep)
        print(f"{name:18s} solved {len(ea):2d}/20  axis median {np.median(ea):.2f} max {ea.max():.2f} deg"
              f"  position median {np.median(ep):.2f} max {ep.max():.2f} mm")
        if need is not None and (len(ea) < need or ea.max() > worst or np.median(ep) > 0.2):
            bad.append(name)
    assert not bad, f"failed: {bad}"
    print("axis_pose: ok")


if __name__ == "__main__":
    _self_check()
