"""Round-trip tests for `conic.py`.

Run: uv run python controller/pose/test_conic.py

These are the gate for the whole pipeline.  Everything downstream -- the
estimator, the zeroing, the synthetic sweep -- assumes the back-projection is
exact.  If these fail, no residual measured anywhere else means anything.

Deliberately dependency-free (no pytest) to match the repo's loose-script
convention, and self-contained: ground truth is generated analytically rather
than read from a fixture.
"""

from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import conic  # noqa: E402

# The real rig's intrinsics, so the tests exercise the numbers we actually use.
K = np.array(
    [[1408.78, 0.0, 497.55], [0.0, 1407.69, 355.70], [0.0, 0.0, 1.0]], dtype=np.float64
)
RADIUS_MM = 10.204  # measured off the STL rim

# The solve is exact; these bound double-precision noise, not modelling error.
# The cone's eigenvalue spread goes as (r/z)^2 -- at r = 10 mm and z = 400 mm the
# smallest eigenvalue is ~6e-4 of the largest, so `eigh` loses a few digits and
# the recovered pose wobbles in the last ones.  Measured worst case over the
# sweep below is ~5e-11 mm and ~2e-6 deg, so these tolerances sit several orders
# above the noise and still some five orders below the 2 deg / 1 mm accuracy the
# synthetic validation actually cares about.  A real regression moves these by
# orders of magnitude, not by a factor of two.
TOL_POS_MM = 1e-6
TOL_ANG_DEG = 1e-4


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _match(poses, center, normal):
    """Best (position error mm, normal error deg) over the candidate solutions.

    The two-fold ambiguity is inherent, so a correct solver is one where *one*
    of its candidates is the truth.  Picking between them is the estimator's
    problem, tested separately.
    """
    best = (np.inf, np.inf)
    for p in poses:
        pos = float(np.linalg.norm(p.center - center))
        ang = math.degrees(math.acos(float(np.clip(p.normal @ _unit(normal), -1, 1))))
        ang = min(ang, 180.0 - ang)  # normal sign is not observable from one conic
        if pos < best[0]:
            best = (pos, ang)
    return best


def test_cone_roundtrip():
    """circle -> cone -> circle, over a spread of tilts, depths and offsets."""
    rng = np.random.default_rng(0xC01C)
    worst_pos = worst_ang = 0.0
    n_two = 0
    trials = 400

    for _ in range(trials):
        tilt = math.radians(rng.uniform(0.0, 70.0))
        az = rng.uniform(0.0, 2 * math.pi)
        normal = _unit([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
        center = np.array(
            [rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(80, 400)], dtype=np.float64
        )

        cone = conic.cone_from_circle(center, normal, RADIUS_MM)
        poses = conic.backproject(cone, RADIUS_MM)
        assert poses, f"no solution for tilt={math.degrees(tilt):.1f} center={center}"
        n_two += len(poses) == 2

        pos_err, ang_err = _match(poses, center, normal)
        worst_pos, worst_ang = max(worst_pos, pos_err), max(worst_ang, ang_err)

    assert worst_pos < TOL_POS_MM, f"position error {worst_pos:.3e} mm"
    assert worst_ang < TOL_ANG_DEG, f"normal error {worst_ang:.3e} deg"
    print(
        f"  cone round-trip     {trials} trials | worst pos {worst_pos:.2e} mm, "
        f"normal {worst_ang:.2e} deg | two solutions in {n_two}/{trials}"
    )


def test_ellipse_conic_roundtrip():
    """conic <-> ellipse conversions agree, and the conic fits its own points.

    This is what makes the OpenCV angle convention a non-issue: rather than
    reasoning about y-down axes, we check numerically that points sampled on the
    ellipse satisfy p^T C p = 0.
    """
    rng = np.random.default_rng(7)
    worst_geo = worst_res = 0.0

    for _ in range(300):
        cx, cy = rng.uniform(100, 900), rng.uniform(100, 600)
        major = rng.uniform(40, 400)
        minor = major * rng.uniform(0.15, 1.0)
        ang = rng.uniform(-180, 180)

        c = conic.conic_from_ellipse(((cx, cy), (major, minor), ang))
        (rx, ry), (rmaj, rmin), rang = conic.ellipse_from_conic(c)

        worst_geo = max(
            worst_geo, abs(rx - cx), abs(ry - cy), abs(rmaj - major), abs(rmin - minor)
        )

        # Sample the ellipse analytically and confirm it lies on the conic.
        t = np.linspace(0, 2 * math.pi, 64, endpoint=False)
        a, b, th = major / 2, minor / 2, math.radians(ang)
        xs = cx + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th)
        ys = cy + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)
        worst_res = max(worst_res, conic.residual(c, np.column_stack([xs, ys])))

    assert worst_geo < 1e-6, f"ellipse geometry drift {worst_geo:.3e} px"
    assert worst_res < 1e-9, f"points do not lie on their own conic: {worst_res:.3e}"
    print(f"  ellipse <-> conic   worst geom {worst_geo:.2e} px, point residual {worst_res:.2e}")


def test_full_pixel_pipeline():
    """The path the estimator actually walks: circle -> pixels -> ellipse -> pose.

    Points are projected through K by hand (no cv2), fitted back to a conic
    analytically, then back-projected.  Exercises `backproject_ellipse`, i.e.
    the K^T C K step, which the cone-only test skips.
    """
    rng = np.random.default_rng(99)
    worst_pos = worst_ang = 0.0

    for _ in range(200):
        tilt = math.radians(rng.uniform(0.0, 65.0))
        az = rng.uniform(0.0, 2 * math.pi)
        normal = _unit([math.sin(tilt) * math.cos(az), math.sin(tilt) * math.sin(az), math.cos(tilt)])
        center = np.array(
            [rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(100, 350)], dtype=np.float64
        )

        # Analytic image ellipse of that circle, then straight back through the
        # public ellipse entry point.
        ellipse = conic.project_circle(center, normal, RADIUS_MM, K)
        poses = conic.backproject_ellipse(ellipse, K, RADIUS_MM)
        assert poses, "pixel pipeline produced no solution"

        pos_err, ang_err = _match(poses, center, normal)
        worst_pos, worst_ang = max(worst_pos, pos_err), max(worst_ang, ang_err)

    assert worst_pos < TOL_POS_MM, f"position error {worst_pos:.3e} mm"
    assert worst_ang < TOL_ANG_DEG, f"normal error {worst_ang:.3e} deg"
    print(f"  pixel pipeline      worst pos {worst_pos:.2e} mm, normal {worst_ang:.2e} deg")


def test_ambiguity_margin_shrinks_head_on():
    """The two solutions must merge as the circle turns to face the camera."""
    margins = []
    for tilt_deg in (40.0, 20.0, 5.0, 0.5):
        t = math.radians(tilt_deg)
        normal = _unit([math.sin(t), 0.0, math.cos(t)])
        poses = conic.backproject(
            conic.cone_from_circle([0.0, 0.0, 200.0], normal, RADIUS_MM), RADIUS_MM
        )
        margins.append(conic.ambiguity_margin_deg(poses))

    assert all(
        margins[i] > margins[i + 1] for i in range(len(margins) - 1)
    ), f"margin not monotonic: {margins}"
    assert margins[-1] < 2.0, f"near head-on margin should collapse, got {margins[-1]:.2f} deg"
    print(f"  ambiguity margin    tilt 40/20/5/0.5 deg -> {[f'{m:.2f}' for m in margins]}")


def test_speed():
    """Timing headroom: the solve must be irrelevant against a 420 fps budget."""
    import time

    ellipse = conic.project_circle([5.0, -3.0, 180.0], _unit([0.3, 0.2, 1.0]), RADIUS_MM, K)
    n = 3000

    for label, tol in (("verified", 1e-6), ("fast", None)):
        conic.backproject_ellipse(ellipse, K, RADIUS_MM, verify_tol=tol)  # warm up
        t0 = time.perf_counter()
        for _ in range(n):
            conic.backproject_ellipse(ellipse, K, RADIUS_MM, verify_tol=tol)
        us = (time.perf_counter() - t0) / n * 1e6
        print(f"  solve speed         {label:8s} {us:7.1f} us/frame  ({1e6/us:8.0f} Hz)")


if __name__ == "__main__":
    print("conic.py round-trip tests")
    fail = 0
    for fn in (
        test_cone_roundtrip,
        test_ellipse_conic_roundtrip,
        test_full_pixel_pipeline,
        test_ambiguity_margin_shrinks_head_on,
        test_speed,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
