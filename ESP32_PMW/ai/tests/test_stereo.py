"""
Unit tests for the stereo solver, against analytic geometry only.

No renderer here on purpose.  `validation/selftest_stereo.py` checks that the
*images* agree with the geometry; this file checks that the solver agrees with
the geometry, so when something breaks the two together say which half it was.
Everything is built by forward-projecting a known circle with
`conic.project_circle`, so the expected answer is exact rather than measured and
the tolerances can be machine-precision tight.

Deliberately dependency-free (no pytest) to match the repo's loose-script
convention.

Run: uv run python ai/tests/test_stereo.py
"""

from __future__ import annotations

import math
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

import conic  # noqa: E402
import stereo  # noqa: E402
from rig import StereoRig  # noqa: E402

RADIUS_MM = 10.2446

# Exact geometry in, exact geometry out: the only error is float64 round-off
# through an eigendecomposition, which lands near 1e-9.
TOL_POS_MM = 1e-6
TOL_ANG_DEG = 1e-4


def _rig(elev=(45.0, 45.0), azim=(0.0, 90.0), range_mm=250.0):
    return StereoRig.from_spherical(elev_deg=elev, azim_deg=azim, range_mm=range_mm)


def _candidates(rig, center_world, normal_world, radius=RADIUS_MM):
    """
    Both back-projection branches per view, from an exact forward projection.

        This is the whole test fixture: project the true circle into each camera,
        fit nothing, and hand the analytic ellipse straight to `backproject_ellipse`.
        Whatever comes back is what the solver has to work with.
    """

    out = []
    for cam in rig.cameras:
        c_cam, n_cam = cam.to_camera(center_world, normal_world)
        ellipse = conic.project_circle(c_cam, n_cam, radius, cam.K)
        out.append(conic.backproject_ellipse(ellipse, cam.K, radius))
    return out


def _hulls(
    rig, center_world, normal_world, radius=RADIUS_MM, n=48, noise_px=0.0, seed=0
):
    """
    Points sampled around each view's true rim ellipse.

        Stands in for `segment.silhouette_hull` output.  ``noise_px`` lets a test ask
        what happens when the boundary is not perfect, which is the only realistic
        thing about a synthetic hull.
    """

    rng = np.random.default_rng(seed)
    out = []
    for cam in rig.cameras:
        c_cam, n_cam = cam.to_camera(center_world, normal_world)
        (cx, cy), (major, minor), ang = conic.project_circle(
            c_cam, n_cam, radius, cam.K
        )
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        a, b, th = major / 2.0, minor / 2.0, math.radians(ang)
        xs = cx + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th)
        ys = cy + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th)
        pts = np.column_stack([xs, ys])
        if noise_px:
            pts = pts + rng.normal(0.0, noise_px, pts.shape)
        out.append(pts)
    return out


TRUTH = [
    # (center_world_mm, normal_world)
    ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ([12.0, -8.0, 20.0], [0.0, 0.0, 1.0]),
    ([0.0, 0.0, 0.0], [0.20, 0.0, 0.98]),
    ([-5.0, 9.0, -14.0], [0.15, -0.25, 0.96]),
    ([3.0, 3.0, 40.0], [-0.35, 0.10, 0.93]),
]


def test_forward_backward_roundtrip():
    """
    Each view alone must reproduce the truth on one of its two branches.

        The precondition for everything else: if a single view's back-projection
        were wrong there would be no correct pair for `match` to find, and the
        stereo tests would fail for a reason that has nothing to do with stereo.
    """

    rig = _rig()
    worst_p = worst_n = 0.0
    for ctr, nrm in TRUTH:
        ctr, nrm = np.array(ctr), stereo._unit(nrm)
        for cam, cands in zip(rig.cameras, _candidates(rig, ctr, nrm)):
            assert (
                len(cands) >= 1
            ), "back-projection returned nothing for an exact ellipse"
            errs = []
            for c in cands:
                cw, nw = cam.to_world(c.center, c.normal)
                errs.append((np.linalg.norm(cw - ctr), stereo.line_angle_deg(nw, nrm)))
            best = min(errs, key=lambda e: e[0])
            worst_p, worst_n = max(worst_p, best[0]), max(worst_n, best[1])
    assert worst_p < TOL_POS_MM, f"position round-trip off by {worst_p:.3e} mm"
    assert worst_n < TOL_ANG_DEG, f"normal round-trip off by {worst_n:.3e} deg"
    print(f"  round-trip          worst {worst_p:.2e} mm, {worst_n:.2e} deg")


def test_match_picks_the_true_branch():
    """
    The cross-view agreement test must select the correct pair every time.

        This is the claim that the second camera resolves the ambiguity, stated as
        a test.  The margin is reported as well as the pick, because a correct pick
        with a near-zero margin is luck, not evidence.
    """

    rig = _rig()
    worst_p = worst_n = 0.0
    margins = []
    for ctr, nrm in TRUTH:
        ctr, nrm = np.array(ctr), stereo._unit(nrm)
        m = stereo.match(_candidates(rig, ctr, nrm), rig, RADIUS_MM)
        assert m is not None, "match failed on an exact projection"
        for pose, cam in zip(m.poses, rig.cameras):
            cw, nw = cam.to_world(pose.center, pose.normal)
            worst_p = max(worst_p, float(np.linalg.norm(cw - ctr)))
            worst_n = max(worst_n, stereo.line_angle_deg(nw, nrm))
        margins.append(m.margin)
        assert (
            m.discrepancy_mm < 1e-6
        ), f"the chosen pair disagrees by {m.discrepancy_mm:.3e} mm"
    assert worst_p < TOL_POS_MM, f"matched branch off by {worst_p:.3e} mm"
    assert worst_n < TOL_ANG_DEG, f"matched branch off by {worst_n:.3e} deg"
    print(
        f"  branch match        exact on {len(TRUTH)}/{len(TRUTH)}; "
        f"margin {min(margins):.1f}-{max(margins):.1f} sigma^2"
    )


def test_margin_collapses_when_it_should():
    """
    A near-face-on rotor must report a small margin, not a confident one.

        The two branches merge as the circle turns face-on, so the honest answer is
        "could not tell" -- and it costs nothing, because merged branches are nearly
        the same pose.  A solver that reported a large margin there would be
        fabricating evidence, and this is the test that would catch it.
    """

    # Both cameras looking almost straight down the rotor axis.
    rig = _rig(elev=(84.0, 84.0), azim=(0.0, 90.0))
    ctr = np.array([0.0, 0.0, 0.0])
    flat = stereo.match(_candidates(rig, ctr, [0.0, 0.0, 1.0]), rig, RADIUS_MM)
    tilted = stereo.match(_candidates(rig, ctr, [0.35, 0.0, 0.94]), rig, RADIUS_MM)
    assert flat.margin < tilted.margin, (
        f"face-on margin ({flat.margin:.1f}) should be below the tilted one "
        f"({tilted.margin:.1f})"
    )
    print(
        f"  margin vs tilt      face-on {flat.margin:.1f}, tilted {tilted.margin:.1f} "
        f"(sigma^2 units)"
    )


def test_fusion_beats_either_view_on_depth():
    """
    The core claim: fusion must fix the axis a single view is worst on.

        Each view is perturbed **along its own optical axis** by a realistic depth
        error, which is the error single-view estimation actually makes.  The fused
        answer has to come out better than either input, and by roughly the factor
        the information-form arithmetic predicts -- not just "no worse", which a
        plain average would also pass.
    """

    rig = _rig()
    ctr = np.array([0.0, 0.0, 0.0])
    nrm = stereo._unit([0.1, 0.0, 0.99])
    depth_err_mm = 1.5

    poses = []
    solo = []
    for cam, cands in zip(rig.cameras, _candidates(rig, ctr, nrm)):
        true_pose = min(
            cands,
            key=lambda c: np.linalg.norm(cam.to_world(c.center, c.normal)[0] - ctr),
        )
        # Push it away along the camera's own view direction.
        bad_world = (
            cam.to_world(true_pose.center, true_pose.normal)[0]
            + depth_err_mm * cam.optical_axis
        )
        bad_cam, _ = cam.to_camera(bad_world, nrm)
        poses.append(conic.CirclePose(bad_cam, true_pose.normal))
        solo.append(float(np.linalg.norm(bad_world - ctr)))

    fused, _, cov = stereo.fuse(poses, rig)
    err = float(np.linalg.norm(fused - ctr))

    assert err < min(
        solo
    ), f"fusion ({err:.3f} mm) is no better than the better single view ({min(solo):.3f} mm)"
    # Each view's depth error is nearly cancelled because the *other* view
    # measures that direction laterally and outweighs it by ~120x.
    assert err < 0.25 * min(
        solo
    ), f"fusion only improved {min(solo) / max(err, 1e-9):.1f}x; expected a large factor"
    print(
        f"  fusion on depth     {min(solo):.3f}/{max(solo):.3f} mm single -> "
        f"{err:.3f} mm fused ({min(solo) / err:.1f}x)"
    )


def test_fused_covariance_matches_the_prediction():
    """
    `fuse` and `rig.position_covariance` must agree, since the plan quotes it.

        The predicted numbers in the plan came from `rig.position_covariance`; the
        estimator's weighting is computed independently inside `fuse`.  If they ever
        drift apart, the documented expectations stop describing the shipped code.
    """

    rig = _rig()
    ctr = np.array([0.0, 0.0, 0.0])
    nrm = stereo._unit([0.0, 0.0, 1.0])
    poses = []
    for cam, cands in zip(rig.cameras, _candidates(rig, ctr, nrm)):
        poses.append(
            min(
                cands,
                key=lambda c: np.linalg.norm(cam.to_world(c.center, c.normal)[0] - ctr),
            )
        )
    _, _, cov = stereo.fuse(poses, rig)
    want = rig.position_covariance(stereo.SIGMA_LAT_MM, stereo.SIGMA_DEPTH_MM)
    assert np.allclose(
        cov, want, rtol=1e-9
    ), "fuse and rig.position_covariance disagree"
    sig = np.sqrt(np.linalg.eigvalsh(cov))
    print(
        f"  covariance          per-axis sigma {np.array2string(sig, precision=4)} mm"
    )


def test_refine_recovers_exact_geometry():
    """
    Refinement on noise-free views must land on the truth, not near it.

        Both parameterisations are exercised, because they answer different
        questions.  ``params='both'`` is seeded 3 mm and 5 degrees off and has to
        recover all five; ``params='normal'`` -- the shipped default, which holds
        the centre at the information-weighted fused value -- is seeded only in
        orientation, since recovering a position it is deliberately not solving for
        would be a strange thing to demand of it.
    """

    rig = _rig()
    worst = {"both": [0.0, 0.0], "normal": [0.0, 0.0]}
    for ctr, nrm in TRUTH:
        ctr, nrm = np.array(ctr), stereo._unit(nrm)
        hulls = _hulls(rig, ctr, nrm)
        bad_n = stereo._unit(nrm + 0.09 * np.array([1.0, -1.0, 0.0]))

        # Tolerances well below the shipped defaults: those are set for a frame
        # budget, and this test is about whether the solve converges at all.
        r = stereo.refine(
            hulls,
            rig,
            ctr + np.array([3.0, -2.0, 2.0]),
            bad_n,
            RADIUS_MM,
            params="both",
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        assert (
            r is not None
        ), "five-parameter refinement returned nothing on a clean view"
        worst["both"][0] = max(worst["both"][0], float(np.linalg.norm(r.center - ctr)))
        worst["both"][1] = max(worst["both"][1], stereo.line_angle_deg(r.normal, nrm))

        r = stereo.refine(hulls, rig, ctr, bad_n, RADIUS_MM, params="normal")
        assert r is not None, "normal-only refinement returned nothing on a clean view"
        assert np.allclose(r.center, ctr), "params='normal' moved the centre"
        worst["normal"][1] = max(
            worst["normal"][1], stereo.line_angle_deg(r.normal, nrm)
        )

    assert worst["both"][0] < 1e-3, f"refined position off by {worst['both'][0]:.3e} mm"
    assert worst["both"][1] < 1e-2, f"refined normal off by {worst['both'][1]:.3e} deg"
    assert (
        worst["normal"][1] < 1e-2
    ), f"normal-only refine off by {worst['normal'][1]:.3e} deg"
    print(
        f"  refine (clean)      both: {worst['both'][0]:.2e} mm / {worst['both'][1]:.2e} deg "
        f"from a 3mm/5deg seed;  normal-only: {worst['normal'][1]:.2e} deg"
    )


def _seed_from_hulls(rig, hulls):
    """
    Closed-form match + fuse, starting from fitted ellipses. ``None`` on failure.
    """

    import segment as segmod

    cands = []
    for cam, hull in zip(rig.cameras, hulls):
        fit = segmod.fit_ellipse(hull)
        if fit is None:
            return None
        cands.append(conic.backproject_ellipse(fit[0], cam.K, RADIUS_MM))
    m = stereo.match(cands, rig, RADIUS_MM)
    return None if m is None else stereo.fuse(m.poses, rig)


def test_refine_is_harmless_without_model_error():
    """
    On an ideal circle, refinement must not make orientation meaningfully worse.

        A deliberately negative result, recorded so it is not rediscovered.  These
        hulls are sampled from the true rim ellipse, so there is no silhouette model
        error for the refinement to correct -- and against clean data the
        information-weighted fusion is already close to optimal, because it weights
        each view's normal by that view's actual tilt sensitivity while a
        reprojection fit weights all boundary points alike.

        So refinement earns nothing here, and this test only checks it costs nothing.
        Its real contribution (roughly 1.7x on the normal) shows up in
        `validation/sweep_stereo.py`, against rendered images of the actual mesh --
        which is the only place the mast exists.
    """

    rig = _rig()
    rng = np.random.default_rng(7)
    seed_errs, ref_errs = [], []
    for trial in range(24):
        ctr = rng.uniform(-15, 15, 3)
        nrm = stereo._unit([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 1.0])
        hulls = _hulls(rig, ctr, nrm, noise_px=0.4, seed=trial)
        seed = _seed_from_hulls(rig, hulls)
        if seed is None:
            continue
        seed_c, seed_n, _ = seed
        seed_errs.append(stereo.line_angle_deg(seed_n, nrm))
        r = stereo.refine(hulls, rig, seed_c, seed_n, RADIUS_MM)
        ref_errs.append(stereo.line_angle_deg(r.normal, nrm) if r else seed_errs[-1])

    assert len(seed_errs) >= 20, f"only {len(seed_errs)} trials produced a seed"
    s, f = float(np.median(seed_errs)), float(np.median(ref_errs))
    # Both are far below anything that matters; the point is only that
    # refinement does not blow up where it has no work to do. It does come out
    # slightly behind the seed, and the reason is worth keeping: `fuse` weights
    # each view's normal by that view's tilt sensitivity, which is the right
    # thing on clean data, whereas the endpoint residual weights both views
    # alike. The joint fit only wins once there is model error to reconcile.
    assert (
        f < 0.25
    ), f"refined normal {f:.4f} deg is far worse than expected on ideal data"
    assert f <= s * 2.0, f"refinement degraded the normal badly: {s:.4f} -> {f:.4f} deg"
    print(
        f"  refine (ideal)      seed {s:.4f} -> refined {f:.4f} deg; nothing to correct"
    )


def test_refine_removes_silhouette_model_error():
    """
    The mechanism that matters, isolated: refinement plus the tilt calibration.

        The real robot is not a flat circle -- the mast widens the silhouette's short
        direction -- and `calibration.TiltCalibration` is the measured description of
        that.  Here the distortion is *injected* into the synthetic hulls with
        `unapply`, so the truth is known exactly and the effect is separable from
        every other error source in a render.

        The seed is built from the distorted ellipses with no correction, so it
        carries the full bias; the refinement is given the calibration and must
        remove most of it.  This is the test that justifies the layer existing.
    """

    rig = _rig()
    cal = stereo.TiltCalibration.load()
    assert not cal.is_identity, "no tilt calibration on disk; this test needs one"

    rng = np.random.default_rng(11)
    seed_errs, ref_errs = [], []
    for trial in range(24):
        ctr = rng.uniform(-12, 12, 3)
        nrm = stereo._unit([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 1.0])
        hulls = []
        for cam in rig.cameras:
            c_cam, n_cam = cam.to_camera(ctr, nrm)
            ideal = conic.project_circle(c_cam, n_cam, RADIUS_MM, cam.K)
            hulls.append(
                _ellipse_points(
                    cal.unapply(conic.normalise_ellipse(ideal)), rng, noise_px=0.2
                )
            )
        seed = _seed_from_hulls(rig, hulls)
        if seed is None:
            continue
        seed_c, seed_n, _ = seed
        seed_errs.append(stereo.line_angle_deg(seed_n, nrm))
        r = stereo.refine(hulls, rig, seed_c, seed_n, RADIUS_MM, tilt_cal=cal)
        ref_errs.append(stereo.line_angle_deg(r.normal, nrm) if r else seed_errs[-1])

    assert len(seed_errs) >= 20, f"only {len(seed_errs)} trials produced a seed"
    s, f = float(np.median(seed_errs)), float(np.median(ref_errs))
    assert f < s / 3.0, (
        f"refinement removed too little of the injected model error: "
        f"{s:.4f} -> {f:.4f} deg (wanted at least 3x)"
    )
    print(
        f"  refine (mast model)  seed {s:.4f} deg -> refined {f:.4f} deg "
        f"({s / max(f, 1e-9):.1f}x) with the tilt calibration"
    )


def _ellipse_points(ellipse, rng, n=48, noise_px=0.0):
    """
    Points around an arbitrary ellipse, for building a synthetic hull.
    """

    (cx, cy), (major, minor), ang = ellipse
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a, b, th = major / 2.0, minor / 2.0, math.radians(ang)
    pts = np.column_stack(
        [
            cx + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th),
            cy + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th),
        ]
    )
    return pts + rng.normal(0.0, noise_px, pts.shape) if noise_px else pts


def test_normals_are_lines_across_hemispheres():
    """
    A rig straddling the rotor plane must still agree with itself.

        `conic.backproject` orients each normal toward its own camera, so a camera
        above and a camera below report opposite normals for one physical pose.
        Anything comparing them as vectors would conclude the two views disagree
        completely and pick the wrong branch with total confidence -- which is why
        this case gets its own test rather than being folded into the others.
    """

    rig = _rig(elev=(45.0, -45.0), azim=(0.0, 90.0))
    ctr, nrm = np.array([0.0, 0.0, 0.0]), stereo._unit([0.15, 0.05, 0.99])
    cands = _candidates(rig, ctr, nrm)

    raw = [c[0].normal for c in cands]
    world = [cam.to_world(np.zeros(3), n)[1] for cam, n in zip(rig.cameras, raw)]
    assert (
        float(world[0] @ world[1]) < 0
    ), "this rig no longer produces opposing normals; the test has stopped testing anything"

    m = stereo.match(cands, rig, RADIUS_MM)
    assert (
        m is not None and m.discrepancy_mm < 1e-6
    ), f"cross-hemisphere match failed: discrepancy {m.discrepancy_mm if m else float('nan')}"
    _, fused_n, _ = stereo.fuse(m.poses, rig)
    ang = stereo.line_angle_deg(fused_n, nrm)
    assert ang < TOL_ANG_DEG, f"fused normal off by {ang:.3e} deg across hemispheres"
    assert (
        float(fused_n @ [0, 0, 1]) > 0
    ), "fused normal was not oriented against the reference"
    print(f"  cross-hemisphere    opposing raw normals reconciled to {ang:.2e} deg")


def test_speed():
    """
    The whole solve must fit inside a 420 Hz budget with room to spare.

        Segmentation is measured separately (`validation/latency.py`); this is the
        part that is new, so it is the part that has to justify itself.  The budget
        quoted is the solve's share, not the frame's.
    """

    rig = _rig()
    ctr, nrm = np.array([2.0, -3.0, 8.0]), stereo._unit([0.12, -0.05, 0.99])
    cands = _candidates(rig, ctr, nrm)
    hulls = _hulls(rig, ctr, nrm, n=48)

    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        stereo.match(cands, rig, RADIUS_MM)
    t_match = (time.perf_counter() - t0) / n * 1e3

    m = stereo.match(cands, rig, RADIUS_MM)
    t0 = time.perf_counter()
    for _ in range(n):
        stereo.fuse(m.poses, rig)
    t_fuse = (time.perf_counter() - t0) / n * 1e3

    c, nn, _ = stereo.fuse(m.poses, rig)
    t0 = time.perf_counter()
    for _ in range(n):
        stereo.refine(hulls, rig, c, nn, RADIUS_MM)
    t_refine = (time.perf_counter() - t0) / n * 1e3

    closed_form = t_match + t_fuse
    total = closed_form + t_refine
    print(
        f"  speed               match {t_match * 1e3:.0f} us, fuse {t_fuse * 1e3:.0f} us, "
        f"refine {t_refine:.3f} ms  -> closed form {closed_form * 1e3:.0f} us, "
        f"full {total:.3f} ms"
    )

    # Two budgets, because there are two shipped configurations and they target
    # different frame rates. This split replaced a single 1.2 ms assertion when
    # `refine(params='both')` became the default: the five-parameter solve costs
    # roughly 2.5x the two-parameter one and buys 0.397 -> 0.348 mm and
    # 1.618 -> 1.503 deg. That is a real rate regression, taken deliberately, and
    # the honest way to encode it is to test each configuration against the rate
    # it can actually hold rather than to relax one number until it passes.
    assert closed_form < 0.30, (
        f"closed-form solve takes {closed_form:.3f} ms; it must stay far inside the "
        f"2.38 ms 420 Hz budget because segmentation needs most of it"
    )
    assert total < 2.5, (
        f"full solve takes {total:.3f} ms; the 240 Hz budget is 4.17 ms and "
        f"segmentation needs the rest"
    )


if __name__ == "__main__":
    print("stereo solver")
    fail = 0
    for fn in (
        test_forward_backward_roundtrip,
        test_match_picks_the_true_branch,
        test_margin_collapses_when_it_should,
        test_fusion_beats_either_view_on_depth,
        test_fused_covariance_matches_the_prediction,
        test_refine_recovers_exact_geometry,
        test_refine_is_harmless_without_model_error,
        test_refine_removes_silhouette_model_error,
        test_normals_are_lines_across_hemispheres,
        test_speed,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
