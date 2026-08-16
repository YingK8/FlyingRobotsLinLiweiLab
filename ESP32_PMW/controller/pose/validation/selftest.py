"""Does the renderer agree with the projection model the estimator assumes?

The synthetic sweep is only meaningful if "ground truth" really is the truth.
If pyrender's camera convention and `conic.py`'s differ by a flipped axis, every
render still looks fine and every residual is quietly wrong.

So before measuring any residual, check the renderer against an independent
analytic projection of the same circle: `conic.project_circle` predicts where
the rim lands in pixels using nothing but K and the geometry, and the rendered
silhouette says where it actually landed.  They must agree.

Run: uv run python controller/pose/validation/selftest.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import conic  # noqa: E402
import render as rendermod  # noqa: E402
import segment as segmod  # noqa: E402

# The renderer is a rasteriser, so agreement is limited by pixel quantisation and
# by the mesh being a faceted approximation of a circle, not by the maths.
TOL_CENTER_PX = 2.5
TOL_AXIS_PX = 6.0

# Past about 45 degrees of tilt the robot stops behaving like a flat circle: the
# mast and magnet, which stick out along the rotor axis, start to dominate the
# silhouette's short direction. The agreement tests therefore run inside that
# envelope, and `test_high_tilt_model_limit` measures what happens outside it
# rather than pretending it does not happen.
MAX_MODEL_TILT_DEG = 40.0


def _fit(sample):
    seg = segmod.segment(sample.image, thresh=segmod.THRESH)
    assert seg is not None, "renderer produced a frame the segmenter cannot see"
    return seg


def test_axis_directions(r):
    """+x in camera coords must move the blob right; +y must move it down.

    This is the mirror check.  OpenCV has +y down, OpenGL +y up; if `CV_TO_GL`
    were missing or wrong, this is the test that fails while the images still
    look correct.
    """
    base = _fit(r.render(0.0, 0.0, [0.0, 0.0, 200.0])).ellipse[0]
    right = _fit(r.render(0.0, 0.0, [15.0, 0.0, 200.0])).ellipse[0]
    down = _fit(r.render(0.0, 0.0, [0.0, 15.0, 200.0])).ellipse[0]
    far = _fit(r.render(0.0, 0.0, [0.0, 0.0, 300.0])).ellipse

    assert right[0] > base[0] + 20, f"+x did not move right: {base} -> {right}"
    assert abs(right[1] - base[1]) < 5, f"+x moved vertically: {base} -> {right}"
    assert down[1] > base[1] + 20, f"+y did not move down: {base} -> {down}"
    assert abs(down[0] - base[0]) < 5, f"+y moved horizontally: {base} -> {down}"

    near_major = _fit(r.render(0.0, 0.0, [0.0, 0.0, 200.0])).ellipse[1][0]
    assert far[1][0] < near_major, "further away did not render smaller"
    print(f"  axis directions     +x right, +y down, +z away  (base centre {base[0]:.0f},{base[1]:.0f})")


def test_matches_analytic_projection(r):
    """Rendered silhouette vs. the analytic rim projection, over many poses."""
    worst_c = worst_a = 0.0
    cases = [
        (0.0, 0.0, [0.0, 0.0, 200.0]),
        (0.0, 0.0, [25.0, -18.0, 260.0]),
        (25.0, 0.0, [0.0, 0.0, 200.0]),
        (25.0, 90.0, [0.0, 0.0, 200.0]),
        (40.0, 45.0, [10.0, 8.0, 220.0]),
        (35.0, 200.0, [-15.0, 12.0, 180.0]),
        (15.0, 300.0, [0.0, 0.0, 150.0]),
    ]

    for tilt, az, ctr in cases:
        s = r.render(tilt, az, ctr, alpha=1.0, bg_level=0.0)
        got = _fit(s).ellipse
        want = conic.project_circle(s.center_mm, s.normal, rendermod.RIM_RADIUS_MM, r.K)

        dc = math.hypot(got[0][0] - want[0][0], got[0][1] - want[0][1])
        da = max(abs(got[1][0] - want[1][0]), abs(got[1][1] - want[1][1]))
        worst_c, worst_a = max(worst_c, dc), max(worst_a, da)

    assert worst_c < TOL_CENTER_PX, f"centre disagrees by {worst_c:.2f} px"
    assert worst_a < TOL_AXIS_PX, f"axis length disagrees by {worst_a:.2f} px"
    print(f"  vs analytic proj    worst centre {worst_c:.2f} px, worst axis {worst_a:.2f} px")


def test_ground_truth_normal_consistent(r):
    """The declared ground-truth normal must reproduce the rendered ellipse.

    Guards the other half of the contract: `pose_matrix` and `normal_from_pose`
    must describe the same rotation the renderer applied.
    """
    worst = 0.0
    for tilt in (0.0, 20.0, 30.0, MAX_MODEL_TILT_DEG):
        for az in (0.0, 120.0, 240.0):
            s = r.render(tilt, az, [0.0, 0.0, 220.0])
            got = _fit(s).ellipse
            ratio_seen = got[1][1] / got[1][0]
            # A circle tilted by theta projects with minor/major ~ cos(theta).
            ratio_want = math.cos(math.radians(tilt))
            worst = max(worst, abs(ratio_seen - ratio_want))
    assert worst < 0.06, f"axis ratio does not follow cos(tilt): off by {worst:.3f}"
    print(f"  tilt -> axis ratio  worst deviation from cos(tilt) {worst:.4f} (to {MAX_MODEL_TILT_DEG:g} deg)")


def test_high_tilt_model_limit(r):
    """Measure where the flat-circle model stops holding, rather than assume it.

    The major axis is the invariant: whatever the tilt, the widest extent of the
    silhouette is still the rim diameter, so it should stay accurate throughout.
    The minor axis is what the protruding mast inflates.  Asserting on the major
    axis while merely reporting the minor keeps the test honest -- a regression
    in the fit shows up, while known geometry does not masquerade as a failure.
    """
    print("  high-tilt limit     tilt |  major fit/true  |  minor fit/true  | minor excess")
    worst_major = 0.0
    for tilt in (40.0, 50.0, 60.0, 70.0):
        s = r.render(tilt, 0.0, [0.0, 0.0, 220.0])
        got = _fit(s).ellipse
        want = conic.project_circle(s.center_mm, s.normal, rendermod.RIM_RADIUS_MM, r.K)
        excess = got[1][1] / want[1][1] - 1.0
        worst_major = max(worst_major, abs(got[1][0] - want[1][0]))
        print(
            f"                      {tilt:4.0f} | {got[1][0]:6.1f}/{want[1][0]:<6.1f}  |"
            f" {got[1][1]:6.1f}/{want[1][1]:<6.1f}  | {excess:+7.1%}"
        )
    assert worst_major < 8.0, f"major axis degraded by {worst_major:.1f} px"


def test_alpha_and_background(r):
    """Opacity and background level must actually reach the image."""
    peaks = [r.render(0.0, 0.0, [0.0, 0.0, 200.0], alpha=a).image.max() for a in (0.3, 1.0)]
    assert peaks[1] > peaks[0], f"alpha had no effect on brightness: {peaks}"

    bgs = [
        int(np.median(r.render(0.0, 0.0, [0.0, 0.0, 200.0], bg_level=b).image[:40, :40]))
        for b in (0.0, 0.3)
    ]
    assert bgs[1] > bgs[0] + 20, f"background level had no effect: {bgs}"

    s = r.render(0.0, 0.0, [0.0, 0.0, 200.0])
    assert s.mask.sum() > 1000, "ground-truth mask is empty"
    print(f"  alpha / background  peak {peaks[0]}->{peaks[1]}, bg {bgs[0]}->{bgs[1]}, mask ok")


def test_estimator_on_render(r):
    """End to end: render a pose, estimate it back, in millimetres and degrees.

    Accuracy is scored against the *better* of the two back-projection branches.
    That is not grading on a curve -- it is the only way to measure the geometry
    separately from the ambiguity.  One conic cannot say which branch is real
    (see `test_ambiguity_is_irreducible`), so mixing the two failure modes into
    one number would mean a perfect fit on the wrong branch looks identical to a
    broken fit.  The sweep reports both, and so does the test below.
    """
    sys.path.insert(0, str(HERE.parent))
    from estimator import PoseEstimator

    est = PoseEstimator(camera_matrix=r.K, dist_coeffs=None, radius_mm=rendermod.RIM_RADIUS_MM)
    worst_pos = worst_ang = 0.0

    for tilt, az in ((0.0, 0.0), (15.0, 0.0), (30.0, 90.0), (MAX_MODEL_TILT_DEG, 210.0)):
        s = r.render(tilt, az, [6.0, -4.0, 210.0], alpha=1.0, bg_level=0.0)
        est.reset()
        p = est.update(s.image)
        assert p is not None, f"estimator lost a clean render at tilt {tilt}"

        worst_pos = max(worst_pos, float(np.linalg.norm(p.xyz_mm - s.center_mm)))
        worst_ang = max(worst_ang, _best_branch_error(p, s))

    print(f"  estimator on render worst pos {worst_pos:.2f} mm, normal {worst_ang:.2f} deg "
          f"(best branch, tilt <= {MAX_MODEL_TILT_DEG:g})")
    assert worst_pos < 4.0, f"position error {worst_pos:.2f} mm on a clean render"
    assert worst_ang < 8.0, f"normal error {worst_ang:.2f} deg on a clean render"


def _best_branch_error(pose, sample):
    """Smallest normal error over the back-projection's candidate branches."""
    cands = pose.extra.get("candidates") or []
    errs = [
        math.degrees(math.acos(float(np.clip(abs(c.normal @ sample.normal), -1.0, 1.0))))
        for c in cands
    ]
    return min(errs) if errs else float("nan")


def test_ambiguity_is_irreducible(r):
    """Record that one frame cannot pick a branch, and that we log the margin.

    A tilted circle and its mirror image about the viewing axis project to the
    identical ellipse.  No amount of fitting separates them; it needs a prior,
    temporal continuity, or a second camera.  This test asserts the estimator
    is honest about it -- that `ambiguity_margin_deg` grows with tilt, so the
    CSV always carries the size of the risk taken on every frame.
    """
    print("  ambiguity           tilt | margin | chosen err | best err")
    margins = []
    for tilt in (5.0, 20.0, 40.0):
        s = r.render(tilt, 0.0, [0.0, 0.0, 220.0])
        est_local = _fresh_estimator(r)
        p = est_local.update(s.image)
        assert p is not None
        chosen = math.degrees(math.acos(float(np.clip(abs(p.normal @ s.normal), -1.0, 1.0))))
        margins.append(p.ambiguity_margin_deg)
        print(
            f"                      {tilt:4.0f} | {p.ambiguity_margin_deg:6.1f} |"
            f" {chosen:10.1f} | {_best_branch_error(p, s):8.1f}"
        )

    assert margins == sorted(margins), f"margin should grow with tilt, got {margins}"
    # The margin is about twice the *read* tilt, and near head-on the read tilt
    # is noise: at a true 5 degrees it lands anywhere in roughly 0-10, because
    # dtheta/d(ratio) = -1/sin(theta) blows up there. So this bound is loose on
    # purpose -- tightening it would be asserting on the noise, and a bound that
    # fails whenever the dice land badly is worse than no bound. It still catches
    # a real regression: a margin of 60 degrees at 5 degrees of tilt would mean
    # the fit had stopped tracking tilt at all.
    assert margins[0] < 25.0, (
        f"near head-on the branches should still be close, got {margins[0]:.1f} deg"
    )


def test_face_on_conditioning(r):
    """Near face-on, tilt is ill-conditioned but position must stay good.

    Tilt comes from foreshortening, ``theta = acos(minor/major)``, so
    ``dtheta/dratio = -1/sin(theta)`` and the projection is stationary at
    theta = 0 -- a circle tilted 2 degrees is only 0.06% narrower than one
    tilted 0.  Every foreshortening-based estimator has this; it is geometry,
    not implementation.

    What must NOT degrade is position, which depends on the ellipse's *size*
    rather than its shape and is well conditioned throughout.  That separation
    is the invariant worth pinning: it is why the estimator is usable for
    position control even where its tilt reading is noisy.
    """
    print("  face-on conditioning tilt | axis err | pos err")
    pos_errs, axis_errs = [], []
    for tilt in (0.0, 5.0, 10.0, 20.0):
        s = r.render(tilt, 30.0, [0.0, 0.0, 220.0])
        est_local = _fresh_estimator(r)
        solved = est_local.solve_camera_frame(s.image)
        assert solved is not None, f"lost a clean render at tilt {tilt}"
        c, n, _ = solved
        ae = math.degrees(math.acos(float(np.clip(abs(n @ s.normal), -1.0, 1.0))))
        pe = float(np.linalg.norm(c - s.center_mm))
        axis_errs.append(ae)
        pos_errs.append(pe)
        print(f"                      {tilt:4.0f} | {ae:8.2f} | {pe:6.2f} mm")

    assert max(pos_errs) < 2.0, f"position degraded near face-on: {max(pos_errs):.2f} mm"
    assert axis_errs[0] > axis_errs[1], "face-on axis error should be the worst"
    assert max(axis_errs[1:]) < 6.0, f"off-axis tilt should be well conditioned: {axis_errs}"


def _fresh_estimator(r):
    sys.path.insert(0, str(HERE.parent))
    from estimator import PoseEstimator

    return PoseEstimator(
        camera_matrix=r.K, dist_coeffs=None, radius_mm=rendermod.RIM_RADIUS_MM
    )


if __name__ == "__main__":
    print("renderer / projection agreement")
    fail = 0
    with rendermod.Renderer() as r:
        print(f"  rim radius measured from mesh: {rendermod.RIM_RADIUS_MM:.4f} mm")
        for fn in (
            test_axis_directions,
            test_matches_analytic_projection,
            test_ground_truth_normal_consistent,
            test_high_tilt_model_limit,
            test_alpha_and_background,
            test_estimator_on_render,
            test_ambiguity_is_irreducible,
            test_face_on_conditioning,
        ):
            try:
                fn(r)
            except AssertionError as e:
                print(f"  FAIL {fn.__name__}: {e}")
                fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
