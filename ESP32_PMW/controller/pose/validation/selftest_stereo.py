"""Does the stereo renderer agree with the geometry the estimator assumes?

Same argument as `selftest.py`, one level up.  There, a flipped axis between
pyrender and `conic.py` would have produced plausible images and silently
mirrored ground truth.  Here the equivalent failure is a **transposed or
inverted extrinsic**: both views still render a perfectly convincing robot, the
pair is simply not the pair the rig describes, and every stereo residual after
that is measuring nothing.

So nothing downstream should be believed until this passes.  Four independent
checks, each with a distinct failure signature:

* the transform itself, against `rig.Camera.to_camera` -- exact, no rendering;
* each rendered silhouette against `conic.project_circle` -- catches a
  convention error that survives the algebra;
* world-fixed lighting, which is the one thing a per-view render can get wrong
  without changing any geometry;
* occluders, which must darken the image without touching the ground-truth mask.

Run: uv run python controller/pose/validation/selftest_stereo.py
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
import render_stereo as rs  # noqa: E402
import segment as segmod  # noqa: E402
from rig import StereoRig  # noqa: E402

# Same limits as the monocular selftest: agreement is bounded by pixel
# quantisation and by the mesh being a faceted circle, not by the maths.
TOL_CENTER_PX = 2.5
# Only the **major** axis is asserted on. It is the rim diameter at every tilt
# and stays accurate throughout, so a regression in the fit shows up here. The
# minor axis is what the mast inflates, and at the tilts a 45-degree rig
# produces that inflation is real geometry, not a renderer fault -- it is
# reported instead, exactly as `selftest.test_high_tilt_model_limit` does.
TOL_MAJOR_PX = 4.0

# Comparing unit vectors through `acos` is hopeless near coincidence: the
# derivative is 1/sqrt(1-x^2), so machine epsilon in the dot product becomes
# ~1e-6 degrees. Comparing the vectors directly is well conditioned, and the
# min over sign is what makes it a comparison of *lines* -- necessary because
# `normal_from_pose` orients each view's normal toward its own camera.
TOL_VEC = 1e-9

# The flat-circle model holds to about 45 degrees, and a 45-degree rig already
# views a level robot at exactly 45. So the agreement poses lean *toward* a
# camera, which reduces the tilt that camera sees. `(15, 225)` was tried and
# dropped from this list on purpose: leaning away from both bearings pushes both
# views to 56 degrees, past the crossover where the mast enters the silhouette
# (tan(theta) > 10.2/7.84 -> 52.5 deg), and it fails on geometry rather than on
# anything this test is meant to catch. It lives in `HIGH_TILT_POSES` instead.
POSES = [
    # (tilt_deg, azimuth_deg, center_world_mm)
    (0.0, 0.0, [0.0, 0.0, 0.0]),
    (10.0, 0.0, [0.0, 0.0, 0.0]),
    (10.0, 90.0, [8.0, -6.0, 12.0]),
    (20.0, 45.0, [0.0, 0.0, -20.0]),
    (12.0, 45.0, [-10.0, 10.0, 25.0]),
]

HIGH_TILT_POSES = [
    (15.0, 225.0, [-10.0, 10.0, 25.0]),
    (25.0, 225.0, [0.0, 0.0, 0.0]),
]


def _fit(sample):
    seg = segmod.segment(sample.image, thresh=segmod.THRESH)
    assert seg is not None, "stereo renderer produced a frame the segmenter cannot see"
    return seg


def _line_distance(u, v):
    """Distance between two unit vectors treated as undirected lines."""
    u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    return float(min(np.linalg.norm(u - v), np.linalg.norm(u + v)))


def _line_angle_deg(u, v):
    """Angle between two unit vectors treated as undirected lines."""
    return math.degrees(math.acos(float(np.clip(abs(np.asarray(u) @ np.asarray(v)), 0.0, 1.0))))


def test_extrinsic_matches_rig(sr):
    """Rendered ground truth must equal the rig's own world->camera transform.

    Pure algebra against `rig.Camera.to_camera`, so it isolates the transform
    from everything the rasteriser could contribute.  A transposed
    ``T_this_ref`` fails here by tens of millimetres while every other check
    still passes, which is precisely why it is first.
    """
    worst_c = worst_n = 0.0
    for tilt, az, ctr in POSES:
        s = sr.render_pair(tilt, az, ctr)
        for cam, view in zip(sr.rig.cameras, s.views):
            want_c, want_n = cam.to_camera(s.center_world, s.normal_world)
            worst_c = max(worst_c, float(np.linalg.norm(view.center_mm - want_c)))
            # As lines: `normal_from_pose` flips the sign to face the camera, so
            # a rig with cameras on opposite sides of the rotor plane legitimately
            # reports opposite normals for the same physical pose.
            worst_n = max(worst_n, _line_distance(view.normal, want_n))

    assert worst_c < 1e-6, f"rendered centre disagrees with the rig transform by {worst_c:.3e} mm"
    assert worst_n < TOL_VEC, f"rendered normal disagrees with the rig transform by {worst_n:.3e}"
    print(f"  extrinsic vs rig    centre {worst_c:.2e} mm, normal {worst_n:.2e} (unit vec)")


def test_identity_view_reproduces_monocular(sr):
    """A rig whose camera sits at the world origin must render the mono image.

    The reduction test: if the stereo path is a generalisation rather than a
    reimplementation, switching off the generalisation has to give back exactly
    what was there before -- pixel for pixel, since it is the same code path
    with an identity matrix in it.
    """
    mono = sr._renderer.render(18.0, 33.0, [4.0, -3.0, 210.0])
    via_view = sr._renderer.render(
        18.0, 33.0, [4.0, -3.0, 210.0], view=rendermod.View()
    )
    diff = int(np.abs(mono.image.astype(int) - via_view.image.astype(int)).max())
    assert diff == 0, f"identity view changed the image by {diff} grey levels"
    assert np.allclose(mono.center_mm, via_view.center_mm), "identity view moved the centre"
    print(f"  identity view       pixel-identical to the monocular path ({diff} levels)")


def test_views_match_analytic_projection(sr):
    """Both rendered silhouettes vs. the analytic rim projection.

    The stereo version of the check `selftest.py` runs on one view.  Each view
    is scored through **its own** intrinsics and its own ground truth, so a rig
    with mismatched cameras would be caught here too.

    Asserts on the centre and the major axis; the minor-axis excess is reported,
    because at the tilts a 45-degree rig produces it is the mast entering the
    silhouette rather than anything the renderer did wrong.
    """
    worst_c = worst_major = 0.0
    worst_minor_excess = -1.0
    for tilt, az, ctr in POSES:
        s = sr.render_pair(tilt, az, ctr)
        for view in s.views:
            got = _fit(view).ellipse
            want = conic.project_circle(
                view.center_mm, view.normal, rendermod.RIM_RADIUS_MM, view.K
            )
            worst_c = max(worst_c, math.hypot(got[0][0] - want[0][0], got[0][1] - want[0][1]))
            worst_major = max(worst_major, abs(got[1][0] - want[1][0]))
            worst_minor_excess = max(worst_minor_excess, got[1][1] / want[1][1] - 1.0)

    assert worst_c < TOL_CENTER_PX, f"centre disagrees by {worst_c:.2f} px"
    assert worst_major < TOL_MAJOR_PX, f"major axis disagrees by {worst_major:.2f} px"
    print(f"  vs analytic proj    worst centre {worst_c:.2f} px, major {worst_major:.2f} px, "
          f"minor excess {worst_minor_excess:+.1%}")


def test_high_tilt_is_geometry_not_renderer(sr):
    """Outside the flat-circle envelope, the major axis must still be right.

    Poses that lean away from both bearings put every view past the ~52 degree
    crossover where the mast widens the silhouette.  The point of measuring it
    separately is to show the failure is one-sided -- the minor axis inflates,
    the major axis does not -- because that is the signature of real geometry
    rather than a broken transform, which would corrupt both.
    """
    print("  high-tilt (reported)  seen  |  major fit/true  | minor excess")
    worst_major = 0.0
    for tilt, az, ctr in HIGH_TILT_POSES:
        s = sr.render_pair(tilt, az, ctr)
        for i, view in enumerate(s.views):
            got = _fit(view).ellipse
            want = conic.project_circle(
                view.center_mm, view.normal, rendermod.RIM_RADIUS_MM, view.K
            )
            seen = math.degrees(math.acos(min(1.0, abs(float(view.normal[2])))))
            worst_major = max(worst_major, abs(got[1][0] - want[1][0]))
            print(f"                        {seen:5.1f} | {got[1][0]:6.1f}/{want[1][0]:<6.1f}  |"
                  f" {got[1][1] / want[1][1] - 1.0:+7.1%}   (view {sr.rig.cameras[i].name or i})")
    assert worst_major < 8.0, f"major axis degraded by {worst_major:.1f} px outside the envelope"


def test_lighting_is_world_fixed(sr):
    """Lights must be anchored to the world, not carried along with each camera.

    This is the one error that changes no geometry at all, so no amount of
    checking transforms will find it.  The signature is specific: a hard
    lateral light and a level robot make the two views nearly identical if the
    light rides with the camera, and clearly different if it does not.  Both
    directions are asserted -- pure ambient is the positive control that the
    difference really is the lighting and not the viewpoint.
    """
    lateral = rendermod.LightRig(
        lateral_deg=(0.0,), intensity=14.0, ambient=0.05, key_from_camera=False
    )
    flat = rendermod.LightRig(lateral_deg=(), intensity=0.0, ambient=0.6,
                              key_from_camera=False)

    def mean_on_mask(sample):
        m = sample.mask
        return float(sample.image[m].mean()) if m.any() else 0.0

    lit = sr.render_pair(0.0, 0.0, [0.0, 0.0, 0.0], light=lateral)
    amb = sr.render_pair(0.0, 0.0, [0.0, 0.0, 0.0], light=flat)

    lit_delta = abs(mean_on_mask(lit.views[0]) - mean_on_mask(lit.views[1]))
    amb_delta = abs(mean_on_mask(amb.views[0]) - mean_on_mask(amb.views[1]))

    assert lit_delta > 4.0, (
        f"the two views are lit alike ({lit_delta:.1f} grey levels apart) under a hard "
        f"lateral light -- the light is riding with the camera instead of the world"
    )
    assert amb_delta < lit_delta, (
        f"ambient light differs more between views ({amb_delta:.1f}) than directional "
        f"({lit_delta:.1f}); the difference is not coming from the lighting"
    )
    print(f"  world-fixed lights  lateral {lit_delta:.1f} vs ambient {amb_delta:.1f} grey levels")


def _lit_inside(sample, mask):
    """Foreground pixels the segmenter would see **within the true silhouette**.

    Counting over the whole frame conflates two different things: how much of
    the robot an occluder hides, and how much foreground the occluder itself
    contributes.  Restricting to the robot's own mask isolates the first;
    `_lit_outside` measures the second.
    """
    return int((sample.image > segmod.THRESH)[mask].sum())


def _lit_outside(sample, mask):
    return int((sample.image > segmod.THRESH)[~mask].sum())


def test_occluder_darkens_image_not_truth(sr):
    """An occluder must change what is seen without rewriting what is true.

    The ground-truth mask is the robot's silhouette; it is the reference the
    IoU and the pose are scored against.  If an occluder edited it, occlusion
    would score as perfect by construction and the sweep would learn nothing.

    The mask is compared as a **pixel set**, not by area: equal areas with
    different pixels is exactly what a subtle depth interaction would produce,
    and it would pass an area check silently.
    """
    clear = sr.render_pair(0.0, 0.0, [0.0, 0.0, 0.0])
    masks = [v.mask.copy() for v in clear.views]

    sr.set_occluders([rs.takeoff_stand(top_z_mm=-11.0)])
    try:
        occ = sr.render_pair(0.0, 0.0, [0.0, 0.0, 0.0])
        for i, (m, v) in enumerate(zip(masks, occ.views)):
            moved = int(np.logical_xor(m, v.mask).sum())
            assert moved == 0, (
                f"the occluder edited view {i}'s ground-truth mask at {moved} pixels"
            )
        inside = [(_lit_inside(c, m), _lit_inside(o, m))
                  for c, o, m in zip(clear.views, occ.views, masks)]
        outside = [(_lit_outside(c, m), _lit_outside(o, m))
                   for c, o, m in zip(clear.views, occ.views, masks)]
    finally:
        sr.set_occluders([])

    # A near-black rod is not a black rod: glTF pins dielectric specular at
    # F0 = 0.04, so under a hard light the stand peaks around grey 155 and
    # crosses the 128 threshold. That is physically right -- a real anodised rod
    # has a sheen -- and it means the stand is a genuine segmentation
    # distractor, not just an occluder. `segment.silhouette_hull` keeps every
    # blob above 2% of the largest, so a bright rod would be hulled in with the
    # robot. Recorded here so the sweep's occlusion results are read correctly.
    for i, ((ci, oi), (co, oo)) in enumerate(zip(inside, outside)):
        print(f"  occluder view {i}     mask intact; lit inside {ci} -> {oi} "
              f"({oi / ci - 1:+.1%}), stand adds {oo - co} px outside")
    assert all(oi <= ci * 1.02 for ci, oi in inside), (
        f"the occluder brightened the robot itself: {inside}"
    )


def test_stand_occludes_from_below_only(sr):
    """The physical claim behind the rig choice, measured rather than asserted.

    A black rod under the robot should remove rim pixels from a camera looking
    up and none from one looking down.  This is what makes the mixed-hemisphere
    rig worth its calibration cost, so it is worth a test rather than a
    paragraph.  Measured **inside the robot's own mask**, so the rod's own
    brightness cannot masquerade as the robot surviving.
    """
    original = sr.rig
    sr.rig = StereoRig.from_spherical(elev_deg=(45.0, -45.0), azim_deg=(0.0, 0.0))
    sr.set_occluders([])
    try:
        clear = sr.render_pair(0.0, 0.0, [0.0, 0.0, 0.0])
        masks = [v.mask.copy() for v in clear.views]
        clear_px = [_lit_inside(v, m) for v, m in zip(clear.views, masks)]
        sr.set_occluders([rs.takeoff_stand(top_z_mm=-11.0)])
        occ = sr.render_pair(0.0, 0.0, [0.0, 0.0, 0.0])
        occ_px = [_lit_inside(v, m) for v, m in zip(occ.views, masks)]
    finally:
        sr.rig = original
        sr.set_occluders([])

    above_loss = 1.0 - occ_px[0] / max(clear_px[0], 1)
    below_loss = 1.0 - occ_px[1] / max(clear_px[1], 1)
    print(f"  stand occlusion     from above {above_loss:+.1%} of the robot's lit px, "
          f"from below {below_loss:+.1%}")
    assert above_loss < 0.02, f"the stand occluded the camera above by {above_loss:.1%}"
    assert below_loss > above_loss + 0.05, (
        f"the stand did not occlude the camera below meaningfully more than the one above "
        f"({below_loss:.1%} vs {above_loss:.1%}) -- the mixed-hemisphere rationale is wrong"
    )


if __name__ == "__main__":
    print("stereo renderer / rig agreement")
    fail = 0
    with rs.StereoRenderer(rs.default_rig()) as sr:
        for k, v in sr.rig.summary().items():
            print(f"  rig {k:20s} {v}")
        for fn in (
            test_extrinsic_matches_rig,
            test_identity_view_reproduces_monocular,
            test_views_match_analytic_projection,
            test_high_tilt_is_geometry_not_renderer,
            test_lighting_is_world_fixed,
            test_occluder_darkens_image_not_truth,
            test_stand_occludes_from_below_only,
        ):
            try:
                fn(sr)
            except AssertionError as e:
                print(f"  FAIL {fn.__name__}: {e}")
                fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
