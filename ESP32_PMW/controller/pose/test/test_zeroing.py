"""Tests for the pose datum.

The contract is one sentence: estimate the reference image again after zeroing
on it, and every channel reads zero.  If that does not hold, "relative to the
pad" means nothing and every plot is offset by an unknown constant.

Run: uv run python controller/pose/test_zeroing.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "validation"))

import conic  # noqa: E402
from estimator import PoseEstimator, _angles_from_normal  # noqa: E402
from zeroing import Zero, average_poses, frame_from_normal  # noqa: E402


def test_frame_from_normal_is_a_rotation():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        r = frame_from_normal(n)
        worst = max(
            worst,
            float(np.abs(r.T @ r - np.eye(3)).max()),
            abs(float(np.linalg.det(r)) - 1.0),
            float(np.abs(r[:, 2] - n).max()),
        )
    assert worst < 1e-12, f"not an orthonormal right-handed frame with +z = n: {worst:.2e}"
    print(f"  frame_from_normal   orthonormal, det=+1, +z=n   (worst {worst:.2e})")


def test_in_plane_pins_azimuth():
    """Supplying an in-plane reference must fix the spin about the normal."""
    n = np.array([0.0, 0.0, 1.0])
    r = frame_from_normal(n, in_plane=np.array([0.0, 1.0, 0.0]))
    assert np.allclose(r[:, 0], [0.0, 1.0, 0.0], atol=1e-12), r[:, 0]
    print("  in-plane reference  pins the azimuth as well as the axis")


def test_identity_passes_through():
    z = Zero.identity()
    c = np.array([3.0, -4.0, 210.0])
    n = np.array([0.1, 0.2, -0.97])
    n /= np.linalg.norm(n)
    xyz, nn = z.apply(c, n)
    assert np.allclose(xyz, c) and np.allclose(nn, n)
    print("  identity datum      leaves camera-frame values untouched")


def test_reference_reads_exactly_zero():
    """The headline contract, on analytic poses across a range of references."""
    worst_pos = worst_theta = worst_psi = 0.0

    for tilt_deg, az_deg, ctr in (
        (0.0, 0.0, [0.0, 0.0, 200.0]),
        (18.0, 40.0, [12.0, -7.0, 240.0]),
        (35.0, 200.0, [-20.0, 15.0, 170.0]),
        (60.0, 300.0, [5.0, 5.0, 300.0]),
    ):
        t, a = math.radians(tilt_deg), math.radians(az_deg)
        normal = np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)])
        center = np.array(ctr, dtype=np.float64)
        psi_ref = 37.5

        in_plane = np.array([math.cos(math.radians(psi_ref)), math.sin(math.radians(psi_ref)), 0.0])
        z = Zero.from_pose(center, normal, psi_deg=psi_ref, in_plane=in_plane)

        xyz, n_rel = z.apply(center, normal)
        theta, _ = _angles_from_normal(n_rel)

        worst_pos = max(worst_pos, float(np.abs(xyz).max()))
        worst_theta = max(worst_theta, abs(theta))
        worst_psi = max(worst_psi, abs(z.apply_psi(psi_ref)))

    assert worst_pos < 1e-9, f"position not zeroed: {worst_pos:.3e} mm"
    assert worst_theta < 1e-6, f"tilt not zeroed: {worst_theta:.3e} deg"
    assert worst_psi < 1e-9, f"psi not zeroed: {worst_psi:.3e} deg"
    print(f"  reference reads 0   pos {worst_pos:.1e} mm, tilt {worst_theta:.1e} deg, "
          f"psi {worst_psi:.1e} deg")


def test_roundtrip_through_a_file():
    z = Zero.from_pose([1.0, 2.0, 300.0], [0.2, -0.1, 0.97], psi_deg=12.5, meta={"note": "unit"})
    with tempfile.TemporaryDirectory() as d:
        p = z.save(Path(d) / "pose_zero.json")
        back = Zero.load(p)
    assert np.allclose(back.R, z.R) and np.allclose(back.t, z.t)
    assert back.psi_ref_deg == z.psi_ref_deg and back.meta.get("note") == "unit"
    assert Zero.load(Path(d) / "does_not_exist.json").is_identity
    print("  save / load         survives a round trip; missing file -> identity")


def test_psi_wraps():
    z = Zero(psi_ref_deg=170.0)
    assert abs(z.apply_psi(170.0)) < 1e-12
    assert abs(z.apply_psi(-170.0) - 20.0) < 1e-9, z.apply_psi(-170.0)
    assert -180.0 <= z.apply_psi(0.0) <= 180.0
    print("  psi wrapping        stays in (-180, 180] across the discontinuity")


def test_average_poses_handles_sign_flapping():
    """Normals from alternating branches must not average to nothing."""
    n = np.array([0.1, 0.0, 0.995])
    n /= np.linalg.norm(n)
    flapping = np.array([n if i % 2 == 0 else -n for i in range(10)])
    centers = np.tile([1.0, 2.0, 200.0], (10, 1))
    c, avg = average_poses(centers, flapping)
    assert np.allclose(c, [1.0, 2.0, 200.0])
    assert abs(abs(float(avg @ n)) - 1.0) < 1e-12, avg
    print("  average_poses       sign-flapping normals average correctly")


def test_zeroed_estimator_on_synthetic_image():
    """Full path: analytic conic -> estimator -> datum -> zeros.

    Uses the estimator's own solver rather than a rendered image, so this test
    stays fast and depends only on the maths, not on a GL context.
    """
    K = np.array([[1408.78, 0, 497.55], [0, 1407.69, 355.70], [0, 0, 1.0]])
    radius = 10.204
    center = np.array([8.0, -5.0, 215.0])
    normal = np.array([0.25, 0.15, 0.956])
    normal /= np.linalg.norm(normal)

    ellipse = conic.project_circle(center, normal, radius, K)
    poses = conic.backproject_ellipse(ellipse, K, radius)
    truth = min(poses, key=lambda p: np.linalg.norm(p.center - center))

    in_plane = np.array([math.cos(math.radians(ellipse[2])), math.sin(math.radians(ellipse[2])), 0])
    z = Zero.from_pose(truth.center, truth.normal, psi_deg=ellipse[2], in_plane=in_plane)

    est = PoseEstimator(camera_matrix=K, dist_coeffs=None, radius_mm=radius, zero=z)
    xyz, n_rel = est.zero.apply(truth.center, truth.normal)
    theta, _ = _angles_from_normal(n_rel)

    assert np.abs(xyz).max() < 1e-9, xyz
    assert abs(theta) < 1e-6, theta
    assert abs(est.zero.apply_psi(ellipse[2])) < 1e-9
    print(f"  estimator + datum   reads {np.abs(xyz).max():.1e} mm, {theta:.1e} deg at reference")


if __name__ == "__main__":
    print("zeroing tests")
    fail = 0
    for fn in (
        test_frame_from_normal_is_a_rotation,
        test_in_plane_pins_azimuth,
        test_identity_passes_through,
        test_reference_reads_exactly_zero,
        test_roundtrip_through_a_file,
        test_psi_wraps,
        test_average_poses_handles_sign_flapping,
        test_zeroed_estimator_on_synthetic_image,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
