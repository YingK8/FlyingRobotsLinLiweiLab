"""Tests for the fitted tilt calibration.

Run: uv run python controller/pose/test_calibration.py

A calibration is a number fitted to data, which makes it the easiest thing in the
pipeline to get quietly wrong -- a refit against different data, or a sign slip,
degrades accuracy without raising anything. These pin the properties that must
hold for *any* refit, plus the measured accuracy of the shipped one.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from calibration import MAX_FIT_TILT_DEG, TiltCalibration  # noqa: E402

# Held-out test-split medians from validation/tune.py at the time of fitting.
# Loose enough not to trip on a legitimate refit, tight enough to catch a
# regression that matters.
SHIPPED_MAX_TILT_MAP_ERR_DEG = 6.0


def test_identity_is_a_no_op():
    cal = TiltCalibration()
    assert cal.is_identity
    e = ((100.0, 200.0), (130.0, 90.0), 25.0)
    assert cal.apply(e) == e
    assert abs(cal.tilt(37.0) - 37.0) < 1e-12
    print("  identity            passes ellipses through untouched")


def test_missing_file_is_identity():
    with tempfile.TemporaryDirectory() as d:
        assert TiltCalibration.load(Path(d) / "nope.json").is_identity
    print("  missing file        loads as identity, not an error")


def test_zero_maps_to_zero():
    """Face-on must read inside the resolution floor -- which is not always zero.

    The quadratic model was built with no constant term so that zero mapped
    exactly to zero. The cylinder model cannot honour that and is right not to:
    an axis ratio of 1 is produced by a tilt of 0 *and* by a tilt of
    ``2 atan(k)``, because the wall widens the silhouette in between. There is no
    observation that distinguishes them, so no inverse can map one to zero
    without being wrong about the other.

    What is still required is that the answer stays inside the band where the
    ambiguity lives. Reporting 5 degrees for a face-on robot is honest when
    5 degrees is the resolution floor; reporting 20 would not be.
    """
    cal = TiltCalibration.load()
    got = cal.tilt(0.0)
    floor = cal.resolution_floor_deg
    if cal.model == "cylinder":
        assert 0.0 <= got <= floor + 1e-9, (got, floor)
        print(f"  theta_raw 0         -> {got:.2f} deg, inside the {floor:.2f} deg "
              f"resolution floor 2*atan(k)")
    else:
        assert abs(got) < 1e-9, got
        print(f"  theta_raw 0         -> {got:.2e} deg (no constant term by construction)")


def test_monotonic_and_bounded():
    """Non-monotonic would make tilt ambiguous; out-of-range would be nonsense."""
    cal = TiltCalibration.load()
    xs = np.linspace(0.0, 90.0, 500)
    ys = np.array([cal.tilt(x) for x in xs])
    assert np.all(np.diff(ys) >= 0), "correction is not monotonic"
    assert ys.min() >= 0.0 and ys.max() <= 90.0, (ys.min(), ys.max())
    print(f"  monotone & bounded  0-90 deg in -> [{ys.min():.2f}, {ys.max():.2f}] deg out")


def test_refuses_non_monotonic_fit():
    """`fit` must reject a curve that would make tilt ambiguous."""
    x = np.linspace(5.0, 70.0, 60)
    try:
        TiltCalibration.fit(x, -3.0 * x)  # strongly decreasing
    except ValueError as e:
        assert "monotonic" in str(e), e
        print("  bad fit rejected    non-monotonic correction raises instead of shipping")
        return
    raise AssertionError("a decreasing correction should have been refused")


def test_fit_recovers_a_known_distortion():
    """Round-trip: distort tilts by a known law, refit, check recovery."""
    rng = np.random.default_rng(11)
    true_tilt = rng.uniform(5.0, 65.0, 500)
    # Pretend the estimator under-reads tilt by 5%, as the mast makes it do.
    raw = true_tilt / 1.05
    cal = TiltCalibration.fit(raw, true_tilt)
    worst = max(abs(cal.tilt(r) - t) for r, t in zip(raw, true_tilt))
    assert worst < 0.05, f"failed to recover a clean 5% scaling: {worst:.4f} deg"
    print(f"  fit round-trip      recovers a known 5% under-read to {worst:.1e} deg")


def test_apply_preserves_major_axis():
    """Only the minor axis is corrected; the major one is the trustworthy one."""
    cal = TiltCalibration.load()
    (cx, cy), (major, minor), ang = cal.apply(((50.0, 60.0), (130.0, 95.0), 33.0))
    assert (cx, cy) == (50.0, 60.0) and ang == 33.0
    assert abs(major - 130.0) < 1e-12, major
    assert minor < 95.0, "correction should shrink the minor axis, raising the implied tilt"
    print(f"  apply()             major fixed at {major:.1f} px, minor 95.0 -> {minor:.2f} px")


def test_apply_never_exceeds_major():
    """A minor axis above the major would be geometrically impossible."""
    cal = TiltCalibration.load()
    for minor in (5.0, 40.0, 90.0, 129.0, 130.0):
        _, (major, out), _ = cal.apply(((0.0, 0.0), (130.0, minor), 0.0))
        assert 0.0 <= out <= major + 1e-9, (minor, out, major)
    print("  apply()             minor stays within [0, major] across the whole range")


def test_shipped_correction_increases_tilt():
    """The mast makes tilt read low, so the correction must push it up."""
    cal = TiltCalibration.load()
    if cal.is_identity:
        print("  shipped direction   (no calibration on disk; skipped)")
        return
    for raw in (10.0, 25.0, 40.0, 55.0):
        assert cal.tilt(raw) > raw, f"correction reduced tilt at {raw}: {cal.tilt(raw)}"
    print("  shipped direction   raises tilt at 10/25/40/55 deg, as the bias requires")


def test_clamps_beyond_fit_range():
    """Beyond the fitted band, stay bounded -- by clamping or by construction.

    The quadratic had to clamp: past its fit range a polynomial does whatever a
    polynomial does. The cylinder model has nothing to extrapolate -- it is an
    exact inverse of a closed-form projection, valid to ``90 - atan(k)`` -- so
    clamping it would discard correct answers. Both are acceptable; running away
    is not, which is what this actually checks.
    """
    cal = TiltCalibration.load()
    beyond = cal.tilt(MAX_FIT_TILT_DEG + 40.0)
    at_edge = cal.tilt(MAX_FIT_TILT_DEG)
    if cal.model == "cylinder":
        assert 0.0 <= beyond <= 90.0 and beyond >= at_edge, (beyond, at_edge)
        print(f"  extrapolation       exact by construction: {MAX_FIT_TILT_DEG:g} -> "
              f"{at_edge:.2f} deg, 115 -> {beyond:.2f} deg, both bounded")
    else:
        assert abs(beyond - at_edge) < 1e-9, (beyond, at_edge)
        print(f"  extrapolation       clamped at {MAX_FIT_TILT_DEG:g} deg -> {at_edge:.2f} deg")


def test_shipped_roundtrip_on_file():
    cal = TiltCalibration.load()
    with tempfile.TemporaryDirectory() as d:
        p = cal.save(Path(d) / "c.json")
        back = TiltCalibration.load(p)
    assert abs(back.a - cal.a) < 1e-12 and abs(back.b - cal.b) < 1e-12
    print(f"  save / load         a={cal.a:.5f} b={cal.b:.6f} survives a round trip")


def test_estimator_uses_it_by_default():
    """A fresh PoseEstimator must pick the calibration up without being asked."""
    from estimator import PoseEstimator

    K = np.array([[1408.78, 0, 497.55], [0, 1407.69, 355.70], [0, 0, 1.0]])
    est = PoseEstimator(camera_matrix=K, dist_coeffs=None)
    disk = TiltCalibration.load()
    assert est.tilt_cal.a == disk.a and est.tilt_cal.b == disk.b
    off = PoseEstimator(camera_matrix=K, dist_coeffs=None, tilt_cal=TiltCalibration())
    assert off.tilt_cal.is_identity, "explicit identity must disable the correction"
    print("  estimator wiring    loads by default; TiltCalibration() disables it")


def test_correction_reduces_tilt_error_on_a_model():
    """End to end on the analytic bias model the calibration exists to remove.

    Simulates the measured effect -- the axis ratio sitting above cos(theta) --
    and checks the shipped curve actually reduces the resulting tilt error.
    """
    cal = TiltCalibration.load()
    if cal.is_identity:
        print("  error reduction     (no calibration on disk; skipped)")
        return

    rng = np.random.default_rng(5)
    tilts = rng.uniform(10.0, 45.0, 400)
    # Excess measured at +0.031 mean over tilt 25-55; scale it with sin(theta).
    excess = 0.031 * np.sin(np.radians(tilts)) / math.sin(math.radians(40.0))
    ratio = np.clip(np.cos(np.radians(tilts)) + excess, 0.0, 1.0)
    raw = np.degrees(np.arccos(ratio))

    before = np.median(np.abs(raw - tilts))
    after = np.median(np.abs([cal.tilt(r) for r in raw] - tilts))
    print(f"  error reduction     median |tilt err| {before:.2f} -> {after:.2f} deg "
          f"({(after/before - 1)*100:+.0f}%)")
    assert after < before, f"calibration made it worse: {before:.3f} -> {after:.3f}"
    assert after < SHIPPED_MAX_TILT_MAP_ERR_DEG, after


if __name__ == "__main__":
    print("tilt calibration tests")
    fail = 0
    for fn in (
        test_identity_is_a_no_op,
        test_missing_file_is_identity,
        test_zero_maps_to_zero,
        test_monotonic_and_bounded,
        test_refuses_non_monotonic_fit,
        test_fit_recovers_a_known_distortion,
        test_apply_preserves_major_axis,
        test_apply_never_exceeds_major,
        test_shipped_correction_increases_tilt,
        test_clamps_beyond_fit_range,
        test_shipped_roundtrip_on_file,
        test_estimator_uses_it_by_default,
        test_correction_reduces_tilt_error_on_a_model,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
