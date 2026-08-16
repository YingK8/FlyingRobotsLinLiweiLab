"""Tests for the pose filter.

Run: uv run python controller/pose/test_filter.py

Checks the behaviours the filter is actually kept for -- velocity, coasting
through dropouts, latency compensation -- rather than position smoothing, which
it deliberately does not do (see `filter.py`).
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
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

from filter import PoseFilter  # noqa: E402


class FakePose:
    """Minimal stand-in for `estimator.Pose` -- the filter only reads three fields."""

    def __init__(self, t, xyz, normal):
        self.t = t
        self.xyz_mm = np.asarray(xyz, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)


def _straight_line(n=400, fps=240.0, speed=(20.0, 0.0, 5.0), noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    v = np.asarray(speed, dtype=np.float64)
    out = []
    for i in range(n):
        t = i * dt
        true = np.array([0.0, 0.0, 220.0]) + v * t
        meas = true + (rng.normal(0, noise, 3) if noise else 0.0)
        out.append((t, true, meas))
    return out, dt, v


def test_returns_none_before_first_detection():
    f = PoseFilter()
    assert f.update(None, t=0.0) is None
    assert f.predict_ahead(0.01) is None
    print("  cold start          returns None until something is seen")


def test_seeds_on_first_measurement():
    """It must jump to the first fix, not slide in from the origin."""
    f = PoseFilter()
    xyz, vel, n = f.update(FakePose(0.0, [5.0, -3.0, 210.0], [0, 0, -1.0]))
    assert np.allclose(xyz, [5.0, -3.0, 210.0], atol=1e-9), xyz
    assert np.allclose(vel, 0.0), vel
    print("  seeding             first fix is adopted exactly, velocity starts at zero")


def test_tracks_constant_velocity_exactly():
    """With clean measurements the constant-velocity model should be exact."""
    data, dt, v = _straight_line(noise=0.0)
    f = PoseFilter()
    for t, true, meas in data:
        f.update(FakePose(t, meas, [0, 0, -1.0]))
    xyz, vel, _ = f.update(FakePose(data[-1][0] + dt, data[-1][2] + v * dt, [0, 0, -1.0]))
    assert np.allclose(vel, v, atol=0.5), f"velocity {vel} vs {v}"
    print(f"  constant velocity   recovered {np.round(vel, 2)} against truth {v}")


def test_velocity_beats_finite_difference_on_noise():
    """The whole reason the filter exists, on synthetic noise.

    Uses white noise here deliberately -- this is a property of differencing, not
    of the correlated real residual, and it holds either way. The correlated
    case is measured for real in validation/trajectory.py, where finite
    differencing is worse still.
    """
    data, dt, v = _straight_line(noise=0.4, seed=7)
    f = PoseFilter()
    kal, fd, prev = [], [], None
    for t, true, meas in data:
        _, vel, _ = f.update(FakePose(t, meas, [0, 0, -1.0]))
        kal.append(vel)
        fd.append((meas - prev) / dt if prev is not None else np.zeros(3))
        prev = meas

    warm = len(data) // 4
    k_err = float(np.sqrt(np.mean(np.sum((np.array(kal[warm:]) - v) ** 2, axis=1))))
    f_err = float(np.sqrt(np.mean(np.sum((np.array(fd[warm:]) - v) ** 2, axis=1))))
    print(f"  velocity vs diff    Kalman {k_err:.2f} mm/s vs finite difference {f_err:.1f} mm/s "
          f"({f_err/max(k_err,1e-9):.0f}x better)")
    assert k_err < f_err / 5.0, (k_err, f_err)


def test_coasts_through_a_dropout():
    """A brief loss should extrapolate, not freeze."""
    data, dt, v = _straight_line(n=200, noise=0.0)
    f = PoseFilter()
    for t, true, meas in data:
        f.update(FakePose(t, meas, [0, 0, -1.0]))

    last_t = data[-1][0]
    last_xyz = data[-1][2]
    coasted = None
    for k in range(1, 11):  # ~40 ms of dropout at 240 fps
        coasted = f.update(None, t=last_t + k * dt)
    assert coasted is not None, "should still be tracking after a short gap"

    expected = last_xyz + v * (10 * dt)
    moved = float(np.linalg.norm(coasted[0] - last_xyz))
    assert moved > 0.5, f"did not coast at all: moved {moved:.3f} mm"
    assert np.allclose(coasted[0], expected, atol=1.0), (coasted[0], expected)
    print(f"  dropout coasting    extrapolated {moved:.2f} mm over 10 lost frames")


def test_gives_up_after_a_long_dropout():
    """Extrapolating forever is how an estimator becomes confidently wrong."""
    data, dt, _ = _straight_line(n=100, noise=0.0)
    f = PoseFilter(max_coast_s=0.05)
    for t, _, meas in data:
        f.update(FakePose(t, meas, [0, 0, -1.0]))
    assert f.update(None, t=data[-1][0] + 1.0) is None
    print("  long dropout        gives up rather than extrapolate indefinitely")


def test_reseeds_cleanly_after_giving_up():
    f = PoseFilter(max_coast_s=0.05)
    f.update(FakePose(0.0, [0.0, 0.0, 220.0], [0, 0, -1.0]))
    assert f.update(None, t=5.0) is None
    xyz, vel, _ = f.update(FakePose(5.1, [40.0, 10.0, 300.0], [0, 0, -1.0]))
    assert np.allclose(xyz, [40.0, 10.0, 300.0], atol=1e-9), xyz
    assert np.allclose(vel, 0.0), vel
    print("  re-seeding          adopts the next fix exactly, with no stale velocity")


def test_predict_ahead_leads_the_state():
    data, dt, v = _straight_line(n=300, noise=0.0)
    f = PoseFilter()
    for t, _, meas in data:
        f.update(FakePose(t, meas, [0, 0, -1.0]))

    now = f.update(FakePose(data[-1][0] + dt, data[-1][2] + v * dt, [0, 0, -1.0]))[0]
    ahead = f.predict_ahead(0.0025)[0]  # the measured ~2.5 ms grab-to-pose latency
    assert np.allclose(ahead - now, v * 0.0025, atol=0.05), (ahead - now, v * 0.0025)
    # And it must not have mutated the filter.
    assert np.allclose(f.update(None, t=data[-1][0] + dt)[0], now, atol=0.5)
    print(f"  predict_ahead       leads by {np.linalg.norm(ahead-now):.3f} mm at 2.5 ms, "
          f"non-mutating")


def test_normal_sign_flip_does_not_destabilise():
    """The normal's sign is unobservable; a flipped measurement must not knock it over."""
    f = PoseFilter()
    n = np.array([0.3, 0.1, -0.948])
    n = n / np.linalg.norm(n)
    for i in range(120):
        t = i / 240.0
        meas = n if i % 2 == 0 else -n  # branch flapping
        f.update(FakePose(t, [0.0, 0.0, 220.0], meas))
    _, _, out = f.update(FakePose(0.5, [0.0, 0.0, 220.0], n))
    assert abs(abs(float(out @ n)) - 1.0) < 0.02, out
    assert abs(np.linalg.norm(out) - 1.0) < 1e-9, "normal must stay unit length"
    print("  normal sign flaps   absorbed; output stays unit and aligned")


def test_normal_stays_unit():
    rng = np.random.default_rng(3)
    f = PoseFilter()
    for i in range(200):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        _, _, out = f.update(FakePose(i / 240.0, [0.0, 0.0, 220.0], n))
        assert abs(np.linalg.norm(out) - 1.0) < 1e-9, np.linalg.norm(out)
    print("  normal magnitude    unit to 1e-9 over 200 random updates")


def test_depth_noise_scales_with_range():
    """Depth uncertainty is a fraction of range, so far measurements get less trust."""
    near, far = PoseFilter(), PoseFilter()
    for i in range(60):
        t = i / 240.0
        near.update(FakePose(t, [0.0, 0.0, 150.0], [0, 0, -1.0]))
        far.update(FakePose(t, [0.0, 0.0, 350.0], [0, 0, -1.0]))
    assert far.pos.P[2, 2] > near.pos.P[2, 2], (far.pos.P[2, 2], near.pos.P[2, 2])
    print(f"  range-scaled noise  depth variance {near.pos.P[2,2]:.4f} at 150 mm vs "
          f"{far.pos.P[2,2]:.4f} at 350 mm")


if __name__ == "__main__":
    print("pose filter tests")
    fail = 0
    for fn in (
        test_returns_none_before_first_detection,
        test_seeds_on_first_measurement,
        test_tracks_constant_velocity_exactly,
        test_velocity_beats_finite_difference_on_noise,
        test_coasts_through_a_dropout,
        test_gives_up_after_a_long_dropout,
        test_reseeds_cleanly_after_giving_up,
        test_predict_ahead_leads_the_state,
        test_normal_sign_flip_does_not_destabilise,
        test_normal_stays_unit,
        test_depth_noise_scales_with_range,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
