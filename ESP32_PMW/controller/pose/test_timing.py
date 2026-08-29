#!/usr/bin/env python3
"""
The live loop's answer to two cameras that do not fire together.

`calib/calibrate.py` avoids skew: the board is static, so the rig re-reads until the two
frames land within 2 ms and the operator waits. None of that is available here. The robot
does not hold still, there is no second chance at a frame, and a frame of latency is the
one thing this loop cannot spend. So `stereo.fuse` prices the skew instead of avoiding it,
and this checks that the price is right.

Cheap accuracy is not the point -- an inflated covariance is trivially "safe" if you make it
big enough. The test that matters is **consistency**: the normalised error
``e^T Sigma^-1 e`` must average the number of degrees of freedom, 3. Below that the filter
is being lied to in the pessimistic direction and will ignore good measurements; above it,
the optimistic direction, and it will trust skewed ones. Section 17 of `theory.md` derives
the terms.

    python test_timing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Same layering `stereo.py` sets up: pose is stage 3, calib is stage 2.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import rig as rigmod  # noqa: E402
import stereo  # noqa: E402

SKEW_S = 0.008          # one frame period at 119 fps, the worst case rather than the median
SPEED_MM_S = 40.0       # a brisk translation; hover is 15-22
SIGMA_V = 5.0           # how well the filter knows the velocity, mm/s
N_TRIALS = 400
DOF = 3


class _View:
    """What `fuse` needs of a pose: a centre and a normal in camera coordinates."""

    def __init__(self, center, normal):
        self.center = center
        self.normal = normal


def _trial(rig, truth, normal, velocity, seed, corrected):
    """One frame: each camera sees the robot where it was at that camera's own instant."""

    rng = np.random.default_rng(seed)
    poses, stamps = [], []
    for i, cam in enumerate(rig.cameras):
        t_i = 0.0 if i == 0 else SKEW_S
        world = truth + velocity * t_i
        r, t = cam.T_world_cam[:3, :3], cam.T_world_cam[:3, 3]
        c_cam = r.T @ (world - t) + rng.normal(0.0, 0.08, 3)
        poses.append(_View(c_cam, r.T @ normal))
        stamps.append(t_i)

    if not corrected:
        return stereo.fuse(poses, rig)
    return stereo.fuse(poses, rig, stamps=stamps, velocity=velocity,
                       vel_cov=np.eye(3) * SIGMA_V**2)


def run():
    rig = rigmod.StereoRig.from_spherical(elev_deg=(45, 45), azim_deg=(0, 90), range_mm=300)
    truth = np.array([5.0, -3.0, 2.0])
    normal = np.array([0.0, 0.0, 1.0])
    velocity = np.array([SPEED_MM_S, 0.0, 0.0])
    # `fuse` returns the estimate at the mean of the stamps, so that is what to score against.
    reference = truth + velocity * (SKEW_S / 2)

    out = {}
    for label, corrected in (("uncorrected", False), ("shift + inflate", True)):
        errs, nis = [], []
        for seed in range(N_TRIALS):
            centre, _, cov = _trial(rig, truth, normal, velocity, seed, corrected)
            e = centre - reference
            errs.append(float(np.linalg.norm(e)))
            nis.append(float(e @ np.linalg.inv(cov) @ e))
        out[label] = (float(np.mean(errs)), float(np.mean(nis)))
        print(f"{label:16s} error {out[label][0]:6.3f} mm   "
              f"normalised error {out[label][1]:5.2f} (want {DOF})")

    err_raw, nis_raw = out["uncorrected"]
    err_fix, nis_fix = out["shift + inflate"]

    assert err_fix < err_raw, "the shift must reduce the error, not just the confidence"
    assert nis_raw > 1.5 * DOF, (
        f"uncorrected fusion should look overconfident here ({nis_raw:.2f}); if it does not, "
        f"the skew is below the noise floor and this test proves nothing")
    assert 0.6 * DOF < nis_fix < 1.6 * DOF, (
        f"corrected covariance is not honest: normalised error {nis_fix:.2f} against {DOF}")

    # With stamps omitted nothing may change: every existing caller relies on it.
    a = stereo.fuse([_View(np.array([1.0, 2.0, 300.0]), np.array([0.0, 0.0, 1.0]))] * 2, rig)
    b = stereo.fuse([_View(np.array([1.0, 2.0, 300.0]), np.array([0.0, 0.0, 1.0]))] * 2, rig,
                    stamps=None, velocity=None, vel_cov=None)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2]), "default path moved"
    print("\nuntimed fuse is unchanged\nall timing checks passed")


if __name__ == "__main__":
    run()
