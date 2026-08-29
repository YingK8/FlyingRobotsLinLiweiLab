"""
Constant-velocity Kalman filter over the 5-DOF pose.

**Take velocity from here, not position.** The filtered position is 1.4% worse
than the raw measurement, because per-frame error is not white noise: depth error
autocorrelates at r = 0.966 after one frame and stays above 0.5 for 408 ms. It is
a smooth function of pose, and averaging does not remove a persistent bias. Use
`estimator.Pose.xyz_mm` for position unless you need the coasting or prediction.

What it is for:

    velocity     a raw finite difference gives 64.0 mm/s RMSE on a trajectory whose
                 true speed is 7.6-24.4 mm/s; this gives **3.5 mm/s**. Position
                 error being correlated is exactly why differencing it is so bad.
    coasting     through dropouts, instead of freezing on the last fix
    latency      `predict_ahead` hands a controller a current estimate rather than
                 one ~2.5 ms stale

Two independent filters:

    position     6-state [x, y, z, vx, vy, vz]
    orientation  6-state on the normal vector and its rate. Filtering the normal
                 as a 3-vector and renormalising avoids the wrap and singularity
                 that (theta, phi) would hit -- phi is undefined at zero tilt.

Measurement noise comes from `noise.py` when a static calibration has been recorded,
and from the rendered fallbacks below when it has not. Depth noise is range dependent
and rebuilt each update; a fixed value would over-trust depth far away and under-trust
it up close. Given a rig, R is the *fused* 3x3 of the two views rather than a triple:
cameras 90 degrees apart in azimuth do not give a world-axis-aligned covariance.
"""

from __future__ import annotations

import numpy as np

# Per-frame 1-sigma noise, from the held-out split of *rendered* data.
#
# These are the fallbacks, not the answer. `noise.py` measures the real thing on a
# stationary robot and writes `noise_model.json`; when that file exists the defaults
# below are replaced by what the bench actually does. They stay here because the
# first static run has to be estimated by something, and because a rendered number
# that is labelled as one is honest -- silently shipping it as "measured" was not.
SIGMA_LATERAL_MM = 0.13
SIGMA_DEPTH_FRAC = 0.0036  # of range
SIGMA_NORMAL = 0.022  # unit-vector components, ~= sin(1.24 deg)

# Process noise as an acceleration, in mm/s^2, tuned for velocity RMSE against
# analytic truth on a rendered trajectory:
#     4000 -> 5.3 mm/s    1500 -> 3.8    500 -> 3.5    150 -> 3.9
# 500 is the optimum and sits comfortably above the robot's own ~1400 mm/s^2
# peak climb divided by the margin a constant-velocity model needs, so a real
# manoeuvre is not lagged.
ACCEL_MM_S2 = 500.0
ACCEL_NORMAL = 5.0  # unit-vector components per s^2, scaled the same way

#: How far a measurement may sit from the filter's prediction before it is refused, in
#: sigmas of the innovation's own covariance, and how many refusals may run consecutively.
#:
#: The estimator hands over frames that are simply wrong -- a quarter-turn branch flip is
#: 84 degrees out and an ellipse fitted to a collapsed seed can be hundreds of
#: millimetres out (`pose/theory.md` 16.23) -- and fusing one drags the track for as long
#: as the filter takes to recover. `S = H P H' + R` already says how far a measurement
#: is *allowed* to fall, so the test costs a solve of a 3x3 that was being computed
#: anyway and thrown away.
#:
#: Refusing is not dropping. The state has already been predicted forward, so a refused
#: frame keeps the constant-velocity extrapolation, which is the right answer for one
#: frame at 60 fps and the reason the model is there.
#:
#: `MAX_GATED` is the safety catch: a manoeuvre the model did not anticipate would
#: otherwise be refused forever, with the filter growing more confident in its own
#: extrapolation every frame. A gate that can lock on is worse than no gate.
GATE_SIGMA = 4.0
MAX_GATED = 5


class _ConstantVelocity:
    """
    A 3-channel constant-velocity Kalman filter, state ``[p(3), v(3)]``.

        Channels are kept independent (block-diagonal), which is exact here: the
        measurement errors in x, y and z come from different parts of the ellipse
        fit -- centroid for lateral, size for depth -- and are uncorrelated.
    """

    def __init__(self, accel, p0_pos, p0_vel):
        self.n = 3
        self.x = np.zeros(6)
        self.P = np.diag([p0_pos] * 3 + [p0_vel] * 3).astype(float)
        self.accel = float(accel)
        self.initialised = False
        self.n_gated = 0

    def reset(self):
        self.x[:] = 0.0
        self.initialised = False
        self.n_gated = 0

    def _predict_matrices(self, dt):
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        # Continuous white-noise acceleration, discretised exactly.
        q = np.zeros((6, 6))
        s = self.accel**2
        q[:3, :3] = np.eye(3) * (dt**4 / 4.0) * s
        q[:3, 3:] = np.eye(3) * (dt**3 / 2.0) * s
        q[3:, :3] = np.eye(3) * (dt**3 / 2.0) * s
        q[3:, 3:] = np.eye(3) * (dt**2) * s
        return f, q

    def predict(self, dt):
        if dt <= 0:
            return
        f, q = self._predict_matrices(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q

    def update(self, z, sigma, gate=None):
        """
        Fuse a measurement with per-channel standard deviations ``sigma``.

            ``sigma`` may instead be a full 3x3 covariance. Two cameras 90 degrees
            apart in azimuth do not produce a covariance aligned to the world axes,
            so the triple is the *monocular* geometry and the matrix is the stereo
            one -- see `noise.NoiseModel.sigma_pos`.

            ``gate`` rejects a measurement whose innovation is more than that many
            sigmas from where the filter expected it, in the innovation's own metric --
            `y' S^-1 y` normalised by the three degrees of freedom, where ``S`` is the
            innovation covariance this already computes and then threw away. A rejected
            frame is not a lost frame: the state has already been predicted forward by
            the caller, so the filter simply keeps the extrapolation, which is what a
            constant-velocity model is for.

            Bounded by `MAX_GATED` consecutive rejections, after which the measurement
            is taken regardless. Without that, a genuine manoeuvre the model did not
            anticipate would be rejected forever -- the filter would grow more confident
            in its extrapolation every frame while drifting further from the robot, and
            a gate that can lock itself on is worse than no gate.

            Returns ``(value, accepted)``.
        """

        z = np.asarray(z, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        r = np.diag(sigma**2) if sigma.ndim == 1 else sigma

        if not self.initialised:
            # Seed from the first measurement rather than from zero, so the
            # filter does not spend its first second sliding in from the origin.
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P[:3, :3] = r
            self.initialised = True
            self.n_gated = 0
            return self.x[:3].copy(), True

        h = np.zeros((3, 6))
        h[:, :3] = np.eye(3)
        y = z - h @ self.x
        s = h @ self.P @ h.T + r
        if gate is not None and self.n_gated < MAX_GATED:
            try:
                d = float(y @ np.linalg.solve(s, y)) / 3.0
            except np.linalg.LinAlgError:
                d = 0.0
            if d > gate * gate:
                self.n_gated += 1
                return self.x[:3].copy(), False
        self.n_gated = 0
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(6) - k @ h) @ self.P
        return self.x[:3].copy(), True

    @property
    def value(self):
        return self.x[:3].copy()

    @property
    def rate(self):
        return self.x[3:].copy()

    @property
    def rate_cov(self):
        """
        Covariance of `rate`, which is what an extrapolation costs.

            Shifting a measurement by ``dt`` using this velocity adds
            ``rate_cov * dt**2`` to its covariance -- see `stereo.fuse`, which uses it
            to price the skew between two free-running cameras.
        """

        return self.P[3:, 3:].copy()

    def peek(self, dt):
        """
        Where the state will be in ``dt`` seconds, without mutating it.
        """

        f, _ = self._predict_matrices(dt)
        return (f @ self.x)[:3].copy()


class PoseFilter:
    """
    Filters `estimator.Pose` measurements into a smoothed state.

        Usage is one call per frame::

            filt = PoseFilter()
            state = filt.update(pose)        # None until the first detection

        Dropouts are handled by continuing to predict: `update(None)` still advances
        the state, so a brief occlusion coasts on velocity instead of freezing. After
        ``max_coast_s`` the filter gives up and re-seeds on the next detection,
        because extrapolating a constant velocity indefinitely is how an estimator
        ends up confidently wrong.
    """

    def __init__(
        self,
        accel_mm_s2=ACCEL_MM_S2,
        accel_normal=ACCEL_NORMAL,
        sigma_lateral_mm=None,
        sigma_depth_frac=None,
        sigma_normal=None,
        max_coast_s=0.15,
        gate=GATE_SIGMA,
        noise=None,
        rig=None,
    ):
        # `noise` is a `noise.NoiseModel`; absent, the measured one on disk is used,
        # and absent that, the rendered fallbacks above. An explicit sigma always
        # wins over all three -- callers that pass one mean it.
        if noise is None:
            from noise import NoiseModel
            noise = NoiseModel.load()
        self.noise = noise
        # With a rig, R is the fused covariance of two cameras that are not aligned
        # to the world axes; without one it is the axis-aligned triple. Held here
        # rather than resolved per frame: it does not change during a run.
        self.rig = rig

        self.pos = _ConstantVelocity(accel_mm_s2, p0_pos=100.0, p0_vel=1e4)
        self.nrm = _ConstantVelocity(accel_normal, p0_pos=1.0, p0_vel=100.0)
        self.sigma_lateral_mm = (
            noise.sigma_lateral_mm if sigma_lateral_mm is None else sigma_lateral_mm)
        self.sigma_depth_frac = (
            noise.sigma_depth_frac_world if sigma_depth_frac is None
            else sigma_depth_frac)
        self.sigma_normal = noise.sigma_normal if sigma_normal is None else sigma_normal
        self.max_coast_s = max_coast_s
        # `None` disables the innovation gate and fuses every frame, which is what the
        # filter did before.
        self.gate = gate
        self.n_gated = 0
        self._t = None
        self._last_seen = None

    def reset(self):
        self.pos.reset()
        self.nrm.reset()
        self._t = None
        self._last_seen = None

    def update(self, pose, t=None):
        """
        Fuse one frame. ``pose`` may be ``None`` for a lost frame.

                Returns ``(xyz, velocity, normal)`` or ``None`` if nothing is tracked
                yet.
        """

        now = (
            (pose.t if pose is not None else t)
            if (pose is not None or t is not None)
            else None
        )
        if now is None:
            return None

        if self._t is not None:
            dt = now - self._t
            if dt > 0:
                self.pos.predict(dt)
                self.nrm.predict(dt)
        self._t = now

        if pose is None:
            if self._last_seen is None or (now - self._last_seen) > self.max_coast_s:
                self.reset()
                self._t = now
                return None
            return self._state()

        # Depth noise scales with range; lateral noise does not.
        z = max(1.0, abs(float(pose.xyz_mm[2])))
        if self.rig is None:
            sigma_pos = (
                self.sigma_lateral_mm,
                self.sigma_lateral_mm,
                self.sigma_depth_frac * z,
            )
        else:
            # `rig.position_covariance` combines the two views' anisotropic error in
            # information form and returns the 3x3 in world coordinates. The triple
            # above is what that reduces to for one camera looking down world z; for
            # this rig, whose cameras sit 90 degrees apart in azimuth, it does not.
            sigma_pos = self.rig.position_covariance(
                self.noise.sigma_lat_mm, self.noise.sigma_depth_mm(z))

        _, took_pos = self.pos.update(
            np.asarray(pose.xyz_mm, dtype=np.float64), sigma_pos, gate=self.gate)
        n = np.asarray(pose.normal, dtype=np.float64)
        # Keep the measurement on the same side as the current estimate: the
        # normal's sign is not observable, and a flip would read as a huge
        # innovation and knock the filter over.
        if self.nrm.initialised and float(n @ self.nrm.value) < 0:
            n = -n
        _, took_nrm = self.nrm.update(n, (self.sigma_normal,) * 3, gate=self.gate)
        self.n_gated += not (took_pos and took_nrm)

        self._last_seen = now
        return self._state()

    def _state(self):
        if not self.pos.initialised:
            return None
        n = self.nrm.value
        norm = np.linalg.norm(n)
        return self.pos.value, self.pos.rate, (n / norm if norm > 1e-9 else n)

    def predict_ahead(self, dt):
        """
        State extrapolated ``dt`` seconds forward, for latency compensation.

                The measured grab-to-pose latency is ~2.5 ms at 640x480; handing a
                controller ``predict_ahead(latency)`` gives it an estimate for now rather
                than for when the shutter closed.
        """

        if not self.pos.initialised:
            return None
        n = self.nrm.peek(dt)
        norm = np.linalg.norm(n)
        return self.pos.peek(dt), self.pos.rate, (n / norm if norm > 1e-9 else n)


def _check():
    """The gate refuses one bad frame and yields to a real manoeuvre."""

    class _P:
        def __init__(self, x, t):
            self.xyz_mm = np.asarray(x, dtype=np.float64)
            self.normal = np.array([0.0, 0.0, 1.0])
            self.t = t

    f = PoseFilter()
    for i in range(30):
        f.update(_P([i * 0.5, 0.0, 200.0], i / 60.0))
    settled = f.pos.value.copy()

    f.update(_P([500.0, 500.0, 200.0], 30 / 60.0))
    moved = float(np.linalg.norm(f.pos.value - settled))
    assert f.n_gated == 1, f"outlier was fused, n_gated={f.n_gated}"
    assert moved < 5.0, f"a refused frame still dragged the track {moved:.1f} mm"

    # Sustained, so it is a manoeuvre and not an outlier: MAX_GATED has to let it in.
    for i in range(31, 31 + 2 * MAX_GATED + 4):
        f.update(_P([500.0, 500.0, 200.0], i / 60.0))
    moved = float(np.linalg.norm(f.pos.value - settled))
    assert moved > 100.0, f"the gate locked on: track moved only {moved:.1f} mm"
    print("filter self-check ok (gate refuses an outlier, yields to a manoeuvre)")

    # A full 3x3 R tracks the same target as the equivalent triple. Correlated R is
    # the whole point of passing a rig, so the check uses one with real off-diagonal
    # terms rather than a diagonal matrix that would pass either way.
    class _Rig:
        def position_covariance(self, lat, dep):
            c = np.diag([lat**2, lat**2, dep**2])
            c[0, 2] = c[2, 0] = 0.5 * lat * dep
            return c

    rig = _Rig()
    g = PoseFilter(rig=rig)
    g.update(_P([0.0, 0.0, 200.0], 0.0))
    # The seed sets P[:3,:3] = R, so this is a direct read-back: the off-diagonal
    # proves the matrix went in whole rather than being flattened to its diagonal.
    want = rig.position_covariance(g.noise.sigma_lat_mm, g.noise.sigma_depth_mm(200.0))
    assert np.allclose(g.pos.P[:3, :3], want), g.pos.P[:3, :3]
    assert abs(want[0, 2]) > 0, "the test rig must have an off-diagonal term"
    for i in range(1, 30):
        g.update(_P([i * 0.5, 0.0, 200.0], i / 60.0))
    d = float(np.linalg.norm(g.pos.value - settled))
    assert d < 5.0, f"the 3x3 R path tracks elsewhere than the triple: {d:.2f} mm"
    assert g.n_gated == 0, f"the 3x3 R path gated a clean track, n_gated={g.n_gated}"
    print(f"  3x3 R reaches the filter whole (off-diagonal {want[0, 2]:.4f}), "
          f"tracks to {d:.3f} mm")


if __name__ == "__main__":
    _check()
