"""Constant-velocity Kalman filter over the 5-DOF pose.

**This filter does not reduce the position residual, and it is not meant to.**
That was the original motivation and measurement killed it, so the reasoning is
recorded here to stop anyone re-deriving it.

The hypothesis was that per-frame error is white noise, and that with motion at
sub-Hz against a 240-420 fps camera there would be enormous averaging available.
The error is not white.  Measured along a rendered trajectory, the depth error
autocorrelates at **r = 0.966 after one frame** and stays above 0.5 for
**408 ms** -- it is a smooth function of pose, not measurement noise, and no
amount of averaging removes a bias that persists.  Filtering position was tried
across process noise from 15 to 4000 mm/s^2 and never beat the raw measurement:
the best result was 1.01x, and tighter settings made it monotonically worse.

Blade phase does not rescue it either.  Re-rendering with the rotor spinning at
330 Hz -- so blade phase is fully aliased, as on the real robot -- only moved the
one-frame autocorrelation from 0.966 to 0.886.  The reason is geometric: the
segmenter takes a convex hull, the hull is set by the outermost features, and
the outermost feature is the **rotationally symmetric duct rim**.  Spinning the
blades inside the ring barely changes the silhouette.

What the filter is genuinely for:

* **Velocity.**  This is the real justification.  The LQR in
  `ai/simulate_hover.py` consumes velocity, and a raw finite difference is
  hopeless: measured against analytic truth it gives **64.0 mm/s RMSE** on a
  trajectory whose true speed is only 7.6-24.4 mm/s -- the noise is several times
  the signal.  The 5 Hz IIR that `simulate_hover.py` uses gets that to 6.8 mm/s;
  this filter gets **3.5 mm/s**, twice better again and 18x better than the bare
  difference.  Position error being correlated is precisely why differencing it
  is so bad and why a model-based rate estimate is so much better.
* **Coasting through dropouts**, instead of freezing on the last fix.
* **Latency compensation**, via `predict_ahead` -- see below.

Two independent constant-velocity filters:

* **position** -- a 6-state ``[x, y, z, vx, vy, vz]``.  Take the *velocity* from
  here.  The filtered position is 1.4% worse than the raw measurement (see
  above), so use `estimator.Pose.xyz_mm` for position unless you specifically
  need the coasting or prediction behaviour.
* **orientation** -- a 6-state on the normal vector and its rate.  Filtering the
  normal as a 3-vector and renormalising avoids the wrap and the singularity
  that filtering (theta, phi) would hit: phi is undefined at zero tilt, and no
  amount of filtering fixes a coordinate that does not exist there.

Measurement noise defaults come from `validation/tune.py` on held-out data --
0.13 mm lateral, 0.36% of range in depth, 1.24 degrees on the normal -- so the
gains are set by measurement rather than by taste.  Depth noise is range
dependent and is rebuilt each update, which matters: a fixed value would
over-trust depth far away and under-trust it up close.

`predict(dt)` extrapolates forward, which is how the measured ~2.5 ms of
grab-to-pose latency gets handed to a controller as a current estimate rather
than a stale one.
"""

from __future__ import annotations

import numpy as np

# Measured 1-sigma per-frame noise, from the held-out test split.
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


class _ConstantVelocity:
    """A 3-channel constant-velocity Kalman filter, state ``[p(3), v(3)]``.

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

    def reset(self):
        self.x[:] = 0.0
        self.initialised = False

    def _predict_matrices(self, dt):
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        # Continuous white-noise acceleration, discretised exactly.
        q = np.zeros((6, 6))
        s = self.accel ** 2
        q[:3, :3] = np.eye(3) * (dt ** 4 / 4.0) * s
        q[:3, 3:] = np.eye(3) * (dt ** 3 / 2.0) * s
        q[3:, :3] = np.eye(3) * (dt ** 3 / 2.0) * s
        q[3:, 3:] = np.eye(3) * (dt ** 2) * s
        return f, q

    def predict(self, dt):
        if dt <= 0:
            return
        f, q = self._predict_matrices(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q

    def update(self, z, sigma):
        """Fuse a measurement with per-channel standard deviations ``sigma``."""
        z = np.asarray(z, dtype=np.float64)
        r = np.diag(np.asarray(sigma, dtype=np.float64) ** 2)

        if not self.initialised:
            # Seed from the first measurement rather than from zero, so the
            # filter does not spend its first second sliding in from the origin.
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P[:3, :3] = r
            self.initialised = True
            return self.x[:3].copy()

        h = np.zeros((3, 6))
        h[:, :3] = np.eye(3)
        y = z - h @ self.x
        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(6) - k @ h) @ self.P
        return self.x[:3].copy()

    @property
    def value(self):
        return self.x[:3].copy()

    @property
    def rate(self):
        return self.x[3:].copy()

    def peek(self, dt):
        """Where the state will be in ``dt`` seconds, without mutating it."""
        f, _ = self._predict_matrices(dt)
        return (f @ self.x)[:3].copy()


class PoseFilter:
    """Filters `estimator.Pose` measurements into a smoothed state.

    Usage is one call per frame::

        filt = PoseFilter()
        state = filt.update(pose)        # None until the first detection

    Dropouts are handled by continuing to predict: `update(None)` still advances
    the state, so a brief occlusion coasts on velocity instead of freezing. After
    ``max_coast_s`` the filter gives up and re-seeds on the next detection,
    because extrapolating a constant velocity indefinitely is how an estimator
    ends up confidently wrong.
    """

    def __init__(self, accel_mm_s2=ACCEL_MM_S2, accel_normal=ACCEL_NORMAL,
                 sigma_lateral_mm=SIGMA_LATERAL_MM, sigma_depth_frac=SIGMA_DEPTH_FRAC,
                 sigma_normal=SIGMA_NORMAL, max_coast_s=0.15):
        self.pos = _ConstantVelocity(accel_mm_s2, p0_pos=100.0, p0_vel=1e4)
        self.nrm = _ConstantVelocity(accel_normal, p0_pos=1.0, p0_vel=100.0)
        self.sigma_lateral_mm = sigma_lateral_mm
        self.sigma_depth_frac = sigma_depth_frac
        self.sigma_normal = sigma_normal
        self.max_coast_s = max_coast_s
        self._t = None
        self._last_seen = None

    def reset(self):
        self.pos.reset()
        self.nrm.reset()
        self._t = None
        self._last_seen = None

    def update(self, pose, t=None):
        """Fuse one frame. ``pose`` may be ``None`` for a lost frame.

        Returns ``(xyz, velocity, normal)`` or ``None`` if nothing is tracked
        yet.
        """
        now = (pose.t if pose is not None else t) if (pose is not None or t is not None) else None
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
        sigma_pos = (self.sigma_lateral_mm, self.sigma_lateral_mm, self.sigma_depth_frac * z)

        self.pos.update(np.asarray(pose.xyz_mm, dtype=np.float64), sigma_pos)
        n = np.asarray(pose.normal, dtype=np.float64)
        # Keep the measurement on the same side as the current estimate: the
        # normal's sign is not observable, and a flip would read as a huge
        # innovation and knock the filter over.
        if self.nrm.initialised and float(n @ self.nrm.value) < 0:
            n = -n
        self.nrm.update(n, (self.sigma_normal,) * 3)

        self._last_seen = now
        return self._state()

    def _state(self):
        if not self.pos.initialised:
            return None
        n = self.nrm.value
        norm = np.linalg.norm(n)
        return self.pos.value, self.pos.rate, (n / norm if norm > 1e-9 else n)

    def predict_ahead(self, dt):
        """State extrapolated ``dt`` seconds forward, for latency compensation.

        The measured grab-to-pose latency is ~2.5 ms at 640x480; handing a
        controller ``predict_ahead(latency)`` gives it an estimate for now rather
        than for when the shutter closed.
        """
        if not self.pos.initialised:
            return None
        n = self.nrm.peek(dt)
        norm = np.linalg.norm(n)
        return self.pos.peek(dt), self.pos.rate, (n / norm if norm > 1e-9 else n)
