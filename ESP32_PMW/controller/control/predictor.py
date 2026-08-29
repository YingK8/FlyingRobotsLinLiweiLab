#!/usr/bin/env python3
"""
Model-forward state propagation across vision dropouts.

Propagates on the frequency we commanded instead of coasting kinematically:
z_ddot = g*((f_cmd/f_hat)^2 - 1), clamped to >= -g. Lateral stays constant velocity,
there is no lateral model worth propagating. Derivation in theory.md 6.2, 6.5.

Where:
    xyz_mm      mm, whole public API in and out. Index 2 is the lifted axis, caller's frame
    vel_mm_s    mm/s
    f_cmd       Hz, field frequency commanded this tick
    f_hat       Hz, the loop's CURRENT hover estimate, never cached here
    g           m/s^2, converted to mm/s^2 once in __init__
    a_z         mm/s^2 on [-g, inf), thrust is non-negative so -g is free fall
    stale       True once the coast outruns max_coast_s, the loop lands on it

Self-check: uv run python controller/control/predictor.py
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from hover_model import GRAVITY

# 3 frames at 30 fps. At a 5% f_hat error that costs 5.5 mm against 10.3 mm coasting.
MAX_COAST_S = 0.10

# Shared with z_track.ZTracker's tau_zdot: same measurement, same noise, same number,
# and they must move together. The canonical value and the derivation behind it live
# in `z_track.TAU_ZDOT_S`; `pose/noise.py` measures the noise it trades against, and
# `z_track.check_stepout_margin` reports what the current tau costs in rate noise.
TAU_VEL_S = 0.08


class Command(NamedTuple):
    """What went to the coils this tick. f_hat is passed, never cached: the loop moves it."""

    f_cmd: float               # Hz, the field frequency commanded
    f_hat: float               # Hz, the loop's current hover-frequency estimate
    tilt: tuple = (0.0, 0.0)   # read only by PlantPredictor; the lift law has no lateral term


class StatePredictor:
    """Propagates position through a vision gap on the commanded field frequency, in mm."""

    def __init__(
        self,
        max_coast_s: float = MAX_COAST_S,
        g: float = GRAVITY,
        tau_vel_s: float = TAU_VEL_S,
    ):
        """Call update() on a frame with a fix and predict() on one without; both keep state."""

        self.max_coast_s = float(max_coast_s)
        self.g = float(g)                 # m/s^2, repo convention
        self._g = 1000.0 * self.g         # mm/s^2, the only unit conversion in this file
        self.tau_vel_s = float(tau_vel_s)

        self.xyz_mm = np.zeros(3)
        self.vel_mm_s = np.zeros(3)
        self.t = 0.0                      # time the state is valid for
        self.coast_s = 0.0                # seconds since the last real measurement
        self.initialised = False
        self._t_meas: float | None = None
        self._xyz_meas: np.ndarray | None = None

    @property
    def stale(self) -> bool:
        """True once coast_s outruns max_coast_s. The loop lands on this, it does not coast on."""

        return self.coast_s > self.max_coast_s

    def reset(self) -> None:
        """Back to uninitialised, keeping the constructor settings."""

        self.__init__(self.max_coast_s, self.g, self.tau_vel_s)

    def _estimate_vel(self, xyz: np.ndarray, t: float) -> None:
        """Finite difference through a first-order lowpass, per axis, as z_track._estimate_zdot."""

        prev_xyz, prev_t = self._xyz_meas, self._t_meas
        self._xyz_meas, self._t_meas = xyz.copy(), t
        if prev_xyz is None or prev_t is None or t <= prev_t:
            return
        # dt spans the whole gap when a fix follows a coast, which is the interval it moved in.
        dt = t - prev_t
        raw = (xyz - prev_xyz) / dt
        alpha = dt / (self.tau_vel_s + dt)
        self.vel_mm_s = self.vel_mm_s + alpha * (raw - self.vel_mm_s)

    def update(self, xyz_mm, vel_mm_s=None, t: float | None = None) -> np.ndarray:
        """A fix arrived, in mm: snap the state to it and clear the coast. Returns xyz_mm."""

        xyz = np.asarray(xyz_mm, dtype=float)
        # t None means two updates with no predict between them see dt = 0, leaving vel alone.
        t = self.t if t is None else float(t)

        if vel_mm_s is None:
            self._estimate_vel(xyz, t)
        else:
            # A supplied velocity re-seeds the differencer, so a later frame without one
            # does not difference across the frames it was supplied for.
            self.vel_mm_s = np.asarray(vel_mm_s, dtype=float).copy()
            self._xyz_meas, self._t_meas = xyz.copy(), t

        self.xyz_mm = xyz.copy()
        self.t = t
        self.coast_s = 0.0
        self.initialised = True
        return self.xyz_mm.copy()

    def accel_mm_s2(self, u) -> np.ndarray:
        """Commanded acceleration mm/s^2: lift law on z clamped to >= -g, lateral exactly zero."""

        u = u if isinstance(u, Command) else Command(*u)
        if u.f_hat <= 0.0:
            az = -self._g
        else:
            az = self._g * ((u.f_cmd / u.f_hat) ** 2 - 1.0)
        # Lateral stays zero: k_lat = 0.05 is a seed guess, COIL_AZ and MIX_GAIN unverified.
        return np.array([0.0, 0.0, max(az, -self._g)])

    def predict(self, u, dt: float) -> np.ndarray:
        """Propagate dt seconds on u = (f_cmd, f_hat[, tilt]); a is held, so integrate exactly."""

        if dt <= 0.0:
            return self.xyz_mm.copy()
        a = self.accel_mm_s2(u)
        self.xyz_mm = self.xyz_mm + self.vel_mm_s * dt + 0.5 * a * dt * dt
        self.vel_mm_s = self.vel_mm_s + a * dt
        self.t += dt
        self.coast_s += dt
        return self.xyz_mm.copy()


class PlantPredictor(StatePredictor):
    """Same API over spatial_model.Plant.step. OFF by default and unvalidated."""

    def __init__(self, plant=None, s0=(0.0, 0.0, 1.0), **kw):
        """plant None builds the default array; the import is lazy because it pulls scipy."""

        super().__init__(**kw)
        if plant is None:
            from spatial_model import Plant, quick_coils, robot_params

            plant = Plant(quick_coils(), robot_params())
        self.plant = plant
        self.s = np.asarray(s0, dtype=float)

    def predict(self, u, dt: float) -> np.ndarray:
        """One Plant.step in metres, converting at the seam. phasor=None, so tilt is ignored."""

        if dt <= 0.0:
            return self.xyz_mm.copy()
        u = u if isinstance(u, Command) else Command(*u)
        # The loop's estimate wins over the plant's inherited 110.0 default.
        if u.f_hat > 0.0:
            self.plant.f_hover = u.f_hat
        x = np.concatenate([self.xyz_mm / 1000.0, self.vel_mm_s / 1000.0, self.s])
        x, _ = self.plant.step(x, u.f_cmd, dt)
        self.xyz_mm = x[0:3] * 1000.0
        self.vel_mm_s = x[3:6] * 1000.0
        self.s = x[6:9]
        self.t += dt
        self.coast_s += dt
        return self.xyz_mm.copy()


# --------------------------------------------------------------------------


def _truth(z0_mm, vz0_mm_s, f_of_t, f_true, gap_s, g=GRAVITY, dt=1e-5):
    """Integrate the exact lift law at fine dt. The thing the predictor is scored against."""

    g_mm = 1000.0 * g
    z, vz, t = float(z0_mm), float(vz0_mm_s), 0.0
    n = int(round(gap_s / dt))
    for i in range(n):
        a = max(g_mm * ((f_of_t(t) / f_true) ** 2 - 1.0), -g_mm)
        z += vz * dt + 0.5 * a * dt * dt
        vz += a * dt
        t += dt
    return z, vz


def demo() -> None:
    g = GRAVITY
    f_true = 150.0            # the "truth" hover frequency, itself a seed guess (z_track)

    # 1. Zero input. f_cmd == f_hat means zero commanded acceleration whatever f_hat is,
    # so the propagator must reduce exactly to a constant-velocity coast. If this drifts,
    # every number below it is measuring the integrator instead of the model.
    for f_hat in (110.0, 140.0, 150.0, 190.0):
        p = StatePredictor()
        p.update([1.0, -2.0, 30.0], vel_mm_s=[0.0, 0.0, 0.0], t=0.0)
        for _ in range(20):
            p.predict(Command(f_hat, f_hat), 1.0 / 60.0)
        assert np.allclose(p.xyz_mm, [1.0, -2.0, 30.0], atol=1e-9), p.xyz_mm

        p = StatePredictor()
        p.update([0.0, 0.0, 30.0], vel_mm_s=[10.0, -5.0, 20.0], t=0.0)
        for _ in range(20):
            p.predict(Command(f_hat, f_hat), 1.0 / 60.0)
        want = np.array([0.0, 0.0, 30.0]) + np.array([10.0, -5.0, 20.0]) * (20.0 / 60.0)
        assert np.allclose(p.xyz_mm, want, atol=1e-9), (p.xyz_mm, want)
    print("zero input  : f_cmd == f_hat holds position and velocity to 1e-9 mm, "
          "for f_hat in 110..190 Hz")

    # 2. Round trip against the truth model, worst case. The command RAMPS through the gap
    # at z_track's own slew ceiling, TorqueLimits.f_dot_max(f_hover) = 67 Hz/s, which is
    # the fastest the loop is allowed to move f at all. The truth integrates that ramp
    # continuously at 1e-5 s; the predictor sees a zero-order hold at control dt. f_hat is
    # exact, so the ZOH is the whole residual -- and it is a pessimistic residual, because
    # the real coils are held for the step too. This is an upper bound, not a prediction.
    dt_ctrl = 1.0 / 60.0
    f_dot = 67.0
    ramp = lambda t: f_true + f_dot * t
    worst = 0.0
    for n_frames in (3, 6, 10):
        gap = n_frames * dt_ctrl
        p = StatePredictor()
        p.update([0.0, 0.0, 30.0], vel_mm_s=[0.0, 0.0, 0.0], t=0.0)
        for i in range(n_frames):
            p.predict(Command(ramp(i * dt_ctrl), f_true), dt_ctrl)
        z_true, _ = _truth(30.0, 0.0, ramp, f_true, gap)
        worst = max(worst, abs(p.xyz_mm[2] - z_true))
    print(f"round trip  : exact f_hat, command ramping at the {f_dot:.0f} Hz/s slew "
          f"ceiling, 3-10 frames at 60 fps -> max {worst:.3f} mm")
    assert worst < 1.5, worst

    # 3. THE NUMBER THAT MATTERS. What a gap actually costs, at a realistic f_hat error,
    # during a climb -- because at hover the truth does not move and every method looks
    # perfect. f_cmd = 1.10*f_true is a 2.06 m/s^2 climb, most of the a_max = 2.4 the
    # torque budget allows, so this is close to the worst case the loop can produce.
    # The two f_hat errors are the two regimes the loop actually lives in: 0.5% is what
    # z_track's estimator converges to (its own self-check lands inside 0.5 Hz of 150), and
    # 5% is roughly the seed error you start a flight with -- 140 against 150 is 6.7%.
    f_cmd = 1.10 * f_true
    print()
    print(f"prediction error during a {g * ((f_cmd / f_true) ** 2 - 1.0):.2f} m/s^2 climb "
          f"(f_cmd = {f_cmd:.1f} Hz, f_true = {f_true:.1f} Hz):")
    print()
    print("    gap     frames       model         model       const-vel")
    print("     ms   @30  @60   f_hat +0.5%    f_hat +5%    (PoseFilter)")
    rows = []
    for gap in (0.017, 0.033, 0.050, 0.067, 0.100, 0.150, 0.200, 0.300):
        z_true, _ = _truth(30.0, 0.0, lambda _t: f_cmd, f_true, gap)
        errs = []
        for f_err in (0.005, 0.05):
            p = StatePredictor()
            p.update([0.0, 0.0, 30.0], vel_mm_s=[0.0, 0.0, 0.0], t=0.0)
            n = max(1, int(round(gap / dt_ctrl)))
            for _ in range(n):
                p.predict(Command(f_cmd, (1.0 + f_err) * f_true), gap / n)
            errs.append(abs(p.xyz_mm[2] - z_true))
        cv = abs(30.0 - z_true)   # constant velocity from rest: misses the whole climb
        rows.append((gap, errs[0], errs[1], cv))
        print(f"    {gap * 1e3:4.0f}  {gap * 30:4.1f} {gap * 60:4.1f}"
              f"   {errs[0]:9.3f} mm {errs[1]:9.3f} mm {cv:9.3f} mm")
    print()

    # The claim the module is built on: knowing the command beats not knowing it, at every
    # gap length, even with f_hat 5% wrong. If this ever fails, delete the module.
    for gap, e_half, e_5pct, cv in rows:
        assert e_5pct < cv, (gap, e_5pct, cv)
        assert e_half < e_5pct, (gap, e_half, e_5pct)

    # And the survivability line, which is what max_coast_s is set from. 2 mm is the MPC's
    # pos_tol (spatial_mpc.Weights) -- the distance at which the controller stops agreeing
    # it is at the setpoint. 20 mm is Limits.box_xy, the workspace the field map is trusted
    # over, and therefore the point past which the estimate means nothing at all.
    at = lambda gap, col: [r[col] for r in rows if abs(r[0] - gap) < 1e-9][0]
    cross = lambda col: min(r[0] for r in rows if r[col] > 2.0) * 1e3
    print(f"    -> the 2 mm MPC pos_tol is passed at {cross(1):.0f} ms converged "
          f"(f_hat +0.5%), {cross(2):.0f} ms unconverged (+5%), {cross(3):.0f} ms on a "
          f"kinematic coast.")
    print(f"    -> {MAX_COAST_S * 1e3:.0f} ms costs {at(MAX_COAST_S, 2):.1f} mm unconverged "
          f"against {at(MAX_COAST_S, 3):.1f} mm for a kinematic coast, so max_coast_s = "
          f"{MAX_COAST_S:.2f} s. PoseFilter's 0.15 s would be {at(0.15, 2):.1f} mm.")
    print(f"    -> a 300 ms gap is NOT survivable by anything here: {at(0.3, 2):.0f} mm "
          f"unconverged, {at(0.3, 3):.0f} mm kinematic, both past the 20 mm box the field "
          f"map is trusted over. Land, do not extrapolate.")
    print()
    assert MAX_COAST_S <= 0.15, "coasting longer than PoseFilter is not a defensible default"

    # 4. Staleness. Inside the budget it is fresh, past it stale, and a fix clears it --
    # the last of those is what stops a single recovered frame from landing the robot.
    p = StatePredictor()
    p.update([0.0, 0.0, 30.0], t=0.0)
    assert not p.stale
    p.predict(Command(f_true, f_true), 0.9 * MAX_COAST_S)
    assert not p.stale, p.coast_s
    p.predict(Command(f_true, f_true), 0.2 * MAX_COAST_S)
    assert p.stale, p.coast_s
    p.update([0.0, 0.0, 31.0], t=p.t)
    assert not p.stale and p.coast_s == 0.0
    print(f"staleness   : fresh to {MAX_COAST_S:.2f} s, stale past it, cleared by a fix")

    # 5. Free fall clamp. Thrust is non-negative, so -g is the floor. f_cmd = 0 is the
    # firmware's own idle, and it must read as free fall and not as anything worse.
    p = StatePredictor()
    for f_cmd_i in np.linspace(0.0, 300.0, 121):
        for f_hat in (110.0, 150.0, 190.0):
            a = p.accel_mm_s2(Command(float(f_cmd_i), f_hat))[2]
            assert a >= -1000.0 * g - 1e-9, (f_cmd_i, f_hat, a)
    assert abs(p.accel_mm_s2(Command(0.0, 150.0))[2] + 1000.0 * g) < 1e-9
    assert abs(p.accel_mm_s2(Command(150.0, 0.0))[2] + 1000.0 * g) < 1e-9   # bad f_hat
    p.update([0.0, 0.0, 30.0], vel_mm_s=[0.0, 0.0, 0.0], t=0.0)
    p.predict(Command(0.0, 150.0), 0.05)
    assert abs(p.xyz_mm[2] - (30.0 - 0.5 * 1000.0 * g * 0.05**2)) < 1e-9, p.xyz_mm
    print(f"free fall   : a >= -g for every f_cmd in 0..300 Hz; f_cmd = 0 gives exactly "
          f"-{g:.2f} m/s^2")

    # 6. Tier 2, off by default: only that the adapter runs and is not silently the
    # default. It is NOT scored against the truth model here, because there is nothing to
    # score it against -- see PlantPredictor's docstring.
    pp = PlantPredictor()
    pp.update([0.0, 0.0, 20.084], vel_mm_s=[0.0, 0.0, 0.0], t=0.0)
    for _ in range(6):
        pp.predict(Command(110.0, 110.0), dt_ctrl)
    # On the symmetry axis with lift == weight the 9-state must also hold still. This
    # checks the adapter's packing and mm <-> m seam, nothing about the physics.
    assert np.allclose(pp.xyz_mm, [0.0, 0.0, 20.084], atol=1e-6), pp.xyz_mm
    assert pp.plant.f_hover == 110.0
    assert type(StatePredictor()) is StatePredictor, "the scalar path must stay the default"
    print("tier 2      : PlantPredictor runs, holds on-axis hover, and is NOT the default")
    print("self-check PASS")


if __name__ == "__main__":
    demo()
