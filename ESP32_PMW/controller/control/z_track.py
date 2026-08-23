#!/usr/bin/env python3
"""
Altitude waypoint tracking by rotating-field frequency modulation.

The plant (theory.md 6.2) carries no robot mass at all, and the input enters as a pure
quadratic, so it inverts exactly. No trim linearization, no A/B matrices:

    z_ddot  = g*((f_robot/f_hover)^2 - 1)
    f_robot = f_hover*sqrt(1 + a_des/g),     a_des >= -g

The domain limit a_des >= -g is the actuator, not the model: thrust is non-negative, so free
fall is the hardest available downward acceleration. Exact inversion rather than the linearized
2g/f_hover gain because every trajectory starts and ends on the pad, where the linear model is
worst (100% error at rest) and its gain drifts 83-111% across a 125-167 Hz band.

Torque-ratio limiting (theory.md 5.6, App. B) is applied in ACCELERATION space before
inverting, so the emitted frequency is feasible by construction:

    sin(delta) = k_drag*f^2 / tau_max(f) <= s_lim        -> a_max
    2*pi*I*f_dot + k_drag*f^2 <= s_lim*tau_max(f)        -> a_dot_max

The cascade assumes f_robot ~ f_field, which the ~112 rad/s phase-lock mode licenses. Keep
OMEGA_N well below it.

Self-check: uv run python controller/control/z_track.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from hover_model import GRAVITY, I_ROBOT, fit_k_drag

# Series-RLC coil channel. tau_max moves with drive frequency, which is what makes the
# torque ceiling a function of f rather than a constant.
INDUCTANCE_H = 1.4e-3
CAPACITANCE_F = 500e-6
RESISTANCE_OHM = 1.7

F_HOVER_HZ = 150.0     # what actually flies; hover_model.py defaults to the GUI's 140
F_STEPOUT_HZ = 190.0   # MEASURED: highest f the field still holds. See f_ceiling().
S_LIM = 0.8            # sin(delta) ceiling, below 1 for the Q~22 phase swing at ramp ends

OMEGA_N = 2.0          # rad/s. 56:1 below the phase-lock mode, so f_robot ~ f_field holds.
ZETA = 0.9
KI = 0.5               # read as a slow f_hover trim estimator, not a classical integral


def coil_gain(f_hz: float) -> float:
    """|I(f)|/|I(f_res)| for the series RLC channel. 1 at resonance, 0 at DC."""

    if f_hz <= 0.0:
        return 0.0
    w = 2.0 * math.pi * f_hz
    reactance = w * INDUCTANCE_H - 1.0 / (w * CAPACITANCE_F)
    return RESISTANCE_OHM / math.hypot(RESISTANCE_OHM, reactance)


@dataclass
class TorqueLimits:
    """Phase-lock torque budget, expressed as limits on vertical acceleration."""

    f_hover: float = F_HOVER_HZ       # Hz
    f_stepout: float = F_STEPOUT_HZ   # Hz, measured. tau_max is pinned to this, NOT to a
                                      # datasheet B_max: that gives sin(delta) > 1 at hover,
                                      # so the limiter clamps below f_hover and never flies.
    s_lim: float = S_LIM              # sin(delta) ceiling
    k_drag: float = field(default_factory=lambda: fit_k_drag()[0])   # N m/Hz^2
    i_robot: float = I_ROBOT          # kg m^2
    g: float = GRAVITY                # m/s^2

    def tau_max(self, f_hz: float) -> float:
        """Peak magnetic torque m*B(f) available at drive frequency f_hz."""

        scale = self.k_drag * self.f_stepout**2 / coil_gain(self.f_stepout)
        return scale * coil_gain(f_hz)

    def sin_delta(self, f_hz: float) -> float:
        """Torque ratio at steady spin. >= 1 means step-out."""

        tau = self.tau_max(f_hz)
        return math.inf if tau <= 0.0 else self.k_drag * f_hz**2 / tau

    def f_ceiling(self) -> float:
        """Highest f with k_drag*f^2 <= s_lim*tau_max(f)."""

        surplus = lambda f: self.s_lim * self.tau_max(f) - self.k_drag * f**2
        if surplus(self.f_hover) <= 0.0:
            raise ValueError(
                f"no torque headroom at f_hover={self.f_hover:.0f} Hz "
                f"(sin(delta)={self.sin_delta(self.f_hover):.2f} vs s_lim={self.s_lim}). "
                f"f_stepout={self.f_stepout:.0f} Hz is too low to fly closed-loop: "
                "re-measure it, or revisit k_drag / the drive."
            )
        # surplus is positive at f_hover and -> -inf as gain(f) rolls off, so a
        # sign change is bracketed. Coarse scan then bisect; no scipy needed.
        grid = np.arange(self.f_hover, 10.0 * self.f_hover, 0.5)
        below = grid[[surplus(float(f)) > 0.0 for f in grid]]
        lo, hi = float(below[-1]), float(below[-1]) + 0.5
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if surplus(mid) > 0.0 else (lo, mid)
        return 0.5 * (lo + hi)

    def a_max(self) -> float:
        """Largest sustainable upward acceleration, from the magnitude clamp."""

        return self.g * ((self.f_ceiling() / self.f_hover) ** 2 - 1.0)

    def f_dot_max(self, f_hz: float) -> float:
        """Ramp rate whose spin-up torque still fits under s_lim*tau_max."""

        headroom = self.s_lim * self.tau_max(f_hz) - self.k_drag * f_hz**2
        return max(0.0, headroom / (2.0 * math.pi * self.i_robot))

    def a_dot_max(self, f_hz: float) -> float:
        """Slew clamp mapped into acceleration units: d/dt of g*((f/f_h)^2-1)."""

        return 2.0 * self.g * f_hz * self.f_dot_max(f_hz) / self.f_hover**2


def load_waypoints(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read [[t_s, z_m], ...] and return (times, heights), sorted by time."""

    pairs = np.asarray(json.loads(open(path).read()), dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: expected a list of [t_s, z_m] pairs")
    order = np.argsort(pairs[:, 0])
    return pairs[order, 0], pairs[order, 1]


def z_ref(t: float, times: np.ndarray, heights: np.ndarray) -> float:
    """Linearly interpolated setpoint; held flat outside the waypoint span."""

    return float(np.interp(t, times, heights))


class ZTracker:
    """PID on altitude -> clamped acceleration -> exact lift inversion."""

    def __init__(
        self,
        times: np.ndarray,
        heights: np.ndarray,
        limits: TorqueLimits | None = None,
        tau_zdot: float = 0.08,
    ):
        self.times, self.heights = times, heights
        self.lim = limits or TorqueLimits()
        self.a_ceiling = self.lim.a_max()  # raises early on bad calibration
        self.tau_zdot = tau_zdot
        self.kp, self.kv = OMEGA_N**2, 2.0 * ZETA * OMEGA_N
        self.ki = KI
        self._z_prev: float | None = None
        self._zdot = 0.0
        self._integral = 0.0
        self._a_prev = 0.0
        self.f_cmd = self.lim.f_hover

    def _estimate_zdot(self, z: float, dt: float) -> float:
        """Finite difference through a first-order lowpass."""

        if self._z_prev is None or dt <= 0.0:
            self._z_prev = z
            return self._zdot
        raw = (z - self._z_prev) / dt
        self._z_prev = z
        alpha = dt / (self.tau_zdot + dt)
        self._zdot += alpha * (raw - self._zdot)
        return self._zdot

    def step(self, t: float, z: float, dt: float, z_dot: float | None = None) -> float:
        """One control tick. Returns the field frequency to command, in Hz."""

        zdot = self._estimate_zdot(z, dt) if z_dot is None else z_dot
        error = z_ref(t, self.times, self.heights) - z

        # Damp the velocity ERROR, not absolute velocity. The reference is
        # piecewise linear, so tracking a climb needs zdot == zdot_ref != 0; a
        # kv*(0 - zdot) term would fight exactly the motion being asked for.
        # Central difference smooths the waypoint corners at no cost.
        zdot_ref = (
            z_ref(t + dt, self.times, self.heights)
            - z_ref(t - dt, self.times, self.heights)
        ) / (2.0 * dt)
        a_des = self.kp * error + self.kv * (zdot_ref - zdot) + self.ki * self._integral

        # Clamp in acceleration space, then invert -- the frequency is then
        # feasible by construction. -g is free fall (thrust = 0).
        a_lo, a_hi = -self.lim.g, self.a_ceiling
        a_rate = self.lim.a_dot_max(self.f_cmd)
        a_lo = max(a_lo, self._a_prev - a_rate * dt)
        a_hi = min(a_hi, self._a_prev + a_rate * dt)
        a_cmd = min(max(a_des, a_lo), a_hi)

        # Conditional integration: only accumulate when not saturated, so a
        # clamped climb cannot wind the integrator up.
        if a_lo < a_des < a_hi:
            self._integral += error * dt

        self._a_prev = a_cmd
        self.f_cmd = self.lim.f_hover * math.sqrt(1.0 + a_cmd / self.lim.g)
        return self.f_cmd


def demo() -> None:
    lim = TorqueLimits()
    k_drag = lim.k_drag
    f_ceil, a_max = lim.f_ceiling(), lim.a_max()

    print(f"k_drag      = {k_drag:.4e} N m/Hz^2   (reused from hover_model.py)")
    print(f"f_hover     = {lim.f_hover:.1f} Hz")
    print(
        f"f_resonance = {1.0 / (2 * math.pi * math.sqrt(INDUCTANCE_H * CAPACITANCE_F)):.1f} Hz"
    )
    print(f"f_stepout   = {lim.f_stepout:.1f} Hz (measured knob), s_lim = {lim.s_lim}")
    print(f"f_ceiling   = {f_ceil:.1f} Hz")
    print(f"a_max       = {a_max:.3f} m/s^2  ({a_max / lim.g:.2f} g)")
    print(f"a_dot_max   = {lim.a_dot_max(lim.f_hover):.2f} m/s^3 at f_hover")
    print(
        f"sin(delta)  = {lim.sin_delta(lim.f_hover):.3f} at f_hover, "
        f"{lim.sin_delta(f_ceil):.3f} at f_ceiling"
    )

    # The calibration trap: a limiter that cannot reach hover cannot fly.
    assert f_ceil > lim.f_hover, f"{f_ceil} <= {lim.f_hover}"
    assert abs(lim.sin_delta(f_ceil) - lim.s_lim) < 1e-3, lim.sin_delta(f_ceil)

    # Exact inversion round-trips: a -> f -> z_ddot returns a.
    for a in np.linspace(-lim.g, a_max, 25):
        f = lim.f_hover * math.sqrt(1.0 + a / lim.g)
        assert abs(lim.g * ((f / lim.f_hover) ** 2 - 1.0) - a) < 1e-9, a
    # ...and the linearization it replaces does not, by 8% at 25 Hz out.
    df = 25.0
    exact = lim.g * (((lim.f_hover + df) / lim.f_hover) ** 2 - 1.0)
    linear = 2.0 * lim.g / lim.f_hover * df
    assert abs(linear - exact) / exact > 0.07, (linear, exact)

    # A step demand must respect both clamps and never command DC.
    times, heights = np.array([0.0, 5.0, 20.0]), np.array([0.0, 0.15, 0.15])
    trk = ZTracker(times, heights, lim)
    dt, z, zdot = 1.0 / 30.0, 0.0, 0.0
    f_prev = trk.f_cmd
    for i in range(int(20.0 / dt)):
        t = i * dt
        f = trk.step(t, z, dt, z_dot=zdot)
        assert 0.0 < f <= f_ceil + 1e-6, f
        assert lim.sin_delta(f) <= lim.s_lim + 1e-6, (f, lim.sin_delta(f))
        # Integrate the truth plant, with the pad as a unilateral constraint.
        a = lim.g * ((f / lim.f_hover) ** 2 - 1.0)
        zdot, z = zdot + a * dt, z + zdot * dt
        if z <= 0.0 and a <= 0.0:
            z, zdot = 0.0, 0.0
        f_prev = f
    print(f"after 20 s: z = {z * 1000:.1f} mm (target 150.0), f = {f_prev:.1f} Hz")
    assert abs(z - 0.15) < 0.02, z

    # Saturating demand: 2 m in 1 ms is unreachable, so only the clamps hold it
    # back. This is the path the gentle trajectory above never exercises.
    trk = ZTracker(np.array([0.0, 0.001]), np.array([0.0, 2.0]), lim)
    z = zdot = f_max = 0.0
    for i in range(90):
        f = trk.step(i * dt, z, dt, z_dot=zdot)
        f_max = max(f_max, f)
        assert lim.sin_delta(f) <= lim.s_lim + 1e-6, (f, lim.sin_delta(f))
        a = lim.g * ((f / lim.f_hover) ** 2 - 1.0)
        zdot, z = zdot + a * dt, z + zdot * dt
    print(
        f"saturated:  f_max = {f_max:.2f} Hz vs f_ceiling {f_ceil:.2f} Hz, "
        f"sin(delta) = {lim.sin_delta(f_max):.3f}"
    )
    assert f_max <= f_ceil + 1e-9, (f_max, f_ceil)
    assert f_max > 0.98 * f_ceil, f"clamp far too conservative: {f_max} << {f_ceil}"
    print("self-check PASS")


if __name__ == "__main__":
    demo()
