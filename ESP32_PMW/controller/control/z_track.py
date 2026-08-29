#!/usr/bin/env python3
"""
Altitude tracking by rotating-field frequency modulation, exactly inverted and clamped in
acceleration space by the phase-lock torque budget. f_hover is a seed guess, so ZTracker
trims it in FREQUENCY space (f_hat) rather than as an integral in acceleration space.
Lift law and torque cascade: theory.md 6.2. Trim parameterisation: theory.md 6.6.

Where:
    z, z_ref  m, +z up in the datum frame       a_*    m/s^2, domain a >= -g (thrust >= 0)
    f_*       Hz, field frequency               gamma  Hz per (m*s)
    z_ddot = g*((f_robot/f_hover)^2 - 1)        f_robot = f_hover*sqrt(1 + a_des/g)

Self-check: uv run python controller/control/z_track.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hover_model import GRAVITY, I_ROBOT, fit_k_drag

HERE = Path(__file__).resolve().parent

# Series-RLC coil channel. tau_max moves with drive frequency, which is what makes the
# torque ceiling a function of f rather than a constant.
INDUCTANCE_H = 1.4e-3
CAPACITANCE_F = 500e-6
RESISTANCE_OHM = 1.7

F_HOVER_HZ = 150.0     # SEED GUESS, the firmware's ramp target. ZTracker identifies it.
F_STEPOUT_HZ = 190.0   # SEED GUESS: 190 Hz is the coil's electrical match, not a step-out.
S_LIM = 0.8            # sin(delta) ceiling, below 1 for the Q~22 phase swing at ramp ends

OMEGA_N = 2.0          # rad/s. 56:1 below the phase-lock mode, so f_robot ~ f_field holds.
ZETA = 0.9

GAMMA_F = 4.0          # Hz/(m*s). Trim pole gamma*(2g/f_hover)/kp ~ 1/(7.6 s), 3x under OMEGA_N.
F_HAT_LO_HZ = 125.0    # f_hat band, matching the firmware freq= guard: the estimator, not the
F_HAT_HI_HZ = 175.0    # seed, must not be what stops identification short of the true f_hover
F_HAT_REFRESH = 0.01   # refresh the ceiling per 1% of f_hat move; f_ceiling() bisects 50x

# Step-out: all available torque commanded while z falls anyway. The threshold has to
# sit far enough above the rate noise that noise alone never trips it, and 3 frames of
# agreement rides a dropout.
#
# The "~9 mm/s, so this is 16 sigma" this comment used to claim came from an assumed
# 0.5 mm per frame that nobody ever measured. `pose/noise.py` measures it; `demo()`
# below recomputes the multiple from whatever model is on disk and fails if it has
# fallen under `STEPOUT_MIN_SIGMA`. Until a static calibration is recorded the number
# stays a guess, and the check says so rather than passing quietly.
STEPOUT_F_FRAC = 0.98
STEPOUT_ZDOT_MPS = -0.15
STEPOUT_FRAMES = 3

# How many sigmas of rate noise the step-out threshold must clear. Tripping step-out
# spuriously commands full torque at a robot that is flying correctly, so the margin
# is deliberately large; 6 sigma on a per-frame test at 60 fps is about one false
# trip per 15 hours, and `STEPOUT_FRAMES` consecutive agreements on top of that.
STEPOUT_MIN_SIGMA = 6.0

# Time constant of the zdot low-pass, seconds. Shared with `predictor.TAU_VEL_S`:
# same measurement, same noise, same number, and they must move together.
#
# The tradeoff it settles is rate noise against lag, and `noise.NoiseModel`
# quantifies both ends of it -- `velocity_sigma_mm_s(tau)` for what a tau costs in
# noise, `tau_for_velocity_sigma` for the inverse. **Only the white part of the
# position error reaches the rate**: the correlated part is common to both ends of
# the difference and cancels, which is why raising tau past the measured correlation
# time buys lag and nothing else.
TAU_ZDOT_S = 0.08


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

    f_hover: float = F_HOVER_HZ       # Hz. Every method below takes an override for it.
    f_stepout: float = F_STEPOUT_HZ   # Hz. Sets tau_max; pinning it to a datasheet B_max
                                      # instead gives sin(delta) > 1 at hover and never flies.
    s_lim: float = S_LIM              # sin(delta) ceiling
    k_drag: float = field(default_factory=lambda: fit_k_drag()[0])   # N m/Hz^2
    i_robot: float = I_ROBOT          # kg m^2
    g: float = GRAVITY                # m/s^2

    def tau_max(self, f_hz: float) -> float:
        """Peak magnetic torque m*B(f) available at drive frequency f_hz, in N m."""

        scale = self.k_drag * self.f_stepout**2 / coil_gain(self.f_stepout)
        return scale * coil_gain(f_hz)

    def sin_delta(self, f_hz: float) -> float:
        """Torque ratio at steady spin. >= 1 means step-out."""

        tau = self.tau_max(f_hz)
        return math.inf if tau <= 0.0 else self.k_drag * f_hz**2 / tau

    def f_ceiling(self, f_hover: float | None = None) -> float:
        """Highest f in Hz with k_drag*f^2 <= s_lim*tau_max(f); raises if none above f_hover."""

        f_h = self.f_hover if f_hover is None else f_hover
        surplus = lambda f: self.s_lim * self.tau_max(f) - self.k_drag * f**2
        if surplus(f_h) <= 0.0:
            raise ValueError(
                f"no torque headroom at f_hover={f_h:.0f} Hz "
                f"(sin(delta)={self.sin_delta(f_h):.2f} vs s_lim={self.s_lim}). "
                f"f_stepout={self.f_stepout:.0f} Hz is too low to fly closed-loop: "
                "re-measure it, or revisit k_drag / the drive."
            )
        # surplus is positive at f_hover and -> -inf as gain(f) rolls off, so a
        # sign change is bracketed. Coarse scan then bisect; no scipy needed.
        grid = np.arange(f_h, 10.0 * f_h, 0.5)
        below = grid[[surplus(float(f)) > 0.0 for f in grid]]
        lo, hi = float(below[-1]), float(below[-1]) + 0.5
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if surplus(mid) > 0.0 else (lo, mid)
        return 0.5 * (lo + hi)

    def a_max(self, f_hover: float | None = None) -> float:
        """Largest sustainable upward acceleration in m/s^2, from the magnitude clamp."""

        f_h = self.f_hover if f_hover is None else f_hover
        return self.g * ((self.f_ceiling(f_h) / f_h) ** 2 - 1.0)

    def f_dot_max(self, f_hz: float) -> float:
        """Ramp rate in Hz/s whose spin-up torque still fits under s_lim*tau_max."""

        headroom = self.s_lim * self.tau_max(f_hz) - self.k_drag * f_hz**2
        return max(0.0, headroom / (2.0 * math.pi * self.i_robot))

    def a_dot_max(self, f_hz: float, f_hover: float | None = None) -> float:
        """Slew clamp in m/s^3: d/dt of g*((f/f_hover)^2 - 1) at the f_dot_max ramp rate."""

        f_h = self.f_hover if f_hover is None else f_hover
        return 2.0 * self.g * f_hz * self.f_dot_max(f_hz) / f_h**2


def load_waypoints(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read [[t_s, z_m], ...] and return (times, heights), sorted by time."""

    pairs = np.asarray(json.loads(open(path).read()), dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: expected a list of [t_s, z_m] pairs")
    order = np.argsort(pairs[:, 0])
    return pairs[order, 0], pairs[order, 1]


def z_ref(t: float, times: np.ndarray, heights: np.ndarray) -> float:
    """Linearly interpolated setpoint in m; held flat outside the waypoint span."""

    return float(np.interp(t, times, heights))


class ZTracker:
    """PD on altitude + an f_hover estimator -> clamped acceleration -> exact lift inversion."""

    def __init__(
        self,
        times: np.ndarray,
        heights: np.ndarray,
        limits: TorqueLimits | None = None,
        tau_zdot: float = TAU_ZDOT_S,
        f_seed: float | None = None,
    ):
        """f_seed is the starting hover-frequency estimate in Hz; None falls back to limits.f_hover."""

        self.times, self.heights = times, heights
        self.lim = limits or TorqueLimits()
        self.f_seed = float(f_seed) if f_seed else self.lim.f_hover
        self.f_hat = self.f_seed
        self.f_lo, self.f_hi = F_HAT_LO_HZ, F_HAT_HI_HZ
        self._f_ceiling_at = 0.0
        self._recompute_ceiling()  # raises early on bad calibration
        self.tau_zdot = tau_zdot
        self.kp, self.kv = OMEGA_N**2, 2.0 * ZETA * OMEGA_N
        self.gamma = GAMMA_F
        self._z_prev: float | None = None
        self._zdot = 0.0
        self._a_prev = 0.0
        self.a_fb = 0.0
        self.f_cmd = self.f_hat
        self.stepout = False
        self._falling = 0

    def _recompute_ceiling(self) -> None:
        """Set f_ceil (Hz) and a_ceiling (m/s^2) at the current f_hat; pin f_hat if infeasible."""

        try:
            self.f_ceil = self.lim.f_ceiling(self.f_hat)
        except ValueError:
            # No torque headroom at this f_hat, so keep the last feasible pair and stop the
            # estimator there: an f_hat with no computable ceiling must not reach the coils.
            self.f_hat = self.f_hi = self._f_ceiling_at
            return
        self.a_ceiling = self.lim.g * ((self.f_ceil / self.f_hat) ** 2 - 1.0)
        self._f_ceiling_at = self.f_hat

    def _estimate_zdot(self, z: float, dt: float) -> float:
        """Finite difference through a first-order lowpass. Returns m/s."""

        if self._z_prev is None or dt <= 0.0:
            self._z_prev = z
            return self._zdot
        raw = (z - self._z_prev) / dt
        self._z_prev = z
        alpha = dt / (self.tau_zdot + dt)
        self._zdot += alpha * (raw - self._zdot)
        return self._zdot

    def step(
        self,
        t: float,
        z: float,
        dt: float,
        z_dot: float | None = None,
        z_target: float | None = None,
    ) -> float:
        """One tick. z, z_target in m; returns the field frequency to command, in Hz."""

        zdot = self._estimate_zdot(z, dt) if z_dot is None else z_dot
        if z_target is None:
            error = z_ref(t, self.times, self.heights) - z
            # Damp the velocity ERROR, not absolute velocity. The reference is
            # piecewise linear, so tracking a climb needs zdot == zdot_ref != 0; a
            # kv*(0 - zdot) term would fight exactly the motion being asked for.
            # Central difference smooths the waypoint corners at no cost.
            zdot_ref = (
                z_ref(t + dt, self.times, self.heights)
                - z_ref(t - dt, self.times, self.heights)
            ) / (2.0 * dt)
        else:
            error, zdot_ref = z_target - z, 0.0  # a live setpoint is a step, not a trajectory

        a_fb = self.kp * error + self.kv * (zdot_ref - zdot)  # feedback only; trim is f_hat

        # Clamp in acceleration space, then invert -- the frequency is then
        # feasible by construction. -g is free fall (thrust = 0).
        a_lo, a_hi = -self.lim.g, self.a_ceiling
        a_rate = self.lim.a_dot_max(self.f_cmd, self.f_hat)
        a_lo = max(a_lo, self._a_prev - a_rate * dt)
        a_hi = min(a_hi, self._a_prev + a_rate * dt)
        a_cmd = min(max(a_fb, a_lo), a_hi)

        # Conditional adaptation: against a clamped actuator a steady error measures the
        # clamp, not f_hover, and winds f_hat up exactly as the old integral wound up.
        if a_lo < a_fb < a_hi and not self.stepout:
            self.f_hat = min(max(self.f_hat + self.gamma * error * dt, self.f_lo), self.f_hi)
            if abs(self.f_hat - self._f_ceiling_at) > F_HAT_REFRESH * self._f_ceiling_at:
                self._recompute_ceiling()

        self._a_prev = a_cmd
        self.a_fb = a_fb
        self.f_cmd = self.f_hat * math.sqrt(1.0 + a_cmd / self.lim.g)

        # Step-out is the one failure more frequency deepens, so fall back to trim and let
        # the runner land rather than chasing it.
        near_ceiling = self.f_cmd >= STEPOUT_F_FRAC * self.f_ceil
        self._falling = self._falling + 1 if (near_ceiling and zdot <= STEPOUT_ZDOT_MPS) else 0
        if self._falling >= STEPOUT_FRAMES:
            self.stepout = True
        if self.stepout:
            self.f_cmd = self.f_hat
        return self.f_cmd


def check_stepout_margin(tau_s: float = TAU_ZDOT_S, dt: float = 1.0 / 60.0):
    """
    How many sigmas of rate noise `STEPOUT_ZDOT_MPS` sits above zero.

        Reads `pose/noise.py`'s measured model, so the threshold is checked against
        the bench rather than against the assumption it was originally written from.
        Returns ``(sigma_mm_s, multiple, measured)``; ``measured`` is False when no
        static calibration has been recorded and the numbers are still rendered.
    """

    import sys as _sys
    _sys.path[:0] = [str(HERE.parent / "pose"), str(HERE.parent / "calib"),
                     str(HERE.parent / "camera")]
    from noise import NoiseModel

    m = NoiseModel.load()
    sigma = m.velocity_sigma_mm_s("z", tau_s=tau_s, dt=dt)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan"), float("inf"), m.measured
    return sigma, abs(STEPOUT_ZDOT_MPS) * 1e3 / sigma, m.measured


def demo() -> None:
    lim = TorqueLimits()
    k_drag = lim.k_drag
    f_ceil, a_max = lim.f_ceiling(), lim.a_max()
    f_true = lim.f_hover

    print(f"k_drag      = {k_drag:.4e} N m/Hz^2   (reused from hover_model.py)")
    print(f"f_hover     = {lim.f_hover:.1f} Hz")
    print(
        f"f_resonance = {1.0 / (2 * math.pi * math.sqrt(INDUCTANCE_H * CAPACITANCE_F)):.1f} Hz"
    )
    print(f"f_stepout   = {lim.f_stepout:.1f} Hz (SEED GUESS), s_lim = {lim.s_lim}")
    print(f"f_ceiling   = {f_ceil:.1f} Hz")
    print(f"a_max       = {a_max:.3f} m/s^2  ({a_max / lim.g:.2f} g)")
    print(f"a_dot_max   = {lim.a_dot_max(lim.f_hover):.2f} m/s^3 at f_hover")
    print(
        f"sin(delta)  = {lim.sin_delta(lim.f_hover):.3f} at f_hover, "
        f"{lim.sin_delta(f_ceil):.3f} at f_ceiling"
    )

    sigma, mult, measured = check_stepout_margin()
    if measured and np.isfinite(sigma):
        print(f"zdot noise  = {sigma:.2f} mm/s at tau={TAU_ZDOT_S * 1e3:.0f} ms "
              f"(measured)")
        print(f"step-out    = {abs(STEPOUT_ZDOT_MPS) * 1e3:.0f} mm/s, {mult:.1f} "
              f"sigma (floor {STEPOUT_MIN_SIGMA:.0f})")
        assert mult >= STEPOUT_MIN_SIGMA, (
            f"step-out sits at {mult:.1f} sigma of the measured rate noise, under the "
            f"{STEPOUT_MIN_SIGMA} floor: raise STEPOUT_ZDOT_MPS or lower TAU_ZDOT_S")
    else:
        print(f"zdot noise  = not measured, so the {abs(STEPOUT_ZDOT_MPS) * 1e3:.0f} "
              f"mm/s step-out threshold is unjustified.")
        print("              Record a static calibration:\n"
              "                uv run python controller/pose/noise.py --record\n"
              "                or noise.record_live(stations=4) in run.ipynb")

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
        assert 0.0 < f <= trk.f_ceil + 1e-6, f
        assert lim.sin_delta(f) <= lim.s_lim + 1e-6, (f, lim.sin_delta(f))
        # Integrate the truth plant, with the pad as a unilateral constraint.
        a = lim.g * ((f / f_true) ** 2 - 1.0)
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
        a = lim.g * ((f / f_true) ** 2 - 1.0)
        zdot, z = zdot + a * dt, z + zdot * dt
    print(
        f"saturated:  f_max = {f_max:.2f} Hz vs f_ceiling {f_ceil:.2f} Hz, "
        f"sin(delta) = {lim.sin_delta(f_max):.3f}"
    )
    assert f_max <= f_ceil + 1e-9, (f_max, f_ceil)
    assert f_max > 0.98 * f_ceil, f"clamp far too conservative: {f_max} << {f_ceil}"

    # The re-parameterization's whole claim: seed f_hat 10 Hz low against the exact lift law
    # and the error ends up in f_hat, not as a permanent bias in a_fb. No pad constraint --
    # the sink on the way IS the 1.45 m/s^2 of trim the old form would have carried forever.
    trk = ZTracker(np.array([0.0, 1e9]), np.array([0.06, 0.06]), lim, f_seed=140.0)
    z, zdot, a_fb_tail = 0.06, 0.0, []
    for i in range(int(120.0 / dt)):
        f = trk.step(i * dt, z, dt, z_dot=zdot)
        a = lim.g * ((f / f_true) ** 2 - 1.0)
        zdot, z = zdot + a * dt, z + zdot * dt
        if i * dt > 100.0:
            a_fb_tail.append(trk.a_fb)
    a_fb_rms = float(np.sqrt(np.mean(np.square(a_fb_tail))))
    print(
        f"identify:   f_hat = {trk.f_hat:.2f} Hz from a 140.0 seed (true {f_true:.1f}), "
        f"z = {z * 1000:.1f} mm, a_fb rms = {a_fb_rms:.4f} m/s^2"
    )
    assert abs(trk.f_hat - f_true) < 0.5, trk.f_hat
    assert a_fb_rms < 0.05, a_fb_rms          # centred on zero, the entire point
    assert abs(z - 0.06) < 0.01, z
    assert not trk.stepout, "step-out guard tripped on a healthy run"

    # ...and the guard does trip when the rotor stops responding: free fall while the
    # tracker asks for everything it has.
    trk = ZTracker(np.array([0.0, 1e9]), np.array([1.0, 1.0]), lim)
    z, zdot = 0.0, 0.0
    for i in range(60):
        trk.step(i * dt, z, dt, z_dot=zdot)
        zdot, z = zdot - lim.g * dt, z + zdot * dt
    assert trk.stepout, "step-out guard missed a free fall at the torque ceiling"
    assert abs(trk.f_cmd - trk.f_hat) < 1e-9, trk.f_cmd
    print(f"step-out:   tripped, f_cmd dropped to f_hat = {trk.f_cmd:.2f} Hz")
    print("self-check PASS")


if __name__ == "__main__":
    demo()
