#!/usr/bin/env python3
"""
Closed-loop nonlinear simulation of the hover controller.

Truth plant: the full 6-state nonlinear model integrated with solve_ivp one ZOH interval at a
time. The controller runs at the camera rate through the SAME DiscreteHoverController code path
the hardware runner uses, so velocity estimation, anti-windup, saturation and slew are all
exercised here rather than only in flight.

Scenarios (a|b|c|d|e|all):
  a  10 mm lateral + 10 mm vertical initial offset, clean measurements
  b  (a) + 0.5 mm sensor noise + 15 ms latency
  c  trim mismatch: plant hovers at 143 Hz, controller believes 140 (integrator proof)
  d  k_lat robustness: true k_lat in {0.25x, 1x, 4x} the design value
  e  profile tracking: quadratic-ease 10 mm climb + linear 15 mm lateral translate

Each prints PASS/FAIL and writes hover_sim_<scenario>.png.

Usage: uv run python controller/control/simulate_hover.py [scenario]
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from controller.control.hover_model import make_params, nonlinear_dynamics
from controller.control.reference_profiles import Profile, demo_profile


#: Sentinel: `step(dt=None)` means "no new fix, hold the rate", which is a different
#: thing from "caller did not say", where the old once-per-measurement default applies.
_UNSET = object()


class VelocityEstimator:
    """Finite difference + 1-pole IIR low-pass (default ~5 Hz cutoff).

    The lateral twin of `z_track`'s zdot filter, and the same tradeoff: a 5 Hz
    cutoff is tau = 32 ms, which is *shorter* than the 80 ms those use. What it
    costs in rate noise follows from the measured lateral sigma --
    `noise.NoiseModel.velocity_sigma_mm_s("x", tau_s=1/(2*pi*cutoff_hz))` -- and
    only the white part of the position error reaches the rate at all.
    """

    def __init__(self, ts: float, cutoff_hz: float = 5.0):
        self.ts = ts
        self.tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        self.alpha = ts / (ts + self.tau)
        self.prev_pos = None
        self.vel = np.zeros(2)

    def update(self, pos: np.ndarray, dt: float | None = None) -> np.ndarray:
        """New rate estimate. ``dt`` is the interval the position actually moved over.

            **Not the control period.** The control loop steps far faster than the pose
            pipeline delivers -- 500 Hz against ~100 Hz -- and between fixes the position
            handed in comes from `predictor.StatePredictor`, which advances it by exactly
            `v * dt`. Differencing that returns the velocity that produced it: no
            information, only filter lag. So a stale tick passes ``dt=None`` and the
            estimate is held.

            On the tick a fix DOES land, the jump is the correction accumulated over the
            whole pose interval. Dividing it by `self.ts` -- which is what this did until
            2026-09-01 -- reports a rate too large by `pose_interval / ts`: 2x at 200 Hz
            and 5x at 500 Hz, i.e. the defect gets worse exactly as the clock rises.
            `theory.md` 19.8 names this and puts it second in its ladder; the clock going
            to 500 Hz is what made it due.

            `alpha` follows `dt` for the same reason: a 1-pole low-pass at a fixed
            cutoff is `dt / (dt + tau)`, and pinning it to `ts` would re-introduce the
            rate dependence from the other side.
        """

        if self.prev_pos is None:
            self.prev_pos = pos.copy()
            return self.vel.copy()
        if dt is None:
            return self.vel.copy()      # no new information: hold, do not re-difference
        raw = (pos - self.prev_pos) / dt
        self.prev_pos = pos.copy()
        self.vel += (dt / (dt + self.tau)) * (raw - self.vel)
        return self.vel.copy()


class DiscreteHoverController:
    """u(k) = u_trim + u_ff(k) - K [x_hat - x_ref ; q] -- see design_hover_lqr.py."""

    def __init__(self, gains: dict, profile: Profile):
        self.K = np.array(gains["K"])
        self.ts = gains["design"]["ts"]
        self.f_hover = gains["params"]["f_hover"]
        self.g = gains["params"]["g"]
        self.k_lat = gains["params"]["k_lat"]
        lim = gains["limits"]
        self.mag_max = lim["mag_max"]
        self.freq_min, self.freq_max = lim["freq_min"], lim["freq_max"]
        self.freq_slew = lim["freq_slew_hz_per_s"] * self.ts  # Hz per frame
        self.profile = profile
        self.est = VelocityEstimator(self.ts)
        self.q = np.zeros(2)  # error integrators [int_ex, int_ez]
        self.prev_f_field = self.f_hover  # slew-limit reference

    def step(self, t: float, x_meas: float, z_meas: float,
             dt=_UNSET) -> tuple[float, float]:
        """One control step. ``dt`` is how long the measurement took to change.

            Defaults to `self.ts`, which is right only when a caller steps once per
            measurement -- `simulate_hover`'s scenarios do. The live loop does not, and
            passes the real fix interval, or None on a tick with no new fix. See
            `VelocityEstimator.update`.
        """

        pos = np.array([x_meas, z_meas])
        vel = self.est.update(pos, self.ts if dt is _UNSET else dt)
        ref_p, ref_v, ref_a = self.profile.eval(t)

        err = np.array(
            [pos[0] - ref_p[0], vel[0] - ref_v[0], pos[1] - ref_p[1], vel[1] - ref_v[1]]
        )
        # reference-acceleration feedforward (exact inverse of B's nonzero entries)
        u_ff = np.array(
            [ref_a[0] / (self.g * self.k_lat), ref_a[1] * self.f_hover / (2.0 * self.g)]
        )
        u = (
            np.array([0.0, self.f_hover])
            + u_ff
            - self.K @ np.concatenate([err, self.q])
        )

        # saturate: mag symmetric clamp; f_field clamp + slew limit
        mag = float(np.clip(u[0], -self.mag_max, self.mag_max))
        f_tgt = float(np.clip(u[1], self.freq_min, self.freq_max))
        f_field = float(
            np.clip(
                f_tgt,
                self.prev_f_field - self.freq_slew,
                self.prev_f_field + self.freq_slew,
            )
        )
        self.prev_f_field = f_field

        # Conditional-integration anti-windup: integrate only when the unsaturated command
        # is in range, or when the error points back toward it. Slew limiting is a transient
        # and does NOT freeze the integrator; only hard clamps do.
        self._integrate(0, err[0], u[0], -self.mag_max, self.mag_max)
        self._integrate(1, err[2], u[1], self.freq_min, self.freq_max)
        return mag, f_field

    def _integrate(self, i: int, e: float, u_unsat: float, lo: float, hi: float):
        if lo <= u_unsat <= hi or (u_unsat > hi and e > 0) or (u_unsat < lo and e < 0):
            self.q[i] += self.ts * e


@dataclass
class Scenario:
    name: str
    duration: float = 10.0
    x0_mm: float = 0.0  # initial lateral offset
    z0_mm: float = 0.0  # initial vertical offset
    sensor_sigma_m: float = 0.0
    # Sensor-to-command latency in SECONDS, not in control steps. It used to be
    # `latency_frames`, which meant 5 ms at 200 Hz and 2 ms at 500 -- so raising the
    # clock made every latency scenario quietly easier while claiming to test the same
    # thing. That is the same class of error as 19.9's phase-lock check measuring its own
    # sample grid. Physical delay does not care what rate the loop runs at.
    latency_s: float = 0.0
    dist_accel: float = 0.0  # constant lateral disturbance accel m/s^2
    k_lat_true_mult: float = 1.0  # plant k_lat vs design k_lat
    plant_f_hover: float | None = None  # true lift-balance frequency (mismatch test)
    profile: Profile = field(default_factory=lambda: Profile.hold())
    # PASS criteria
    settle_s: float = 4.0
    tol_mm: float = 2.0
    track_tol_mm: float | None = None  # if set, error bound over the whole run


def simulate(sc: Scenario, gains: dict, seed: int = 0) -> dict:
    ctrl = DiscreteHoverController(gains, sc.profile)
    ts = ctrl.ts
    design = gains["params"]
    p_true = make_params(
        f_hover=sc.plant_f_hover or design["f_hover"],
        k_lat=design["k_lat"] * sc.k_lat_true_mult,
        margin=design["margin"],
    )
    rng = np.random.default_rng(seed)

    # start at the plant's true hover trim, plus position offsets
    s = np.array(
        [
            sc.x0_mm * 1e-3,
            0.0,
            sc.z0_mm * 1e-3,
            0.0,
            p_true.delta_trim,
            p_true.omega_trim,
        ]
    )
    meas_queue = deque(maxlen=max(1, int(round(sc.latency_s / ts)) + 1))
    n = int(round(sc.duration / ts))
    out = {
        k: np.zeros(n)
        for k in ("t", "x", "z", "mag", "f_field", "delta", "x_ref", "z_ref")
    }

    def dyn(t, y, u):
        d = nonlinear_dynamics(t, y, u, p_true)
        d[1] += sc.dist_accel
        return d

    u = (0.0, ctrl.f_hover)
    for k in range(n):
        t = k * ts
        meas = np.array([s[0], s[2]]) + rng.normal(0.0, sc.sensor_sigma_m, 2)
        meas_queue.append(meas)
        m = meas_queue[0]  # oldest = delayed measurement
        u = ctrl.step(t, m[0], m[1])

        ref_p, _, _ = sc.profile.eval(t)
        out["t"][k], out["x"][k], out["z"][k] = t, s[0], s[2]
        out["mag"][k], out["f_field"][k] = u
        out["x_ref"][k], out["z_ref"][k] = ref_p

        sol = solve_ivp(
            dyn, (t, t + ts), s, args=(u,), max_step=ts / 8, rtol=1e-8, atol=1e-9
        )
        # The PEAK phase error over the step, not the value at its start. The plant was
        # always integrated finely (max_step=ts/8); only the recording was on the control
        # grid, so the phase-lock check measured whatever `ts` happened to sample. That
        # made the metric rate-dependent for no physical reason: the same 143 Hz mismatch
        # read 46.6 deg at 30 Hz and 73.5 at 50-200 Hz, and the 60 deg gate was set
        # against the coarser view. See `theory.md` 18.12.
        span = sol.y[4, :]
        out["delta"][k] = span[np.argmax(np.abs(
            np.arctan2(np.sin(span), np.cos(span))))]
        s = sol.y[:, -1]
    return out


def evaluate(sc: Scenario, out: dict, gains: dict) -> tuple[bool, list[str]]:
    ex = (out["x"] - out["x_ref"]) * 1e3  # mm
    ez = (out["z"] - out["z_ref"]) * 1e3
    dw = np.degrees(np.arctan2(np.sin(out["delta"]), np.cos(out["delta"])))
    msgs, ok = [], True

    settled = out["t"] >= sc.settle_s
    e_after = np.maximum(np.abs(ex[settled]), np.abs(ez[settled]))
    cond = e_after.max() < sc.tol_mm
    ok &= cond
    msgs.append(
        f"  settle: max err after {sc.settle_s:.0f}s = {e_after.max():.2f} mm "
        f"(< {sc.tol_mm} mm) {'PASS' if cond else 'FAIL'}"
    )

    peak = np.abs(dw).max()
    cond = peak < 90.0
    ok &= cond
    grade = "PASS" if peak < 60.0 else ("THIN" if cond else "FAIL")
    msgs.append(
        f"  phase lock: max |delta| = {peak:.1f} deg (margin 60, lock lost 90) {grade}"
    )

    mag_max = gains["limits"]["mag_max"]
    sat_frac = np.mean(np.abs(out["mag"]) >= mag_max * 0.999)
    cond = sat_frac < 0.2
    ok &= cond
    msgs.append(
        f"  saturation: mag saturated {sat_frac*100:.0f}% of run (< 20%) "
        f"{'PASS' if cond else 'FAIL'}"
    )

    if sc.track_tol_mm is not None:
        warm = out["t"] >= 2.0  # skip initial transient
        e_all = np.maximum(np.abs(ex[warm]), np.abs(ez[warm]))
        cond = e_all.max() < sc.track_tol_mm
        ok &= cond
        msgs.append(
            f"  tracking: max err = {e_all.max():.2f} mm "
            f"(< {sc.track_tol_mm} mm) {'PASS' if cond else 'FAIL'}"
        )
    return ok, msgs


def plot(sc: Scenario, out: dict, gains: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    lim = gains["limits"]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 9))

    axes[0].plot(out["t"], out["x"] * 1e3, label="x")
    axes[0].plot(out["t"], out["x_ref"] * 1e3, "--", label="x ref")
    axes[0].plot(out["t"], out["z"] * 1e3, label="z")
    axes[0].plot(out["t"], out["z_ref"] * 1e3, "--", label="z ref")
    for ref in (out["x_ref"], out["z_ref"]):
        axes[0].fill_between(
            out["t"], (ref - 2e-3) * 1e3, (ref + 2e-3) * 1e3, alpha=0.08, color="gray"
        )
    axes[0].set_ylabel("pos [mm]")
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(out["t"], out["mag"])
    for y in (lim["mag_max"], -lim["mag_max"]):
        axes[1].axhline(y, color="r", ls=":", lw=0.8)
    axes[1].set_ylabel("mag (signed)")

    axes[2].plot(out["t"], out["f_field"])
    for y in (lim["freq_min"], lim["freq_max"]):
        axes[2].axhline(y, color="r", ls=":", lw=0.8)
    axes[2].set_ylabel("f_field [Hz]")

    dw = np.degrees(np.arctan2(np.sin(out["delta"]), np.cos(out["delta"])))
    axes[3].plot(out["t"], dw)
    axes[3].axhline(90, color="r", ls="--", lw=1.0, label="pull-out")
    axes[3].set_ylabel("delta [deg]")
    axes[3].set_xlabel("t [s]")
    axes[3].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.4)
    axes[0].set_title(f"hover sim: scenario {sc.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def build_scenarios() -> dict[str, list[Scenario]]:
    return {
        "a": [Scenario("a", x0_mm=10, z0_mm=10)],
        "b": [
            # 15 ms: `control/theory.md` 19.1's pose-pipeline figure, which is the age
            # of a fix by the time a command based on it reaches the coils.
            Scenario("b", x0_mm=10, z0_mm=10, sensor_sigma_m=0.5e-3, latency_s=0.015)
        ],
        "c": [Scenario("c", plant_f_hover=143.0, duration=15.0, settle_s=8.0)],
        "d": [
            Scenario(f"d_klat{m}x", x0_mm=10, z0_mm=10, k_lat_true_mult=m)
            for m in (0.25, 1.0, 4.0)
        ],
        "e": [
            Scenario(
                "e",
                duration=14.0,
                profile=demo_profile(),
                track_tol_mm=3.0,
                settle_s=2.0,
            )
        ],
    }


def run(scenario="all", gains=None, plots=True):
    """Simulate the scenarios and report pass/fail. Returns ``(all_ok, results)``."""

    gains_path = gains or os.path.join(
        os.path.dirname(__file__), "hover_controller.json"
    )
    with open(gains_path) as f:
        gains_d = json.load(f)

    groups = build_scenarios()
    keys = list(groups) if scenario == "all" else [scenario]
    all_ok, results = True, []
    for key in keys:
        for sc in groups[key]:
            out = simulate(sc, gains_d)
            ok, msgs = evaluate(sc, out, gains_d)
            all_ok &= ok
            results.append((sc, out, ok, msgs))
            print(f"scenario {sc.name}: {'PASS' if ok else 'FAIL'}")
            print("\n".join(msgs))
            if plots:
                path = f"hover_sim_{sc.name}.png"
                plot(sc, out, gains_d, path)
                print(f"  wrote {path}")
    print("=" * 40)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return all_ok, results


def _check_velocity_estimator():
    """The rate estimate must not depend on how fast the CONTROL loop steps.

        This is `theory.md` 19.8's defect, made falsifiable: the control clock runs far
        faster than the pose pipeline, so a fix lands only every k-th tick. Dividing that
        fix's jump by the control period reports a rate k times too large, and k rises
        with the clock -- 500 Hz against ~100 Hz pose is k = 5.
    """

    truth = np.array([12.0, -7.0])       # mm/s, constant
    for ts, k in ((0.005, 2), (0.002, 5), (0.002, 20)):
        est = VelocityEstimator(ts)
        pos, t_fix = np.zeros(2), 0.0
        for i in range(4000):
            t = i * ts
            if i % k == 0:               # a fix lands: position jumps by a whole interval
                dt = t - t_fix if i else None
                pos, t_fix = truth * t, t
                v = est.update(pos.copy(), dt)
            else:                        # predictor tick: no new information
                v = est.update(pos.copy(), None)
        err = np.max(np.abs(v - truth) / np.abs(truth))
        assert err < 0.01, (
            f"ts={ts} k={k}: rate {v} against {truth}, {err:.1%} off -- the estimator "
            f"is reading the control period instead of the fix interval")

    # And the held ticks really are held, not silently re-differenced to zero: with the
    # position frozen between fixes, dividing by ts would drag the estimate toward 0.
    est = VelocityEstimator(0.002)
    est.update(np.zeros(2), None)
    est.update(truth * 0.01, 0.01)
    v0 = est.vel.copy()
    for _ in range(500):
        est.update(truth * 0.01, None)
    assert np.array_equal(est.vel, v0), "a stale tick moved the rate estimate"
    print("simulate_hover: rate estimate is independent of the control clock "
          "(ts 5/2 ms, 2-20 ticks per fix) and stale ticks hold\n  ok")


if __name__ == "__main__":
    import sys

    _check_velocity_estimator()
    ok, _ = run(sys.argv[1] if len(sys.argv) > 1 else "all")
    sys.exit(0 if ok else 1)
