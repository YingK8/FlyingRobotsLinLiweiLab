#!/usr/bin/env python3
"""Closed-loop nonlinear simulation of the state-space hover controller.

Truth plant: the full 6-state nonlinear model (ai/hover_model.py) integrated
with solve_ivp one ZOH interval at a time; controller runs at the camera rate
with the SAME DiscreteHoverController code path the hardware runner uses
(velocity estimation, integrators + anti-windup, saturation, slew).

Scenarios (--scenario a|b|c|d|e|all):
  a  10 mm lateral + 10 mm vertical initial offset, clean measurements
  b  (a) + 0.5 mm sensor noise + 1 frame latency
  c  trim mismatch: plant hovers at 143 Hz, controller believes 140
     (integrator proof)
  d  k_lat robustness: true k_lat in {0.25x, 1x, 4x} the design value
  e  profile tracking: quadratic-ease 10 mm climb + linear 15 mm lateral
     translate (reference feedforward proof)

Each run prints PASS/FAIL and writes hover_sim_<scenario>.png: x/z vs
reference (+-2 mm band), inputs with saturation lines, and wrapped delta(t)
with the 90 deg pull-out line (validity check for the phase-locked
model reduction).

Usage: uv run python ai/simulate_hover.py [--scenario all]
       [--gains ai/hover_controller.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from hover_model import HoverParams, make_params, nonlinear_dynamics
from reference_profiles import Profile, demo_profile


class VelocityEstimator:
    """Finite difference + 1-pole IIR low-pass (default ~5 Hz cutoff)."""

    def __init__(self, ts: float, cutoff_hz: float = 5.0):
        self.ts = ts
        tau = 1.0 / (2.0 * math.pi * cutoff_hz)
        self.alpha = ts / (ts + tau)
        self.prev_pos = None
        self.vel = np.zeros(2)

    def update(self, pos: np.ndarray) -> np.ndarray:
        if self.prev_pos is None:
            self.prev_pos = pos.copy()
            return self.vel.copy()
        raw = (pos - self.prev_pos) / self.ts
        self.prev_pos = pos.copy()
        self.vel += self.alpha * (raw - self.vel)
        return self.vel.copy()


class DiscreteHoverController:
    """u(k) = u_trim + u_ff(k) - K [x_hat - x_ref ; q]  -- see design_hover_lqr.py.

    Shared verbatim between this simulator and hover_controller_runner.py.
    step() expects to be called once per camera frame (fixed rate ts)."""

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
        self.q = np.zeros(2)             # error integrators [int_ex, int_ez]
        self.prev_f_field = self.f_hover  # slew-limit reference

    def step(self, t: float, x_meas: float, z_meas: float) -> tuple[float, float]:
        pos = np.array([x_meas, z_meas])
        vel = self.est.update(pos)
        ref_p, ref_v, ref_a = self.profile.eval(t)

        err = np.array([pos[0] - ref_p[0], vel[0] - ref_v[0],
                        pos[1] - ref_p[1], vel[1] - ref_v[1]])
        # reference-acceleration feedforward (exact inverse of B's nonzero entries)
        u_ff = np.array([ref_a[0] / (self.g * self.k_lat),
                         ref_a[1] * self.f_hover / (2.0 * self.g)])
        u = np.array([0.0, self.f_hover]) + u_ff - self.K @ np.concatenate([err, self.q])

        # saturate: mag symmetric clamp; f_field clamp + slew limit
        mag = float(np.clip(u[0], -self.mag_max, self.mag_max))
        f_tgt = float(np.clip(u[1], self.freq_min, self.freq_max))
        f_field = float(np.clip(f_tgt, self.prev_f_field - self.freq_slew,
                                self.prev_f_field + self.freq_slew))
        self.prev_f_field = f_field

        # Conditional-integration anti-windup. u depends on q through -K, so
        # integrating error e moves u by ~ -K_int*ts*e: when the unsaturated
        # command is above the clamp we need e > 0 to come back into range,
        # below the clamp we need e < 0. Slew limiting is a transient and
        # does NOT freeze the integrator -- only hard clamps do.
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
    x0_mm: float = 0.0            # initial lateral offset
    z0_mm: float = 0.0            # initial vertical offset
    sensor_sigma_m: float = 0.0
    latency_frames: int = 0
    dist_accel: float = 0.0       # constant lateral disturbance accel m/s^2
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
    p_true = make_params(f_hover=sc.plant_f_hover or design["f_hover"],
                         k_lat=design["k_lat"] * sc.k_lat_true_mult,
                         margin=design["margin"])
    rng = np.random.default_rng(seed)

    # start at the plant's true hover trim, plus position offsets
    s = np.array([sc.x0_mm * 1e-3, 0.0, sc.z0_mm * 1e-3, 0.0,
                  p_true.delta_trim, p_true.omega_trim])
    meas_queue = deque(maxlen=sc.latency_frames + 1)
    n = int(round(sc.duration / ts))
    out = {k: np.zeros(n) for k in ("t", "x", "z", "mag", "f_field", "delta",
                                    "x_ref", "z_ref")}

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
        out["delta"][k] = s[4]
        out["x_ref"][k], out["z_ref"][k] = ref_p

        sol = solve_ivp(dyn, (t, t + ts), s, args=(u,), max_step=ts / 8,
                        rtol=1e-8, atol=1e-9)
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
    msgs.append(f"  settle: max err after {sc.settle_s:.0f}s = {e_after.max():.2f} mm "
                f"(< {sc.tol_mm} mm) {'PASS' if cond else 'FAIL'}")

    cond = np.abs(dw).max() < 60.0
    ok &= cond
    msgs.append(f"  phase lock: max |delta| = {np.abs(dw).max():.1f} deg (< 60) "
                f"{'PASS' if cond else 'FAIL'}")

    mag_max = gains["limits"]["mag_max"]
    sat_frac = np.mean(np.abs(out["mag"]) >= mag_max * 0.999)
    cond = sat_frac < 0.2
    ok &= cond
    msgs.append(f"  saturation: mag saturated {sat_frac*100:.0f}% of run (< 20%) "
                f"{'PASS' if cond else 'FAIL'}")

    if sc.track_tol_mm is not None:
        warm = out["t"] >= 2.0  # skip initial transient
        e_all = np.maximum(np.abs(ex[warm]), np.abs(ez[warm]))
        cond = e_all.max() < sc.track_tol_mm
        ok &= cond
        msgs.append(f"  tracking: max err = {e_all.max():.2f} mm "
                    f"(< {sc.track_tol_mm} mm) {'PASS' if cond else 'FAIL'}")
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
        axes[0].fill_between(out["t"], (ref - 2e-3) * 1e3, (ref + 2e-3) * 1e3,
                             alpha=0.08, color="gray")
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
        "b": [Scenario("b", x0_mm=10, z0_mm=10, sensor_sigma_m=0.5e-3,
                       latency_frames=1)],
        "c": [Scenario("c", plant_f_hover=143.0, duration=15.0, settle_s=8.0)],
        "d": [Scenario(f"d_klat{m}x", x0_mm=10, z0_mm=10, k_lat_true_mult=m)
              for m in (0.25, 1.0, 4.0)],
        "e": [Scenario("e", duration=14.0, profile=demo_profile(),
                       track_tol_mm=3.0, settle_s=2.0)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all",
                    choices=["a", "b", "c", "d", "e", "all"])
    ap.add_argument("--gains", default=os.path.join(os.path.dirname(__file__),
                                                    "hover_controller.json"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    with open(args.gains) as f:
        gains = json.load(f)

    groups = build_scenarios()
    keys = list(groups) if args.scenario == "all" else [args.scenario]
    all_ok = True
    for key in keys:
        for sc in groups[key]:
            out = simulate(sc, gains)
            ok, msgs = evaluate(sc, out, gains)
            all_ok &= ok
            print(f"scenario {sc.name}: {'PASS' if ok else 'FAIL'}")
            print("\n".join(msgs))
            if not args.no_plots:
                path = f"hover_sim_{sc.name}.png"
                plot(sc, out, gains, path)
                print(f"  wrote {path}")
    print("=" * 40)
    print("ALL PASS" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
