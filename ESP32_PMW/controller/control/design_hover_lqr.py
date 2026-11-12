#!/usr/bin/env python3
"""
Discrete-LQR design for the hover controller. Writes hover_controller.json.

Design model: the reduced 4-state double-integrator pair from hover_model.linearize_reduced,
ZOH-discretized at the camera rate, augmented with integrators on the x and z position errors:

    x_a = [x, xd, z, zd, int_ex, int_ez]
    A_a = [[Ad, 0], [C*Ts, I]],  B_a = [[Bd], [0]],  C = [[1,0,0,0],[0,0,1,0]]

Control law, implemented in simulate_hover.DiscreteHoverController and shared with the runner:

    u(k) = u_trim + u_ff(k) - K [ x_hat(k)-x_ref(k) ; q(k) ]

With the augmented double integrators u_ss = 0 for any constant setpoint, so no Nbar is needed.

Weights are Bryson's rule tuned for LATENCY ROBUSTNESS, not for speed: both axes land at
~0.78 Hz. Tighter designs pass on paper and fail in closed loop, at 1.33 Hz through a ~2.2 Hz
limit cycle from one frame of latency, and at 3.9 Hz by demanding infeasible frequency slew.
Hard-fails if any closed-loop pole exceeds rate/6.

Usage: uv run python controller/control/design_hover_lqr.py
"""

from __future__ import annotations

import datetime
import json
import os

import numpy as np
from scipy.linalg import solve_discrete_are

from controller.control import constants as C
from controller.control.hover_model import make_params, linearize_reduced, discretize

C_MEAS = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])


def augment_integrators(
    Ad: np.ndarray, Bd: np.ndarray, C: np.ndarray, ts: float
) -> tuple[np.ndarray, np.ndarray]:
    """q(k+1) = q(k) + Ts*(C x(k) - y_ref): forward-Euler error integrators."""

    n, m = Ad.shape[0], Bd.shape[1]
    p = C.shape[0]
    Aa = np.block([[Ad, np.zeros((n, p))], [C * ts, np.eye(p)]])
    Ba = np.vstack([Bd, np.zeros((p, m))])
    return Aa, Ba


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Discrete LQR: K = (R + B'PB)^-1 B'PA with P from the discrete ARE."""

    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig = np.linalg.eigvals(A - B @ K)
    return K, P, eig


def _noise_provenance(rate_hz):
    """
    The measured noise these gains assume, for the gain file's own record.

        Degrades to ``{"measured": False}`` when no static calibration exists --
        the design does not depend on it, and refusing to synthesise gains until
        someone has clamped the robot would be the wrong trade.
    """

    import sys as _sys
    from pathlib import Path as _Path

    here = _Path(__file__).resolve().parent
    try:
        from controller.pose.noise import NoiseModel
    except Exception as e:
        return {"measured": False, "why": f"noise model unavailable: {e}"}

    m = NoiseModel.load()
    if not m.measured:
        return {"measured": False,
                "why": "no static calibration recorded; run noise.record_live() "
                       "or controller/pose/noise.py --record"}

    dt = 1.0 / rate_hz
    return {
        "measured": True,
        "condition": m.meta.get("condition"),
        "created": m.meta.get("created"),
        "sigma_lateral_mm": m.sigma_lateral_mm,
        "sigma_depth_frac": m.sigma_depth_frac_world,
        "sigma_normal": m.sigma_normal,
        "sigma_white_mm": m.sigma_white,
        "rho1": m.rho1,
        "tau_corr_s": m.tau_s,
        # What the loop actually sees on its rate channels at this rate.
        "velocity_sigma_mm_s": {
            ax: m.velocity_sigma_mm_s(ax, tau_s=0.08, dt=dt) for ax in "xyz"
        },
    }


def design(
    # The rate the SHIPPED gain file is designed at, so re-running this module as its own
    # docstring documents regenerates what is already there. It was 30.0 -- a stale
    # default left behind when the loop moved to 200 Hz -- and `__main__` calls `design()`
    # bare, so following the documented command silently overwrote the shipped file with
    # gains for a rate the loop has not run at since 2026-08-30. The closed loop is
    # invariant in `rate_hz` (`theory.md` 19.8), so this never announced itself as wrong.
    rate_hz=500.0,
    # The LINEARISATION POINT for the gains, not a claim about where the rig hovers --
    # `_anchor` overwrites the trim at runtime with the frequency the ramp actually
    # reached, so this only has to be close enough for the linearisation to hold. 160 Hz
    # from the 2026-08-29 takeoffs. Was 140.0, a MATLAB GUI text-box default that was
    # never a measurement of this rig. See `theory.md` 18.12 for why this, z_track's
    # F_HOVER_HZ and constants.RAMP_TARGET_HZ are all different and all correct.
    f_hover=C.F_HOVER_DESIGN_HZ,
    k_lat=0.05,
    margin=5.0,
    # No power ceiling in software: the bench supply's current limit is the only one.
    # mag_max is the firmware's own range for `mag=<0..1>`.
    mag_max=1.0,
    # +/- about f_hover. Wide on purpose: a narrow window is a *control* envelope, and
    # centred on the design f_hover it clamps hard the moment the rig is run anywhere
    # else -- at a 60 Hz ramp target the old +/-15 would have commanded 125 Hz on arming.
    freq_delta_max=139.0,
    freq_slew=200.0,
    # Bryson's rule sets `r = 1 / u_max^2`, where `u_max` is the control excursion worth
    # one unit of cost. This scales that excursion: >1 says a larger command is
    # acceptable, which buys closed-loop bandwidth. It is the ONE knob for the
    # speed-vs-latency-robustness trade, kept as a knob rather than folded into the
    # literals so the sweep behind the shipped value is reproducible. See `theory.md`
    # 19.13 for what was measured across it.
    authority=1.0,
    out=None,
):
    """Synthesise the hover LQR and write the gain file. Returns the gain dict."""

    ts = 1.0 / rate_hz
    p = make_params(f_hover=f_hover, k_lat=k_lat, margin=margin)
    A, B = linearize_reduced(p)
    Ad, Bd = discretize(A, B, ts)
    Aa, Ba = augment_integrators(Ad, Bd, C_MEAS, ts)

    # Bryson weights (see module docstring for why NOT tighter)
    q_diag = [
        1 / 0.010**2,
        1 / 0.05**2,
        1 / 0.030**2,
        1 / 0.15**2,
        1 / 0.010**2,
        1 / 0.030**2,
    ]
    r_diag = [1 / (0.5 * authority) ** 2, 1 / (3.0 * authority) ** 2]
    K, _, eig = dlqr(Aa, Ba, np.diag(q_diag), np.diag(r_diag))

    # discrete eigenvalue -> equivalent continuous rate in Hz
    poles_hz = np.abs(np.log(eig.astype(complex))) / ts / (2 * np.pi)

    np.set_printoptions(precision=3, suppress=True)
    print(
        f"design @ {rate_hz:.0f} Hz (Ts={ts*1000:.1f} ms), "
        f"f_hover={f_hover}, k_lat={k_lat} (seed), margin={margin}"
    )
    print("K (rows: mag_signed, delta_f_hz | cols: x xd z zd int_ex int_ez):")
    print(K)
    print("closed-loop pole rates [Hz]:", np.sort(poles_hz.real))

    limit_hz = rate_hz / 6.0
    if np.any(poles_hz > limit_hz):
        raise ValueError(
            f"closed-loop pole(s) exceed {limit_hz:.1f} Hz (= rate/6) -- "
            f"detune Q/R or raise rate_hz"
        )

    noise_block = _noise_provenance(rate_hz)
    if noise_block.get("measured") and noise_block.get("sigma_lateral_mm"):
        # Bryson weights the x state by 1/0.010^2, i.e. 10 mm is the deviation worth
        # one unit of cost. A measurement whose own noise is a large fraction of that
        # makes the controller spend authority chasing noise.
        frac = noise_block["sigma_lateral_mm"] / 1e3 / 0.010
        if frac > 0.1:
            print(f"WARNING: measured lateral noise is {frac:.0%} of the 10 mm Bryson "
                  f"weight on x.\n         The x/int_ex weights are chasing noise; "
                  f"loosen them or filter harder.")

    gains = {
        "meta": {
            "generated_by": "controller/control/design_hover_lqr.design",
            "date": datetime.date.today().isoformat(),
            "args": {
                "rate_hz": rate_hz,
                "f_hover": f_hover,
                "k_lat": k_lat,
                "margin": margin,
                "mag_max": mag_max,
                "freq_delta_max": freq_delta_max,
                "freq_slew": freq_slew,
            },
        },
        "params": {
            "I_robot": p.I_robot,
            "k_drag": p.k_drag,
            "f_hover": p.f_hover,
            "k_lat": p.k_lat,
            "margin": p.margin,
            "g": p.g,
            "tau_max": p.tau_max,
            "delta_trim_rad": p.delta_trim,
        },
        "design": {"rate_hz": rate_hz, "ts": ts, "Q_diag": q_diag, "R_diag": r_diag},
        # What noise these gains were tuned against. Provenance, not an LQG R: the
        # full state is already measured (`pose/filter.py`), and `control/theory.md`
        # S11 says no further observer is needed. A gain file that does not record
        # the noise it assumed cannot be audited when the noise changes.
        "noise": noise_block,
        "K": K.tolist(),
        "u_ff": {"mag": 0.0, "f_field_hz": p.f_hover},
        "limits": {
            "mag_max": mag_max,
            "freq_min": p.f_hover - freq_delta_max,
            "freq_max": p.f_hover + freq_delta_max,
            "freq_slew_hz_per_s": freq_slew,
        },
        "closed_loop_poles_hz": sorted(float(x) for x in poles_hz.real),
    }
    path = out or os.path.join(os.path.dirname(__file__), "hover_controller.json")
    with open(path, "w") as f:
        json.dump(gains, f, indent=2)
    print(f"wrote {path}")
    return gains


if __name__ == "__main__":
    design()
