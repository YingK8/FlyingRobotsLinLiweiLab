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
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

from hover_model import make_params, linearize_reduced, discretize

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


def design(
    rate_hz=30.0,
    f_hover=140.0,
    k_lat=0.05,
    margin=5.0,
    mag_max=0.8,
    freq_delta_max=15.0,
    freq_slew=200.0,
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
    r_diag = [1 / 0.5**2, 1 / 3.0**2]
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
