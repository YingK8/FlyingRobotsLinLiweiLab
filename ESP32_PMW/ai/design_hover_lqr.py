#!/usr/bin/env python3
"""Discrete-LQR design for the state-space hover controller.

Design model: reduced 4-state double-integrator pair from ai/hover_model.py
(linearize_reduced), ZOH-discretized at the camera rate, augmented with
integrators on the x and z position errors (6 design states):

  x_a = [x, xd, z, zd, int_ex, int_ez]
  A_a = [[Ad, 0], [C*Ts, I]],  B_a = [[Bd], [0]],  C = [[1,0,0,0],[0,0,1,0]]

Gain K (2x6) from scipy.linalg.solve_discrete_are. Control law (implemented
in simulate_hover.DiscreteHoverController, shared with the runner):

  u(k) = u_trim + u_ff(k) - K [ x_hat(k)-x_ref(k) ; q(k) ]

u_trim = (mag=0, f_field=f_hover); u_ff is the analytic reference-acceleration
feedforward from ai/reference_profiles.py. With the augmented double
integrators u_ss = 0 for any constant setpoint, so no Nbar matrix is needed.

Weights: Bryson's rule -- tolerate 10 mm error on both axes, 5 cm/s lateral
/ 10 cm/s vertical velocity, integrator states scaled to position tolerance
x 1 s; R = diag(1/0.5^2, 1/5^2). Chosen so BOTH axes land near ~1.3 Hz
dominant poles: a faster vertical loop (e.g. 4 Hz with the tighter 5 mm /
1/10^2 weighting) demands frequency slews ~300 Hz/s and limit-cycles against
the slew limiter (verified in simulate_hover.py scenario a). Hard-fails if
any closed-loop pole is faster than rate/6 -- the loop must stay well below
Nyquist and the 15.6 Hz phase-lock mode.

Slew default 200 Hz/s: the 1.33 Hz vertical loop needs up to
2*pi*1.33*15 ~ 125 Hz/s, while phase-lock pull-out headroom at margin 5
allows (tau_max - drag)/(2*pi*I) ~ 1250 Hz/s -- 6x safety factor.

Usage: uv run python ai/design_hover_lqr.py [--rate 30] [--k-lat 0.05] ...
Writes ai/hover_controller.json (convention: compute_model_gains.py).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are

from hover_model import make_params, linearize_reduced, discretize

C_MEAS = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])


def augment_integrators(Ad: np.ndarray, Bd: np.ndarray, C: np.ndarray,
                        ts: float) -> tuple[np.ndarray, np.ndarray]:
    """q(k+1) = q(k) + Ts*(C x(k) - y_ref): forward-Euler error integrators."""
    n, m = Ad.shape[0], Bd.shape[1]
    p = C.shape[0]
    Aa = np.block([[Ad, np.zeros((n, p))],
                   [C * ts, np.eye(p)]])
    Ba = np.vstack([Bd, np.zeros((p, m))])
    return Aa, Ba


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Discrete LQR: K = (R + B'PB)^-1 B'PA with P from the discrete ARE."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    eig = np.linalg.eigvals(A - B @ K)
    return K, P, eig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=30.0, help="camera/controller rate Hz")
    ap.add_argument("--f-hover", type=float, default=140.0,
                    help="lift=weight frequency (GUI default 140; firmware HOVER_HZ=150)")
    ap.add_argument("--k-lat", type=float, default=0.05,
                    help="SEED GUESS tilt per unit mag -- re-identify on rig")
    ap.add_argument("--margin", type=float, default=5.0, help="torque margin")
    ap.add_argument("--mag-max", type=float, default=0.8)
    ap.add_argument("--freq-delta-max", type=float, default=15.0,
                    help="max |f_field - f_hover| Hz")
    ap.add_argument("--freq-slew", type=float, default=200.0,
                    help="Hz per second (see docstring for the pull-out bound)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "hover_controller.json"))
    args = ap.parse_args()

    ts = 1.0 / args.rate
    p = make_params(f_hover=args.f_hover, k_lat=args.k_lat, margin=args.margin)
    A, B = linearize_reduced(p)
    Ad, Bd = discretize(A, B, ts)
    Aa, Ba = augment_integrators(Ad, Bd, C_MEAS, ts)

    # Bryson weights (see module docstring)
    q_diag = [1 / 0.010**2, 1 / 0.05**2, 1 / 0.010**2, 1 / 0.10**2,
              1 / 0.010**2, 1 / 0.010**2]
    r_diag = [1 / 0.5**2, 1 / 5.0**2]
    K, _, eig = dlqr(Aa, Ba, np.diag(q_diag), np.diag(r_diag))

    # discrete eigenvalue -> equivalent continuous rate in Hz
    poles_hz = np.abs(np.log(eig.astype(complex))) / ts / (2 * np.pi)

    np.set_printoptions(precision=3, suppress=True)
    print(f"design @ {args.rate:.0f} Hz (Ts={ts*1000:.1f} ms), "
          f"f_hover={args.f_hover}, k_lat={args.k_lat} (seed), margin={args.margin}")
    print("K (rows: mag_signed, delta_f_hz | cols: x xd z zd int_ex int_ez):")
    print(K)
    print("closed-loop pole rates [Hz]:", np.sort(poles_hz.real))

    limit_hz = args.rate / 6.0
    if np.any(poles_hz > limit_hz):
        sys.exit(f"FAIL: closed-loop pole(s) exceed {limit_hz:.1f} Hz (= rate/6) -- "
                 f"detune Q/R or raise --rate")

    out = {
        "meta": {
            "generated_by": "ai/design_hover_lqr.py",
            "date": datetime.date.today().isoformat(),
            "args": vars(args),
        },
        "params": {
            "I_robot": p.I_robot, "k_drag": p.k_drag, "f_hover": p.f_hover,
            "k_lat": p.k_lat, "margin": p.margin, "g": p.g,
            "tau_max": p.tau_max, "delta_trim_rad": p.delta_trim,
        },
        "design": {"rate_hz": args.rate, "ts": ts,
                   "Q_diag": q_diag, "R_diag": r_diag},
        "K": K.tolist(),
        "u_ff": {"mag": 0.0, "f_field_hz": p.f_hover},
        "limits": {"mag_max": args.mag_max,
                   "freq_min": p.f_hover - args.freq_delta_max,
                   "freq_max": p.f_hover + args.freq_delta_max,
                   "freq_slew_hz_per_s": args.freq_slew},
        "closed_loop_poles_hz": sorted(float(x) for x in poles_hz.real),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
