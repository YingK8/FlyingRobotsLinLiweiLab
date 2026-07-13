#!/usr/bin/env python3
"""Design a discrete-time LQR gain from the coupled 4-channel plant model
(tools/build_state_space_model.py -> state_space_model.json).

IMPORTANT finding from this rig's fitted model: Ad ~ 0 (all entries ~1e-61).
The electrical settling time (~7ms, from the fitted L/R) is far faster than
the achievable control period (Ts=50ms, current-sense filter bandwidth
limited), so by every sample the state has already fully forgotten the
previous one -- x_{k+1} ~= Bd @ u_k, independent of x_k (and Bd numerically
matches the directly-measured DC gain matrix from
tools/fit_coupling_matrix.py, an end-to-end cross-check that the model is
right). A PLAIN discrete LQR on this system correctly returns K ~ 0: state
feedback is optimally worthless when the state has zero predictive power
over the next sample. But K~0 gives a controller indistinguishable from
open-loop feedforward -- no correction for model error, disturbances, or
duty saturation nonlinearity.

Fix: augment the state with the INTEGRAL of tracking error (standard
technique for near-static/fast-settling plants that still need closed-loop
correction -- "LQI"), and OPTIONALLY also with the PREVIOUS tracking error,
so a discrete-derivative term can be penalized in the same augmented-Riccati
framework (a proper "PID"-style K_x/K_d/K_z split, not a bolted-on heuristic
derivative). Let e_k = x_k - x_ref (assumed ~constant over the LQR's design
horizon), e_prev_k = e_{k-1}, and z_k = integral of e. Then:

  e_{k+1}      = Ad e_k + Bd u_k
  e_prev_{k+1} = e_k                  (pure one-step delay)
  z_{k+1}      = z_k + Ts * e_k

Augmented system: [e; e_prev; z]_{k+1} =
  [[Ad, 0, 0], [I, 0, 0], [Ts*I, 0, I]] [e; e_prev; z]_k + [Bd; 0; 0] u_k.

Cost function: J = sum_k e_k^T Q e_k
                  + q_derivative * ((e_k - e_prev_k)/Ts)^T Q ((e_k - e_prev_k)/Ts)
                  + q_integral * z_k^T Q z_k
                  + u_k^T R u_k.
Q is built from two weights so "penalize spread between channels" is an
explicit, separate knob from "track the shared target":

  Q = q_track * I + q_spread * (n*I - ones(n,n))

The second term is the (scaled) graph Laplacian of the complete graph on the
4 channels: x^T(n*I - J)x = sum_{i<j} (x_i - x_j)^2 -- it penalizes pairwise
differences ONLY (null space = the uniform vector), i.e. "equalize the 4
channels" as an explicit cost term. The integral and derivative terms reuse
the same Q shape (scaled by --q-integral / --q-derivative respectively) so
steady-state SPREAD (integral) and transient SPREAD RINGING (derivative) are
both driven down, not just steady-state tracking.

R = r_duty * I penalizes actuation effort (duty deviation from feedforward).

Solving the augmented Riccati equation gives a gain split into K_e (on e_k),
K_eprev (on e_prev_k), and K_z (on z_k). Firmware doesn't track e_prev as a
raw state -- it tracks a filtered discrete derivative derr=(e-e_prev)/Ts
directly (see main_state_space.cpp), so this script converts the solved
(K_e, K_eprev) pair into an equivalent (K_x, K_d) pair via the substitution
e_prev = e - Ts*derr:

  -K_e @ e - K_eprev @ e_prev = -K_e @ e - K_eprev @ (e - Ts*derr)
                              = -(K_e + K_eprev) @ e + Ts*K_eprev @ derr
  => K_x = K_e + K_eprev,   K_d = -Ts * K_eprev

so that -K_x @ e - K_d @ derr reproduces the original augmented control law
exactly (verified numerically below before any header is written). At
--q-derivative 0.0 (the default), the derivative cost block vanishes, K_eprev
solves to ~0, hence K_d~0 and K_x~K_e -- i.e. this reduces exactly to the
plain LQI design used before this option existed (verify this when in doubt:
diff the K_x output against a pre-derivative lqr_gains.json at the same
q_track/q_spread/q_integral/r_duty).

Usage:
  uv run python tools/design_lqr_gains.py --model tools/state_space_model.json \
      --q-track 1.0 --q-spread 50.0 --q-integral 5.0 --q-derivative 0.0 \
      --r-duty 0.1 --out tools/lqr_gains.json --header ../src/lqr_gains.h
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.linalg import solve_discrete_are


def build_Q(n: int, q_track: float, q_spread: float) -> np.ndarray:
    return q_track * np.eye(n) + q_spread * (n * np.eye(n) - np.ones((n, n)))


def design_lqi(Ad: np.ndarray, Bd: np.ndarray, Ts: float, Q: np.ndarray,
               q_integral: float, q_derivative: float,
               R: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integral- (and optionally derivative-) augmented discrete LQR.
    Returns (K_x, K_d, K_z, closed_loop_eig)."""
    n = Ad.shape[0]
    Z = np.zeros((n, n))
    I = np.eye(n)

    Ad_aug = np.block([[Ad, Z, Z],
                        [I, Z, Z],
                        [Ts * I, Z, I]])
    Bd_aug = np.vstack([Bd, Z, Z])

    qd = q_derivative / (Ts ** 2)
    Q_aug = np.block([[Q + qd * Q, -qd * Q, Z],
                       [-qd * Q, qd * Q, Z],
                       [Z, Z, q_integral * Q]])

    P = solve_discrete_are(Ad_aug, Bd_aug, Q_aug, R)
    K_aug = np.linalg.solve(R + Bd_aug.T @ P @ Bd_aug, Bd_aug.T @ P @ Ad_aug)
    K_e, K_eprev, K_z = K_aug[:, :n], K_aug[:, n:2 * n], K_aug[:, 2 * n:]

    K_x = K_e + K_eprev
    K_d = -Ts * K_eprev

    # Self-check: -K_e@e - K_eprev@e_prev must equal -K_x@e - K_d@derr for
    # derr=(e-e_prev)/Ts, on random test points -- catches any algebra
    # mistake in the (K_e,K_eprev)->(K_x,K_d) conversion before it ever
    # reaches a header.
    rng = np.random.default_rng(0)
    for _ in range(5):
        e = rng.normal(size=n)
        e_prev = rng.normal(size=n)
        derr = (e - e_prev) / Ts
        lhs = -K_e @ e - K_eprev @ e_prev
        rhs = -K_x @ e - K_d @ derr
        assert np.allclose(lhs, rhs, atol=1e-8), \
            "K_x/K_d conversion does not reproduce the augmented control law"

    eig_cl = np.linalg.eigvals(Ad_aug - Bd_aug @ K_aug)
    return K_x, K_d, K_z, eig_cl


def write_header(path: str, K_x: np.ndarray, K_d: np.ndarray, K_z: np.ndarray,
                  name_suffix: str = "") -> None:
    n = K_x.shape[0]

    def mat_lines(name, K):
        rows = []
        for i in range(n):
            row_vals = ", ".join("{:.8f}f".format(v) for v in K[i])
            rows.append("  {" + row_vals + "}")
        return [f"const float {name}{name_suffix}[NUM_CHANNELS][NUM_CHANNELS] = {{",
                ",\n".join(rows), "};", ""]

    lines = [
        "// GENERATED by tools/design_lqr_gains.py -- do not hand-edit.",
        "// Integral- (and optionally derivative-) augmented LQR (see this",
        "// script's docstring -- the fitted plant is near-memoryless at this",
        "// control rate, so plain state feedback K_x is expected to be weak;",
        "// K_z (integral-of-error feedback) is where most of the real",
        "// closed-loop correction comes from. K_d (derivative-of-error",
        "// feedback) is ~0 unless --q-derivative was nonzero when this was",
        "// generated.",
        "// u = u_ff(r) - LQR_KX @ (x - x_ref) - LQR_KD @ derr_filt - LQR_KZ @ z_integral",
        "// z_integral[k+1] = z_integral[k] + Ts * (x_ref - x[k])  (anti-windup: freeze on saturation)",
        "// derr_filt = low-pass-filtered (err - err_prev) / Ts  (see main_state_space.cpp)",
        "#pragma once",
        "",
    ]
    lines += mat_lines("LQR_KX", K_x)
    lines += mat_lines("LQR_KD", K_d)
    lines += mat_lines("LQR_KZ", K_z)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "state_space_model.json"))
    ap.add_argument("--q-track", type=float, default=1.0,
                     help="cost weight on tracking the shared target r (default: %(default)s)")
    ap.add_argument("--q-spread", type=float, default=50.0,
                     help="cost weight on pairwise spread between channels -- the main "
                          "'equalize' knob; raise this relative to q-track to prioritize "
                          "spread over tracking speed (default: %(default)s)")
    ap.add_argument("--q-integral", type=float, default=5.0,
                     help="cost weight on the integral-of-error state relative to Q -- "
                          "this is what makes K_z nonzero (plain state feedback K_x is "
                          "~0 for this plant, see docstring); raise to tighten steady-"
                          "state tracking/spread at the cost of slower integral response "
                          "(default: %(default)s)")
    ap.add_argument("--q-derivative", type=float, default=0.0,
                     help="cost weight on the discrete derivative of tracking error -- "
                          "makes K_d nonzero, penalizing transient spread RINGING (as seen "
                          "during RAMP_UP) rather than steady-state spread/tracking. "
                          "Default 0.0 (D term off, reduces exactly to the prior plain-LQI "
                          "design) -- tune this manually and deliberately, a mistuned "
                          "derivative term can amplify current-sense noise into duty "
                          "chatter (default: %(default)s)")
    ap.add_argument("--r-duty", type=float, default=0.1,
                     help="cost weight on duty actuation effort (default: %(default)s)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "lqr_gains.json"))
    ap.add_argument("--header", default=os.path.join(os.path.dirname(__file__), "..", "src", "lqr_gains.h"))
    ap.add_argument("--name-suffix", default="",
                     help="appended to the generated LQR_KX/LQR_KD/LQR_KZ array names (e.g. "
                          "'_CCW') so a CW and a CCW header can both be #include'd without "
                          "clashing -- coupling (and thus K_x/K_d/K_z) is direction-dependent "
                          "on this rig, confirmed on hardware: CW-fit gains applied to CCW "
                          "operation did not converge (channel saturated, spread ~1.9A "
                          "instead of settling)")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} not found -- run tools/build_state_space_model.py first")

    with open(args.model) as f:
        model = json.load(f)
    Ad = np.array(model["Ad"])
    Bd = np.array(model["Bd"])
    Ts = model["ts"]
    n = Ad.shape[0]

    Q = build_Q(n, args.q_track, args.q_spread)
    R = args.r_duty * np.eye(n)
    K_x, K_d, K_z, eig_cl = design_lqi(Ad, Bd, Ts, Q, args.q_integral, args.q_derivative, R)

    max_mag = np.max(np.abs(eig_cl))
    print(f"augmented closed-loop eigenvalues (discrete, must be inside unit circle): {eig_cl}")
    print(f"max |eigenvalue| = {max_mag:.4f}  -> {'STABLE' if max_mag < 1.0 else 'UNSTABLE -- do not flash this K'}")
    print(f"K_x (state feedback, expected weak for this near-memoryless plant) =\n{K_x}")
    print(f"K_d (derivative feedback, ~0 unless --q-derivative > 0) =\n{K_d}")
    print(f"K_z (integral feedback, the term that actually corrects) =\n{K_z}")

    if max_mag >= 1.0:
        raise SystemExit("designed K is not stabilizing for this model -- refusing to write "
                          "output. Try raising --r-duty or lowering --q-integral/--q-derivative.")

    out = {
        "K_x": K_x.tolist(),
        "K_d": K_d.tolist(),
        "K_z": K_z.tolist(),
        "q_track": args.q_track,
        "q_spread": args.q_spread,
        "q_integral": args.q_integral,
        "q_derivative": args.q_derivative,
        "r_duty": args.r_duty,
        "closed_loop_eig": [[float(v.real), float(v.imag)] for v in eig_cl],
        "channels": model["channels"],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    write_header(args.header, K_x, K_d, K_z, args.name_suffix)
    print(f"wrote {args.header}")


if __name__ == "__main__":
    main()
