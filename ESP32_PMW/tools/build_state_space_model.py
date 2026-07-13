#!/usr/bin/env python3
"""Assemble the coupled 4-channel state-space plant model from the fitted
per-channel RLC model (tools/fit_rlc_model.py -> rlc_fit.json) and, for the
steady-state coupling term, the fitted DC gain matrix
(tools/fit_coupling_matrix.py -> coupling_fit.json).

Full state-space model, x = [i_A,i_B,i_C,i_D] (coil current, the CONTROLLED
state), u = [d_A..d_D] (carrier duty %, 0-100), y = CS pin voltage (the
MEASURED/observed quantity, distinct from x -- see C below):

  L*(dx/dt) + R*x = V(u)          V(u) = diag(Kv_i)*u,  Kv_i = (4/pi)*V_supply/100
  dx/dt = A x + B u
  A = -L^-1 R
  B =  L^-1 diag(Kv)
  y = C x + D u
  C = diag(1/SENS_i)   (SENS_i is the VNH5019 CS gain, A per V, so this maps
                         state current back to the CS pin's sensed voltage)
  D = 0                (duty has no direct/instantaneous path to the CS
                         voltage that bypasses the coil current state)

R is now a FULL (non-diagonal) matrix, not diag(R_i) -- see
tools/fit_coupling_matrix.py: an empirical step-response test showed real
steady-state cross-channel coupling (e.g. stepping channel D moved A/B/C by
24-33% of D's own current shift) that a pure per-channel R-L model (with
coupling only through mutual inductance M_ij, a di/dt-only effect) cannot
produce. Rather than invent additional M_ij states, this folds the measured
coupling into R directly: at steady state, R x = V(u) => x = R^-1 V(u), so a
non-diagonal R^-1 (fit directly from measured duty->current gains) is
exactly the mechanism needed to reproduce the observed cross-channel shifts.
This is a lumped, modeling choice (the real mechanism -- shared supply
loading vs. magnetic coupling too fast to resolve as di/dt at this control
rate -- is not distinguished), not fundamental physics; see
tools/fit_coupling_matrix.py's docstring. If tools/coupling_fit.json is not
supplied, this falls back to the old diagonal-R (fully decoupled) model, and
prints a loud warning -- that fallback should not be trusted for LQR design.

Mutual inductance M_ij (tools/fit_mutual_inductance.py -> m_matrix.json) is
supported for L's off-diagonal but is NOT needed for the coupling above --
see the state-space plan: at the achievable ~20Hz control rate, the coil's
electrical settling time (~7ms, from the fitted L/R) is far faster than the
sample period, so any M_ij-driven transient has already fully decayed by the
next tick and contributes ~0 to the discretized Ad/Bd regardless of its true
value. Missing pairs default to 0 (a known gap, not a measured zero).

The series capacitor (fit_rlc_model.py fits R-L-C, not R-L) is deliberately
NOT included as a third dynamic state, matching the precedent already set by
tools/compute_model_gains.py's IMC design (tau_elec = 2L/R): first-order-per-
channel envelope model, not a full 2nd-order electrical model.

The discretization below (Ad = expm(A*Ts), Bd = A^-1(Ad-I)B) discretizes the
4-state model directly. It does NOT separately model the current-sense 50ms
EMA filter (CurrentSense.h) as an extra state -- Ts should be chosen >=
tau_filter_ms as a conservative approximation (the filter's own lag is
treated as already "inside" the achievable control period, not as a distinct
pole to observe around). See the plan's caveats section.

Usage:
  uv run python tools/build_state_space_model.py \
      --rlc-fit tools/rlc_fit.json --coupling-fit tools/coupling_fit.json \
      --v-supply 11.9 --ts 0.05 --out tools/state_space_model.json \
      --header ../src/plant_model.h
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
from scipy.linalg import expm

# VNH5019 CS gain, A per V -- per-board calibration, matches SENS[] in
# src/main_experiment.cpp / src/main_current_pid.cpp. Used only for the
# output equation y=Cx (C=diag(1/SENS)); NOT part of the A/B dynamics.
DEFAULT_SENS = {"A": 15.26, "B": 15.28, "C": 15.57, "D": 15.34}


def build_L(rlc: dict, m_matrix: dict, channels: list[str]) -> tuple[np.ndarray, list[str]]:
    n = len(channels)
    L = np.zeros((n, n))
    for i, ch in enumerate(channels):
        L[i, i] = rlc[ch]["L"]
    missing_pairs = []
    for i, j in itertools.combinations(range(n), 2):
        key = "".join(sorted([channels[i], channels[j]]))
        if key in m_matrix:
            L[i, j] = L[j, i] = m_matrix[key]
        else:
            missing_pairs.append(key)
    return L, missing_pairs


def build_R(rlc: dict, coupling_fit: dict | None, channels: list[str]) -> tuple[np.ndarray, bool]:
    """Full R matrix. Uses the fitted (non-diagonal) coupling matrix if
    available (tools/fit_coupling_matrix.py); otherwise falls back to
    diag(R_i) -- a fully decoupled model that should not be trusted for LQR
    design (see this module's docstring)."""
    if coupling_fit is not None:
        fit_channels = coupling_fit["channels"]
        R_full = np.array(coupling_fit["R_full"])
        # reorder/subset to match `channels`, in case they differ
        idx = [fit_channels.index(ch) for ch in channels]
        R = R_full[np.ix_(idx, idx)]
        return R, True
    return np.diag([rlc[ch]["R"] for ch in channels]), False


def build_C(channels: list[str], sens: dict) -> np.ndarray:
    """Output equation y = Cx (+ Du, D=0): maps state current [A] to the
    VNH5019 CS pin's sensed voltage [V]. SENS_i is A/V, so C_ii = 1/SENS_i."""
    return np.diag([1.0 / sens[ch] for ch in channels])


def build_state_space(rlc: dict, m_matrix: dict, coupling_fit: dict | None,
                       v_supply: float, channels: list[str], sens: dict) -> dict:
    n = len(channels)
    L, missing_pairs = build_L(rlc, m_matrix, channels)
    R, coupling_from_fit = build_R(rlc, coupling_fit, channels)
    Kv = np.array([(4.0 / np.pi) * v_supply / 100.0 for _ in channels])  # A/duty% -> V, same for all channels (duty-normalized)

    L_inv = np.linalg.inv(L)
    A = -L_inv @ R
    B = L_inv @ np.diag(Kv)
    C = build_C(channels, sens)
    D = np.zeros((n, n))

    return {"A": A, "B": B, "C": C, "D": D, "L": L, "R": R, "Kv": Kv,
            "missing_pairs": missing_pairs, "coupling_from_fit": coupling_from_fit}


def discretize(A: np.ndarray, B: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    Ad = expm(A * Ts)
    Bd = np.linalg.solve(A, (Ad - np.eye(A.shape[0])) @ B)
    return Ad, Bd


def write_header(path: str, channels: list[str], rlc: dict, Kv: np.ndarray) -> None:
    r_vals = ", ".join("{:.6f}f".format(rlc[ch]["R"]) for ch in channels)
    l_vals = ", ".join("{:.8f}f".format(rlc[ch]["L"]) for ch in channels)
    kv_vals = ", ".join("{:.6f}f".format(k) for k in Kv)
    lines = [
        "// GENERATED by tools/build_state_space_model.py -- do not hand-edit.",
        "// Per-channel plant constants for the LQR feedforward term",
        "// (u_ff_i = R_i * r / Kv_i, the steady-state duty->current relation).",
        "#pragma once",
        "",
        "const float PLANT_R_OHM[NUM_CHANNELS] = {" + r_vals + "};",
        "const float PLANT_L_H[NUM_CHANNELS]   = {" + l_vals + "};",
        "const float PLANT_KV[NUM_CHANNELS]    = {" + kv_vals + "};",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rlc-fit", default=os.path.join(os.path.dirname(__file__), "rlc_fit.json"))
    ap.add_argument("--m-matrix", default=os.path.join(os.path.dirname(__file__), "m_matrix.json"),
                     help="missing pairs default to 0 coupling and are reported, not fatal "
                          "(NOT needed for the steady-state coupling term -- see --coupling-fit)")
    ap.add_argument("--coupling-fit", default=os.path.join(os.path.dirname(__file__), "coupling_fit.json"),
                     help="tools/fit_coupling_matrix.py output -- full (non-diagonal) R matrix "
                          "fit from measured step-response data. Falls back to diag(R_i) "
                          "(fully decoupled, NOT trustworthy for LQR) if missing.")
    ap.add_argument("--v-supply", type=float, required=True,
                     help="bench supply voltage to assume for the Kv (duty->voltage) term")
    ap.add_argument("--channels", nargs="+", default=list("ABCD"), choices=list("ABCD"))
    ap.add_argument("--sens", nargs="+", type=float, default=None,
                     help="VNH5019 CS gain (A/V) per channel, in --channels order, for the "
                          "output equation y=Cx (default: main_experiment.cpp's SENS[] values)")
    ap.add_argument("--ts", type=float, default=0.05,
                     help="discretization sample time, seconds -- keep >= the current-sense "
                          "filter's tau (50ms default in CurrentSense.h) (default: %(default)s)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "state_space_model.json"))
    ap.add_argument("--header", default=os.path.join(os.path.dirname(__file__), "..", "src", "plant_model.h"))
    args = ap.parse_args()

    if not os.path.exists(args.rlc_fit):
        raise SystemExit(f"{args.rlc_fit} not found -- run tools/fit_rlc_model.py first "
                          "(see the state-space plan's Phase A)")

    with open(args.rlc_fit) as f:
        rlc = json.load(f)
    missing_rlc = [ch for ch in args.channels if ch not in rlc]
    if missing_rlc:
        raise SystemExit(f"rlc_fit.json is missing channel(s) {missing_rlc}")

    m_matrix = {}
    if os.path.exists(args.m_matrix):
        with open(args.m_matrix) as f:
            m_matrix = json.load(f)
    # else: mutual inductance stays 0 -- expected/deliberate, not a gap (see
    # this module's docstring: M_ij is invisible at the achievable control
    # rate regardless of its true value, so it's not worth measuring).

    coupling_fit = None
    if os.path.exists(args.coupling_fit):
        with open(args.coupling_fit) as f:
            coupling_fit = json.load(f)
    else:
        print(f"WARNING: {args.coupling_fit} not found -- falling back to diag(R_i) "
              "(fully decoupled model). Run tools/fit_coupling_matrix.py on a "
              "step-response capture first; empirically this rig has real "
              "steady-state cross-channel coupling that diag(R_i) cannot represent.")

    sens = DEFAULT_SENS if args.sens is None else dict(zip(args.channels, args.sens))

    model = build_state_space(rlc, m_matrix, coupling_fit, args.v_supply, args.channels, sens)
    if model["missing_pairs"]:
        print(f"note: mutual inductance pair(s) {model['missing_pairs']} left at 0 "
              "(deliberate -- see docstring, not a measurement gap).")
    if not model["coupling_from_fit"]:
        print("WARNING: R is diag(R_i) -- fully decoupled steady state. This model "
              "will NOT reproduce this rig's measured cross-channel coupling.")

    Ad, Bd = discretize(model["A"], model["B"], args.ts)
    eig_A = np.linalg.eigvals(model["A"])
    print(f"continuous A eigenvalues (should be real, negative -- stable RL decay): {eig_A}")
    print(f"L matrix (H):\n{model['L']}")
    print(f"R matrix (ohm-like{', fitted coupling' if model['coupling_from_fit'] else ', diagonal only'}):\n{model['R']}")
    print(f"C matrix (V/A, output = CS pin voltage):\n{model['C']}")

    out = {
        "channels": args.channels,
        "v_supply": args.v_supply,
        "ts": args.ts,
        "A": model["A"].tolist(),
        "B": model["B"].tolist(),
        "C": model["C"].tolist(),
        "D": model["D"].tolist(),
        "Ad": Ad.tolist(),
        "Bd": Bd.tolist(),
        "L": model["L"].tolist(),
        "R": model["R"].tolist(),
        "Kv": model["Kv"].tolist(),
        "missing_coupling_pairs": model["missing_pairs"],
        "coupling_from_fit": model["coupling_from_fit"],
        "sens": sens,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")

    write_header(args.header, args.channels, rlc, model["Kv"])
    print(f"wrote {args.header}")


if __name__ == "__main__":
    main()
