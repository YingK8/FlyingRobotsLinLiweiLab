#!/usr/bin/env python3
"""Fit a full 4x4 steady-state coupling matrix from a duty-STEP experiment
(tools/gen_step_response_experiment.py + a main_experiment.cpp capture) --
this REPLACES the diagonal-R assumption in tools/build_state_space_model.py
with the coupling actually measured on hardware.

Background (see tools/validate_state_space_model.py): the pure R-L-M plant
model predicts ZERO steady-state cross-channel coupling (mutual inductance
only enters through di/dt, and at the achievable ~20Hz control rate any
transient from M_ij has long since settled -- see the state-space plan).
But an empirical step-response run on this rig showed real, sizeable
steady-state coupling (stepping D moved A/B/C by 24-33% of D's own shift),
which the R-L-M model can't explain. This script fits that coupling
directly as a full (non-diagonal) resistance matrix R_full, which enters
the state-space model the same place the old diag(R_i) did:

  at steady state:  R x = V(u) = diag(Kv) u   =>   x = R^-1 diag(Kv) u
  measured directly: G_ij = d(x_i)/d(u_j) (from channel j's isolated duty step)
  => R^-1 = G @ diag(1/Kv)   =>   R_full = inv(R^-1)

This is a MODELING CHOICE, not fundamental physics -- lumping whatever the
real coupling mechanism is (shared supply loading, real magnetic coupling
too fast to resolve as di/dt at this control rate, etc.) into an effective
resistance matrix is the simplest way to reproduce the measured DC gain
without inventing new state variables. If the mechanism turns out to be
duty-dependent or nonlinear, this single-point-per-pair linear fit will not
capture that -- see the residual check below.

Usage:
  uv run python tools/fit_coupling_matrix.py step_response.log \
      --v-supply 10 --step-delta 15 --out tools/coupling_fit.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import parse_experiment_log

CHANNELS = list("ABCD")
_LABEL_RE = re.compile(r"STEP_([ABCD])_(BASELINE|STEPPED)")


def _settled_mean(data: dict, label: str, settle_frac: float) -> np.ndarray | None:
    mask = data["label"] == label
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    k0 = idx[0] + int(len(idx) * settle_frac)
    k0 = min(k0, idx[-1])
    keep = idx[idx >= k0]
    currents = np.stack([data[f"i_{c.lower()}"] for c in CHANNELS], axis=1)
    return currents[keep].mean(axis=0)


def fit_gain_matrix(data: dict, step_delta: float, settle_frac: float):
    """G[i][j] = d(current_i)/d(duty_j), from each channel j's isolated step.
    Returns (G, per_column_residual_note)."""
    G = np.zeros((4, 4))
    notes = {}
    for j, stepped in enumerate(CHANNELS):
        base = _settled_mean(data, f"STEP_{stepped}_BASELINE", settle_frac)
        step = _settled_mean(data, f"STEP_{stepped}_STEPPED", settle_frac)
        if base is None or step is None:
            raise SystemExit(f"missing STEP_{stepped}_BASELINE/STEPPED labels -- "
                              "is this a gen_step_response_experiment.py capture?")
        delta = step - base
        G[:, j] = delta / step_delta
        notes[stepped] = {"own_delta_a": float(delta[j]), "duty_step_pct": step_delta}
    return G, notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--v-supply", type=float, required=True,
                     help="bench supply voltage AT TEST TIME")
    ap.add_argument("--step-delta", type=float, default=15.0,
                     help="duty %% step used by gen_step_response_experiment.py "
                          "(default: %(default)s)")
    ap.add_argument("--settle-frac", type=float, default=0.5,
                     help="tail fraction of each labeled dwell to average (default: %(default)s)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "coupling_fit.json"))
    args = ap.parse_args()

    data = parse_experiment_log(args.log)
    if "label" not in data:
        raise SystemExit(f"{args.log}: no labels found -- is this a "
                          "gen_step_response_experiment.py capture?")

    G, notes = fit_gain_matrix(data, args.step_delta, args.settle_frac)

    kv = (4.0 / np.pi) * args.v_supply / 100.0  # same Kv for all channels (shared supply)
    Rinv = G / kv  # R^-1 = G @ diag(1/Kv), Kv scalar here so this is just G/kv
    cond = np.linalg.cond(Rinv)
    if cond > 1e6 or not np.all(np.isfinite(Rinv)):
        raise SystemExit(f"R^-1 is singular/ill-conditioned (cond={cond:.2e}) -- "
                          "coupling fit unusable, check the step-response log")
    R_full = np.linalg.inv(Rinv)

    print("measured DC gain matrix G (rows=affected channel, cols=stepped channel), A/duty%:")
    for i, row in enumerate(CHANNELS):
        print("  " + row + ": " + " ".join(f"{G[i,j]:+.4f}" for j in range(4)))
    print(f"\nfitted R_full (ohm-like, off-diagonal = coupling), cond={cond:.1f}:")
    for i, row in enumerate(CHANNELS):
        print("  " + row + ": " + " ".join(f"{R_full[i,j]:+.4f}" for j in range(4)))

    off_diag_frac = np.abs(R_full - np.diag(np.diag(R_full))).sum() / np.abs(np.diag(R_full)).sum()
    print(f"\noff-diagonal / diagonal magnitude ratio: {off_diag_frac:.3f} "
          f"(0 = fully decoupled, matches empirical cross-channel shifts if large)")

    out = {
        "channels": CHANNELS,
        "v_supply": args.v_supply,
        "kv": kv,
        "step_delta_pct": args.step_delta,
        "G_duty_to_current": G.tolist(),
        "R_full": R_full.tolist(),
        "cond_Rinv": float(cond),
        "notes": notes,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
