#!/usr/bin/env python3
"""Cross-check the fitted state-space plant model (tools/state_space_model.json)
against an empirical duty-step response log (tools/gen_step_response_experiment.py
+ a main_experiment.cpp capture) -- the gate before trusting an LQR K on
hardware (state-space plan, Phase E).

The model (see tools/build_state_space_model.py) predicts NO steady-state
cross-channel coupling: L*(dx/dt)+R*x=V(u) decouples completely at dx/dt=0,
so stepping one channel's duty should, in the model, change ONLY that
channel's settled current. This script measures, for each stepped channel,
the settled current shift on every OTHER channel and reports it as a
fraction of the stepped channel's own shift -- if that fraction is not
small (the historical coupling-matrix data suggests it may not be, see
project memory: k~0.24 solo-vs-ALL redistribution), the model is missing a
steady-state coupling term and K should not be trusted until that's
resolved.

Usage:
  uv run python tools/validate_state_space_model.py step_response.log \
      --model tools/state_space_model.json --settle-frac 0.5
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

_LABEL_RE = re.compile(r"STEP_([ABCD])_(BASELINE|STEPPED|RECOVER)")
CHANNELS = list("ABCD")


def _settled_mean(data: dict, label: str, settle_frac: float) -> np.ndarray | None:
    """Mean [i_a,i_b,i_c,i_d] over the last settle_frac of the labeled dwell's
    samples, or None if the label never appears."""
    mask = data["label"] == label
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return None
    k0 = idx[0] + int(len(idx) * settle_frac)
    k0 = min(k0, idx[-1])
    keep = idx[idx >= k0]
    currents = np.stack([data[f"i_{c.lower()}"] for c in CHANNELS], axis=1)
    return currents[keep].mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="serial log from tools/trigger_reset_log.py capturing "
                                 "a tools/gen_step_response_experiment.py run")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "state_space_model.json"),
                     help="tools/state_space_model.json -- used only to report the model's "
                          "(zero) predicted cross-coupling for comparison, not required")
    ap.add_argument("--settle-frac", type=float, default=0.5,
                     help="keep this fraction (tail) of each labeled dwell as settled (default: %(default)s)")
    ap.add_argument("--cross-coupling-warn-frac", type=float, default=0.05,
                     help="flag a channel pair if |other channel's shift| exceeds this "
                          "fraction of the stepped channel's own shift (default: %(default)s)")
    args = ap.parse_args()

    data = parse_experiment_log(args.log)
    if "label" not in data:
        raise SystemExit(f"{args.log}: no labels found -- is this a "
                          "gen_step_response_experiment.py capture?")

    model_note = "model predicts exactly 0 for all cross-channel entries (steady-state decoupled)"
    if os.path.exists(args.model):
        with open(args.model) as f:
            json.load(f)  # just confirm it's readable; nothing else needed for this check

    print(f"({model_note})\n")
    any_flag = False
    for stepped in CHANNELS:
        base = _settled_mean(data, f"STEP_{stepped}_BASELINE", args.settle_frac)
        step = _settled_mean(data, f"STEP_{stepped}_STEPPED", args.settle_frac)
        if base is None or step is None:
            print(f"{stepped}: no BASELINE/STEPPED samples found, skipping")
            continue
        delta = step - base
        own_idx = CHANNELS.index(stepped)
        own_delta = delta[own_idx]
        print(f"step on {stepped}: own current shift = {own_delta:+.3f}A")
        for i, ch in enumerate(CHANNELS):
            if ch == stepped:
                continue
            frac = abs(delta[i] / own_delta) if abs(own_delta) > 1e-6 else float("nan")
            flag = frac == frac and frac > args.cross_coupling_warn_frac
            any_flag = any_flag or flag
            print(f"    {ch}: shift = {delta[i]:+.3f}A  ({frac * 100:5.1f}% of {stepped}'s own shift)"
                  f"{'  <-- MODEL MISMATCH' if flag else ''}")
        print()

    if any_flag:
        print("RESULT: measurable steady-state cross-channel coupling found -- the "
              "state-space model (which predicts 0 here) is missing a steady-state "
              "coupling term. Do NOT trust an LQR K designed from it until this is "
              "resolved -- see the state-space plan's caveat on resistive/supply "
              "coupling vs. pure mutual inductance.")
    else:
        print("RESULT: no significant steady-state cross-channel coupling found -- "
              "consistent with the model. OK to proceed to LQR design/hardware test.")


if __name__ == "__main__":
    main()
