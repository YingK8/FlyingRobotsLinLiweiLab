#!/usr/bin/env python3
"""IMC-PID gain computation from the fitted RLC model (tools/rlc_fit.json),
replacing hand-picked KP/KI/KD seeds with a model-derived starting point.

Plant model: the loop-relevant signal is ENVELOPE current vs. duty, not
instantaneous coil current -- the CS reading passes through the 50ms EMA
filter (CurrentSense.h) before the PID ever sees it, comparable to or
slower than the coil's own electrical time constant. Modeled as a 2nd-order
lag and tuned via standard IMC-PID (Rivera/Morari):

  K(omega)   = (4/pi)*V_supply / (100*|Z_i(omega)|)      [A per duty-%]
  tau_filter = 0.050s (TAU_FILTER_MS in CurrentSense.h)
  tau_elec   = 2*L/R                                      [envelope decay time]
  Kc = (tau_filter+tau_elec) / (K*lambda)
  Ti = tau_filter+tau_elec
  Td = tau_filter*tau_elec / (tau_filter+tau_elec)

Converting to this codebase's actual KP/KI/KD is NOT a 1:1 continuous-time
mapping: main_current_pid.cpp's integrator/derivative are rate-scaled by
dt/NOMINAL_TICK_MS, not raw continuous time. With TICK_HZ =
1000/NOMINAL_TICK_MS:

  KP = Kc
  KI = Kc / (Ti * TICK_HZ)
  KD = Kc * Td * TICK_HZ

Computes gains at every swept frequency (visibility + groundwork for future
gain-scheduling) and recommends a single seed triple at the WORST-CASE
(highest-gain, i.e. at/near resonance) frequency, since main_current_pid.cpp
currently locks gains for the whole run.

Usage:
  uv run python tools/compute_model_gains.py tools/rlc_fit.json \
      --v-supply 11.9 --channel A --out tools/model_gains.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def imc_pid(K: float, tau1: float, tau2: float, lam: float):
    Kc = (tau1 + tau2) / (K * lam)
    Ti = tau1 + tau2
    Td = (tau1 * tau2) / (tau1 + tau2)
    return Kc, Ti, Td


def continuous_to_firmware_gains(Kc: float, Ti: float, Td: float, nominal_tick_ms: float):
    tick_hz = 1000.0 / nominal_tick_ms
    return Kc, Kc / (Ti * tick_hz), Kc * Td * tick_hz  # KP, KI, KD


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rlc_fit", help="tools/rlc_fit.json from tools/fit_rlc_model.py")
    ap.add_argument("--v-supply", type=float, required=True,
                     help="bench supply voltage to assume for the gain computation "
                          "(should match the intended operating condition)")
    ap.add_argument("--channel", choices=list("ABCD"), required=True)
    ap.add_argument("--lambda-frac", type=float, default=0.5,
                     help="IMC robustness knob, as a fraction of tau_filter -- "
                          "smaller = more aggressive (default: %(default)s)")
    ap.add_argument("--tau-filter-ms", type=float, default=50.0,
                     help="MUST match CurrentSense.h's TAU_FILTER_MS default (default: %(default)s)")
    ap.add_argument("--nominal-tick-ms", type=float, default=2.0,
                     help="MUST match main_current_pid.cpp's NOMINAL_TICK_MS (default: %(default)s)")
    ap.add_argument("--freqs", type=float, nargs="+",
                     default=list(range(1, 211, 5)),
                     help="frequency grid to evaluate gains across (default: 1-210Hz step 5)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "model_gains.json"))
    args = ap.parse_args()

    with open(args.rlc_fit) as f:
        rlc = json.load(f)[args.channel]
    R, L, C = rlc["R"], rlc["L"], rlc["C"]
    tau_filter = args.tau_filter_ms / 1000.0
    lam = args.lambda_frac * tau_filter

    table = []
    for f_hz in args.freqs:
        omega = 2 * np.pi * f_hz
        z = np.sqrt(R ** 2 + (omega * L - 1.0 / (omega * C)) ** 2)
        K = (4.0 / np.pi) * args.v_supply / (100.0 * z)  # A per duty%
        tau_elec = 2 * L / R
        Kc, Ti, Td = imc_pid(K, tau_filter, tau_elec, lam)
        KP, KI, KD = continuous_to_firmware_gains(Kc, Ti, Td, args.nominal_tick_ms)
        table.append({"freq_hz": f_hz, "Z_ohm": float(z), "K_a_per_pct": float(K),
                       "KP": float(KP), "KI": float(KI), "KD": float(KD)})

    worst = max(table, key=lambda r: r["K_a_per_pct"])  # highest gain = at/near resonance
    print(f"recommended seed (worst-case @ {worst['freq_hz']:.0f} Hz, "
          f"near f0={rlc['f0_hz']:.1f} Hz): "
          f"KP={worst['KP']:.3f} KI={worst['KI']:.4f} KD={worst['KD']:.4f}")

    out = {"channel": args.channel, "table": table, "recommended": worst}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
