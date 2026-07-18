#!/usr/bin/env python3
"""Falsifiable check: does the fitted series-RLC model actually predict WHERE
the real min/max-PID controller's resonance-band spread peaks?

Compares ai/rlc_fit.json's per-channel f0_i = 1/(2*pi*sqrt(L*C)) (and, if
ai/m_matrix.json is given, the coupling-severity peak
kappa_ij(omega) = omega*M_ij / |Z_i(omega)|, expected to also peak near f0_i
-- see the plan's dynamic-model section) against
ai/pid_metrics.py's resonance_peak_freq_hz from a REAL
main_current_pid.cpp HOLD/RAMP_UP log -- the frequency at which that run's
spread actually peaked during the 90-190Hz band. This is the concrete test
of whether the series-LC resonance hypothesis explains the historically
hard-to-control band, rather than just asserting it.

Usage:
  uv run python ai/validate_resonance_model.py current_pid_run.log \
      --rlc-fit ai/rlc_fit.json [--m-matrix ai/m_matrix.json] --tol-hz 15
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import compute_metrics, parse_log


def z_of(rlc: dict, omega: np.ndarray) -> np.ndarray:
    R, L, C = rlc["R"], rlc["L"], rlc["C"]
    return np.sqrt(R ** 2 + (omega * L - 1.0 / (omega * C)) ** 2)


def kappa_peak_freq(rlc_all: dict, m_matrix: dict, channel: str,
                     freqs_hz: np.ndarray) -> float:
    """Peak location of kappa_ij(omega) = omega*M_ij/|Z_i(omega)| over j != i,
    maximized across this channel's coupled neighbors -- the frequency where
    this channel injects (or receives, by the same |Z| term) the worst
    disturbance. See the plan's dynamic-model section 1.3."""
    omega = 2 * np.pi * freqs_hz
    z = z_of(rlc_all[channel], omega)
    best = np.zeros_like(omega)
    for pair, m_ij in m_matrix.items():
        if channel not in pair:
            continue
        kappa = omega * m_ij / z
        best = np.maximum(best, kappa)
    return float(freqs_hz[np.argmax(best)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="main_current_pid.cpp serial log (a real HOLD/RAMP_UP run)")
    ap.add_argument("--rlc-fit", required=True, help="ai/rlc_fit.json")
    ap.add_argument("--m-matrix", default=None,
                     help="ai/m_matrix.json -- if given, also checks the full "
                          "coupling-severity peak, not just raw f0_i")
    ap.add_argument("--tol-hz", type=float, default=15.0,
                     help="pass if |predicted - measured| <= this (default: %(default)s)")
    ap.add_argument("--freqs", type=float, nargs="+",
                     default=list(range(1, 211, 1)),
                     help="frequency grid for the kappa-peak search (default: 1-210Hz step 1)")
    args = ap.parse_args()

    data = parse_log(args.log)
    metrics = compute_metrics(data)
    measured = metrics["resonance_peak_freq_hz"]
    if measured is None:
        raise SystemExit(f"{args.log}: no RAMP_UP samples in the 90-190Hz resonance "
                          "band found -- can't validate against this log")

    with open(args.rlc_fit) as f:
        rlc_all = json.load(f)

    m_matrix = None
    if args.m_matrix:
        with open(args.m_matrix) as f:
            m_matrix = json.load(f)

    freqs_hz = np.asarray(args.freqs, dtype=float)
    print(f"measured resonance_peak_freq_hz = {measured:.1f} Hz "
          f"(resonance_peak spread = {metrics['resonance_peak']:.3f} A)\n")

    any_pass = False
    for ch, rlc in rlc_all.items():
        f0 = rlc["f0_hz"]
        diff_f0 = abs(f0 - measured)
        pass_f0 = diff_f0 <= args.tol_hz
        any_pass = any_pass or pass_f0
        line = f"{ch}: f0={f0:.1f}Hz  |predicted-measured|={diff_f0:.1f}Hz  {'PASS' if pass_f0 else 'fail'}"

        if m_matrix is not None:
            f_kappa = kappa_peak_freq(rlc_all, m_matrix, ch, freqs_hz)
            diff_kappa = abs(f_kappa - measured)
            pass_kappa = diff_kappa <= args.tol_hz
            any_pass = any_pass or pass_kappa
            line += (f"  |  kappa_peak={f_kappa:.1f}Hz  "
                     f"|predicted-measured|={diff_kappa:.1f}Hz  {'PASS' if pass_kappa else 'fail'}")
        else:
            line += "  (no --m-matrix given -- only raw f0_i checked, not full coupling-severity)"
        print(line)

    print(f"\noverall: {'PASS' if any_pass else 'FAIL'} (tol={args.tol_hz:.0f}Hz) -- "
          f"{'at least one channel' if any_pass else 'no channel'} predicts the "
          "measured resonance-band peak location")


if __name__ == "__main__":
    main()
