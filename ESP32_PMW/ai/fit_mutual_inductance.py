#!/usr/bin/env python3
"""Fit one M_ij entry (henries) from a PicoScope capture of the induced EMF
on an undriven neighbor coil while another channel is driven solo at a known
frequency (mutual-inductance tier of the RLC/coupling characterization plan).

Why this needs a probe move, not the onboard CS pin: the VNH5019 CS pin
mirrors current through the bridge's OWN conducting output stage. An
undriven neighbor (carrier=0%, Hi-Z) has no closed current path through its
own bridge, so its CS reading is ~0 regardless of the real induced EMF at
its coil terminals -- a hardware topology fact, not fixable in firmware.
This tier requires physically relocating a PicoScope probe to read directly
across the UNDRIVEN coil's terminals (see docs/PCB_Design_Documentation.md
sec.6 for the J10/J5/etc. connector map), not the CS shunt.

Procedure: drive channel `--driven` solo at `--freq-hz`/`--duty` (reuse
ai/gen_solo_sweep_experiment.py with a single frequency and a long dwell,
e.g. --freqs 150 --dwell-ms 8000); probe `--probed`'s coil terminals with the
PicoScope and capture with ai/picoscope_capture.py; then run this script.
The driven channel's true coil current is computed from ai/rlc_fit.json
(no re-measurement needed): I_driven = V_fund / |Z_driven(omega)|. The
probed voltage's fundamental gives V_induced via FFT (reusing
ai/picoscope_capture.py's compute_fft/measure_fundamental). Then:

  M_ij = |V_induced,j| / (omega * I_driven,i)

Usage:
  uv run python ai/fit_mutual_inductance.py picoscope_capture.csv \
      --driven A --probed B --v-channel "Channel B" --freq-hz 150 --duty 30 \
      --v-supply 11.9 --rlc-fit ai/rlc_fit.json --out ai/m_matrix.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from picoscope_capture import compute_fft, measure_fundamental


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="PicoScope capture CSV (Time,Channel A..D header, "
                                 "units row -- same convention as coupling_matrix.py)")
    ap.add_argument("--driven", choices=list("ABCD"), required=True)
    ap.add_argument("--probed", choices=list("ABCD"), required=True)
    ap.add_argument("--v-channel", required=True,
                     help="PicoScope CSV column probed across the undriven "
                          "coil's terminals, e.g. 'Channel B'")
    ap.add_argument("--freq-hz", type=float, required=True,
                     help="commutation frequency the driven channel was held at")
    ap.add_argument("--duty", type=float, required=True,
                     help="carrier duty %% the driven channel was held at")
    ap.add_argument("--v-supply", type=float, required=True,
                     help="bench supply voltage at test time")
    ap.add_argument("--rlc-fit", required=True,
                     help="ai/rlc_fit.json from ai/fit_rlc_model.py -- "
                          "supplies the driven channel's R,L,C to compute its "
                          "true coil current without re-measuring it")
    ap.add_argument("--settle-s", type=float, default=1.0,
                     help="skip this much of the capture as transient (default: %(default)s)")
    ap.add_argument("--band-frac", type=float, default=0.1,
                     help="fundamental search band as +/- this fraction of "
                          "--freq-hz (default: %(default)s)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "m_matrix.json"))
    args = ap.parse_args()

    if args.driven == args.probed:
        raise SystemExit("--driven and --probed must differ")

    with open(args.rlc_fit) as f:
        rlc = json.load(f)[args.driven]
    R, L, C = rlc["R"], rlc["L"], rlc["C"]
    omega = 2 * np.pi * args.freq_hz
    z_driven = np.sqrt(R ** 2 + (omega * L - 1.0 / (omega * C)) ** 2)
    v_fund = (4.0 / np.pi) * (args.duty / 100.0) * args.v_supply
    i_driven = v_fund / z_driven

    df = pd.read_csv(args.csv, skiprows=[1])
    t = df["Time"].to_numpy(dtype=float)
    v = df[args.v_channel].to_numpy(dtype=float)
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    keep = t >= (t[0] + args.settle_s)
    freqs, mag, _ = compute_fft(v[keep], fs)
    band = (args.freq_hz * (1 - args.band_frac), args.freq_hz * (1 + args.band_frac))
    f0_found, v_induced = measure_fundamental(freqs, mag, band=band)

    m_ij = v_induced / (omega * i_driven)
    print(f"I_driven({args.driven})={i_driven:.3f}A  "
          f"V_induced({args.probed})={v_induced * 1000:.2f}mV @ {f0_found:.1f}Hz  "
          f"-> M_{args.driven}{args.probed} = {m_ij * 1e3:.4f} mH")

    m_data = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            m_data = json.load(f)
    key = "".join(sorted([args.driven, args.probed]))
    m_data[key] = m_ij
    with open(args.out, "w") as f:
        json.dump(m_data, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
