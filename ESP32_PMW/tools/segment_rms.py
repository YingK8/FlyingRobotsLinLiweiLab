#!/usr/bin/env python3
"""Per-channel RMS over named time windows of a picoscope_record.py CSV.

Usage:
  uv run python tools/segment_rms.py CSV --win soloA:13.5:15.0 --win base:22.7:23.3

Prints one row per window: label, then RMS (mV) per scope channel. RMS is
computed after removing each channel's floor (median of the whole capture's
quietest 10%), so gaps read ~0 and active segments read the burst magnitude.
"""
import argparse

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--win", action="append", required=True,
                   help="label:t0:t1 (seconds)")
    args = p.parse_args()

    df = pd.read_csv(args.csv, skiprows=[1])
    t = df.iloc[:, 0].to_numpy()
    chans = df.columns[1:]

    print(f"{'window':>12}  " + "  ".join(f"{c[-1]:>8}" for c in chans) + "   (RMS mV)")
    for spec in args.win:
        label, t0, t1 = spec.split(":")
        m = (t >= float(t0)) & (t <= float(t1))
        if not m.any():
            print(f"{label:>12}  (no samples)")
            continue
        vals = []
        for c in chans:
            v = df[c].to_numpy()[m] * 1000.0  # mV
            vals.append(np.sqrt(np.mean(v ** 2)))
        print(f"{label:>12}  " + "  ".join(f"{v:8.1f}" for v in vals))


if __name__ == "__main__":
    main()
