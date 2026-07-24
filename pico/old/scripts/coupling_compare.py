"""Compare per-channel coil current SOLO vs ALL-TOGETHER to separate a coupling /
shared-supply interaction from a per-channel drive imbalance.

Capture (with picoscope_record.py + main_coupling_test firmware) five runs at
constant 190 Hz / 100% duty:
    solo A, solo B, solo C, solo D, and all four together.

Usage:
    python3 coupling_compare.py soloA.csv soloB.csv soloC.csv soloD.csv all.csv

For each channel it reports the steady current when driven ALONE vs when all
channels run together:
  * solo ~= combined  -> channels independent (imbalance is per-channel drive)
  * combined != solo   -> coupling / common-impedance interaction when run together
"""
import os
import sys
import numpy as np
import pandas as pd

CH = ["Channel A", "Channel B", "Channel C", "Channel D"]
G = {"Channel A": 15.26, "Channel B": 15.28, "Channel C": 15.57, "Channel D": 15.34}
# Zero-current CS offset per channel (V), from the multipoint calibration.
# These captures are constant-drive (no drive-off window), so use the known
# offset rather than estimating a baseline from the start of the file.
OFFSET = {"Channel A": 0.0, "Channel B": -0.0404,
          "Channel C": 0.0, "Channel D": -0.0408}
W = 41


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def steady_current(csv, ch):
    """Calibrated current of `ch` in `csv`, averaged over the steady middle 60%."""
    df = pd.read_csv(csv, skiprows=[1])
    t = df["Time"].values
    cur = np.clip(env(G[ch] * (df[ch].values - OFFSET[ch])), 0, None)
    lo, hi = t.min() + 0.2 * (t.max() - t.min()), t.min() + 0.8 * (t.max() - t.min())
    m = (t >= lo) & (t <= hi)
    return float(np.mean(cur[m]))


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        raise SystemExit("need 5 CSVs: soloA soloB soloC soloD all")
    solo_files = dict(zip(CH, sys.argv[1:5]))
    all_file = sys.argv[5]

    solo = {ch: steady_current(solo_files[ch], ch) for ch in CH}
    comb = {ch: steady_current(all_file, ch) for ch in CH}

    print(f"\n{'ch':3}{'solo A':>9}{'combined A':>12}{'delta %':>10}")
    for ch in CH:
        d = 100.0 * (comb[ch] - solo[ch]) / solo[ch] if solo[ch] else float('nan')
        print(f"{ch[-1]:3}{solo[ch]:9.2f}{comb[ch]:12.2f}{d:+10.1f}")

    solo_spread = max(solo.values()) / min(solo.values()) if min(solo.values()) else float('inf')
    comb_spread = max(comb.values()) / min(comb.values()) if min(comb.values()) else float('inf')
    print(f"\nspread (max/min):  solo {solo_spread:.2f}x   combined {comb_spread:.2f}x")
    print("\ninterpretation:")
    print("  solo matched + combined diverges  -> COUPLING / shared-supply interaction")
    print("  solo already diverges             -> per-channel DRIVE (wiring/bridge/supply)")


if __name__ == "__main__":
    main()
