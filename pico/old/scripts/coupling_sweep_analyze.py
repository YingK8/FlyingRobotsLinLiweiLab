"""Analyze ONE long PicoScope recording of the automated coupling sweep
(main_coupling_test firmware: A -> B -> C -> D -> ALL, with off-gaps).

Auto-detects each driven burst by which channels are active, so the recording
start time need not be aligned to the firmware. Classifies each burst as a SOLO
(one channel) or ALL (>=3 channels) segment, then reports each channel's current
SOLO vs COMBINED:

  * solo ~= combined  -> channels independent (imbalance is per-channel drive)
  * combined != solo   -> coupling / shared-supply interaction when run together

Usage:  python3 coupling_sweep_analyze.py sweep.csv
"""
import os
import sys
import numpy as np
import pandas as pd

CH = ["Channel A", "Channel B", "Channel C", "Channel D"]
G = {"Channel A": 15.26, "Channel B": 15.28, "Channel C": 15.57, "Channel D": 15.34}
OFFSET = {"Channel A": 0.0, "Channel B": -0.0404,
          "Channel C": 0.0, "Channel D": -0.0408}
W = 41
ON_A = 2.0          # a channel counts as "driven" above this current (A)
MIN_BURST_S = 1.0   # ignore bursts shorter than this (partial at file ends)


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def find_bursts(active, t):
    """Yield (i0, i1) index spans where `active` (bool array) is True."""
    spans, i, n = [], 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            if t[j - 1] - t[i] >= MIN_BURST_S:
                spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit("need 1 argument: the sweep CSV")
    csv = sys.argv[1]
    df = pd.read_csv(csv, skiprows=[1])
    t = df["Time"].values
    cur = {ch: np.clip(env(G[ch] * (df[ch].values - OFFSET[ch])), 0, None) for ch in CH}
    total = np.sum([cur[ch] for ch in CH], axis=0)

    # "on" = total current well above the 4-channel idle envelope floor.
    bursts = find_bursts(total > ON_A * 1.5, t)
    if not bursts:
        raise SystemExit("no driven bursts found — check capture / thresholds.")

    solo = {ch: [] for ch in CH}
    comb = {ch: [] for ch in CH}
    print(f"detected {len(bursts)} bursts:")
    for (i0, i1) in bursts:
        lo = i0 + int(0.25 * (i1 - i0))
        hi = i0 + int(0.75 * (i1 - i0))           # steady middle of the burst
        means = {ch: float(np.mean(cur[ch][lo:hi])) for ch in CH}
        act = [ch for ch in CH if means[ch] > ON_A]
        tag = "".join(c[-1] for c in act)
        print(f"  {t[i0]:6.1f}-{t[i1-1]:5.1f}s  active=[{tag:4}]  " +
              "  ".join(f"{c[-1]}={means[c]:.2f}" for c in CH))
        if len(act) == 1:
            solo[act[0]].append(means[act[0]])
        elif len(act) >= 3:
            for ch in CH:
                comb[ch].append(means[ch])

    def avg(d, ch):
        return float(np.mean(d[ch])) if d[ch] else float("nan")

    print(f"\n{'ch':3}{'solo A':>9}{'combined A':>12}{'delta %':>10}")
    s, c = {}, {}
    for ch in CH:
        s[ch], c[ch] = avg(solo, ch), avg(comb, ch)
        d = 100.0 * (c[ch] - s[ch]) / s[ch] if s[ch] else float("nan")
        print(f"{ch[-1]:3}{s[ch]:9.2f}{c[ch]:12.2f}{d:+10.1f}")

    sv = [s[ch] for ch in CH if not np.isnan(s[ch])]
    cv = [c[ch] for ch in CH if not np.isnan(c[ch])]
    if sv and cv:
        print(f"\nspread (max/min):  solo {max(sv)/min(sv):.2f}x   "
              f"combined {max(cv)/min(cv):.2f}x")
    print("\ninterpretation:")
    print("  solo matched + combined diverges -> COUPLING / shared-supply interaction")
    print("  solo already diverges            -> per-channel DRIVE (wiring/bridge/supply)")


if __name__ == "__main__":
    main()
