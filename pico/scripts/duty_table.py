"""Tabulate carrier duty cycle vs per-channel raw CS voltage and current.

For each channel we take its solo carrier ramp (0->100% over 8 s after a 2 s gap),
map capture time to duty %, and report the rolling-envelope CS voltage (as measured)
and the calibrated current I = g * (V_CS - baseline). Each channel is read from the
data column that actually ramps in its window, which absorbs the B/D probe swap.

Writes pico/data/dutyramp/<stem>_duty_table.csv.

Usage:  python3 scripts/duty_table.py <capture.csv>
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.dirname(HERE)

# Per data column: g [A/V] and the solo-ramp segment start time [s].
G = {"A": 15.10, "B": 15.10, "C": 15.09, "D": 15.43}
SEG_START = {"A": 0.0, "C": 10.0, "D": 30.0, "B": 40.0}
GAP_S, RAMP_S, W = 2.0, 8.0, 41
DUTY = np.arange(0, 101, 5)          # duty grid, %


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("need 1 argument: the duty-ramp capture CSV")
    csv = sys.argv[1]
    df = pd.read_csv(csv, skiprows=[1])
    t = df["Time"].values

    out = {"Duty (%)": DUTY}
    for ch in ["A", "B", "C", "D"]:
        col = f"Channel {ch}"
        cs = env(df[col].values)                      # CS voltage envelope (V)
        base = float(np.percentile(cs, 5))            # idle floor of the envelope
        start = SEG_START[ch] + GAP_S
        m = (t >= start) & (t <= start + RAMP_S)
        duty = 100.0 * (t[m] - start) / RAMP_S
        cs_seg = cs[m]
        # Median CS within +/-2.5% of each duty grid point.
        cs_grid, i_grid = [], []
        for d in DUTY:
            sel = np.abs(duty - d) <= 2.5
            vcs = float(np.median(cs_seg[sel])) if sel.any() else np.nan
            cs_grid.append(vcs)
            i_grid.append(G[ch] * (vcs - base))
        out[f"{ch} CS (V)"] = np.round(cs_grid, 4)
        out[f"{ch} I (A)"] = np.round(i_grid, 2)

    table = pd.DataFrame(out)
    stem = os.path.splitext(os.path.basename(csv))[0]
    dst = os.path.join(PICO, "data", "dutyramp", f"{stem}_duty_table.csv")
    table.to_csv(dst, index=False)
    print("saved", dst, "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
