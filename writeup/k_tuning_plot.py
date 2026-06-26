#!/usr/bin/env python3
"""VNH5019 current-sense (CS) calibration plot.

Reads k-tuning.csv and, for each channel, plots the measured CS voltage
(pico_voltage, mV) against the load/supply current (A) and fits a line.

  V_CS = K * I_load + V_offset

The slope K (mV/A) is the current-sense gain; its inverse (A/V) is what the
firmware uses to convert an ADC reading back into a current.

With only two points per channel the "fit" is just the line through them.
Once more points exist per channel it automatically switches to a
least-squares fit and reports R^2.
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "ESP32_PMW", "src", "k-tuning.csv")
OUT = os.path.join(HERE, "k_tuning.png")


def load(path):
    """Return {channel: (currents[A], cs_voltages[mV])} from the CSV."""
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            ch = (row.get("channel") or "").strip()
            try:
                v_cs = float(row["pico_voltage"])
                i_load = float(row["supply current"])
            except (TypeError, ValueError):
                continue  # blank / placeholder row
            data.setdefault(ch, []).append((i_load, v_cs))
    return {ch: (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
            for ch, pts in sorted(data.items())}


def main():
    data = load(CSV)
    if not data:
        raise SystemExit("No usable data points found in " + CSV)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"A": "tab:blue", "B": "tab:orange",
              "C": "tab:green", "D": "tab:red"}

    for ch, (i_load, v_cs) in data.items():
        if len(i_load) < 2:
            continue
        color = colors.get(ch, None)
        slope, intercept = np.polyfit(i_load, v_cs, 1)

        # R^2 (only meaningful with >2 points; trivially 1.0 for two)
        resid = v_cs - (slope * i_load + intercept)
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((v_cs - v_cs.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # fit line, extended down toward 0 A to show the offset
        x_lo = min(0.0, i_load.min())
        xs = np.linspace(x_lo, i_load.max() * 1.05, 100)
        ax.scatter(i_load, v_cs, color=color, zorder=3,
                   label=f"Ch {ch} data")
        ax.plot(xs, slope * xs + intercept, color=color, lw=1.5, alpha=0.8,
                label=(f"Ch {ch} fit: K={slope:.1f} mV/A, "
                       f"off={intercept:+.0f} mV"
                       + (f", R²={r2:.3f}" if len(i_load) > 2 else "")))

    ax.set_xlabel("Load current  I$_{OUT}$  (A)")
    ax.set_ylabel("Current-sense voltage  V$_{CS}$  (mV)")
    ax.set_title("VNH5019 Current-Sense Calibration")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print("wrote", OUT)

    # also print the numbers for the writeup
    for ch, (i_load, v_cs) in data.items():
        if len(i_load) < 2:
            continue
        slope, intercept = np.polyfit(i_load, v_cs, 1)
        print(f"Ch {ch}: K = {slope:.2f} mV/A  ({1000.0/slope:.2f} A/V), "
              f"offset = {intercept:+.1f} mV, n = {len(i_load)}")


if __name__ == "__main__":
    main()
