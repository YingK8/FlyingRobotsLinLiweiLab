#!/usr/bin/env python3
"""Interpretable current-sense view: rolling RMS running average.

The coil current is AC (H-bridge drive), so the raw trace swings +/- and its
mean is ~0. The RMS *is* the physical current magnitude. We compute a running
RMS -- sqrt(moving-average(x^2)) -- which smooths the AC ripple into a clean
per-channel current envelope over the whole run.

Usage: uv run python ai/plot_rms.py [CSV] [--win-ms 500]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="tilt_test.csv")
ap.add_argument("--win-ms", type=float, default=500.0, help="RMS window (ms)")
args = ap.parse_args()

df = pd.read_csv(args.csv, skiprows=[1])
t = df["Time"].values
dt = np.median(np.diff(t)); fs = 1.0 / dt
chs = ["Channel A", "Channel B", "Channel C", "Channel D"]
colors = {"Channel A": "#e15759", "Channel B": "#4e79a7",
          "Channel C": "#59a14f", "Channel D": "#f28e2b"}
win = max(1, int(round(args.win_ms * 1e-3 * fs)))

def rolling_rms(x, w):
    # remove DC (AC signal), then sqrt of moving-average of the square
    x = x - np.mean(x)
    ms = pd.Series(x * x).rolling(w, min_periods=1, center=True).mean().values
    return np.sqrt(ms)

rms = {c: rolling_rms(df[c].values, win) * 1000.0 for c in chs}  # -> mV

print(f"{args.csv}: {len(df)} samp, {t[-1]:.1f}s, {fs:.0f} S/s, "
      f"RMS window {win} samp ({args.win_ms:.0f} ms)")
print(f"{'ch':<10}{'overall RMS(mV)':>16}{'peak RMS(mV)':>14}")
for c in chs:
    print(f"{c:<10}{np.sqrt(np.mean((df[c].values-df[c].values.mean())**2))*1000:>16.3f}"
          f"{np.nanmax(rms[c]):>14.3f}")

fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

# Per-channel rolling RMS
ax = axs[0]
for c in chs:
    ax.plot(t, rms[c], lw=1.1, color=colors[c], label=c)
ax.set_title(f"Rolling-RMS current-sense envelope — {args.csv}  "
             f"({args.win_ms:.0f} ms window)")
ax.set_ylabel("RMS current sense (mV)")
ax.grid(alpha=0.3); ax.legend(ncol=4, fontsize=9); ax.set_xlim(0, t[-1])

# Board-pair sums (balanced target = A+B on one supply, C+D on the other)
ax = axs[1]
ax.plot(t, rms["Channel A"] + rms["Channel B"], lw=1.3, color="#8c564b",
        label="A + B")
ax.plot(t, rms["Channel C"] + rms["Channel D"], lw=1.3, color="#17becf",
        label="C + D")
ax.set_title("Board-pair RMS sum — closer curves = better balanced supplies")
ax.set_xlabel("time (s)"); ax.set_ylabel("summed RMS (mV)")
ax.grid(alpha=0.3); ax.legend(fontsize=10); ax.set_xlim(0, t[-1])

fig.tight_layout()
out = args.csv.rsplit(".", 1)[0] + "_rms.png"
fig.savefig(out, dpi=110)
print("wrote", out)
