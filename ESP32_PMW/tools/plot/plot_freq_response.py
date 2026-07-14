#!/usr/bin/env python3
"""Frequency response from a current-sense stream: RMS current vs drive freq.

The firmware sweeps the drive frequency start->end over `ramp_s` using an
ease-in-out (smoothstep) profile:  f(t) = start + (end-start)*u^2*(3-2u),
u = t/ramp_s.  So time during the ramp maps monotonically to drive frequency.
We take short-window RMS of the (AC) current sense, restrict to the ramp window,
map each window's center time to its drive frequency, and plot RMS vs frequency
-- i.e. each coil's frequency response, with peaks at its resonance(s).

Usage: uv run python tools/plot_freq_response.py [CSV]
         [--t0 S] [--ramp-s S] [--f0 HZ] [--f1 HZ] [--win-ms MS]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default="picoscope_stream_20260703_200227.csv")
ap.add_argument("--t0", type=float, default=0.0, help="ramp start time in CSV (s)")
ap.add_argument("--ramp-s", type=float, default=30.0, help="ramp duration (s)")
ap.add_argument("--f0", type=float, default=1.0, help="start freq (Hz)")
ap.add_argument("--f1", type=float, default=210.0, help="end freq (Hz)")
ap.add_argument("--win-ms", type=float, default=250.0, help="RMS window (ms)")
args = ap.parse_args()

df = pd.read_csv(args.csv, skiprows=[1])
t = df["Time"].values
dt = np.median(np.diff(t)); fs = 1.0 / dt
chs = ["Channel A", "Channel B", "Channel C", "Channel D"]
colors = {"Channel A": "#e15759", "Channel B": "#4e79a7",
          "Channel C": "#59a14f", "Channel D": "#f28e2b"}

def drive_freq(tc):
    """map ramp-relative time -> drive frequency via smoothstep ease."""
    u = np.clip((tc - args.t0) / args.ramp_s, 0.0, 1.0)
    return args.f0 + (args.f1 - args.f0) * (u * u * (3.0 - 2.0 * u))

# non-overlapping RMS windows across the ramp region
win = max(1, int(round(args.win_ms * 1e-3 * fs)))
mask = (t >= args.t0) & (t <= args.t0 + args.ramp_s)
i0, i1 = np.argmax(mask), len(mask) - np.argmax(mask[::-1])
nwin = (i1 - i0) // win
fc, rms = [], {c: [] for c in chs}
for k in range(nwin):
    sl = slice(i0 + k * win, i0 + (k + 1) * win)
    tc = t[sl].mean()
    fc.append(drive_freq(tc))
    for c in chs:
        x = df[c].values[sl]; x = x - x.mean()
        rms[c].append(np.sqrt(np.mean(x * x)) * 1000.0)   # -> mV
fc = np.array(fc)

print(f"{args.csv}: {len(df)} samp, {fs:.0f} S/s | ramp {args.f0}->{args.f1} Hz "
      f"over {args.ramp_s}s from t0={args.t0}s | {nwin} windows ({args.win_ms:.0f} ms)")
for c in chs:
    r = np.array(rms[c])
    print(f"  {c}: peak {r.max():.1f} mV at {fc[np.argmax(r)]:.0f} Hz")

fig, ax = plt.subplots(figsize=(12, 6))
for c in chs:
    ax.plot(fc, rms[c], "-", lw=1.4, color=colors[c], label=c)
ax.set_title(f"Frequency response — {args.csv}\nRMS current sense vs drive frequency "
             f"({args.win_ms:.0f} ms windows over the {args.f0:.0f}-{args.f1:.0f} Hz ramp)")
ax.set_xlabel("drive frequency (Hz)"); ax.set_ylabel("RMS current sense (mV)")
ax.set_xlim(args.f0, args.f1); ax.grid(alpha=0.3); ax.legend(fontsize=10)
fig.tight_layout()
out = args.csv.rsplit(".", 1)[0] + "_freqresp.png"
fig.savefig(out, dpi=110)
print("wrote", out)
