#!/usr/bin/env python3
"""Plot raw current-sense voltage vs time for a capture (per-channel + zoom)."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

csv = sys.argv[1] if len(sys.argv) > 1 else "coupling_sweep_20260704_142357.csv"
df = pd.read_csv(csv, skiprows=[1]); t = df["Time"].values
chs = ["Channel A","Channel B","Channel C","Channel D"]
col = {"Channel A":"#e15759","Channel B":"#4e79a7","Channel C":"#59a14f","Channel D":"#f28e2b"}

fig, axs = plt.subplots(5, 1, figsize=(13, 11))
# per-channel raw voltage vs time (AC fills as bands; envelope = drive bursts)
for ax, c in zip(axs[:4], chs):
    ax.plot(t, df[c].values, lw=0.3, color=col[c])
    ax.set_ylabel(f"{c[-1]} (V)", color=col[c]); ax.set_xlim(0, t[-1])
    ax.grid(alpha=0.3)
    ax.tick_params(labelbottom=False)
axs[0].set_title(f"Raw current-sense voltage vs time — {csv}")

# zoom: 60 ms inside the first solo-A burst, to show the actual 190 Hz waveform
zc = t[(t >= 21.0)]
z0 = 21.0
m = (t >= z0) & (t < z0 + 0.06)
axz = axs[4]
for c in chs:
    axz.plot(t[m]*1000, df[c].values[m], lw=1.0, color=col[c], label=c, marker=".", ms=3)
axz.set_title("Zoom: 60 ms inside solo-A burst (raw waveform)")
axz.set_xlabel("time (ms)"); axz.set_ylabel("V"); axz.grid(alpha=0.3)
axz.legend(ncol=4, fontsize=8)

fig.tight_layout()
out = csv.rsplit(".",1)[0] + "_cs_voltage.png"
fig.savefig(out, dpi=110)
print("wrote", out)
