#!/usr/bin/env python3
"""Plot recorded current-sense currents from record_serial.py's CSV.

The CSV columns are: t_s,phase,A,B,C,D,dutyA,dutyB,dutyC,dutyD
where A..D are the per-channel current-sense readings in amps.

Usage: uv run python ai/plot_cs_record.py data/cs_YYYYMMDD_HHMMSS.csv
"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv = sys.argv[1]
df = pd.read_csv(csv)
t = df["t_s"].values
chs = ["A", "B", "C", "D"]
col = {"A": "#e15759", "B": "#4e79a7", "C": "#59a14f", "D": "#f28e2b"}

fig, ax = plt.subplots(figsize=(13, 6))
for c in chs:
    ax.plot(t, df[c].values, lw=1.0, color=col[c], label=f"I{c}")
ax.set_xlabel("time (s)")
ax.set_ylabel("current-sense (A)")
ax.set_xlim(t[0], t[-1])
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=9)
ax.set_title(f"Current-sense currents vs time — {csv}")

# quick per-channel summary to stdout
print(f"{csv}: {len(df)} samples over {t[-1]-t[0]:.1f}s")
for c in chs:
    v = df[c].values
    print(f"  I{c}: mean={v.mean():.3f}A  min={v.min():.3f}A  max={v.max():.3f}A")

fig.tight_layout()
out = csv.rsplit(".", 1)[0] + "_cs.png"
fig.savefig(out, dpi=110)
print("wrote", out)
