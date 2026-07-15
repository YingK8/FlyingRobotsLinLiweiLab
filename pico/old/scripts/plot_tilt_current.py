"""Graph calibrated per-channel current from a tilt/ramp capture on one axis.

Usage:  python3 plot_tilt_current.py [capture.csv]
        (default: tilt/20260621.csv)

Uses the multipoint CS calibration (g, A/V) and per-channel drive-off baseline.
Prints the opposite-pair balance the user is targeting: A vs D and B vs C.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CH = ["Channel A", "Channel B", "Channel C", "Channel D"]
COL = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
G = {"Channel A": 15.26, "Channel B": 15.28, "Channel C": 15.57, "Channel D": 15.34}
W = 41

HERE = os.path.dirname(os.path.abspath(__file__))
csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tilt", "20260621.csv")


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


df = pd.read_csv(csv, skiprows=[1])
t = df["Time"].values
cur = {}
for ch in CH:
    base = float(np.median(df[ch].values[t < 1.0]))
    e = env(G[ch] * (df[ch].values - base))
    cur[ch] = np.clip(e - float(np.mean(e[t < 1.0])), 0, None)

fig, ax = plt.subplots(figsize=(12, 6))
for ch in CH:
    ax.plot(t, cur[ch], color=COL[ch], lw=1.8, label=f"{ch}  ({G[ch]:.2f} A/V)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Load current (A)")
ax.set_title(f"Calibrated per-channel current — {os.path.basename(csv)}")
ax.set_xlim(t.min(), t.max())
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
out = os.path.splitext(csv)[0] + "_current.png"
fig.tight_layout()
fig.savefig(out, dpi=140)
print("saved", out)

# steady window = last 25% of the run (or full if short)
m = t > (t.min() + 0.75 * (t.max() - t.min()))
mean = {ch: float(np.mean(cur[ch][m])) for ch in CH}
print("\nmean current over last quarter (A):")
for ch in CH:
    print(f"  {ch}: {mean[ch]:.2f}")
print("\nopposite-pair balance:")
print(f"  A vs D: {mean['Channel A']:.2f} / {mean['Channel D']:.2f}  "
      f"(diff {mean['Channel A']-mean['Channel D']:+.2f} A)")
print(f"  B vs C: {mean['Channel B']:.2f} / {mean['Channel C']:.2f}  "
      f"(diff {mean['Channel B']-mean['Channel C']:+.2f} A)")
