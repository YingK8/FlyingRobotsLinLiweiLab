import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "right_ground_17.csv"

# Row 1 holds units, e.g. "(kHz)" or "(Hz)" — read it so we can normalize to Hz.
units_row = pd.read_csv(CSV, skiprows=[0], nrows=1, header=None).iloc[0, 0]
FREQ_SCALE = {"hz": 1.0, "khz": 1e3, "mhz": 1e6}[
    str(units_row).strip().strip("()").lower()
]

df = pd.read_csv(CSV, skiprows=[1])  # row 1 is units
freq = df["Frequency"].values * FREQ_SCALE  # always Hz from here on
channels = ["Channel A", "Channel B", "Channel C", "Channel D"]
colors = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}

W = 101  # smoothing window (~6 Hz)
def smooth(y):
    return pd.Series(y).rolling(W, center=True, min_periods=1).median().values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

# Top: raw spectra (light) + smoothed envelope (bold)
for ch in channels:
    ax1.plot(freq, df[ch], color=colors[ch], lw=0.4, alpha=0.25)
    ax1.plot(freq, smooth(df[ch].values), color=colors[ch], lw=1.6, label=ch)
ax1.set_ylabel("Response (dBu)")
ax1.set_title("Frequency Sweep Response — Four Coils (raw + smoothed envelope)")
ax1.grid(True, alpha=0.3)
ax1.legend(loc="lower right")

# Bottom: smoothed only, with resonance peak (ignore DC, look above 5 Hz)
mask = freq > 5
print("Resonance peaks (smoothed, f > 5 Hz):")
for ch in channels:
    ys = smooth(df[ch].values)
    ax2.plot(freq, ys, color=colors[ch], lw=1.6, label=ch)
    idx = np.where(mask)[0]
    i = idx[np.argmax(ys[idx])]
    ax2.plot(freq[i], ys[i], "o", color=colors[ch], ms=7)
    ax2.annotate(f"{freq[i]:.0f} Hz", (freq[i], ys[i]),
                 textcoords="offset points", xytext=(5, 5),
                 fontsize=9, color=colors[ch])
    print(f"  {ch}: {ys[i]:.2f} dBu @ {freq[i]:.1f} Hz")

ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Response (dBu)")
ax2.set_title("Smoothed envelope with resonance peaks")
ax2.set_xlim(freq.min(), freq.max())
ax2.grid(True, alpha=0.3)
ax2.legend(loc="lower right")

import os
out = os.path.splitext(CSV)[0] + "_sweep.png"
fig.tight_layout()
fig.savefig(out, dpi=140)
print("saved", out)
