import pandas as pd, numpy as np, matplotlib.pyplot as plt, sys, os

CSV = sys.argv[1] if len(sys.argv) > 1 else "seperate_ground.csv"
STEM = os.path.splitext(os.path.basename(CSV))[0]
CHANNELS = ["Channel A", "Channel B", "Channel C", "Channel D"]
COLORS = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
F0 = 190.0  # design resonant frequency (Hz)
W = 121

units = str(pd.read_csv(CSV, skiprows=[0], nrows=1, header=None).iloc[0, 0])
scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6}[units.strip().strip("()").lower()]
df = pd.read_csv(CSV, skiprows=[1])
freq = df["Frequency"].values * scale

def smooth(y): return pd.Series(y).rolling(W, center=True, min_periods=1).median().values
def val_at(ys, f0): return float(np.interp(f0, freq, ys))

S = {ch: smooth(df[ch].values) for ch in CHANNELS}

# ---- Figure 1: all channels combined ----
fig1, ax = plt.subplots(figsize=(11, 6))
for ch in CHANNELS:
    ax.plot(freq, df[ch], color=COLORS[ch], lw=0.4, alpha=0.18)
    ax.plot(freq, S[ch], color=COLORS[ch], lw=1.7, label=ch)
ax.axvline(F0, color="k", ls="--", lw=1.2, label=f"f0 = {F0:.0f} Hz")
ax.set_xlim(0, freq.max()); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Response (dBu)")
ax.set_title("Separate grounds — all channels")
ax.grid(alpha=0.3); ax.legend(loc="upper right")
fig1.tight_layout(); fig1.savefig(f"{STEM}_all.png", dpi=140)

# ---- Figure 2: 2x2 grid, value labeled at 190 Hz ----
fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
print(f"Value at {F0:.0f} Hz (separate grounds):")
for ax, ch in zip(axes.ravel(), CHANNELS):
    ax.plot(freq, df[ch], color=COLORS[ch], lw=0.4, alpha=0.18)
    ax.plot(freq, S[ch], color=COLORS[ch], lw=1.7)
    v = val_at(S[ch], F0)
    ax.axvline(F0, color="k", ls="--", lw=1.0)
    ax.plot(F0, v, "o", color=COLORS[ch], ms=8, zorder=5)
    ax.annotate(f"{v:.1f} dBu @ {F0:.0f} Hz", (F0, v),
                textcoords="offset points", xytext=(10, 8),
                fontsize=10, fontweight="bold", color=COLORS[ch])
    ax.set_title(ch); ax.grid(alpha=0.3); ax.set_xlim(0, freq.max())
    print(f"  {ch}: {v:.2f} dBu")
for ax in axes[:, 0]: ax.set_ylabel("Response (dBu)")
for ax in axes[1, :]: ax.set_xlabel("Frequency (Hz)")
fig2.suptitle(f"Separate grounds — per channel (value at f0={F0:.0f} Hz)")
fig2.tight_layout(); fig2.savefig(f"{STEM}_grid.png", dpi=140)
print(f"saved {STEM}_all.png and {STEM}_grid.png")
