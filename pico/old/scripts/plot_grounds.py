import pandas as pd, numpy as np, matplotlib.pyplot as plt, os

CONFIGS = [
    ("left_ground.csv",     "all GND left",  "C0"),
    ("right_ground_17.csv", "all GND right", "C1"),
    ("seperate_ground.csv", "separate PSU",  "C2"),
]
CHANNELS = ["Channel A", "Channel B", "Channel C", "Channel D"]
W = 121

def load(csv):
    units = str(pd.read_csv(csv, skiprows=[0], nrows=1, header=None).iloc[0, 0])
    scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6}[units.strip().strip("()").lower()]
    df = pd.read_csv(csv, skiprows=[1])
    df["Frequency"] = df["Frequency"] * scale
    return df

def smooth(y): return pd.Series(y).rolling(W, center=True, min_periods=1).median().values

data = {label: load(csv) for csv, label, _ in CONFIGS}

# resonance + notch detection over a clean sub-band
def features(f, ys):
    m = (f > 5) & (f < 1500); idx = np.where(m)[0]
    pk = idx[np.argmax(ys[idx])]
    # notch: deepest local min between peak and 2x peak freq, vs surrounding maxima
    fm = (f > f[pk]) & (f < min(f[pk]*2.2, 1500)); j = np.where(fm)[0]
    if len(j):
        dmin = j[np.argmin(ys[j])]
        depth = ys[pk] - ys[dmin]
    else:
        dmin, depth = pk, 0.0
    return f[pk], ys[pk], f[dmin], depth

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
print(f"{'Channel':10} {'config':14} {'peak':>16} {'notch@/depth':>18}")
for ax, ch in zip(axes.ravel(), CHANNELS):
    for csv, label, color in CONFIGS:
        df = data[label]; f = df["Frequency"].values
        ys = smooth(df[ch].values)
        ax.plot(f, ys, color=color, lw=1.5, label=label)
        fp, vp, fd, depth = features(f, ys)
        print(f"{ch:10} {label:14} {vp:6.1f} dBu @ {fp:5.0f} Hz   "
              f"{fd:5.0f} Hz / {depth:4.1f} dB")
    ax.set_title(ch); ax.grid(alpha=0.3); ax.set_xlim(0, 1500)
    ax.set_ylabel("Response (dBu)")
axes[0, 0].legend(loc="upper right", fontsize=8)
for ax in axes[1]: ax.set_xlabel("Frequency (Hz)")
fig.suptitle("Grounding configuration vs. coil response (smoothed)")
fig.tight_layout()
fig.savefig("ground_compare.png", dpi=140)
print("saved ground_compare.png")
