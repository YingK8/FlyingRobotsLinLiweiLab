import pandas as pd, numpy as np, matplotlib.pyplot as plt

ABCD = "seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv"
AONLY = "seperate_ground_10sec_linear_carrier_ramp_15kHz_A.csv"
CHANNELS = ["Channel A", "Channel B", "Channel C", "Channel D"]
COLORS = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
W = 41  # envelope window

def load(csv):
    df = pd.read_csv(csv, skiprows=[1])  # row 1 is units
    return df
def env(y):  # upper envelope of the (rectified) growing signal
    return pd.Series(y).rolling(W, center=True, min_periods=1).max() \
             .rolling(W, center=True, min_periods=1).mean().values

# ---------- ABCD: all channels together ----------
df = load(ABCD); t = df["Time"].values
fig1, ax = plt.subplots(figsize=(11, 6))
for ch in CHANNELS:
    ax.plot(t, df[ch], color=COLORS[ch], lw=0.4, alpha=0.2)
    ax.plot(t, env(df[ch].values), color=COLORS[ch], lw=1.8, label=ch)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Output (V)")
ax.set_title("Linear carrier ramp 0->100% @ 190 Hz — all channels (separate grounds)")
ax.set_xlim(t.min(), t.max()); ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig1.tight_layout(); fig1.savefig("ramp_ABCD_all.png", dpi=140)

# ---------- ABCD: 4-grid separate ----------
fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, ch in zip(axes.ravel(), CHANNELS):
    ax.plot(t, df[ch], color=COLORS[ch], lw=0.4, alpha=0.25)
    ax.plot(t, env(df[ch].values), color=COLORS[ch], lw=1.8)
    ax.set_title(ch); ax.grid(alpha=0.3); ax.set_xlim(t.min(), t.max())
for ax in axes[:, 0]: ax.set_ylabel("Output (V)")
for ax in axes[1, :]: ax.set_xlabel("Time (s)")
fig2.suptitle("Linear carrier ramp 0->100% @ 190 Hz — per channel (separate grounds)")
fig2.tight_layout(); fig2.savefig("ramp_ABCD_grid.png", dpi=140)

# ---------- A only ----------
dfa = load(AONLY); ta = dfa["Time"].values
fig3, ax = plt.subplots(figsize=(11, 5))
ax.plot(ta, dfa["Channel A"], color="C0", lw=0.5, alpha=0.3, label="raw")
ax.plot(ta, env(dfa["Channel A"].values), color="C0", lw=2.0, label="envelope")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Output (V)")
ax.set_title("Linear carrier ramp 0->100% @ 190 Hz — Channel A only")
ax.set_xlim(ta.min(), ta.max()); ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig3.tight_layout(); fig3.savefig("ramp_A_only.png", dpi=140)

print("peak output (V), max of envelope:")
for ch in CHANNELS:
    print(f"  {ch}: {np.nanmax(env(df[ch].values)):.3f} V")
print("saved ramp_ABCD_all.png, ramp_ABCD_grid.png, ramp_A_only.png")
