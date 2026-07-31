"""Convert VNH5019 current-sense (CS) voltage to actual current using the
per-channel calibration sensitivity (A/V) backed out from the single-point
calibration (supply current vs CS voltage at 190 Hz, 100% duty).

    I_load [A] = SENS[ch] * V_CS [V]

NOTE: A and B sensitivities are flagged as low-confidence (A is an outlier,
both likely carry significant sense offset). C and D agree to 0.3%.
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Per-channel calibration: A/V (= K_eff / R_shunt = I_supply / V_CS)
SENS = {"Channel A": 14.38, "Channel B": 8.94, "Channel C": 9.89, "Channel D": 9.87}
TRUST = {"Channel A": False, "Channel B": False, "Channel C": True, "Channel D": True}

ABCD  = "seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv"
AONLY = "seperate_ground_10sec_linear_carrier_ramp_15kHz_A.csv"
CHANNELS = ["Channel A", "Channel B", "Channel C", "Channel D"]
COLORS = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
W = 41

def load(csv): return pd.read_csv(csv, skiprows=[1])  # row 1 is units
def env(y):    # upper envelope of the rectified ramp
    return pd.Series(y).rolling(W, center=True, min_periods=1).max() \
             .rolling(W, center=True, min_periods=1).mean().values
def lbl(ch):   return f"{ch}" + ("" if TRUST[ch] else " (low-conf)")

# ---------- ABCD: all channels, current ----------
df = load(ABCD); t = df["Time"].values
fig1, ax = plt.subplots(figsize=(11, 6))
for ch in CHANNELS:
    I = df[ch].values * SENS[ch]
    ax.plot(t, I, color=COLORS[ch], lw=0.4, alpha=0.2)
    ax.plot(t, env(I), color=COLORS[ch], lw=1.8,
            ls="-" if TRUST[ch] else "--", label=lbl(ch))
ax.set_xlabel("Time (s)"); ax.set_ylabel("Load current (A)")
ax.set_title("Calibrated current — linear carrier ramp 0->100% @ 190 Hz (separate grounds)")
ax.set_xlim(t.min(), t.max()); ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig1.tight_layout(); fig1.savefig("current_ABCD_all.png", dpi=140)

# ---------- ABCD: 4-grid current ----------
fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, ch in zip(axes.ravel(), CHANNELS):
    I = df[ch].values * SENS[ch]
    ax.plot(t, I, color=COLORS[ch], lw=0.4, alpha=0.25)
    ax.plot(t, env(I), color=COLORS[ch], lw=1.8)
    ax.set_title(lbl(ch) + f"   [{SENS[ch]:.2f} A/V]")
    ax.grid(alpha=0.3); ax.set_xlim(t.min(), t.max())
for ax in axes[:, 0]: ax.set_ylabel("Load current (A)")
for ax in axes[1, :]: ax.set_xlabel("Time (s)")
fig2.suptitle("Calibrated current per channel — carrier ramp (separate grounds)")
fig2.tight_layout(); fig2.savefig("current_ABCD_grid.png", dpi=140)

# ---------- A only, current ----------
dfa = load(AONLY); ta = dfa["Time"].values
Ia = dfa["Channel A"].values * SENS["Channel A"]
fig3, ax = plt.subplots(figsize=(11, 5))
ax.plot(ta, Ia, color="C0", lw=0.5, alpha=0.3, label="raw")
ax.plot(ta, env(Ia), color="C0", lw=2.0, label="envelope")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Load current (A)")
ax.set_title("Calibrated current — Channel A (low-conf cal: 14.38 A/V)")
ax.set_xlim(ta.min(), ta.max()); ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig3.tight_layout(); fig3.savefig("current_A_only.png", dpi=140)

print("peak calibrated current (A):")
for ch in CHANNELS:
    print(f"  {ch}: {np.nanmax(env(df[ch].values*SENS[ch])):.2f} A   ({'trust' if TRUST[ch] else 'low-conf'})")
print("saved current_ABCD_all.png, current_ABCD_grid.png, current_A_only.png")
