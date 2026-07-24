"""Calibrated peak current for channels A and C only (board 1, both zero-offset).
Gains are the §5 single-point (avg-anchored) values — provisional until a probe
peak/RMS calibration is done. A is flagged low-confidence (gain anomaly)."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt

RAMP = "seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv"
G   = {"Channel A": 14.38, "Channel C": 9.89}   # A/V, offset = 0 for both
COL = {"Channel A": "C0", "Channel C": "C2"}
W = 41
def env(y): return pd.Series(y).rolling(W,center=True,min_periods=1).max() \
                    .rolling(W,center=True,min_periods=1).mean().values

df = pd.read_csv(RAMP, skiprows=[1]); t = df["Time"].values
fig, ax = plt.subplots(figsize=(11, 6))
for ch in ["Channel A", "Channel C"]:
    I = df[ch].values * G[ch]
    ax.plot(t, I, color=COL[ch], lw=0.4, alpha=0.2)
    ax.plot(t, env(I), color=COL[ch], lw=2.0, ls="-", label=ch)
    print(f"{ch}: peak {np.nanmax(env(I)):.2f} A   (g={G[ch]} A/V)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Peak coil current (A)")
ax.set_title("Calibrated peak current — Channels A & C (carrier ramp, separate grounds)")
ax.set_xlim(t.min(), t.max()); ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("current_AC.png", dpi=140)
print("saved current_AC.png")
