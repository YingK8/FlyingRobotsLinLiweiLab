"""Two-parameter (gain + offset) VNH5019 current-sense calibration.

  offset  : zero-current CS voltage, from the carrier-ramp baseline (drive off)
  100%pt  : single-point (supply current, CS voltage) at 190 Hz / 100% duty
  gain g  : g = I_100 / (V_100 - offset)          [A/V]
  model   : I_load(V_CS) = g * (V_CS - offset)
  K_eff   : R_shunt * g    (effective sense ratio for this waveform)

The carrier-ramp is also used to (a) measure the offset and (b) check that CS is
linear over the current range (R^2 of CS-envelope vs duty).
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt

CH     = ["Channel A", "Channel B", "Channel C", "Channel D"]
COLORS = {"Channel A":"C0","Channel B":"C1","Channel C":"C2","Channel D":"C3"}
R      = {"Channel A":2538.,"Channel B":2540.,"Channel C":2550.,"Channel D":2545.}
I100   = {"Channel A":5.466,"Channel B":3.618,"Channel C":5.234,"Channel D":5.247}  # supply A
V100   = {"Channel A":0.380,"Channel B":0.4049,"Channel C":0.529,"Channel D":0.5315} # CS V
ABCD   = "seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv"
W = 41

df = pd.read_csv(ABCD, skiprows=[1]); t = df["Time"].values
def env(y): return pd.Series(y).rolling(W,center=True,min_periods=1).max() \
                    .rolling(W,center=True,min_periods=1).mean().values

# ---- offsets from ramp baseline (drive off, t<1s) ----
offset = {ch: float(np.median(df[ch].values[t < 1.0])) for ch in CH}

# ---- global ramp window from averaged normalized envelope ----
norm = np.zeros_like(t, dtype=float)
E = {}
for ch in CH:
    e = env(df[ch].values); E[ch] = e
    plat = np.median(e[t > t.max()-2.0])
    norm += np.clip((e-offset[ch])/(plat-offset[ch]), 0, 1)
norm /= len(CH)
t0 = t[np.argmax(norm > 0.05)]
t1 = t[np.argmax(norm > 0.95)]
duty = np.clip((t - t0)/(t1 - t0), 0, 1)

# ---- calibration table + linearity check ----
print(f"ramp window t0={t0:.1f}s t1={t1:.1f}s (duration {t1-t0:.1f}s)\n")
print(f"{'ch':3}{'offset mV':>10}{'gain A/V':>10}{'K_eff':>8}{'lin R^2':>9}{'old A/V':>9}")
cal = {}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
OLD = {"Channel A":14.38,"Channel B":8.94,"Channel C":9.89,"Channel D":9.87}
for ax, ch in zip(axes.ravel(), CH):
    g = I100[ch] / (V100[ch] - offset[ch])     # A/V, offset-corrected
    Keff = R[ch] * g
    cal[ch] = (offset[ch], g)
    # linearity: CS-envelope vs duty in the rising region
    m = (duty > 0.05) & (duty < 0.98)
    p = np.polyfit(duty[m], E[ch][m], 1)
    resid = E[ch][m] - np.polyval(p, duty[m])
    r2 = 1 - np.var(resid)/np.var(E[ch][m])
    print(f"{ch[-1]:3}{offset[ch]*1000:10.1f}{g:10.2f}{Keff:8.0f}{r2:9.3f}{OLD[ch]:9.2f}")
    # plot: synthetic current axis (duty*I100) vs CS, with fit line
    I_axis = duty * I100[ch]
    ax.plot(I_axis[m], E[ch][m], ".", color=COLORS[ch], ms=2, alpha=0.5)
    xx = np.array([0, I100[ch]])
    ax.plot(xx, offset[ch] + xx/g, "k-", lw=1.5,
            label=f"V = {offset[ch]*1000:.0f}mV + I/{g:.2f}")
    ax.axhline(offset[ch], color="gray", ls=":", lw=1)
    ax.set_title(f"{ch}  (g={g:.2f} A/V, R^2={r2:.3f})")
    ax.set_xlabel("Load current (A, duty-anchored)"); ax.set_ylabel("CS envelope (V)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Two-parameter CS calibration: CS voltage vs current (gain + offset)")
fig.tight_layout(); fig.savefig("calibration_fit.png", dpi=140)

# ---- regenerate calibrated-current ramp plots WITH offset correction ----
TRUST = {"Channel A":False,"Channel B":True,"Channel C":True,"Channel D":True}
def cur(ch, v):
    o, g = cal[ch]; return g * (v - o)
fig2, ax = plt.subplots(figsize=(11,6))
for ch in CH:
    I = cur(ch, df[ch].values)
    ax.plot(t, I, color=COLORS[ch], lw=0.4, alpha=0.2)
    ax.plot(t, env(I), color=COLORS[ch], lw=1.8,
            ls="-" if TRUST[ch] else "--",
            label=ch + ("" if TRUST[ch] else " (low-conf)"))
ax.set_xlabel("Time (s)"); ax.set_ylabel("Load current (A)")
ax.set_title("Offset-corrected calibrated current — carrier ramp (separate grounds)")
ax.set_xlim(t.min(), t.max()); ax.grid(alpha=0.3); ax.legend(loc="upper left")
fig2.tight_layout(); fig2.savefig("current_ABCD_all_offsetcorr.png", dpi=140)

print("\npeak offset-corrected current (A):")
for ch in CH:
    print(f"  {ch}: {np.nanmax(env(cur(ch, df[ch].values))):.2f} A")
print("saved calibration_fit.png, current_ABCD_all_offsetcorr.png")
