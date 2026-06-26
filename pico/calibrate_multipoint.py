"""Multipoint VNH5019 current-sense (CS) calibration.

Supersedes the single-point `calibrate.py` / hard-coded `SENS` in `plot_current.py`.

Input:  ../ESP32_PMW/src/k-calibration.csv
        Per channel A/B/C/D, a supply-voltage sweep with columns:
          POW V (supply volts), POW I (supply current = load current, A),
          CS V (current-sense voltage, mV).

Model (matches the rest of the pipeline):
        V_CS = offset + I / g          fit V_CS[V] vs I[A]
        I_load(V_CS) = g * (V_CS - offset)
        g     [A/V]  : sensitivity used by plot_current
        K_eff        : R_shunt * g   (effective VNH5019 sense ratio)

The fit is robust: an ordinary least-squares pass, then drop points whose
residual exceeds 3*MAD and refit (kills the known 6.00 V supply-I outliers on
A and B). Dropped points are drawn as hollow x's so it's transparent.

Outputs (in pico/):
  kcal_fit.png                 raw CS-vs-current points + calibrated fit, 4-grid
  current_ABCD_all_kcal.png    ramp current re-derived with the multipoint cal
  current_ABCD_grid_kcal.png   same, per-channel grid
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

HERE = os.path.dirname(os.path.abspath(__file__))
KCAL = os.path.join(HERE, "..", "ESP32_PMW", "src", "k-calibration.csv")
RAMP = os.path.join(HERE, "seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv")

CH = ["Channel A", "Channel B", "Channel C", "Channel D"]
COLORS = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
R = {"Channel A": 2532., "Channel B": 2530., "Channel C": 2543., "Channel D": 2540.}
OLD = {"Channel A": 14.38, "Channel B": 8.94, "Channel C": 9.89, "Channel D": 9.87}
W = 41  # envelope window (matches the rest of the pipeline)


def load_kcal(path):
    """Parse the 4-block CSV into {channel: (I[A], V_CS[V])}."""
    raw = pd.read_csv(path, header=None, skiprows=1)
    out = {}
    for i, ch in enumerate(CH):
        base = i * 5  # blocks: label, POW V, POW I, CS V, CS R
        I = pd.to_numeric(raw[base + 2], errors="coerce").values
        V = pd.to_numeric(raw[base + 3], errors="coerce").values / 1000.0  # mV -> V
        m = np.isfinite(I) & np.isfinite(V)
        out[ch] = (I[m], V[m])
    return out


def robust_fit(I, V):
    """Theil-Sen fit of V = m*I + b (robust to high-leverage outliers).

    Flags points whose residual from the robust line exceeds 3*MAD; R^2 is
    reported over the inliers.
    """
    m, b, _, _ = theilslopes(V, I)
    resid = V - (m * I + b)
    mad = np.median(np.abs(resid - np.median(resid))) or 1e-12
    keep = np.abs(resid) <= 3.0 * 1.4826 * mad
    fit = m * I[keep] + b
    ss_res = float(np.sum((V[keep] - fit) ** 2))
    ss_tot = float(np.sum((V[keep] - V[keep].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"slope": m, "intercept": b, "g": 1.0 / m, "offset_fit": b,
            "r2": r2, "keep": keep, "n": int(keep.sum()), "n_drop": int((~keep).sum())}


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def main():
    data = load_kcal(KCAL)
    cal = {ch: robust_fit(*data[ch]) for ch in CH}

    # baseline (drive-off) offset straight from the ramp capture, per channel
    dfr = pd.read_csv(RAMP, skiprows=[1])
    t = dfr["Time"].values
    base_off = {ch: float(np.median(dfr[ch].values[t < 1.0])) for ch in CH}

    # ---- report ----
    print(f"{'ch':3}{'g A/V':>9}{'K_eff':>8}{'off_fit mV':>11}"
          f"{'off_ramp mV':>12}{'R^2':>8}{'n/drop':>8}{'old A/V':>9}")
    for ch in CH:
        c = cal[ch]
        print(f"{ch[-1]:3}{c['g']:9.2f}{R[ch]*c['g']:8.0f}{c['offset_fit']*1000:11.1f}"
              f"{base_off[ch]*1000:12.1f}{c['r2']:8.4f}"
              f"{str(c['n'])+'/'+str(c['n_drop']):>8}{OLD[ch]:9.2f}")

    # ---- 1) raw relationship + calibrated fit (4-grid) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, ch in zip(axes.ravel(), CH):
        I, V = data[ch]
        c = cal[ch]
        keep = c["keep"]
        ax.scatter(I[keep], V[keep] * 1000, s=20, color=COLORS[ch], zorder=3,
                   label="data")
        if c["n_drop"]:
            ax.scatter(I[~keep], V[~keep] * 1000, s=40, facecolors="none",
                       edgecolors="red", zorder=3, label="dropped (>3 MAD)")
        xx = np.array([0.0, I.max() * 1.05])
        ax.plot(xx, (c["slope"] * xx + c["intercept"]) * 1000, "k-", lw=1.5,
                label=f"V = {c['offset_fit']*1000:.0f}mV + I/{c['g']:.2f}")
        ax.set_title(f"{ch}  g={c['g']:.2f} A/V, K_eff={R[ch]*c['g']:.0f}, "
                     f"R²={c['r2']:.4f}")
        ax.set_xlabel("Load current  I (A)")
        ax.set_ylabel("CS voltage  V$_{CS}$ (mV)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("VNH5019 current-sense calibration — multipoint fit "
                 "(supply-current sweep)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kcal_fit.png"), dpi=140)

    # ---- 2) re-derive ramp current with multipoint cal (gain) + ramp offset ----
    def cur(ch, v):
        return cal[ch]["g"] * (v - base_off[ch])

    def cur_env(ch):
        """Calibrated current envelope, with the baseline noise floor removed.

        env() is a rolling-max, so on the drive-off baseline it rectifies CS
        quantization noise into a positive floor (~0.6 A). Subtract that
        per-channel floor (measured at t<1s) so an idle channel reads ~0.
        """
        e = env(cur(ch, dfr[ch].values))
        floor = float(np.mean(e[t < 1.0]))
        return np.clip(e - floor, 0, None)

    fig2, ax = plt.subplots(figsize=(11, 6))
    for ch in CH:
        I = cur(ch, dfr[ch].values)
        ax.plot(t, I, color=COLORS[ch], lw=0.4, alpha=0.2)
        ax.plot(t, cur_env(ch), color=COLORS[ch], lw=1.8,
                label=f"{ch}  [{cal[ch]['g']:.2f} A/V]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Load current (A)")
    ax.set_title("Calibrated current — carrier ramp (multipoint K, offset-corrected)")
    ax.set_xlim(t.min(), t.max())
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(os.path.join(HERE, "current_ABCD_all_kcal.png"), dpi=140)

    # ---- 3) per-channel grid ----
    fig3, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, ch in zip(axes.ravel(), CH):
        I = cur(ch, dfr[ch].values)
        ax.plot(t, I, color=COLORS[ch], lw=0.4, alpha=0.25)
        ax.plot(t, cur_env(ch), color=COLORS[ch], lw=1.8)
        ax.set_title(f"{ch}   [{cal[ch]['g']:.2f} A/V, R²={cal[ch]['r2']:.3f}]")
        ax.grid(alpha=0.3)
        ax.set_xlim(t.min(), t.max())
    for ax in axes[:, 0]:
        ax.set_ylabel("Load current (A)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")
    fig3.suptitle("Calibrated current per channel — carrier ramp (multipoint K)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(HERE, "current_ABCD_grid_kcal.png"), dpi=140)

    print("\npeak calibrated current (A, baseline-floor removed):")
    for ch in CH:
        print(f"  {ch}: {np.nanmax(cur_env(ch)):.2f} A")
    print("\nsaved kcal_fit.png, current_ABCD_all_kcal.png, current_ABCD_grid_kcal.png")


if __name__ == "__main__":
    main()
