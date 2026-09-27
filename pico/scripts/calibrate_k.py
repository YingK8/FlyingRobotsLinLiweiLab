"""Canonical VNH5019 current-sense (K-gain) calibration.

Input:  ../ESP32_PMW/src/Tue30JunKCalibration.csv
        Per channel A/B/C/D two columns: load current I (A) and CS voltage V (mV).

Model:  V_CS = offset + I / g          (g [A/V] is the sense gain)
        I_load(V_CS) = g * (V_CS - offset)
        K_eff = R_sense * g             (effective VNH5019 sense ratio, dimensionless)

Why the offset is pinned, not fitted:
  The sweep only spans ~6-7.2 A. Over such a narrow, high-current band a free
  (slope, offset) fit is ill-conditioned: it trades slope against intercept and
  lands on a physically-impossible NEGATIVE zero-current offset (the CS pin
  sources current into R_sense, so V_CS >= 0 always). The hardware offset is a
  separate ~0 mV quantity measured with the drive OFF. We therefore report the
  free fit (for transparency) but take the GAIN from a fit through the origin,
  which is what plot_current / plot_duty_ramp consume. Pinned-offset gains are
  consistent across channels (CV ~1%) and agree with prior calibrations.

Output:
  pico/figures/calibration/kcal_fit.png   data points + offset-pinned fit
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.dirname(HERE)               # scripts/ lives under pico/
REPO = os.path.dirname(PICO)
KCAL = os.path.join(REPO, "ESP32_PMW", "src", "Tue30JunKCalibration.csv")

# Minimal, presentation-clean matplotlib defaults.
STYLE = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
    "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
    "legend.frameon": False,
}

CH = ["A", "B", "C", "D"]
COLORS = {"A": "C0", "B": "C1", "C": "C2", "D": "C3"}
R = {"A": 2538., "B": 2540., "C": 2550., "D": 2545.}      # measured shunt, ohm
PRIOR = {"A": 15.26, "B": 15.28, "C": 15.57, "D": 15.34}  # prior cal, console check


def load(path):
    df = pd.read_csv(path)
    out = {}
    for i, ch in enumerate(CH):
        I = pd.to_numeric(df.iloc[:, 2 * i], errors="coerce").values
        V = pd.to_numeric(df.iloc[:, 2 * i + 1], errors="coerce").values  # mV
        m = np.isfinite(I) & np.isfinite(V)
        out[ch] = (I[m], V[m])
    return out


def fit(I, V):
    """Free OLS (drop >3*MAD) for transparency + pinned-offset gain (used)."""
    s, b = np.polyfit(I, V, 1)
    res = V - (s * I + b)
    mad = np.median(np.abs(res - np.median(res))) or 1e-9
    keep = np.abs(res) <= 3 * 1.4826 * mad
    s, b = np.polyfit(I[keep], V[keep], 1)
    r2 = 1 - np.sum((V[keep] - (s * I[keep] + b)) ** 2) / \
        np.sum((V[keep] - V[keep].mean()) ** 2)
    # through origin: V = k*I  ->  k = sum(VI)/sum(II) [mV/A], g = 1000/k [A/V]
    k0 = np.sum(V[keep] * I[keep]) / np.sum(I[keep] ** 2)
    return {"g_free": 1000.0 / s, "offset": b, "r2": r2, "keep": keep,
            "g": 1000.0 / k0, "slope0": k0, "n": int(keep.sum()),
            "ndrop": int((~keep).sum())}


def main():
    data = load(KCAL)
    cal = {ch: fit(*data[ch]) for ch in CH}

    print(f"{'ch':3}{'g A/V':>8}{'K_eff':>8}{'g_free':>8}{'off mV':>8}"
          f"{'R^2':>8}{'n/drop':>8}{'prior g':>9}")
    gs, Ks = [], []
    for ch in CH:
        c = cal[ch]
        K = R[ch] * c["g"]
        gs.append(c["g"]); Ks.append(K)
        nd = f"{c['n']}/{c['ndrop']}"
        print(f"{ch:3}{c['g']:8.2f}{K:8.0f}{c['g_free']:8.2f}{c['offset']:8.1f}"
              f"{c['r2']:8.4f}{nd:>8}{PRIOR[ch]:9.2f}")
    gs, Ks = np.array(gs), np.array(Ks)
    print(f"\ngain  : mean {gs.mean():.2f} A/V  std {gs.std(ddof=1):.3f}  "
          f"CV {gs.std(ddof=1)/gs.mean()*100:.1f}%")
    print(f"K_eff : mean {Ks.mean():.0f}      std {Ks.std(ddof=1):.0f}      "
          f"CV {Ks.std(ddof=1)/Ks.mean()*100:.1f}%")

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, ch in zip(axes.ravel(), CH):
        I, V = data[ch]
        c = cal[ch]
        keep = c["keep"]
        xx = np.array([0.0, I.max() * 1.05])
        ax.plot(xx, c["slope0"] * xx, color="0.15", lw=1.2, zorder=2)
        ax.scatter(I[keep], V[keep], s=18, color=COLORS[ch], alpha=0.55,
                   edgecolors="none", zorder=3)
        if c["ndrop"]:
            ax.scatter(I[~keep], V[~keep], s=30, facecolors="none",
                       edgecolors="0.5", alpha=0.8, zorder=3, label="dropped")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{ch}    g = {c['g']:.2f} A/V    K = {R[ch]*c['g']:.0f}"
                     f"    R² = {c['r2']:.3f}")
        ax.set_xlabel("Load current (A)")
        ax.set_ylabel("Sense voltage (mV)")
        ax.set_xlim(0, I.max() * 1.05)
        ax.set_ylim(0, None)
    fig.suptitle("VNH5019 current-sense calibration", fontsize=12)
    fig.tight_layout()
    out_dir = os.path.join(PICO, "figures", "calibration")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "kcal_fit.png")
    fig.savefig(out, facecolor="white")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
