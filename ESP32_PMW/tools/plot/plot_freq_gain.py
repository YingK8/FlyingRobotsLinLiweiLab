#!/usr/bin/env python3
"""Frequency-gain analysis of a PicoScope current-sense stream.

The rig drives a swept sine (frequency ramp 1->210 Hz over ~30 s), so the coil
current is an AC *chirp*, not DC. A whole-record FFT smears a chirp across the
whole band (it looks like a flat noise floor -- it isn't). The right view is a
spectrogram: the swept tone shows up as a rising ridge in time-frequency, and
because time maps to drive frequency during the ramp, the ridge amplitude vs
frequency is the coil's frequency gain (transfer-function magnitude).

Usage: uv run python tools/plot_freq_gain.py [CSV]
"""
import sys
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "picoscope_stream_20260703_200227.csv"
df = pd.read_csv(path, skiprows=[1])
t = df["Time"].values
dt = np.median(np.diff(t)); fs = 1.0 / dt
chs = ["Channel A", "Channel B", "Channel C", "Channel D"]
colors = {"Channel A": "#e15759", "Channel B": "#4e79a7",
          "Channel C": "#59a14f", "Channel D": "#f28e2b"}

FMAX = 260.0            # Hz, view/analysis ceiling (end_freq=210 + margin)
MAINS = (60.0, 120.0, 180.0)  # US line + harmonics to mask during ridge track
nperseg = 512          # ~0.5 s window -> ~2 Hz resolution at 1 kS/s
noverlap = nperseg * 3 // 4

fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# ---- 1) Spectrogram of one channel (visual proof of the swept ridge) -------
ref = "Channel A"
v = df[ref].values - df[ref].values.mean()
f, tt, Sxx = spectrogram(v, fs=fs, nperseg=nperseg, noverlap=noverlap,
                         scaling="spectrum", mode="magnitude")
band = f <= FMAX
ax = axs[0]
pcm = ax.pcolormesh(tt, f[band], 20*np.log10(Sxx[band] + 1e-9),
                    shading="gouraud", cmap="magma")
ax.set_title(f"Spectrogram — {ref}   (swept-sine ridge = the chirp, NOT noise)")
ax.set_ylabel("frequency (Hz)"); ax.set_xlabel("time (s)"); ax.set_ylim(0, FMAX)
fig.colorbar(pcm, ax=ax, label="dB (mag)")

# ---- 2) Ridge frequency vs time (confirms the 1->210 Hz ramp) --------------
# Per channel, track the dominant non-mains frequency in each STFT column.
def ridge(v):
    f, tt, S = spectrogram(v, fs=fs, nperseg=nperseg, noverlap=noverlap,
                           scaling="spectrum", mode="magnitude")
    keep = (f > 2.0) & (f <= FMAX)
    for m in MAINS:                        # notch mains +-2 Hz so it can't win
        keep &= ~((f > m - 2) & (f < m + 2))
    fk, Sk = f[keep], S[keep]
    idx = np.argmax(Sk, axis=0)
    rf = fk[idx]                           # ridge frequency per time column
    ramp = np.take_along_axis(Sk, idx[None], 0)[0]  # ridge magnitude per column
    return tt, rf, ramp

ax = axs[1]
ridges = {}
for c in chs:
    vv = df[c].values - df[c].values.mean()
    tt, rf, mag = ridge(vv)
    ridges[c] = (tt, rf, mag)
    ax.plot(tt, rf, ".", ms=3, color=colors[c], label=c, alpha=0.7)
ax.set_title("Tracked ridge frequency vs time — the drive-frequency ramp")
ax.set_ylabel("ridge freq (Hz)"); ax.set_xlabel("time (s)")
ax.set_ylim(0, FMAX); ax.grid(alpha=0.3); ax.legend(ncol=4, fontsize=8)

# ---- 3) FREQUENCY GAIN: ridge amplitude vs ridge frequency -----------------
# Only the rising (monotone) part of the ramp is a clean freq sweep; keep the
# window where the ridge climbs, then plot amplitude against its own frequency.
ax = axs[2]
for c in chs:
    tt, rf, mag = ridges[c]
    # keep the ramp region: from first frame to the ridge-frequency peak
    top = int(np.argmax(rf))
    if top < 3:
        top = len(rf) - 1
    fsel, gsel = rf[:top + 1], mag[:top + 1] * 1000.0   # -> mV
    order = np.argsort(fsel)
    fsel, gsel = fsel[order], gsel[order]
    # light binning to smooth the ridge amplitude
    nb = 60
    edges = np.linspace(1, FMAX, nb + 1)
    bi = np.clip(np.digitize(fsel, edges) - 1, 0, nb - 1)
    fb = 0.5 * (edges[:-1] + edges[1:])
    gb = np.array([gsel[bi == k].mean() if np.any(bi == k) else np.nan
                   for k in range(nb)])
    ax.plot(fb, gb, "-", lw=1.6, color=colors[c], label=c)
ax.set_title("Frequency gain — swept-tone current amplitude vs drive frequency")
ax.set_xlabel("drive frequency (Hz)"); ax.set_ylabel("current-sense amplitude (mV)")
ax.set_xlim(0, 220); ax.grid(alpha=0.3); ax.legend(ncol=4, fontsize=9)

fig.tight_layout()
out = path.rsplit(".", 1)[0] + "_freqgain.png"
fig.savefig(out, dpi=110)
print("wrote", out)
