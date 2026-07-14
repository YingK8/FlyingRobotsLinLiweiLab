#!/usr/bin/env python3
"""Extract per-channel max-current-vs-frequency from a main_tilt_pi run whose
frequency ramp was driven at 100% duty (i.e. an open-loop max-current run like
I_TARGET_A above every coil's ceiling). During that ramp every channel is
saturated at 100% duty, so the measured current at each instant IS the coil's
maximum achievable current at that commutation frequency -- the ramp doubles as
a frequency sweep. Reconstructs the EASE ramp frequency vs board-time and finds
where each channel peaks.

Usage: uv run python tools/plot/plot_freq_sweep.py <run.log> [ramp_start_ms] [ramp_ms] [f0] [f1]
"""
import re
import sys

import matplotlib.pyplot as plt

LINE = re.compile(
    r"t=(\d+) \| carrier=([\d.]+)% .*?"
    r"I\[A\]: A=([\-\d.]+) B=([\-\d.]+) C=([\-\d.]+) D=([\-\d.]+) \| "
    r"duty\[%\]: A=([\-\d.]+) B=([\-\d.]+) C=([\-\d.]+) D=([\-\d.]+)"
)
COL = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

path = sys.argv[1]
ramp_start_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 4033.0
ramp_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 30000.0
f0 = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
f1 = float(sys.argv[5]) if len(sys.argv) > 5 else 200.0


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


freq = []
I = {k: [] for k in "ABCD"}
with open(path) as f:
    for ln in f:
        m = LINE.search(ln)
        if not m:
            continue
        g = m.groups()
        tms, carrier = int(g[0]), float(g[1])
        # Only the ramp portion (carrier still 100%, within ramp window) is a
        # clean 100%-duty sweep; stop once carrier steps down.
        if carrier < 100.0 - 1e-3:
            continue
        elapsed = tms - ramp_start_ms
        if elapsed < 0 or elapsed > ramp_ms:
            continue
        fhz = f0 + ease(elapsed / ramp_ms) * (f1 - f0)
        freq.append(fhz)
        for j, k in enumerate("ABCD"):
            I[k].append(float(g[2 + j]))

# Per-channel peak
print(f"{'ch':>3} {'peak I (A)':>10} {'@ freq (Hz)':>12}   (current @200Hz)")
for k in "ABCD":
    if not I[k]:
        continue
    imax = max(I[k])
    fpk = freq[I[k].index(imax)]
    # nearest sample to 200Hz
    i200 = min(range(len(freq)), key=lambda i: abs(freq[i] - f1))
    print(f"{k:>3} {imax:>10.2f} {fpk:>12.0f}   {I[k][i200]:.2f}")

fig, ax = plt.subplots(figsize=(10, 6))
for k in "ABCD":
    ax.plot(freq, I[k], color=COL[k], marker=".", ms=4, lw=1, label=f"I_{k}")
    if I[k]:
        imax = max(I[k])
        fpk = freq[I[k].index(imax)]
        ax.plot([fpk], [imax], "o", color=COL[k], ms=10, mfc="none", mew=2)
ax.set_xlabel("commutation frequency (Hz)  [reconstructed from EASE ramp]")
ax.set_ylabel("coil current @ 100% duty (A)")
ax.set_title(f"{path}\nper-channel MAX current vs frequency (open circles = peak)")
ax.legend()
ax.grid(alpha=0.3)
out = path.rsplit(".", 1)[0] + "_freqsweep.png"
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"wrote {out}")
