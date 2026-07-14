#!/usr/bin/env python3
"""Plot a main_tilt_pi balance run: parses the telemetry lines
(t / carrier% / ref / per-channel I / per-channel duty) main_tilt_pi.cpp prints
and renders current-balance + duty + carrier/ref panels.
Usage: uv run python tools/plot/plot_tilt_balance.py <run.log>
"""
import re
import sys

import matplotlib.pyplot as plt

LINE = re.compile(
    r"t=(\d+) \| carrier=([\d.]+)% ref=([\-\d.]+)A \| "
    r"I\[A\]: A=([\-\d.]+) B=([\-\d.]+) C=([\-\d.]+) D=([\-\d.]+) \| "
    r"duty\[%\]: A=([\-\d.]+) B=([\-\d.]+) C=([\-\d.]+) D=([\-\d.]+)"
)
COL = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}

path = sys.argv[1]
t, carrier, ref = [], [], []
I = {k: [] for k in "ABCD"}
duty = {k: [] for k in "ABCD"}
with open(path) as f:
    for ln in f:
        m = LINE.search(ln)
        if not m:
            continue
        g = m.groups()
        t.append(int(g[0]) / 1000.0)
        carrier.append(float(g[1]))
        ref.append(float(g[2]))
        for j, k in enumerate("ABCD"):
            I[k].append(float(g[3 + j]))
            duty[k].append(float(g[7 + j]))

t0 = t[0]
t = [x - t0 for x in t]

fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

# Panel 1: measured current vs the shared balanced reference
for k in "ABCD":
    ax[0].plot(t, I[k], color=COL[k], label=f"I_{k}")
ax[0].plot(t, ref, "k--", lw=1.5, label="reference (all channels)")
ax[0].set_ylabel("current (A)")
ax[0].set_title(f"{path}\nbalance mode [1,1,1,1]: all channels track the dashed reference")
ax[0].legend(ncol=5, fontsize=8)
ax[0].grid(alpha=0.3)

# Panel 2: per-channel carrier duty (weakest channel should sit highest)
for k in "ABCD":
    ax[1].plot(t, duty[k], color=COL[k], label=f"duty_{k}")
ax[1].set_ylabel("carrier duty (%)")
ax[1].legend(ncol=4, fontsize=8)
ax[1].grid(alpha=0.3)

# Panel 3: scheduled carrier % and reference (amps)
ax[2].plot(t, carrier, "k-", label="scheduled carrier %")
ax[2].plot(t, ref, "c-", label="reference = (carrier/100) x I_TARGET_A (A)")
ax[2].set_ylabel("carrier% / amps")
ax[2].set_xlabel("time since ARMED (s)")
ax[2].legend(ncol=2, fontsize=8)
ax[2].grid(alpha=0.3)

out = path.rsplit(".", 1)[0] + ".png"
fig.tight_layout()
fig.savefig(out, dpi=110)
print(f"wrote {out}")
