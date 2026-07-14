#!/usr/bin/env python3
"""Plot a balanced_experiment run (main_tilt / main_ceiling / ...). Parses the
current-PID telemetry line format emitted by src/balanced_experiment.cpp:
  t=<ms> phase=<n> freq=<Hz> | I[A]: A=.. B=.. C=.. D=.. | duty[%]: A=.. ... | spread=..
Renders current, per-channel carrier duty, and freq/spread vs time.
Usage: python3 tools/plot/plot_balanced_run.py <run.log>
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"t=(\d+) phase=(\d+) freq=([\-\d.]+) \| "
    r"I\[A\]: A=([\-\d.]+) B=([\-\d.]+) C=([\-\d.]+) D=([\-\d.]+) \| "
    r"duty\[%\]: A=([\-\d.]+) B=([\-\d.]+) C=([\-\d.]+) D=([\-\d.]+) \| "
    r"spread=([\-\d.]+)"
)
COL = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}
# Distinct linestyle + marker per channel so lines that overlap exactly (e.g.
# all four at 100% duty in the open-loop ceiling run) stay individually visible.
LS = {"A": "-", "B": "--", "C": "-.", "D": ":"}
MK = {"A": "o", "B": "s", "C": "^", "D": "D"}

path = sys.argv[1]
t, ph, freq, spread = [], [], [], []
I = {k: [] for k in "ABCD"}
duty = {k: [] for k in "ABCD"}
with open(path) as f:
    for ln in f:
        m = LINE.search(ln)
        if not m:
            continue
        g = m.groups()
        t.append(int(g[0]) / 1000.0)
        ph.append(int(g[1]))
        freq.append(float(g[2]))
        for j, k in enumerate("ABCD"):
            I[k].append(float(g[3 + j]))
            duty[k].append(float(g[7 + j]))
        spread.append(float(g[11]))

if not t:
    sys.exit("no telemetry lines parsed")
t0 = t[0]
t = [x - t0 for x in t]

# markevery spaces markers out so they read as line styles, not clutter
me = max(1, len(t) // 25)
fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for k in "ABCD":
    ax[0].plot(t, I[k], color=COL[k], ls=LS[k], marker=MK[k], markevery=me,
               ms=5, label=f"I_{k}")
ax[0].set_ylabel("current (A)")
ax[0].set_title(f"{path}\nmeasured current (top), carrier duty (mid), freq (bot)")
ax[0].legend(ncol=4, fontsize=8); ax[0].grid(alpha=0.3)

for k in "ABCD":
    ax[1].plot(t, duty[k], color=COL[k], ls=LS[k], marker=MK[k], markevery=me,
               ms=5, label=f"duty_{k}")
ax[1].set_ylabel("carrier duty (%)")
ax[1].set_ylim(-5, 108)  # so a line pinned at 100% isn't clipped at the frame
ax[1].legend(ncol=4, fontsize=8); ax[1].grid(alpha=0.3)

ax[2].plot(t, freq, "k-", label="drive freq (Hz)")
ax2b = ax[2].twinx()
ax2b.plot(t, spread, "m-", alpha=0.6, label="spread (A)")
ax2b.set_ylabel("spread (A)", color="m")
ax[2].set_ylabel("freq (Hz)"); ax[2].set_xlabel("time since ARM (s)")
ax[2].legend(loc="upper left", fontsize=8); ax[2].grid(alpha=0.3)

out = path.rsplit(".", 1)[0] + ".png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"wrote {out}")
