#!/usr/bin/env python3
"""Plot ESP32 tilt telemetry from a serial log.

Lines look like:  state=1 freq=7.2 target=100.0 ledc0=0 ledc3=0
emitted at ~1 Hz, so line order = time in seconds. Plots drive frequency and
duty target over time, with the sweep-state as a shaded background band.

Usage: uv run python tools/plot_serial_log.py LOG
"""
import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("log")
args = ap.parse_args()

STATES = {0: "IDLE", 1: "RAMP_UP", 2: "HOLD", 3: "TRANSITION",
          4: "DONE", 5: "ENDING", 6: "STOPPED"}
pat = re.compile(r"state=(\d+)\s+freq=([\d.]+)\s+target=([\d.]+)"
                 r"(?:\s+ledc0=(\d+)\s+ledc3=(\d+))?")

rows = []
for line in open(args.log):
    m = pat.search(line)
    if m:
        rows.append((int(m[1]), float(m[2]), float(m[3])))
if not rows:
    raise SystemExit("no telemetry lines found")
state = np.array([r[0] for r in rows])
freq = np.array([r[1] for r in rows])
target = np.array([r[2] for r in rows])
t = np.arange(len(rows))  # ~1 Hz -> seconds

print(f"{args.log}: {len(rows)} telemetry samples (~{len(rows)}s)")
for s in sorted(set(state)):
    print(f"  state {s} {STATES.get(s,'?'):<11} {np.sum(state==s):>4} s")

fig, ax1 = plt.subplots(figsize=(12, 6))

# shade state regions
cmap = plt.get_cmap("Pastel2")
seen = set()
start = 0
for i in range(1, len(state) + 1):
    if i == len(state) or state[i] != state[start]:
        s = state[start]
        lbl = STATES.get(s, str(s)) if s not in seen else None
        ax1.axvspan(t[start], t[i-1] + 1, color=cmap(s % 8), alpha=0.6, label=lbl)
        seen.add(s)
        start = i

ax1.plot(t, freq, color="#1f77b4", lw=1.8, label="drive freq (Hz)")
ax1.set_xlabel("time (s)"); ax1.set_ylabel("drive frequency (Hz)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_xlim(0, t[-1]); ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t, target, color="#d62728", lw=1.8, label="duty target (%)")
ax2.set_ylabel("duty target (%)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.set_ylim(-5, 105)

ax1.set_title(f"ESP32 tilt telemetry — {args.log}")
ax1.legend(loc="center left", fontsize=8, framealpha=0.9, title="state / signal")
fig.tight_layout()
out = args.log.rsplit(".", 1)[0] + "_telemetry.png"
fig.savefig(out, dpi=110)
print("wrote", out)
