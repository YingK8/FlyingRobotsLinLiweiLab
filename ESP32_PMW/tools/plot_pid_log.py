#!/usr/bin/env python3
"""Plot main_current_pid.cpp telemetry from a serial log (no PicoScope needed).

Lines look like (emitted at ~2 Hz):
  t=12345 phase=3 freq=142.1 | I[A]: A=4.98 B=4.91 C=5.03 D=5.10 | duty[%]: A=55.0 B=52.0 C=54.0 D=50.0 | spread=0.190

Log with tools/trigger_reset_log.py, e.g.:
  uv run python tools/trigger_reset_log.py --log-seconds 65 --out current_pid_run.log
  uv run python tools/plot_pid_log.py current_pid_run.log

Also prints a machine-parsable "METRICS ..." summary line (see
tools/pid_metrics.py) that tools/pid_autotune.py scores across trials.

Usage: uv run python tools/plot_pid_log.py LOG
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import STATES, HOLD_PHASE, compute_metrics, format_metrics_line, parse_log

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--spread-limit", type=float, default=0.4,
                 help="validation bar in amps (default 0.4: the balance goal is "
                      "the four channel currents within 0.4A for any schedule)")
args = ap.parse_args()

data = parse_log(args.log)
t, state, freq = data["t"], data["state"], data["freq"]
i_a, i_b, i_c, i_d = data["i_a"], data["i_b"], data["i_c"], data["i_d"]
d_a, d_b, d_c, d_d = data["d_a"], data["d_b"], data["d_c"], data["d_d"]
spread = data["spread"]

print(f"{args.log}: {len(t)} telemetry samples over {t[-1]:.1f}s")
for s in sorted(set(state)):
    print(f"  phase {s} {STATES.get(s, '?'):<8} {sum(state == s):>4} samples")

hold_mask = state == HOLD_PHASE
if hold_mask.any():
    hold_max_spread = spread[hold_mask].max()
    verdict = "PASS" if hold_max_spread < args.spread_limit else "FAIL"
    print(f"  max spread during HOLD = {hold_max_spread:.3f}A -> {verdict} "
          f"vs {args.spread_limit}A bar (full HOLD, incl. settling transient)")

    # The start of HOLD is still the tail of RAMP_UP's settling transient, not
    # steady-state -- a separate settled-window check (last 10s of HOLD) is a
    # fairer read on whether the controller has actually converged.
    hold_t = t[hold_mask]
    settled_mask = hold_mask & (t >= hold_t[-1] - 10.0)
    if settled_mask.any():
        settled_max = spread[settled_mask].max()
        settled_verdict = "PASS" if settled_max < args.spread_limit else "FAIL"
        print(f"  max spread in LAST 10s of HOLD = {settled_max:.3f}A -> "
              f"{settled_verdict} vs {args.spread_limit}A bar (settled, "
              f"excludes the post-ramp settling transient)")
else:
    print("  no HOLD samples captured -- log too short or run didn't reach HOLD")

metrics = compute_metrics(data, spread_limit=args.spread_limit)
print(format_metrics_line(metrics))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

cmap = plt.get_cmap("Pastel2")
seen = set()
start = 0
for i in range(1, len(state) + 1):
    if i == len(state) or state[i] != state[start]:
        s = state[start]
        lbl = STATES.get(s, str(s)) if s not in seen else None
        for ax in (ax1, ax2, ax3, ax4):
            ax.axvspan(t[start], t[i - 1], color=cmap(s % 8), alpha=0.5,
                       label=lbl if ax is ax1 else None)
        seen.add(s)
        start = i

ax1.plot(t, freq, color="#1f77b4", lw=1.8, label="drive freq (Hz)")
ax1.set_ylabel("drive frequency (Hz)")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left", fontsize=8, framealpha=0.9, title="phase / signal")

for name, series in (("A", i_a), ("B", i_b), ("C", i_c), ("D", i_d)):
    ax2.plot(t, series, label=name, lw=1.4)
ax2.set_ylabel("i_meas (A)")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper left", fontsize=8, ncol=4)

for name, series in (("A", d_a), ("B", d_b), ("C", d_c), ("D", d_d)):
    ax3.plot(t, series, label=name, lw=1.4)
ax3.set_ylabel("duty (%)")
ax3.grid(alpha=0.3)
ax3.legend(loc="upper left", fontsize=8, ncol=4)

ax4.plot(t, spread, color="#d62728", lw=1.6, label="spread = max-min(i_meas)")
ax4.axhline(args.spread_limit, color="k", ls="--", lw=1.2,
            label=f"{args.spread_limit}A validation bar")
ax4.set_ylabel("spread (A)")
ax4.set_xlabel("time (s)")
ax4.grid(alpha=0.3)
ax4.legend(loc="upper left", fontsize=8)

fig.suptitle(f"main_current_pid.cpp telemetry — {args.log}")
fig.tight_layout()
out = args.log.rsplit(".", 1)[0] + "_pid_telemetry.png"
fig.savefig(out, dpi=110)
print("wrote", out)
