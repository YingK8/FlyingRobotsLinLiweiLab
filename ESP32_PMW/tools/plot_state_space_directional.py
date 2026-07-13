#!/usr/bin/env python3
"""Plot two main_state_space.cpp runs (e.g. CW vs CCW at the same r_max) side
by side in one figure -- per-channel current + spread vs time, phase-region
shaded, for visual comparison of direction-specific LQI convergence.

Usage:
  uv run python tools/plot_state_space_directional.py \
      state_space_cw_2A.log state_space_ccw_2A_v2.log \
      --labels CW CCW --out state_space_cw_vs_ccw_2A.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import STATES, parse_log

PHASE_COLORS = {0: "#eeeeee", 1: "#e0f0ff", 2: "#e0ffe0", 3: "#fff0e0", 4: "#f5f5f5"}
CHANNEL_COLORS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green", "D": "tab:red"}


def shade_phases(ax, t, state):
    start = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            ax.axvspan(t[start], t[i - 1] if i < len(state) else t[-1],
                       color=PHASE_COLORS.get(int(state[start]), "white"), alpha=0.5, zorder=0)
            start = i


def plot_one(ax_i, ax_spread, data, label):
    t = data["t"]
    shade_phases(ax_i, t, data["state"])
    shade_phases(ax_spread, t, data["state"])
    for ch in "abcd":
        ax_i.plot(t, data[f"i_{ch}"], label=ch.upper(), color=CHANNEL_COLORS[ch.upper()], linewidth=1.2)
    ax_i.set_title(f"{label}: per-channel current")
    ax_i.set_ylabel("I [A]")
    ax_i.legend(loc="upper left", fontsize=8)
    ax_i.grid(alpha=0.3)

    ax_spread.plot(t, data["spread"], color="black", linewidth=1.2)
    ax_spread.set_title(f"{label}: spread (max-min)")
    ax_spread.set_ylabel("spread [A]")
    ax_spread.set_xlabel("time [s]")
    ax_spread.grid(alpha=0.3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs=2, help="two main_state_space.cpp logs to compare")
    ap.add_argument("--labels", nargs=2, default=["run 1", "run 2"])
    ap.add_argument("--out", default="state_space_directional_comparison.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)
    for col, (log, label) in enumerate(zip(args.logs, args.labels)):
        data = parse_log(log)
        plot_one(axes[0, col], axes[1, col], data, label)

    fig.suptitle("main_state_space.cpp: direction-specific LQI convergence")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
