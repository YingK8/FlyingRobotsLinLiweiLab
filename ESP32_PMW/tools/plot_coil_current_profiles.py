#!/usr/bin/env python3
"""Single-coil vs two-coil vs all-four-coil current profile (including the
on/off current ramps), from a real coupling-sweep capture
(data/2026-07-04a_coupling-matrix/coupling_pairwise_20260704_143320.csv).

That file drives 11 solo/pair/ALL segments back-to-back at one current level
(tools/gen_coupling_experiment.py's fixed 5s-slot layout, segment centers
verified against tools/coupling_matrix.py's --t0=12.0/--slot=5.0 defaults).
This script pulls 7 of those segments -- SOLO_A/B/C/D, PAIR_AC, PAIR_BD, and
ALL -- each with enough margin around its ~3s active window to show the full
turn-on/turn-off current ramp, not just the settled plateau.

Each channel is drawn as a translucent raw trace (the actual noisy CS/scope
voltage) with a solid rolling-RMS envelope on top.

Usage:
  uv run python tools/plot_coil_current_profiles.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
CSV = os.path.join(REPO_ROOT, "data", "2026-07-04a_coupling-matrix",
                    "coupling_pairwise_20260704_143320.csv")
OUT = os.path.join(REPO_ROOT, "results", "single_vs_multi_coil",
                    "coil_current_profile_matrix.png")

# fixed 4-slot categorical order (blue/aqua/yellow/red) -- dataviz skill palette
CHAN_COLOR = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100", "D": "#e34948"}
ENV_WIN_MS = 15  # rolling-RMS envelope window, ~3 commutation cycles at 210Hz

# segment centers (s), per tools/coupling_matrix.py's --t0=12.0/--slot=5.0
# fixed-timing layout: SOLO_A,B,C,D=12,17,22,27; PAIR_AB,AC,AD=32,37,42;
# PAIR_BC,BD,CD=47,52,57; ALL=62.
ROW1 = [  # single-coil
    ("Single coil: A", 12.0),
    ("Single coil: B", 17.0),
    ("Single coil: C", 22.0),
    ("Single coil: D", 27.0),
]
ROW2 = [  # two-coil / all-four
    ("Two coils: A+C", 37.0),
    ("Two coils: B+D", 52.0),
    ("All four coils", 62.0),
]
HALF_WINDOW_S = 2.0  # +/- margin around center -- covers the ~3s active
                      # window plus the abrupt on/off current ramps at its edges


def envelope(x_v: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x_v)
    return np.sqrt((s ** 2).rolling(win, center=True, min_periods=1).mean()).to_numpy()


def draw_panel(ax, df, t, title, center, y_max_box):
    m = (t >= center - HALF_WINDOW_S) & (t <= center + HALF_WINDOW_S)
    t_win = (t[m] - center) * 1000.0  # ms, centered
    for ch in "ABCD":
        raw_mv = df[ch].to_numpy()[m] * 1000.0
        env_mv = envelope(raw_mv, ENV_WIN_MS)
        ax.plot(t_win, raw_mv, color=CHAN_COLOR[ch], alpha=0.25, linewidth=0.6)
        ax.plot(t_win, env_mv, color=CHAN_COLOR[ch], alpha=1.0, linewidth=2.0,
                label=f"channel {ch}")
        y_max_box[0] = max(y_max_box[0], np.nanmax(np.abs(raw_mv)))
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("time relative to segment center [ms]")
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.grid(True, alpha=0.25)


def main() -> None:
    df = pd.read_csv(CSV, skiprows=2, header=None, names=["t", "A", "B", "C", "D"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df["t"].to_numpy()

    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 12, hspace=0.55, wspace=0.35)
    fig.suptitle("Coil current profile, including turn-on/turn-off current ramps:\n"
                 "single coil vs two coils vs all four coils\n"
                 "source: coupling_pairwise_20260704_143320.csv (CW, one current level)",
                 fontsize=15, y=1.01)

    y_max = [0.0]
    row1_axes = []
    for i, (title, center) in enumerate(ROW1):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
        draw_panel(ax, df, t, title, center, y_max)
        row1_axes.append(ax)

    row2_axes = []
    for i, (title, center) in enumerate(ROW2):
        ax = fig.add_subplot(gs[1, i * 4:(i + 1) * 4])
        draw_panel(ax, df, t, title, center, y_max)
        row2_axes.append(ax)

    all_axes = row1_axes + row2_axes
    ylim = y_max[0] * 1.15
    for ax in all_axes:
        ax.set_ylim(-ylim * 0.15, ylim)
    row1_axes[0].set_ylabel("CS voltage [mV]  (proxy for coil current)")
    row2_axes[0].set_ylabel("CS voltage [mV]  (proxy for coil current)")

    handles, labels = row1_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
