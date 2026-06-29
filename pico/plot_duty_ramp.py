"""Plot calibrated per-channel current from a duty-ramp capture three ways.

The capture comes from the `test_current` firmware (main_test_current.cpp):
carrier duty ramps 0->100% in three arrangements with off-gaps, then shutdown:
    {A,B} (1 ch/board) -> {C,D} (1 ch/board) -> {A,B,C,D} (2 ch/board).
Two boards: board 1 = A,C ; board 2 = B,D.

Produces three figures next to the CSV:
  <stem>_individual.png  2x2 grid, one channel per axis
  <stem>_pairs.png       A&C (board 1) | B&D (board 2) side by side
  <stem>_all.png         all four channels on one axis

Usage:  python3 plot_duty_ramp.py <capture.csv>
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CH = ["Channel A", "Channel B", "Channel C", "Channel D"]
COL = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
G = {"Channel A": 15.26, "Channel B": 15.28, "Channel C": 15.57, "Channel D": 15.34}
W = 41


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit("need 1 argument: the duty-ramp capture CSV")
    csv = sys.argv[1]
    stem = os.path.splitext(csv)[0]

    df = pd.read_csv(csv, skiprows=[1])
    t = df["Time"].values

    # Per-channel current. Baseline from a low percentile of the raw signal (the
    # idle gaps), so this is alignment-agnostic, then drop the post-env floor.
    cur = {}
    for ch in CH:
        base = float(np.percentile(df[ch].values, 5))
        e = env(G[ch] * (df[ch].values - base))
        cur[ch] = np.clip(e - float(np.percentile(e, 5)), 0, None)

    name = os.path.basename(csv)

    # 1) Individual — 2x2 grid, raw trace under bold envelope.
    raw = {ch: G[ch] * (df[ch].values - float(np.percentile(df[ch].values, 5)))
           for ch in CH}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, ch in zip(axes.flat, CH):
        ax.plot(t, raw[ch], color=COL[ch], lw=0.4, alpha=0.2)
        ax.plot(t, cur[ch], color=COL[ch], lw=1.8, label=f"{ch}  ({G[ch]:.2f} A/V)")
        ax.set_xlim(t.min(), t.max())
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Current (A)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(f"Duty-ramp current per channel — {name}")
    fig.tight_layout()
    out = stem + "_individual.png"
    fig.savefig(out, dpi=140)
    print("saved", out)

    # 2) Pairs — board 1 (A&C) | board 2 (B&D).
    pairs = [("board 1", ["Channel A", "Channel C"]),
             ("board 2", ["Channel B", "Channel D"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, (title, chs) in zip(axes, pairs):
        for ch in chs:
            ax.plot(t, cur[ch], color=COL[ch], lw=1.8, label=ch)
        ax.set_title(f"{title}: {' & '.join(c[-1] for c in chs)}")
        ax.set_xlabel("Time (s)")
        ax.set_xlim(t.min(), t.max())
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Current (A)")
    fig.suptitle(f"Duty-ramp current by board pair — {name}")
    fig.tight_layout()
    out = stem + "_pairs.png"
    fig.savefig(out, dpi=140)
    print("saved", out)

    # 3) All four on one axis.
    fig, ax = plt.subplots(figsize=(11, 6))
    for ch in CH:
        ax.plot(t, cur[ch], color=COL[ch], lw=1.8, label=f"{ch}  ({G[ch]:.2f} A/V)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title(f"Duty-ramp current, all channels — {name}")
    ax.set_xlim(t.min(), t.max())
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = stem + "_all.png"
    fig.savefig(out, dpi=140)
    print("saved", out)


if __name__ == "__main__":
    main()
