"""Plot calibrated per-channel current from a duty-ramp capture three ways.

The capture comes from the `test_current` firmware (main_test_current.cpp):
carrier duty ramps 0->100% one board at a time, then shutdown. Segment order
A, C, A+C, B, D, B+D. Two boards: board 1 = A,C ; board 2 = B,D.

Writes to pico/figures/dutyramp/:
  <stem>_individual.png  2x2 grid, one channel per axis
  <stem>_pairs.png       board 1 (A&C) | board 2 (B&D)
  <stem>_all.png         all four channels on one axis

Usage:  python3 scripts/plot_duty_ramp.py <capture.csv>
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.dirname(HERE)

STYLE = {
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 10,
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
    "legend.frameon": False,
}

CH = ["Channel A", "Channel B", "Channel C", "Channel D"]
COL = {"Channel A": "C0", "Channel B": "C1", "Channel C": "C2", "Channel D": "C3"}
# Sense gain A/V, from calibrate_k.py (Tue30Jun cal, offset-pinned fit).
G = {"Channel A": 15.10, "Channel B": 15.10, "Channel C": 15.09, "Channel D": 15.43}
W = 41


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit("need 1 argument: the duty-ramp capture CSV")
    csv = sys.argv[1]
    out_dir = os.path.join(PICO, "figures", "dutyramp")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(csv))[0])

    plt.rcParams.update(STYLE)
    df = pd.read_csv(csv, skiprows=[1])
    t = df["Time"].values

    # Per-channel current: baseline from the idle-gap 5th percentile (alignment
    # agnostic), then drop the rolling-max envelope's noise floor.
    raw, cur = {}, {}
    for ch in CH:
        base = float(np.percentile(df[ch].values, 5))
        raw[ch] = G[ch] * (df[ch].values - base)
        e = env(raw[ch])
        cur[ch] = np.clip(e - float(np.percentile(e, 5)), 0, None)

    def save(fig, suffix):
        fig.tight_layout()
        out = stem + suffix
        fig.savefig(out, facecolor="white")
        print("saved", out)

    # 1) Individual: raw trace (faint) under the current envelope.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, ch in zip(axes.flat, CH):
        ax.plot(t, raw[ch], color=COL[ch], lw=0.4, alpha=0.2)
        ax.plot(t, cur[ch], color=COL[ch], lw=1.6, label=f"{ch}  ({G[ch]:.2f} A/V)")
        ax.set_xlim(t.min(), t.max())
        ax.legend(loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("Current (A)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Duty-ramp current per channel")
    save(fig, "_individual.png")

    # 2) Pairs: board 1 (A&C) | board 2 (B&D).
    pairs = [("board 1: A & C", ["Channel A", "Channel C"]),
             ("board 2: B & D", ["Channel B", "Channel D"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, chs) in zip(axes, pairs):
        for ch in chs:
            ax.plot(t, cur[ch], color=COL[ch], lw=1.6, alpha=0.9, label=ch)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(t.min(), t.max())
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Current (A)")
    save(fig, "_pairs.png")

    # 3) All four on one axis.
    fig, ax = plt.subplots(figsize=(11, 5))
    for ch in CH:
        ax.plot(t, cur[ch], color=COL[ch], lw=1.6, alpha=0.85,
                label=f"{ch}  ({G[ch]:.2f} A/V)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Duty-ramp current, all channels")
    ax.set_xlim(t.min(), t.max())
    ax.legend(loc="upper left", ncol=2)
    save(fig, "_all.png")


if __name__ == "__main__":
    main()
