"""Two single-peak duty-ramp figures from a capture.

  <stem>_A_solo.png    channel A during its solo ramp (0-10 s)
  <stem>_AC_pair.png   channels A and C during the opposed-pair ramp (20-30 s)

Current is I = g * (V_CS - baseline), enveloped as in plot_duty_ramp.py. Both use
segment-relative time so the two peaks line up.

Usage:  python3 scripts/plot_peaks.py <capture.csv>
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
COL = {"Channel A": "C0", "Channel C": "C2"}
G = {"Channel A": 15.10, "Channel C": 15.09}
W = 41


def env(y):
    return (pd.Series(y).rolling(W, center=True, min_periods=1).max()
            .rolling(W, center=True, min_periods=1).mean().values)


def current(df, ch):
    base = float(np.percentile(df[ch].values, 5))
    raw = G[ch] * (df[ch].values - base)
    e = env(raw)
    return raw, np.clip(e - float(np.percentile(e, 5)), 0, None)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("need 1 argument: the duty-ramp capture CSV")
    csv = sys.argv[1]
    out_dir = os.path.join(PICO, "figures", "dutyramp")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(csv))[0])

    plt.rcParams.update(STYLE)
    df = pd.read_csv(csv, skiprows=[1])
    t = df["Time"].values

    def window(start, span=10.6):
        return (t >= start) & (t <= start + span)

    # 1) Channel A solo (segment starts at 0 s).
    m = window(0.0)
    raw, cur = current(df, "Channel A")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t[m], raw[m], color=COL["Channel A"], lw=0.4, alpha=0.2)
    ax.plot(t[m], cur[m], color=COL["Channel A"], lw=1.8, label="Channel A")
    ax.set_xlabel("Time since segment start (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Channel A solo ramp (0-10 s)")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(stem + "_A_solo.png", facecolor="white")
    print("saved", stem + "_A_solo.png")

    # 2) A and C together (opposed-pair segment starts at 20 s). Skip the first
    # 0.5 s so the tail of the preceding C-solo peak (ends at 20 s) is excluded.
    start = 20.0
    m = (t >= start + 0.5) & (t <= start + 10.6)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ch in ["Channel A", "Channel C"]:
        raw, cur = current(df, ch)
        ax.plot(t[m] - start, raw[m], color=COL[ch], lw=0.4, alpha=0.2)
        ax.plot(t[m] - start, cur[m], color=COL[ch], lw=1.8, label=ch)
    ax.set_xlabel("Time since segment start (s)")
    ax.set_ylabel("Current (A)")
    ax.set_title("A and C opposed-pair ramp (20-30 s)")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(stem + "_AC_pair.png", facecolor="white")
    print("saved", stem + "_AC_pair.png")


if __name__ == "__main__":
    main()
