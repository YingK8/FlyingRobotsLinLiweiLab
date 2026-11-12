"""Plot a duty-ramp capture as RAW sense voltages, with segment annotations.

Unlike plot_duty_ramp.py (which gain-scales to current), this plots the raw
PicoScope voltages as captured. Dashed vertical lines mark the firmware segment
boundaries; each segment is labelled with the channel(s) driven during it.

Firmware schedule (main_test_current.cpp): 6 segments, each 2 s off + 8 s ramp,
one board at a time. Board 1 = A,C ; board 2 = B,D.
    A | C | A+C | B | D | B+D   ->  shutdown

Writes to pico/figures/dutyramp/:
  <stem>_raw_individual.png  2x2 grid, one channel per axis
  <stem>_raw_pairs.png       board 1 (A&C) | board 2 (B&D)
  <stem>_raw_all.png         all four channels on one axis

Usage:  python3 scripts/plot_duty_ramp_raw.py <capture.csv>
"""
import os
import sys
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

# Identity map = plot raw columns as captured. (The firmware drives B during the
# 30-40 s segment and D during 40-50 s, but this capture shows the reverse; the
# board-2 scope probes were cross-plugged. Left as-is; swap here if desired.)
SRC_COL = {ch: ch for ch in CH}

GAP_S, RAMP_S = 2.0, 8.0
SLOT_S = GAP_S + RAMP_S
SEG_LABELS = ["A", "C", "A+C", "B", "D", "B+D"]
SEG_ACTIVE = [["Channel A"], ["Channel C"], ["Channel A", "Channel C"],
              ["Channel B"], ["Channel D"], ["Channel B", "Channel D"]]


def annotate_segments(ax, t, labels=False):
    """Light dashed slot boundaries; optional active-channel label per segment."""
    for i in range(len(SEG_LABELS)):
        ax.axvline(i * SLOT_S, color="0.8", ls="--", lw=0.8, zorder=0)
    t_end = len(SEG_LABELS) * SLOT_S
    if t.max() > t_end:
        ax.axvline(t_end, color="0.8", ls="--", lw=0.8, zorder=0)
    if not labels:
        return
    # Add headroom above the data and place labels in it (axes-fraction y).
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y0 + (y1 - y0) * 1.12)
    trans = ax.get_xaxis_transform()
    for i, label in enumerate(SEG_LABELS):
        chs = SEG_ACTIVE[i]
        c = COL[chs[0]] if len(chs) == 1 else "0.3"
        ax.text(i * SLOT_S + SLOT_S / 2.0, 0.98, label, transform=trans,
                ha="center", va="top", fontsize=9, color=c)


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
    raw = {ch: df[SRC_COL[ch]].values for ch in CH}

    def save(fig, suffix):
        fig.tight_layout()
        out = stem + suffix
        fig.savefig(out, facecolor="white")
        print("saved", out)

    # 1) Individual: one raw channel per axis.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for ax, ch in zip(axes.flat, CH):
        ax.plot(t, raw[ch], color=COL[ch], lw=0.5, alpha=0.8, label=ch)
        ax.set_xlim(t.min(), t.max())
        ax.legend(loc="upper right")
        annotate_segments(ax, t)
    for ax in axes[:, 0]:
        ax.set_ylabel("Sense voltage (V)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (s)")
    fig.suptitle("Duty-ramp raw sense voltage per channel")
    save(fig, "_raw_individual.png")

    # 2) Pairs: board 1 (A&C) | board 2 (B&D).
    pairs = [("board 1: A & C", ["Channel A", "Channel C"]),
             ("board 2: B & D", ["Channel B", "Channel D"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (title, chs) in zip(axes, pairs):
        for ch in chs:
            ax.plot(t, raw[ch], color=COL[ch], lw=0.5, alpha=0.75, label=ch)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_xlim(t.min(), t.max())
        ax.legend(loc="upper right")
        annotate_segments(ax, t)
    axes[0].set_ylabel("Sense voltage (V)")
    fig.suptitle("Duty-ramp raw sense voltage by board")
    save(fig, "_raw_pairs.png")

    # 3) All four on one axis.
    fig, ax = plt.subplots(figsize=(12, 5))
    for ch in CH:
        ax.plot(t, raw[ch], color=COL[ch], lw=0.5, alpha=0.7, label=ch)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sense voltage (V)")
    ax.set_title("Duty-ramp raw sense voltage, all channels")
    ax.set_xlim(t.min(), t.max())
    annotate_segments(ax, t, labels=True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    save(fig, "_raw_all.png")


if __name__ == "__main__":
    main()
