#!/usr/bin/env python3
"""Before/after the minSignalA co-ramp fix: per-channel carrier duty and current
through the 1->200Hz spin-up.

The bug: reset() latches channel A as the anchor, which then ramps open-loop to
its ceiling while the other three sit at their 50% start duty -- because at low
drive frequency every channel reads ~0 A, so the PI targets ~0 and never pulls
them up. One coil at 2x the drive of the others => constant off-axis torque =>
the spinning disk precesses.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"t=(\d+) phase=(\d+) freq=([\d.]+) \| I\[A\]: "
    r"A=([\d.]+) B=([\d.]+) C=([\d.]+) D=([\d.]+) \| duty\[%\]: "
    r"A=([\d.]+) B=([\d.]+) C=([\d.]+) D=([\d.]+) \| spread=([\d.]+).*hold=(\d)"
)
CH = ["A", "B", "C", "D"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def parse(path):
    """Spin-up samples only (phase=1, pre-freeze) -- the region the fix targets."""
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m or m.group(2) != "1" or m.group(13) != "0":
            continue
        rows.append(
            dict(
                t=int(m.group(1)) / 1000.0,
                freq=float(m.group(3)),
                i=[float(m.group(k)) for k in range(4, 8)],
                duty=[float(m.group(k)) for k in range(8, 12)],
                spread=float(m.group(12)),
            )
        )
    return rows


def main(before_log, after_log, out):
    runs = [("Before: anchor ramps alone", parse(before_log)),
            ("After: co-ramp below minSignalA", parse(after_log))]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
    fig.suptitle(
        "Spin-up drive symmetry, 1→200 Hz ramp  —  minSignalA co-ramp fix",
        fontsize=14, fontweight="bold",
    )

    for col, (title, rows) in enumerate(runs):
        f = [r["freq"] for r in rows]

        ax = axes[0][col]
        for k in range(4):
            ax.plot(f, [r["duty"][k] for r in rows], color=COLORS[k], lw=1.4,
                    label=f"coil {CH[k]}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("carrier duty [%]")
        ax.set_ylim(40, 105)
        ax.legend(fontsize=8, ncol=4, loc="lower right")

        ax = axes[1][col]
        dsp = [max(r["duty"]) - min(r["duty"]) for r in rows]
        ax.plot(f, dsp, color="black", lw=1.6)
        ax.fill_between(f, dsp, color="black", alpha=0.12)
        ax.set_ylabel("duty spread [%]\n(drive asymmetry)")
        ax.set_ylim(0, 55)
        peak = max(dsp)
        ax.annotate(f"peak {peak:.0f} pts", xy=(f[dsp.index(peak)], peak),
                    xytext=(0.55, 0.8), textcoords="axes fraction", fontsize=9,
                    arrowprops=dict(arrowstyle="->", lw=1))

        ax = axes[2][col]
        for k in range(4):
            ax.plot(f, [r["i"][k] for r in rows], color=COLORS[k], lw=1.2)
        ax.set_ylabel("current [A]")
        ax.set_xlabel("drive frequency [Hz]")
        ax.set_ylim(0, 3.2)

        for row in range(3):
            a = axes[row][col]
            a.grid(alpha=0.25)
            # Region where the loop is blind: currents below minSignalA.
            a.axvspan(0, 25, color="tab:red", alpha=0.07)
        axes[0][col].text(12.5, 102, "no current signal", ha="center", fontsize=8,
                          style="italic", color="tab:red")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
