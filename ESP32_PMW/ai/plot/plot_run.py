#!/usr/bin/env python3
"""General per-run plotter for a balanced_experiment serial log.

Plots the FULL run (spin-up ramp + tilt) against wall time: drive frequency, the
per-channel carrier duty with the commanded V/f ceiling overlaid, the per-channel
current, and the duty spread. The V/f overlay is what makes "commanded vs actual"
readable -- a duty that should track f/VF_CORNER and doesn't is the shape worth
seeing.

  python3 ai/plot/plot_run.py data/2026-07-14_tilt/tilt_cw_vf_run16.log
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(
    r"^\s*([\d.]+)s\s+t=(\d+) phase=(\d+) freq=([\d.\-]+) \| I\[A\]: "
    r"A=([\d.\-]+) B=([\d.\-]+) C=([\d.\-]+) D=([\d.\-]+) \| duty\[%\]: "
    r"A=([\d.\-]+) B=([\d.\-]+) C=([\d.\-]+) D=([\d.\-]+) \| spread=([\d.\-]+)"
    r"(?:.*?\bihold=([\d.\-]+))?"
)
CH = ["A", "B", "C", "D"]
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PHASE_NAME = {0: "ARMING", 1: "RUNNING", 2: "DONE"}


def parse(path):
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        rows.append(
            dict(
                w=float(m.group(1)),
                phase=int(m.group(3)),
                freq=float(m.group(4)),
                i=[float(m.group(k)) for k in range(5, 9)],
                duty=[float(m.group(k)) for k in range(9, 13)],
                spread=float(m.group(13)),
                ihold=float(m.group(14)) if m.group(14) else 0.0,
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default=None, help="default: <log>.png")
    ap.add_argument("--vf-corner", type=float, default=40.0,
                    help="V/f corner Hz to overlay (0 disables)")
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        raise SystemExit(f"no telemetry lines parsed from {args.log}")
    out = args.out or str(Path(args.log).with_suffix(".png"))

    # Everything is plotted against wall time so the phases line up across panels.
    w = [r["w"] for r in rows]
    run = [r for r in rows if r["phase"] == 1]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(Path(args.log).name, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(w, [r["freq"] for r in rows], color="black", lw=1.5)
    ax.set_ylabel("drive freq [Hz]")
    if args.vf_corner > 0:
        ax.axhline(args.vf_corner, color="tab:purple", ls="--", lw=1)
        ax.text(w[-1], args.vf_corner, " V/f corner", va="center", fontsize=8,
                color="tab:purple")

    ax = axes[1]
    for k in range(4):
        ax.plot(w, [r["duty"][k] for r in rows], color=COLORS[k], lw=1.3,
                label=f"coil {CH[k]}")
    if args.vf_corner > 0 and run:
        # The ceiling the firmware commands: min(f/corner, 1) * 100.
        ax.plot([r["w"] for r in run],
                [min(r["freq"] / args.vf_corner, 1.0) * 100 for r in run],
                color="tab:purple", ls="--", lw=1.2, label="V/f ceiling")
    ax.set_ylabel("carrier duty [%]")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, ncol=5, loc="lower right")

    ax = axes[2]
    for k in range(4):
        ax.plot(w, [r["i"][k] for r in rows], color=COLORS[k], lw=1.3)
    ihold = next((r["ihold"] for r in rows if r["ihold"] > 0), 0.0)
    if ihold:
        ax.axhline(ihold, color="grey", ls=":", lw=1.2)
        ax.text(w[0], ihold, f" frozen hold {ihold:.2f}A", va="bottom", fontsize=8,
                color="grey")
    ax.set_ylabel("current [A]")

    ax = axes[3]
    ax.plot(w, [max(r["duty"]) - min(r["duty"]) for r in rows], color="black",
            lw=1.2, label="duty spread [%]")
    ax.plot(w, [r["spread"] for r in rows], color="tab:red", lw=1.2,
            label="current spread [A]")
    ax.set_ylabel("spread")
    ax.set_xlabel("wall time [s]")
    ax.legend(fontsize=8, loc="upper left")

    # Shade the phases once, across every panel.
    for a in axes:
        a.grid(alpha=0.25)
        for ph, color in ((0, "tab:blue"), (2, "tab:grey")):
            seg = [r["w"] for r in rows if r["phase"] == ph]
            if seg:
                a.axvspan(min(seg), max(seg), color=color, alpha=0.07)
    for ph in (0, 1, 2):
        seg = [r["w"] for r in rows if r["phase"] == ph]
        if seg:
            axes[0].text((min(seg) + max(seg)) / 2, axes[0].get_ylim()[1] * 0.9,
                         PHASE_NAME[ph], ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
