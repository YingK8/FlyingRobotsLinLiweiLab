#!/usr/bin/env python3
"""Re-recorded design-B takes at 80-110 Hz against the median trajectory from before.

Top row: angle of the average axis to its final axis; bottom row: cone half-angle -- the two
traces `compare_designs.traces` pools, per take from `alignment_rate.settle_take`. Per frequency:

  solid ink   median of every ORIGINAL take (recorded before the 2026-09-13 re-record session),
              i.e. what traces_1_vs_B.png showed before the exclusions;
  dashed ink  median of the originals still marked ok;
  thin lines  each re-recorded take, coloured by its spin sense on the spin-up
              (report/spin_sense_after_rerecord.csv, ramp log10 power ratio, +/-0.15 cut).

    uv run python ai/alignment/rerecord_traces.py
"""

import collections
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from controller.control import alignment_rate as ar      # noqa: E402
from controller.control import compare_designs as cd     # noqa: E402

ROOT = REPO / "results/alignment_rate/20260910_droneB"
OUT = ROOT / "report/rerecord_vs_before.png"
FREQS = (80, 90, 100, 110)
#: The re-record session's first take is 2026-09-13_101717; every original predates it.
NEW_FROM = "2026-09-13_10"
#: |log10(P+/P-)| below this on the ramp is "no coherent spin" (spin_sense validation, 20-40 Hz).
CUT = 0.15
#: Takes with too few solved frames for the default ramp window, read with RAMP_MIN_N = 60.
RETRY = {"2026-09-13_104421": -0.13, "2026-09-13_105703": 0.44}
#: Categorical slots 1-3 of the dataviz reference palette, in fixed order, validated.
SENSE = {"CCW (normal)": cd.COLORS[0], "CW": cd.COLORS[1], "no coherent spin": "#1baf7a"}


def verdict(ratio):
    if ratio is None:
        return None
    return "CCW (normal)" if ratio >= CUT else "CW" if ratio <= -CUT else "no coherent spin"


def ramp_ratios():
    out = {r["take"]: float(r["ramp_log_ratio"])
           for r in csv.DictReader(open(ROOT / "report/spin_sense_after_rerecord.csv"))
           if r["ramp_log_ratio"] != ""}
    out.update(RETRY)
    return out


def rows_for(f):
    groups = {"orig_all": [], "orig_kept": [], "new": [], "undrawn": [], "table": []}
    for r in csv.DictReader(open(ROOT / f"{f:03d}hz/index.csv")):
        take = Path(r["flight"])
        new = take.name >= NEW_FROM
        usable = r["outcome"] == "ok" or (not new and r["outcome"].startswith("excluded"))
        if not usable or not (take / "axis.csv").exists():
            continue
        got = ar.settle_take(take, r["log"], float(f))
        for g in got:
            g["_take"] = take.name
        if new:
            groups["table"].append({"freq_hz": f, "repeat": r["repeat"], "take": take.name,
                                    "drawn": bool(got)})
        if new and not got:
            # settle_take finds no usable kill: named in the panel title, never silently dropped
            groups["undrawn"].append(f"r{r['repeat']}")
        if new:
            groups["new"] += got
        else:
            groups["orig_all"] += got
            if r["outcome"] == "ok":
                groups["orig_kept"] += got
    return groups


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sense = ramp_ratios()
    data = {f: rows_for(f) for f in FREQS}
    freqs = [f for f in FREQS if data[f]["new"]]
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": cd.GRID, "axes.labelcolor": cd.INK_2,
                         "xtick.color": cd.INK_2, "ytick.color": cd.INK_2, "text.color": cd.INK})
    fig, axes = plt.subplots(2, len(freqs), figsize=(4.4 * len(freqs) + 1.0, 7.8), sharex=True,
                             sharey="row", facecolor=cd.SURFACE, squeeze=False)
    spec = (("delta", "average axis: angle to final axis (deg)"),
            ("env", "cone ('precession') half-angle (deg)"))
    drawn = {"delta": [], "env": []}
    for j, f in enumerate(freqs):
        ff = float(f)
        grid, s_all = cd.trace_stats(data[f]["orig_all"])
        _, s_kept = cd.trace_stats(data[f]["orig_kept"])
        counts = collections.Counter()
        for i, (key, ylab) in enumerate(spec):
            ax = axes[i][j]
            ax.set_facecolor(cd.SURFACE)
            ax.grid(True, color=cd.GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.axvline(0.0, color=cd.INK_2, linewidth=1.0, zorder=1)
            for row in data[f]["new"]:
                _, one = cd.trace_stats([row])
                if ff not in one:
                    continue
                v = verdict(sense.get(row["_take"]))
                y = one[ff]["reps_" + key][0]
                ax.plot(grid, y, color=SENSE.get(v, cd.INK_2), linewidth=1.2, alpha=0.9, zorder=3)
                drawn[key].append(y)
                if i == 0:
                    counts[v or "unread"] += 1
            if ff in s_all:
                ax.plot(grid, s_all[ff][key][0], color=cd.INK, linewidth=2.4, zorder=4)
                drawn[key].append(s_all[ff][key][0])
            if ff in s_kept:
                ax.plot(grid, s_kept[ff][key][0], color=cd.INK, linewidth=1.6, linestyle="--",
                        zorder=4)
            if i == 0:
                n_all = s_all[ff]["n"] if ff in s_all else 0
                n_kept = s_kept[ff]["n"] if ff in s_kept else 0
                tally = ", ".join(f"{n} {k}" for k, n in counts.items())
                miss = data[f]["undrawn"]
                ax.set_title(f"{f} Hz   before: {n_all} takes ({n_kept} kept)\nnew: {tally}"
                             + (f"\nnot drawn, no usable kill: {' '.join(miss)}" if miss else ""),
                             loc="left", fontsize=10, color=cd.INK, pad=6)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("time from the cut (s)")
    for i, (key, _) in enumerate(spec):
        allv = np.concatenate([np.ravel(y) for y in drawn[key]])
        axes[i][0].set_ylim(0.0, 1.08 * float(np.nanpercentile(allv, 99)))
    handles = [Line2D([], [], color=c, linewidth=1.6, label=f"new take: {k}")
               for k, c in SENSE.items()]
    handles += [Line2D([], [], color=cd.INK, linewidth=2.4, label="median, all original takes"),
                Line2D([], [], color=cd.INK, linewidth=1.6, linestyle="--",
                       label="median, originals kept after the exclusions")]
    # Legend on its own row under the title: beside it, it ran into the title at this width.
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.004, 0.955))
    fig.suptitle("Design B: re-recorded takes against the median trajectory from before",
                 x=0.008, ha="left", y=0.99, fontsize=13, color=cd.INK)
    fig.text(0.008, 0.008,
             "Original takes = recorded before the 2026-09-13 re-record session; 'kept' = still "
             "marked ok after 14 were excluded for spin sense or no response. A median is drawn "
             "only where at least half its takes still run, so a 2-take 'kept' median can be "
             "missing. Spin sense of each new take from the spin-up (15-35 Hz drive, log10 power "
             "ratio +/-0.15; ai/alignment/spin_sense.py). y clipped at the 99th percentile.",
             fontsize=8.5, color=cd.INK_2, ha="left", va="bottom", wrap=True)
    # The table view the figure owes its colours (aqua is 2.74:1 on the surface): every new
    # take, drawn or not, with the number its colour stands for.
    with open(OUT.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["freq_hz", "repeat", "take", "drawn",
                                           "ramp_log_ratio", "verdict"])
        w.writeheader()
        for f in freqs:
            for t in data[f]["table"]:
                ratio = sense.get(t["take"])
                w.writerow({**t, "ramp_log_ratio": "" if ratio is None else ratio,
                            "verdict": verdict(ratio) or "unread"})
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(OUT, dpi=150, facecolor=cd.SURFACE)
    plt.close(fig)
    print(f"-> {OUT}")
    for f in freqs:
        print(f"  {f} Hz: {len(data[f]['orig_all'])} original, {len(data[f]['orig_kept'])} kept, "
              f"{len(data[f]['new'])} new")


if __name__ == "__main__":
    assert verdict(0.2) == "CCW (normal)" and verdict(-0.2) == "CW"
    assert verdict(0.0) == "no coherent spin" and verdict(None) is None
    main()
