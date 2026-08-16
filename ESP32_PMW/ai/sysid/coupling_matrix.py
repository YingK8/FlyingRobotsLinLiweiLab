#!/usr/bin/env python3
"""Extract the mutual-coupling matrix from a solo/pairwise/ALL coupling sweep
(4 solos, 6 pairs, ALL; see main_experiment.cpp + ai/gen_coupling_experiment.py,
which repeats this set at several carrier-duty/current levels for CW and
CCW), optionally across multiple current levels ("current vs coupling").
Each coil's current SHIFT in a pair vs its solo current is caused by the
partner's mutual EMF: same-sign shift on both coils is reactive coupling
(M*cos dphi), opposite-sign is real-power transfer (M*sin dphi, the effect
that imbalances the supplies); shift magnitude ~ coupling coefficient.
Handles the scope<->firmware channel swap (B<->D) found empirically.

Two timing modes:
  --t0/--slot (default): fixed segment centers, ONE current level, matches
    the original main_coupling_test.cpp's 11-segment/5s-slot layout.
  --segments-log LOG: parse main_experiment.cpp's "t=.. label=.." telemetry
    (as captured by ai/trigger_reset_log.py) for per-segment timing
    instead -- required for the JSON-driven sweep, since segment count/
    duration now vary by current level and direction. Produces one matrix
    per current level found in the log, plus a coupling-vs-current-level
    summary/plot.

Usage:
  uv run python ai/coupling_matrix.py CSV [--t0 SEC] [--direction cw|ccw]
  uv run python ai/coupling_matrix.py CSV --segments-log LOG [--direction cw|ccw]
"""
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# firmware drive phases (INITIAL_PHASES {A,B,C,D}) per rotation direction --
# matches JsonPhaseSequencer's setDirection / main_current_pid.cpp's
# PHASES_CW/PHASES_CCW conventions.
PHASES = {
    "cw": {"A": 270.0, "B": 90.0, "C": 180.0, "D": 0.0},
    "ccw": {"A": 90.0, "B": 270.0, "C": 180.0, "D": 0.0},
}
SEG_KEY = ["SOLO_A", "SOLO_B", "SOLO_C", "SOLO_D",
           "PAIR_AB", "PAIR_AC", "PAIR_AD", "PAIR_BC", "PAIR_BD", "PAIR_CD", "ALL"]
PAIR_KEYS = [("A", "B", "PAIR_AB"), ("A", "C", "PAIR_AC"), ("A", "D", "PAIR_AD"),
             ("B", "C", "PAIR_BC"), ("B", "D", "PAIR_BD"), ("C", "D", "PAIR_CD")]

ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("csv")
ap.add_argument("--t0", type=float, default=12.0,
                 help="center of solo-A burst (s) -- legacy fixed-timing mode")
ap.add_argument("--slot", type=float, default=5.0,
                 help="segment slot length (s) -- legacy fixed-timing mode")
ap.add_argument("--half", type=float, default=0.8, help="half-window to average (s)")
ap.add_argument("--segments-log", default=None,
                 help="serial log with main_experiment.cpp's 't=.. label=..' telemetry "
                      "(captured via ai/trigger_reset_log.py) -- use labeled segment "
                      "timing instead of fixed --t0/--slot; required for multi-current-"
                      "level sweeps")
ap.add_argument("--reset-delay", type=float, default=4.0,
                 help="seconds between picoscope_record.py starting and the EN-pulse "
                      "reset (matches ai/run_coupling_sweep.py's --delay) -- offsets "
                      "--segments-log timestamps into the scope's time frame")
ap.add_argument("--settle-s", type=float, default=0.5,
                 help="skip this long after a labeled segment starts before sampling "
                      "(--segments-log mode)")
ap.add_argument("--sample-s", type=float, default=1.5,
                 help="window length to average within each labeled segment "
                      "(--segments-log mode)")
ap.add_argument("--direction", choices=["cw", "ccw"], default="cw",
                 help="rotation direction -> phase map, for dphi/reactive-vs-"
                      "real-power labeling (default: %(default)s)")
args = ap.parse_args()

PHASE = PHASES[args.direction]

df = pd.read_csv(args.csv, skiprows=[1])
t = df["Time"].values
scope = ["Channel A", "Channel B", "Channel C", "Channel D"]
dt = np.median(np.diff(t))
fs = 1 / dt
w = int(round(0.3 * fs))


def rms(x):
    x = x - np.mean(x)
    return np.sqrt(pd.Series(x * x).rolling(w, 1, center=True).mean().values) * 1000


R = {c: rms(df[c].values) for c in scope}


def parse_segments_log(path, reset_delay):
    """Returns {level: {seg_key: (t_start, t_end)}} in the scope's time frame,
    parsed from lines like '  12.34s  t=1234 label=CW_I100_SOLO_A | ...'
    (trigger_reset_log.py's per-line wall-clock prefix + the firmware's own
    label field)."""
    time_prefix = re.compile(r"^\s*([\d.]+)s\s")
    label_pat = re.compile(r"label=(\S+)")
    label_fmt = re.compile(r"[A-Z]+_I(\d+)_(.+)")

    events = []
    for line in open(path):
        tm = time_prefix.match(line)
        lm = label_pat.search(line)
        if tm and lm:
            events.append((float(tm[1]) + reset_delay, lm[1]))
    if not events:
        raise SystemExit(f"no 't=.. label=..' telemetry lines found in {path}")

    by_level = {}
    for i, (t_start, label) in enumerate(events):
        t_end = events[i + 1][0] if i + 1 < len(events) else t_start + 1e9
        m = label_fmt.match(label)
        if not m:
            continue
        level, seg_key = int(m.group(1)), m.group(2)
        by_level.setdefault(level, {})[seg_key] = (t_start, t_end)
    return by_level


def detect_fw2scope(seg_time_fn):
    """Auto-detect firmware coil -> scope channel from the 4 solo segments
    (robust to any coil<->channel rewiring): whichever scope channel
    dominates a solo's window is where that coil's current sense is wired."""
    fw2scope = {}
    used = set()
    for coil in "ABCD":
        t0, t1 = seg_time_fn("SOLO_" + coil)
        m = (t >= t0) & (t <= t1)
        means = {c: np.nanmean(R[c][m]) for c in scope}
        best = max((c for c in scope if c not in used), key=lambda c: means[c])
        fw2scope[coil] = best
        used.add(best)
    return fw2scope


def analyze_one(seg_time_fn, fw2scope, label=""):
    """seg_time_fn(seg_key) -> (t_start, t_end) to average currents over.
    Returns (solo, coup, M): solo currents, per-pair coupling %, 4x4 matrix."""
    def cur(coil, seg_key):
        t0, t1 = seg_time_fn(seg_key)
        m = (t >= t0) & (t <= t1)
        return float(np.nanmean(R[fw2scope[coil]][m]))

    solo = {coil: cur(coil, "SOLO_" + coil) for coil in "ABCD"}
    print(f"{label}solo currents (mV):", {k: round(v) for k, v in solo.items()})

    print(f"{'pair':>6}{'dphi':>6} | per-coil shift vs solo        | type")
    coup = {}
    for i, j, seg_key in PAIR_KEYS:
        ci, cj = cur(i, seg_key), cur(j, seg_key)
        si, sj = ci / solo[i] - 1, cj / solo[j] - 1
        dphi = abs((PHASE[i] - PHASE[j] + 180) % 360 - 180)
        typ = "reactive (same-sign)" if si * sj > 0 else "REAL-POWER transfer (opp-sign)"
        print(f"{i+'+'+j:>6}{dphi:>5.0f}° | {i}:{si*100:+5.1f}%  {j}:{sj*100:+5.1f}%   | {typ}")
        coup[(i, j)] = (abs(si) + abs(sj)) / 2 * 100

    allc = {c: cur(c, "ALL") for c in "ABCD"}
    print("ALL segment (mV):", {k: round(v) for k, v in allc.items()},
          " -> A change vs solo:", f"{allc['A']/solo['A']-1:+.0%}")

    coils = ["A", "B", "C", "D"]
    M = np.full((4, 4), np.nan)
    for (i, j), v in coup.items():
        a, b = coils.index(i), coils.index(j)
        M[a, b] = M[b, a] = v
    print("coupling strength matrix (mean |current shift| %, higher = stronger):")
    print("      " + "".join(f"{c:>7}" for c in coils))
    for a, c in enumerate(coils):
        print(f"  {c}  " + "".join(
            f"{M[a,b]:7.1f}" if not np.isnan(M[a, b]) else f"{'-':>7}" for b in range(4)))
    print()
    return solo, coup, M


def plot_matrix(M, coup, out_path, title_suffix=""):
    coils = ["A", "B", "C", "D"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    im = ax1.imshow(np.nan_to_num(M), cmap="magma", vmin=0)
    ax1.set_xticks(range(4)); ax1.set_xticklabels(coils)
    ax1.set_yticks(range(4)); ax1.set_yticklabels(coils)
    for a in range(4):
        for b in range(4):
            if not np.isnan(M[a, b]):
                ax1.text(b, a, f"{M[a,b]:.0f}%", ha="center", va="center", color="w")
    ax1.set_title(f"Coupling strength (mean |current shift| %){title_suffix}")
    fig.colorbar(im, ax=ax1, label="%")

    pairs = [f"{i}+{j}" for (i, j) in coup]
    vals = [coup[k] for k in coup]
    dphis = [abs((PHASE[i] - PHASE[j] + 180) % 360 - 180) for (i, j) in coup]
    colors = ["#4e79a7" if abs(d - 90) < 1 else "#e15759" for d in dphis]
    ax2.bar(pairs, vals, color=colors)
    ax2.set_ylabel("mean |shift| %")
    ax2.set_title("blue = 90° phase (power transfer)   red = 180° (reactive)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print("wrote", out_path)


if args.segments_log:
    by_level = parse_segments_log(args.segments_log, args.reset_delay)
    print(f"{args.csv} (segments from {args.segments_log}, direction={args.direction})")
    print(f"current levels found: {sorted(by_level)}\n")

    level_coup = {}  # level -> coup dict, for the current-vs-coupling summary
    fw2scope = None
    for level in sorted(by_level):
        windows = by_level[level]

        def seg_time_fn(seg_key, _windows=windows):
            t_start, t_end = _windows[seg_key]
            a = t_start + args.settle_s
            b = min(a + args.sample_s, t_end)
            return a, b

        if fw2scope is None:  # detect once, from the lowest current level present
            fw2scope = detect_fw2scope(seg_time_fn)
            print("detected firmware->scope map:", {k: v[-1] for k, v in fw2scope.items()})
            print()

        print(f"=== current level {level}% ===")
        _, coup, M = analyze_one(seg_time_fn, fw2scope, label=f"[I{level}%] ")
        level_coup[level] = coup
        out = args.csv.rsplit(".", 1)[0] + f"_matrix_I{level}.png"
        plot_matrix(M, coup, out, title_suffix=f" @ {level}% ({args.direction.upper()})")

    levels = sorted(level_coup)
    pair_keys = list(level_coup[levels[0]].keys())
    print("coupling % vs current level:")
    print("  pair  " + "".join(f"{l:>7}%" for l in levels))
    for i, j in pair_keys:
        row = [level_coup[l].get((i, j), float("nan")) for l in levels]
        print(f"  {i}+{j}  " + "".join(f"{v:7.1f} " for v in row))

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, j in pair_keys:
        row = [level_coup[l].get((i, j), float("nan")) for l in levels]
        ax.plot(levels, row, marker="o", label=f"{i}+{j}")
    ax.set_xlabel("carrier duty / current level (%)")
    ax.set_ylabel("coupling strength (mean |shift| %)")
    ax.set_title(f"Coupling vs current level ({args.direction.upper()})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = args.csv.rsplit(".", 1)[0] + f"_coupling_vs_current_{args.direction}.png"
    fig.savefig(out, dpi=110)
    print("\nwrote", out)

else:
    # Legacy fixed-timing mode: ONE current level, matches the original
    # main_coupling_test.cpp's fixed 11-segment/5s-slot layout. Kept as a
    # fallback for pre-migration CSVs that have no labeled serial log.
    def seg_time_fn(seg_key):
        k = SEG_KEY.index(seg_key)
        c = args.t0 + args.slot * k
        return c - args.half, c + args.half

    fw2scope = detect_fw2scope(seg_time_fn)
    print("detected firmware->scope map:", {k: v[-1] for k, v in fw2scope.items()})
    print(args.csv)
    _, coup, M = analyze_one(seg_time_fn, fw2scope)
    out = args.csv.rsplit(".", 1)[0] + "_matrix.png"
    plot_matrix(M, coup, out)
