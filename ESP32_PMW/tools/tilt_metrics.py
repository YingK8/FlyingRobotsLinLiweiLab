#!/usr/bin/env python3
"""Utilization + scatter metrics for the balanced tilt/lift firmwares, as
functions of drive frequency.

Two inputs, both serial logs in the current-PID telemetry format emitted by
src/balanced_experiment.cpp (parsed by pid_metrics.parse_log -- frequency is
logged on every line, so no ease-ramp reconstruction is needed):

  --ceiling   an OPEN-LOOP run (env `ceiling`, ceiling_sweep.json: all four
              channels at 100% carrier across a 1->210Hz ramp). Gives each
              channel's peak unregulated current vs frequency, Ceiling_i(f).
  --regulated a PI-BALANCED run (e.g. env `tilt`). Gives the regulated per-
              channel currents Ibar_i(f).

Metrics per frequency bin:
  Utilization  U_i(f) = Ibar_i(f) / Ceiling_i(f)   (per channel, mean, min)
               The anchor (weakest) channel sits near U~=1; stronger channels
               are balanced DOWN so U<1 by design.
  Scatter      S(f)   = max_i Ibar_i(f) - min_i Ibar_i(f)   (amps)

Prints a per-frequency table + a METRICS line, and writes a U(f)/S(f) plot.

Usage (from ESP32_PMW/):
  uv run python tools/tilt_metrics.py --regulated tilt.log --ceiling ceiling.log

A full U(f) curve needs the regulated run to actually drive current across the
sweep. tilt.json drives current only during its step-down at ~210Hz, so its
regulated log yields metrics at ~210Hz; for a swept curve, run a regulated
frequency sweep (e.g. ceiling_sweep.json flashed with piEnabled=true).
"""
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import parse_log, RESONANCE_BAND_HZ  # noqa: E402

# Fixed channel identity/colors, matching the other tools/plot_*.py scripts.
CHANNELS = ["A", "B", "C", "D"]
COLORS = {"A": "#e15759", "B": "#4e79a7", "C": "#59a14f", "D": "#f28e2b"}
IKEYS = ["i_a", "i_b", "i_c", "i_d"]


def _stack(data):
    """(freq[M], currents[M,4]) from a parsed telemetry dict."""
    freq = data["freq"]
    currents = np.column_stack([data[k] for k in IKEYS])
    return freq, currents


def bin_by_freq(freq, currents, bin_hz, fmin, fmax, min_current, settle_frac):
    """Group samples into fixed-width frequency bins. Within each bin keep the
    last `settle_frac` fraction of samples (post-transient, like
    fit_rlc_model's per-label settle window) and average them. Bins whose peak
    current never clears `min_current` (coils off / arming) are dropped.

    Returns (centers[K], means[K,4]) sorted by frequency.
    """
    keep = (freq >= fmin) & (freq <= fmax)
    freq, currents = freq[keep], currents[keep]
    if freq.size == 0:
        return np.array([]), np.zeros((0, 4))

    edges = np.arange(fmin, fmax + bin_hz, bin_hz)
    idx = np.clip(np.digitize(freq, edges) - 1, 0, len(edges) - 2)

    centers, means = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        rows = currents[sel]
        k0 = int(n * (1.0 - settle_frac))  # keep the settled tail of the bin
        rows = rows[k0:] if k0 < n else rows
        m = rows.mean(axis=0)
        if m.max() < min_current:
            continue  # nothing meaningfully driven in this bin
        centers.append(0.5 * (edges[b] + edges[b + 1]))
        means.append(m)
    if not centers:
        return np.array([]), np.zeros((0, 4))
    order = np.argsort(centers)
    return np.array(centers)[order], np.array(means)[order]


def ceiling_curve_from_log(path, bin_hz, fmin, fmax, min_current, settle_frac):
    freq, currents = _stack(parse_log(path))
    return bin_by_freq(freq, currents, bin_hz, fmin, fmax, min_current, settle_frac)


def ceiling_at(f, centers, ceil):
    """Per-channel linear interpolation of the ceiling curve at frequencies f."""
    out = np.empty((len(f), 4))
    for c in range(4):
        out[:, c] = np.interp(f, centers, ceil[:, c])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regulated", required=True,
                    help="PI-balanced run log (Ibar_i(f))")
    ap.add_argument("--ceiling", required=True,
                    help="open-loop 100%%-duty run log (Ceiling_i(f))")
    ap.add_argument("--bin-hz", type=float, default=5.0,
                    help="frequency bin width in Hz (default 5)")
    ap.add_argument("--fmin", type=float, default=1.0)
    ap.add_argument("--fmax", type=float, default=210.0)
    ap.add_argument("--min-current", type=float, default=0.05,
                    help="drop bins whose peak channel current is below this (A)")
    ap.add_argument("--settle-frac", type=float, default=0.5,
                    help="fraction of each bin's tail to average (default 0.5)")
    ap.add_argument("--out", default=None, help="output PNG (default <regulated>_tiltmetrics.png)")
    args = ap.parse_args()

    ceil_f, ceil = ceiling_curve_from_log(
        args.ceiling, args.bin_hz, args.fmin, args.fmax, args.min_current,
        args.settle_frac)
    reg_f, reg = bin_by_freq(*_stack(parse_log(args.regulated)),
                             bin_hz=args.bin_hz, fmin=args.fmin, fmax=args.fmax,
                             min_current=args.min_current,
                             settle_frac=args.settle_frac)
    if ceil_f.size == 0:
        raise SystemExit(f"no usable ceiling samples in {args.ceiling}")
    if reg_f.size == 0:
        raise SystemExit(f"no usable regulated samples in {args.regulated}")

    ceil_at_reg = ceiling_at(reg_f, ceil_f, ceil)
    util = reg / np.where(ceil_at_reg > 1e-6, ceil_at_reg, np.nan)  # U_i(f), [K,4]
    u_mean = np.nanmean(util, axis=1)
    u_min = np.nanmin(util, axis=1)
    scatter = reg.max(axis=1) - reg.min(axis=1)  # S(f)

    # Table.
    print(f"# regulated={os.path.basename(args.regulated)} "
          f"ceiling={os.path.basename(args.ceiling)} bin={args.bin_hz}Hz")
    print(f"# {'f_Hz':>6} | {'Ibar_A':>28} | {'U_A/B/C/D':>28} | "
          f"{'U_mean':>7} {'U_min':>7} {'S_A':>7}")
    for i, f in enumerate(reg_f):
        ib = " ".join(f"{reg[i, c]:6.2f}" for c in range(4))
        uu = " ".join(f"{util[i, c]:6.2f}" for c in range(4))
        print(f"  {f:6.1f} | {ib:>28} | {uu:>28} | "
              f"{u_mean[i]:7.2f} {u_min[i]:7.2f} {scatter[i]:7.3f}")

    # Headline: best band = highest U_min among bins whose scatter is under the
    # 0.4A balance bar (fall back to plain argmax U_min if none qualify).
    ok = scatter <= 0.4
    pool = np.where(ok)[0] if ok.any() else np.arange(len(reg_f))
    best = pool[np.argmax(u_min[pool])]
    print(f"METRICS best_freq_hz={reg_f[best]:.1f} "
          f"u_min_at_best={u_min[best]:.3f} s_at_best={scatter[best]:.3f} "
          f"u_mean_overall={np.nanmean(u_mean):.3f} s_max={scatter.max():.3f}")

    # Plot: U(f) per channel + U_min, and S(f) with the 0.4A bar.
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for c, ch in enumerate(CHANNELS):
        ax1.plot(reg_f, util[:, c], color=COLORS[ch], marker=".", lw=1.4,
                 label=f"U_{ch}")
    ax1.plot(reg_f, u_min, color="k", lw=1.8, ls="--", label="U_min")
    ax1.axhline(1.0, color="gray", lw=1.0, ls=":")
    ax1.axvspan(*RESONANCE_BAND_HZ, color="0.9", zorder=0,
                label=f"resonance {RESONANCE_BAND_HZ[0]:.0f}-{RESONANCE_BAND_HZ[1]:.0f}Hz")
    ax1.set_ylabel("utilization U = Ibar / Ceiling")
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=3, fontsize=8)

    ax2.plot(reg_f, scatter, color="#d62728", lw=1.8, marker=".",
             label="S = max-min(Ibar)")
    ax2.axhline(0.4, color="k", ls="--", lw=1.2, label="0.4A balance bar")
    ax2.axvspan(*RESONANCE_BAND_HZ, color="0.9", zorder=0)
    ax2.set_ylabel("scatter S (A)")
    ax2.set_xlabel("drive frequency (Hz)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    out = args.out or (args.regulated.rsplit(".", 1)[0] + "_tiltmetrics.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
