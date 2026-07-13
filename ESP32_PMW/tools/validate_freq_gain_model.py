#!/usr/bin/env python3
"""Offline validation of freqGainCorrection() (src/FreqGainModel.h) BEFORE
it ever runs on hardware -- see the plan's Phase 3 and progress.md Section 5
for the full derivation.

1. Plots correction(ch, f) = |Z(f)|/R per channel across 1-210Hz (sanity
   check: should be exactly 1.0 at each channel's fitted resonance f0, rise
   on both sides, no NaN/inf).
2. Cross-checks against EXISTING closed-loop logs (no new hardware capture):
   at each logged freq, computes what u_ff would have been WITH vs. WITHOUT
   the correction (mirroring src/CurrentBalanceController.cpp's matvec
   against Bd^-1), and plots both against the ACTUAL total commanded duty
   already in the log. The corrected (increased) curve should track real
   applied duty much more closely than the flat uncorrected one.

Usage:
  uv run python tools/validate_freq_gain_model.py \
      --cw-log state_space_cw_2A_freqgate.log \
      --ccw-log state_space_ccw_2A_freqgate.log \
      --out state_space_freq_gain_validation.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import parse_log

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = list("ABCD")
FREQ_GRID = np.linspace(1.0, 210.0, 400)


def load_rlc(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def correction(rlc: dict, ch: str, f: np.ndarray) -> np.ndarray:
    R, L, C = rlc[ch]["R"], rlc[ch]["L"], rlc[ch]["C"]
    omega = 2.0 * np.pi * f
    x = omega * L - 1.0 / (omega * C)
    z = np.sqrt(R * R + x * x)
    return z / R  # >= 1.0 by construction


def load_ff(model_path: str) -> tuple[np.ndarray, list[str]]:
    with open(model_path) as f:
        model = json.load(f)
    Bd = np.array(model["Bd"])
    return np.linalg.inv(Bd), model["channels"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rlc-fit", default=os.path.join(HERE, "rlc_fit.json"))
    ap.add_argument("--cw-model", default=os.path.join(HERE, "state_space_model.json"))
    ap.add_argument("--ccw-model", default=os.path.join(HERE, "state_space_model_ccw.json"))
    ap.add_argument("--cw-log", default=None)
    ap.add_argument("--ccw-log", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "state_space_freq_gain_validation.png"))
    args = ap.parse_args()

    rlc = load_rlc(args.rlc_fit)

    fig, axes = plt.subplots(3, 1, figsize=(9, 12))

    # --- Panel 1: correction(f) per channel, sanity check ---
    ax = axes[0]
    for ch in CHANNELS:
        corr = correction(rlc, ch, FREQ_GRID)
        ax.plot(FREQ_GRID, corr, label=f"{ch} (f0={rlc[ch]['f0_hz']:.1f}Hz)")
        f0 = rlc[ch]["f0_hz"]
        corr_at_f0 = correction(rlc, ch, np.array([f0]))[0]
        assert abs(corr_at_f0 - 1.0) < 1e-3, f"{ch}: correction at f0 should be 1.0, got {corr_at_f0}"
    ax.set_yscale("log")
    ax.set_xlabel("drive frequency (Hz)")
    ax.set_ylabel("correction = |Z(f)|/R  (log scale)")
    ax.set_title("freqGainCorrection(ch, f) -- should dip to 1.0 at each f0")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    print("[PASS] correction(f0) == 1.0 for all channels (within 1e-3)")
    print(f"correction(1Hz) per channel: "
          f"{ {ch: round(float(correction(rlc, ch, np.array([1.0]))[0]), 1) for ch in CHANNELS} }")
    print(f"correction(60Hz) per channel: "
          f"{ {ch: round(float(correction(rlc, ch, np.array([60.0]))[0]), 2) for ch in CHANNELS} }")

    # --- Panels 2/3: cross-check against real closed-loop logs ---
    for ax, log_path, model_path, label in (
        (axes[1], args.cw_log, args.cw_model, "CW"),
        (axes[2], args.ccw_log, args.ccw_model, "CCW"),
    ):
        if not log_path or not os.path.exists(log_path):
            ax.set_title(f"{label}: no log provided, skipped")
            continue
        FF, channels = load_ff(model_path)
        data = parse_log(log_path)
        freq = data["freq"]
        # r_target isn't in this log format directly usable per-row for older
        # logs; state_space logs carry it as trailing "r=" field via the
        # dir=/kp=... trailer regex, which doesn't match state_space's
        # rmax=/rrate= trailer. Reconstruct r_target from context instead:
        # use each row's own commanded total duty as ground truth and just
        # overlay the STATIC vs CORRECTED feedforward shape (both computed
        # at a fixed r_target=2.0, matching this rig's rmax=2.0 test runs)
        # against real duty -- shape/trend comparison, not exact magnitude.
        r_target_assumed = 2.0
        u_ff_static = np.zeros((len(freq), len(channels)))
        u_ff_corrected = np.zeros((len(freq), len(channels)))
        for i, f in enumerate(freq):
            u_static = FF @ np.full(len(channels), r_target_assumed)
            corr = np.array([correction(rlc, ch, np.array([max(f, 1.0)]))[0] for ch in channels])
            u_ff_static[i] = u_static
            u_ff_corrected[i] = u_static * np.minimum(corr, 10.0)  # matches FREQ_GAIN_CORRECTION_MAX

        real_duty = np.stack([data[f"d_{c.lower()}"] for c in channels], axis=1)
        order = np.argsort(freq)
        ax.plot(freq[order], real_duty.mean(axis=1)[order], "k-", lw=2, label="real mean duty (log)")
        ax.plot(freq[order], u_ff_static.mean(axis=1)[order], "b--", label="u_ff static (uncorrected)")
        ax.plot(freq[order], u_ff_corrected.mean(axis=1)[order], "r--", label="u_ff corrected")
        ax.axhspan(0, 100, alpha=0)
        ax.set_ylim(0, 110)
        ax.set_xlabel("drive frequency (Hz)")
        ax.set_ylabel("duty (%)")
        ax.set_title(f"{label}: real applied duty vs. static/corrected feedforward "
                      f"(r_target={r_target_assumed}A assumed)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
