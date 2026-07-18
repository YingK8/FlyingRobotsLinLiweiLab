#!/usr/bin/env python3
"""Preliminary coil-decoupling feedforward compensation tuning: baseline vs
successive trim iterations, as a matrix of subplots (one panel per stage).

Two independent tuning sequences exist in data/ (see project memory
coupling-compensation.md):
  - CCW recalibration (2026-07-04e): iter0 = untrimmed baseline, iter1/iter2 =
    successive per-channel duty-trim iterations, converging the 4-channel
    spread from ~2.0 down toward ~1.0.
  - CW feedforward calibration (2026-07-04b): comp_ff_iter1/iter2, an earlier
    trim pass on the CW config.

Each panel plots all 4 channels' full-capture current profile: translucent
raw CS/scope voltage, solid rolling-RMS envelope. The per-panel title reports
the whole-capture RMS spread (max/min across channels) as a quick tuning-
progress number -- NOT the exact mid-capture reference-burst numbers quoted
in project memory (those were computed on a specific averaging window from
the original tuning session; this script recomputes a whole-capture proxy
directly from the raw files so the number and the plot are self-consistent).

Usage:
  uv run python ai/plot_pid_tuning_matrix.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
OUT_DIR = os.path.join(REPO_ROOT, "results", "pid_decoupling_tuning")

CHAN_COLOR = {"A": "#2a78d6", "B": "#1baf7a", "C": "#eda100", "D": "#e34948"}
ENV_WIN_MS = 15

CCW_STAGES = [
    ("Baseline (no tuning)", "data/2026-07-04e_recalibration-ccw/comp_ccw_iter0.csv"),
    ("Tune 1", "data/2026-07-04e_recalibration-ccw/comp_ccw_iter1.csv"),
    ("Tune 2 (converged)", "data/2026-07-04e_recalibration-ccw/comp_ccw_iter2.csv"),
]
CW_STAGES = [
    ("Tune 1", "data/2026-07-04b_comp-calibration-cw/comp_ff_iter1.csv"),
    ("Tune 2 (converged)", "data/2026-07-04b_comp-calibration-cw/comp_ff_iter2.csv"),
]


def envelope(x_v: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(x_v)
    return np.sqrt((s ** 2).rolling(win, center=True, min_periods=1).mean()).to_numpy()


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(REPO_ROOT, path), skiprows=2, header=None,
                      names=["t", "A", "B", "C", "D"])
    return df.apply(pd.to_numeric, errors="coerce").dropna()


def spread(df: pd.DataFrame) -> float:
    rms = {ch: float(np.sqrt(np.mean((df[ch].to_numpy() * 1000.0) ** 2))) for ch in "ABCD"}
    return max(rms.values()) / min(rms.values())


def plot_matrix(stages: list[tuple[str, str]], suptitle: str, out_path: str) -> None:
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(suptitle, fontsize=14, y=1.03)

    y_max = 0.0
    for ax, (label, path) in zip(axes, stages):
        df = load(path)
        t_ms = (df["t"].to_numpy() - df["t"].to_numpy()[0]) * 1000.0
        for ch in "ABCD":
            raw_mv = df[ch].to_numpy() * 1000.0
            env_mv = envelope(raw_mv, ENV_WIN_MS)
            ax.plot(t_ms, raw_mv, color=CHAN_COLOR[ch], alpha=0.20, linewidth=0.5)
            ax.plot(t_ms, env_mv, color=CHAN_COLOR[ch], alpha=1.0, linewidth=1.8,
                     label=f"channel {ch}")
            y_max = max(y_max, np.nanmax(raw_mv))
        s = spread(df)
        ax.set_title(f"{label}\n({os.path.basename(path)}, spread max/min = {s:.2f})",
                     fontsize=11, pad=10)
        ax.set_xlabel("time [ms]")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("CS voltage [mV]  (proxy for coil current)")
    for ax in axes:
        ax.set_ylim(0, y_max * 1.15)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=(0, 0.02, 1, 0.90))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_matrix(CCW_STAGES,
                "Coil-decoupling feedforward compensation tuning (CCW)\n"
                "data/2026-07-04e_recalibration-ccw",
                os.path.join(OUT_DIR, "ccw_compensation_tuning_matrix.png"))
    plot_matrix(CW_STAGES,
                "Coil-decoupling feedforward compensation tuning (CW)\n"
                "data/2026-07-04b_comp-calibration-cw -- no separate untrimmed-baseline "
                "capture exists for this sequence, only iter1/iter2",
                os.path.join(OUT_DIR, "cw_compensation_tuning_matrix.png"))


if __name__ == "__main__":
    main()
