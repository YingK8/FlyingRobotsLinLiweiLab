#!/usr/bin/env python3
"""Does counting frame times BACK from the end of the video put the swing at the cut?

For a take with dropped frames, frames.csv has one row per captured frame but the mp4 only
has n_mp4 = rows - dropped of them. The current pairing is forward: mp4 frame i <- row i.
If the drops all happened before the cut, pairing backward -- mp4 frame i <- row
(rows - n_mp4 + i) -- is exact from the last drop to the end, which covers the cut window.

Per take: smooth the axis over two revolutions (nulls the once-per-rev cone), take its angle to
the pre-cut mean axis, and report the first time after -2 s that it stays above ONSET_DEG for
HOLD_S -- under the forward and the backward time base. Drop-free takes are the control.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

REPO = Path("/Users/meli/Desktop/Kevin/UCB/FlyingRobotsLinLiweiLab/ESP32_PMW")
ROOTS = [REPO / "results/alignment_rate/20260910_droneB",
         REPO / "results/alignment_rate/20260912_droneB_rerun"]
ONSET_DEG, HOLD_S, PRE = 4.0, 0.15, (-2.5, -0.5)


def onset(t, n, f):
    fs = 1.0 / float(np.median(np.diff(t)))
    k = max(int(round(2.0 * fs / f)), 1)
    kern = np.ones(k) / k
    sm = np.column_stack([np.convolve(n[:, j], kern, mode="same") for j in range(3)])
    sm /= np.linalg.norm(sm, axis=1)[:, None]
    pre = (t >= PRE[0]) & (t <= PRE[1])
    if pre.sum() < 20:
        return np.nan
    ref = sm[pre].mean(axis=0)
    ref /= np.linalg.norm(ref)
    ang = np.degrees(np.arccos(np.clip(np.abs(sm @ ref), -1.0, 1.0)))
    hold = max(int(round(HOLD_S * fs)), 1)
    idx = np.where(t >= -2.0)[0]
    for i in idx:
        if i + hold < len(ang) and np.all(ang[i:i + hold] > ONSET_DEG):
            return float(t[i])
    return np.nan


def main():
    print("freq rep take               dropped  onset_forward  onset_backward")
    for root in ROOTS:
        for idx in sorted(root.glob("*hz/index.csv")):
            for r in csv.DictReader(open(idx)):
                take = Path(r["flight"])
                if r["outcome"] != "ok" or not (take / "axis.csv").exists():
                    continue
                f = float(r["freq_hz"])
                if f < 30:
                    continue
                dropped = json.load(open(take / "meta.json")).get("dropped", 0)
                tk = float(re.search(r"^\[(\d+\.\d+)\] <- label=KILL_",
                                     Path(r["log"]).read_text(errors="replace"), re.M).group(1))
                rows = np.genfromtxt(take / "frames.csv", delimiter=",", names=True)
                t_rows = rows["t_a"]
                a = np.genfromtxt(take / "axis.csv", delimiter=",", names=True)
                frame = a["frame"].astype(int)
                n = np.c_[a["nx"], a["ny"], a["nz"]]
                n_mp4 = len(t_rows) - dropped
                fwd = t_rows[np.clip(frame, 0, len(t_rows) - 1)] - tk
                back = t_rows[np.clip(len(t_rows) - n_mp4 + frame, 0, len(t_rows) - 1)] - tk
                print(f"{f:4.0f} {r['repeat']:>3} {take.name} {dropped:7d}  "
                      f"{onset(fwd, n, f):+12.2f}  {onset(back, n, f):+13.2f}")


if __name__ == "__main__":
    main()
