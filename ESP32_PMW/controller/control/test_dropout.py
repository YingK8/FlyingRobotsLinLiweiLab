#!/usr/bin/env python3
"""Does controller_loop fly through a vision dropout, and land when it cannot?

Run: uv run python controller/control/test_dropout.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "viz")]

import hover_controller_runner as R
import z_track
from live_viz import NullViz, Tick
from reference_profiles import Profile
from simulate_hover import DiscreteHoverController

GAP_START = 100
FPS = 60.0


class ArmedViz(NullViz):
    """NullViz, but armed and lateral-enabled, as a flying session would be."""

    armed, mag_max = True, 0.8
    setpoint, gain_scale, land = (0.0, 0.0, 60.0), (1.0, 1.0), False


def ticks(gap_frames, progress, n=400):
    """Hover at 60 mm, with frames [GAP_START, GAP_START+gap) lost."""

    viz, t0 = ArmedViz(), time.monotonic()
    for i in range(n):
        while time.monotonic() - t0 < i / FPS:
            time.sleep(0.0005)
        progress[0] = i
        lost = GAP_START <= i < GAP_START + gap_frames
        yield Tick(t=t0 + i / FPS, pose=None, frames=None, lost=int(lost), viz=viz,
                   xyz_mm=None if lost else np.array([0.0, 0.0, 60.0]))


def run(gap_frames, log_dir):
    gains = json.load(open(HERE / "hover_controller.json"))
    cfg = R.RunConfig(dry_run=True, takeoff=False, enable_freq_cmd=True,
                      log=str(Path(log_dir) / f"gap_{gap_frames}.log"))
    ctrl = tuple(DiscreteHoverController(gains, Profile.hold()) for _ in range(2))
    ztrk = z_track.ZTracker(np.array([0.0, 1e9]), np.array([0.06, 0.06]))
    link = R.CommandLink(None, True, cfg.log)
    progress, sent = [0], []
    link.send = lambda c, _s=sent, _p=progress: _s.append((_p[0], c))
    try:
        R.controller_loop(ticks(gap_frames, progress), link, ctrl, cfg, ztrk)
    finally:
        link.close()
    return sent


def demo(log_dir="/tmp"):
    # max_coast_s = 0.10 s, so 6 frames at 60 fps is exactly the limit and must land.
    for gap, expect in ((1, "survive"), (3, "survive"), (5, "survive"),
                        (6, "land"), (60, "land")):
        sent = run(gap, log_dir)
        during = [c for i, c in sent
                  if GAP_START <= i < GAP_START + gap and c.startswith("freq=")]
        after = [c for i, c in sent
                 if i > GAP_START + gap + 5 and c.startswith("freq=")]
        got = "survive" if after else "land"
        print(f"gap {gap:3d} frames ({gap / FPS * 1e3:6.0f} ms): {got:8s} "
              f"{len(during):3d} freq= during, {len(after):3d} after")
        assert got == expect, (gap, got, expect)
    print("self-check PASS: short gaps flown on the model, long gaps land")


if __name__ == "__main__":
    demo(*sys.argv[1:])
