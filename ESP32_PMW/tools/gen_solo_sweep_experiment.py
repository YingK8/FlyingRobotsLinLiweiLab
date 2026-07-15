#!/usr/bin/env python3
"""Generate a solo-channel frequency-sweep JSON experiment for
main_experiment.cpp (system-ID: joint R/L/C fit -- see
tools/fit_rlc_model.py and the plan's dynamic-model section).

For each channel: drive it alone (activateChannels, others at 0%) at a fixed
duty, and step the GLOBAL commutation frequency through a discrete grid (no
chirp primitive exists on-device -- JsonPhaseSequencer only supports
piecewise linear/ease ramps and instant sets, see lib/JsonPhaseSequencer).
Each frequency point is reached via a short addLinearRampTask and held via
addWaitTask, long enough for the CurrentSense 50ms EMA filter (and any
resonant ringing, unknown a priori -- start generous) to settle before
main_experiment.cpp's periodic telemetry samples it. Each dwell is tagged
with a label like "SWEEP_A_F150" so tools/fit_rlc_model.py can extract the
settled current per (channel, frequency) point.

Usage:
  uv run python tools/gen_solo_sweep_experiment.py --duty 30 \
      --dwell-ms 2500 --initial-freq 190 --direction cw --out-dir spiffs_data
"""
from __future__ import annotations

import argparse
import json
import os

CHANNEL_BIT = {"A": 0, "B": 1, "C": 2, "D": 3}
DEFAULT_FREQS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,
                  140, 150, 160, 170, 180, 190, 200, 210]


def build_events(channels: list[str], freqs: list[float], duty: float,
                  dwell_ms: int, transition_ms: int, gap_ms: int,
                  initial_freq: float, direction: str) -> list[dict]:
    assert direction in ("cw", "ccw")
    events: list[dict] = [
        {"method": "setDirection", "value": 0 if direction == "cw" else 1}
    ]
    cur_freq = initial_freq
    for ch in channels:
        mask = 1 << CHANNEL_BIT[ch]
        events.append({"method": "activateChannels", "mask": mask, "value": float(duty)})
        for f in freqs:
            events.append({"method": "label", "value": f"SWEEP_{ch}_F{f:g}"})
            events.append({"method": "addLinearRampTask", "from": cur_freq, "to": f,
                            "duration_ms": transition_ms})
            events.append({"method": "addWaitTask", "duration_ms": dwell_ms})
            cur_freq = f
        events.append({"method": "activateChannels", "mask": 0, "value": 0.0})
        events.append({"method": "label", "value": f"SWEEP_{ch}_OFF"})
        events.append({"method": "addWaitTask", "duration_ms": gap_ms})
    return events


def total_duration_ms(channels: list[str], freqs: list[float], dwell_ms: int,
                       transition_ms: int, gap_ms: int) -> int:
    return len(channels) * (len(freqs) * (dwell_ms + transition_ms) + gap_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=list("ABCD"), choices=list("ABCD"))
    ap.add_argument("--freqs", type=float, nargs="+", default=DEFAULT_FREQS,
                     help="commutation frequencies to sweep, Hz (default: sparse "
                          "1-210Hz grid -- run a first coarse pass to locate each "
                          "channel's approximate resonance, then optionally a "
                          "denser pass concentrated near it)")
    ap.add_argument("--duty", type=float, default=30.0,
                     help="carrier duty %% during the sweep -- keep conservative "
                          "on the FIRST pass, resonance current is unknown a "
                          "priori (default: %(default)s)")
    ap.add_argument("--dwell-ms", type=int, default=2500,
                     help=">= 5x the CurrentSense EMA tau (50ms) AND long enough "
                          "for resonant ringing (~2L/R) to settle -- unknown "
                          "until after the first pass, so start generous "
                          "(default: %(default)s)")
    ap.add_argument("--transition-ms", type=int, default=300,
                     help="ramp duration between frequency points (default: %(default)s)")
    ap.add_argument("--gap-ms", type=int, default=3000,
                     help="off-gap between channels, letting current fully "
                          "decay before the next channel's solo drive (default: %(default)s)")
    ap.add_argument("--initial-freq", type=float, default=190.0,
                     help="MUST match main_experiment.cpp's DRIVE_FREQ constant "
                          "(default: %(default)s)")
    ap.add_argument("--direction", choices=["cw", "ccw"], default="cw")
    ap.add_argument("--out-name", default="solo_sweep.json",
                     help="output filename within --out-dir (default: %(default)s); "
                          "copy/rename to experiment.json before uploadfs to make "
                          "it the active experiment")
    ap.add_argument("--out-dir",
                     default=os.path.join(os.path.dirname(__file__), "..", "spiffs_data"))
    args = ap.parse_args()

    total_ms = total_duration_ms(args.channels, args.freqs, args.dwell_ms,
                                  args.transition_ms, args.gap_ms)
    print(f"{len(args.channels)} channels x {len(args.freqs)} freqs -> "
          f"{total_ms / 1000:.1f}s total (record >= {total_ms / 1000 + 5:.1f}s)")

    events = build_events(args.channels, args.freqs, args.duty, args.dwell_ms,
                           args.transition_ms, args.gap_ms, args.initial_freq,
                           args.direction)
    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"wrote {out_path}: {len(events)} events")


if __name__ == "__main__":
    main()
