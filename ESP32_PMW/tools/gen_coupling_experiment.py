#!/usr/bin/env python3
"""Generate the CW/CCW coil-coupling characterization JSON experiments.

Emits task_sequences/coupling_cw.json and task_sequences/coupling_ccw.json: for each
direction, sweep every solo/pairwise/all-4 activation combo (the 11 segments
main_coupling_test.cpp used to hardcode) at several carrier-duty ("current")
levels, so tools/coupling_matrix.py --segments-log can extract a coupling
matrix per level from the resulting capture. Only emits objects the on-device
JsonPWMSequencer parser already understands (setDirection / activateChannels
/ label / addWaitTask); this script's only job is unrolling the combinatorics
(repeats aren't a queue primitive), not adding capability the firmware lacks.

Usage:
    uv run python tools/gen_coupling_experiment.py
    uv run python tools/gen_coupling_experiment.py --levels 50 100 --active-ms 2000
"""
from __future__ import annotations

import argparse
import json
import os

# 4 solos + 6 pairs + ALL, as a 4-bit channel bitmask (bit0=A, bit1=B, bit2=C,
# bit3=D) -- the same segment list main_coupling_test.cpp hardcoded as SEQ[].
SEGMENTS = [
    (0b0001, "SOLO_A"), (0b0010, "SOLO_B"), (0b0100, "SOLO_C"), (0b1000, "SOLO_D"),
    (0b0011, "PAIR_AB"), (0b0101, "PAIR_AC"), (0b1001, "PAIR_AD"),
    (0b0110, "PAIR_BC"), (0b1010, "PAIR_BD"), (0b1100, "PAIR_CD"),
    (0b1111, "ALL"),
]

DEFAULT_LEVELS = [25, 50, 75, 100]
DEFAULT_ACTIVE_MS = 3000
DEFAULT_GAP_MS = 2000


def build_events(direction: str, levels: list[int], active_ms: int,
                  gap_ms: int) -> list[dict]:
    assert direction in ("cw", "ccw")
    events: list[dict] = [
        {"method": "setDirection", "value": 0 if direction == "cw" else 1}
    ]
    for level in levels:
        for mask, name in SEGMENTS:
            label = f"{direction.upper()}_I{level}_{name}"
            events.append({"method": "label", "value": label})
            events.append({"method": "activateChannels", "mask": mask,
                            "value": float(level)})
            events.append({"method": "addWaitTask", "duration_ms": active_ms})
            events.append({"method": "activateChannels", "mask": 0, "value": 0.0})
            events.append({"method": "addWaitTask", "duration_ms": gap_ms})
    return events


def total_duration_ms(levels: list[int], active_ms: int, gap_ms: int) -> int:
    return len(levels) * len(SEGMENTS) * (active_ms + gap_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
                     help="carrier-duty / current levels to sweep, percent "
                          "(default: %(default)s)")
    ap.add_argument("--active-ms", type=int, default=DEFAULT_ACTIVE_MS,
                     help="drive duration per segment, ms (default: %(default)s)")
    ap.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS,
                     help="off-gap between segments, ms (default: %(default)s)")
    ap.add_argument("--out-dir",
                     default=os.path.join(os.path.dirname(__file__), "..", "task_sequences"),
                     help="directory to write coupling_cw.json/coupling_ccw.json into")
    args = ap.parse_args()

    total_ms = total_duration_ms(args.levels, args.active_ms, args.gap_ms)
    print(f"{len(args.levels)} levels x {len(SEGMENTS)} segments -> "
          f"{total_ms / 1000:.1f}s per direction (record >= {total_ms / 1000 + 5:.1f}s)")

    for direction in ("cw", "ccw"):
        events = build_events(direction, args.levels, args.active_ms, args.gap_ms)
        out_path = os.path.join(args.out_dir, f"coupling_{direction}.json")
        with open(out_path, "w") as f:
            json.dump(events, f, indent=2)
        print(f"wrote {out_path}: {len(events)} events")


if __name__ == "__main__":
    main()
