#!/usr/bin/env python3
"""Generate a per-channel duty-STEP JSON experiment for main_experiment.cpp
(state-space model validation -- see tools/validate_state_space_model.py and
the state-space plan's Phase E). Holds all 4 channels at a baseline duty
(fixed DRIVE_FREQ, no ramp), then steps ONE channel's duty by --step-delta
while the other 3 stay at baseline, holds long enough to settle, returns it
to baseline, and moves to the next channel. Reuses the SAME labeled-dwell
pattern as tools/gen_solo_sweep_experiment.py so tools/pid_metrics.parse_experiment_log
+ label matching can extract the settled window before/after each step.

The point of this experiment (distinct from the coupling-matrix sweeps in
data/2026-07-04a) is a clean single-channel perturbation: the fitted
state-space model (tools/build_state_space_model.py) predicts that, at
STEADY STATE, only the stepped channel's current should change -- coupling
only enters through di/dt (see the plan). If the OTHER 3 channels' settled
current shifts measurably in response to one channel's duty step, that's
direct evidence of a steady-state coupling mechanism (e.g. shared supply
loading) the model doesn't capture, and tools/validate_state_space_model.py
will flag it.

Usage:
  uv run python tools/gen_step_response_experiment.py --baseline-duty 30 \
      --step-delta 15 --hold-ms 3000 --direction cw --out-dir task_sequences
"""
from __future__ import annotations

import argparse
import json
import os

CHANNEL_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def build_events(channels: list[str], baseline_duty: float, step_delta: float,
                  baseline_hold_ms: int, step_hold_ms: int, gap_ms: int,
                  direction: str) -> list[dict]:
    assert direction in ("cw", "ccw")
    events: list[dict] = [
        {"method": "setDirection", "value": 0 if direction == "cw" else 1}
    ]
    for ch in "ABCD":
        events.append({"method": "addCarrierDutyCycleTask", "channel": CHANNEL_IDX[ch],
                        "value": float(baseline_duty)})

    for ch in channels:
        events.append({"method": "label", "value": f"STEP_{ch}_BASELINE"})
        events.append({"method": "addWaitTask", "duration_ms": baseline_hold_ms})

        events.append({"method": "addCarrierDutyCycleTask", "channel": CHANNEL_IDX[ch],
                        "value": float(baseline_duty + step_delta)})
        events.append({"method": "label", "value": f"STEP_{ch}_STEPPED"})
        events.append({"method": "addWaitTask", "duration_ms": step_hold_ms})

        events.append({"method": "addCarrierDutyCycleTask", "channel": CHANNEL_IDX[ch],
                        "value": float(baseline_duty)})
        events.append({"method": "label", "value": f"STEP_{ch}_RECOVER"})
        events.append({"method": "addWaitTask", "duration_ms": gap_ms})
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=list("ABCD"), choices=list("ABCD"))
    ap.add_argument("--baseline-duty", type=float, default=30.0,
                     help="carrier duty %% all 4 channels sit at between/before "
                          "steps (default: %(default)s)")
    ap.add_argument("--step-delta", type=float, default=15.0,
                     help="duty %% added to the stepped channel (default: %(default)s)")
    ap.add_argument("--baseline-hold-ms", type=int, default=3000,
                     help=">= 5x the CurrentSense EMA tau (50ms) so the pre-step "
                          "settled current is well defined (default: %(default)s)")
    ap.add_argument("--step-hold-ms", type=int, default=3000,
                     help="post-step settle time before returning to baseline (default: %(default)s)")
    ap.add_argument("--gap-ms", type=int, default=1000,
                     help="recovery time at baseline before the next channel's step (default: %(default)s)")
    ap.add_argument("--direction", choices=["cw", "ccw"], default="cw")
    ap.add_argument("--out-name", default="step_response.json",
                     help="output filename within --out-dir (default: %(default)s); "
                          "copy/rename to experiment.json before uploadfs to make "
                          "it the active experiment")
    ap.add_argument("--out-dir",
                     default=os.path.join(os.path.dirname(__file__), "..", "task_sequences"))
    args = ap.parse_args()

    total_ms = len(args.channels) * (args.baseline_hold_ms + args.step_hold_ms + args.gap_ms)
    print(f"{len(args.channels)} channels -> {total_ms / 1000:.1f}s total "
          f"(record >= {total_ms / 1000 + 5:.1f}s)")

    events = build_events(args.channels, args.baseline_duty, args.step_delta,
                           args.baseline_hold_ms, args.step_hold_ms, args.gap_ms,
                           args.direction)
    out_path = os.path.join(args.out_dir, args.out_name)
    with open(out_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"wrote {out_path}: {len(events)} events")


if __name__ == "__main__":
    main()
