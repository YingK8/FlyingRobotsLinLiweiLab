#!/usr/bin/env python3
"""Generate the drone swim JSON schedule.

Emits spiffs_data/swim.json: spin the rotating field up to a swim frequency
(1 -> 30 Hz over 20 s by default), then undulate around it (30 <-> 22 Hz) to
produce a stroke, then cut the coils. 30 Hz is far below the 150-210 Hz flight
regime, so the stroke bounds here are seeds to tune on hardware -- that is why
they are flags rather than constants.

Only emits objects the on-device JsonPhaseSequencer parser already understands
(addCarrierDutyCycleTask / add*RampTask / activateChannels / addWaitTask /
label); this script's only job is unrolling the stroke cycles (repeats aren't a
queue primitive), not adding capability the firmware lacks.

Usage:
    uv run python ai/gen_swim_experiment.py
    uv run python ai/gen_swim_experiment.py --strokes 3 --ramp-mode ease
    uv run python ai/gen_swim_experiment.py --spinup-hz 25 --stroke-low-hz 18
"""
from __future__ import annotations

import argparse
import os

from json_schedule import write_experiment

# add*RampTask variants, all ramping the global commutation frequency.
RAMP_METHOD = {
    "linear": "addLinearRampTask",       # POLYNOMIAL; shape = power, 1 = straight
    "ease": "addEaseRampTask",           # symmetric S-curve, shape = k >= 1
    "exp": "addExponentialRampTask",     # shape > 0 ease-in, < 0 ease-out
}

DEFAULT_START_HZ = 1.0
DEFAULT_SPINUP_HZ = 30.0
DEFAULT_SPINUP_MS = 20000
DEFAULT_STROKE_LOW_HZ = 22.0
DEFAULT_STROKE_DOWN_MS = 1000
DEFAULT_STROKE_UP_MS = 1000
DEFAULT_STROKES = 5
DEFAULT_CARRIER = 100.0
SHUTDOWN_MS = 500


def build_events(start_hz: float, spinup_hz: float, spinup_ms: int,
                 ramp_mode: str, ramp_shape: float | None,
                 stroke_low_hz: float, stroke_down_ms: int, stroke_up_ms: int,
                 strokes: int, carrier: float) -> list[dict]:
    spinup = RAMP_METHOD[ramp_mode]

    def ramp(method: str, frm: float, to: float, duration_ms: int) -> dict:
        step = {"method": method, "from": frm, "to": to, "duration_ms": duration_ms}
        if ramp_shape is not None and method == spinup:
            step["shape"] = ramp_shape
        return step

    events: list[dict] = [
        {"method": "addCarrierDutyCycleTask", "channels": [0, 1, 2, 3],
         "value": carrier},
        {"method": "label",
         "value": f"SWIM_SPINUP_{start_hz:g}_{spinup_hz:g}HZ"},
        ramp(spinup, start_hz, spinup_hz, spinup_ms),
    ]
    # Strokes are unrolled: the sequencer has no loop primitive. Consecutive
    # ramps compose because the loader carries curFreq = to across steps.
    for i in range(1, strokes + 1):
        events.append({"method": "label", "value": f"SWIM_STROKE_{i:02d}_DOWN"})
        events.append(ramp("addLinearRampTask", spinup_hz, stroke_low_hz,
                           stroke_down_ms))
        events.append({"method": "label", "value": f"SWIM_STROKE_{i:02d}_UP"})
        events.append(ramp("addLinearRampTask", stroke_low_hz, spinup_hz,
                           stroke_up_ms))
    events.append({"method": "label", "value": "SWIM_OFF"})
    events.append({"method": "activateChannels", "mask": 15, "value": 0.0})
    events.append({"method": "addWaitTask", "duration_ms": SHUTDOWN_MS})
    return events


def total_duration_ms(spinup_ms: int, stroke_down_ms: int, stroke_up_ms: int,
                      strokes: int) -> int:
    return spinup_ms + strokes * (stroke_down_ms + stroke_up_ms) + SHUTDOWN_MS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-hz", type=float, default=DEFAULT_START_HZ,
                    help="frequency the spin-up starts from (default: %(default)s)")
    ap.add_argument("--spinup-hz", type=float, default=DEFAULT_SPINUP_HZ,
                    help="swim frequency at the top of the ramp, Hz "
                         "(default: %(default)s)")
    ap.add_argument("--spinup-ms", type=int, default=DEFAULT_SPINUP_MS,
                    help="spin-up ramp duration, ms (default: %(default)s)")
    ap.add_argument("--ramp-mode", choices=sorted(RAMP_METHOD), default="linear",
                    help="spin-up curve family (default: %(default)s)")
    ap.add_argument("--ramp-shape", type=float, default=None,
                    help="curve parameter for the spin-up ramp; omitted means "
                         "the firmware default (linear p=1, ease k=2, exp k=2)")
    ap.add_argument("--stroke-low-hz", type=float, default=DEFAULT_STROKE_LOW_HZ,
                    help="bottom of the undulation, Hz (default: %(default)s)")
    ap.add_argument("--stroke-down-ms", type=int, default=DEFAULT_STROKE_DOWN_MS,
                    help="down-stroke duration, ms (default: %(default)s)")
    ap.add_argument("--stroke-up-ms", type=int, default=DEFAULT_STROKE_UP_MS,
                    help="up-stroke duration, ms (default: %(default)s)")
    ap.add_argument("--strokes", type=int, default=DEFAULT_STROKES,
                    help="number of stroke cycles (default: %(default)s)")
    ap.add_argument("--carrier", type=float, default=DEFAULT_CARRIER,
                    help="carrier duty on all four channels, percent "
                         "(default: %(default)s)")
    ap.add_argument("--direction", choices=("cw", "ccw"), default="ccw",
                    help="phase convention seeded at compile (default: %(default)s)")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "spiffs_data"),
                    help="directory to write the schedule into")
    ap.add_argument("--out-name", default="swim.json",
                    help="output filename (default: %(default)s)")
    args = ap.parse_args()

    events = build_events(args.start_hz, args.spinup_hz, args.spinup_ms,
                          args.ramp_mode, args.ramp_shape, args.stroke_low_hz,
                          args.stroke_down_ms, args.stroke_up_ms, args.strokes,
                          args.carrier)
    total_ms = total_duration_ms(args.spinup_ms, args.stroke_down_ms,
                                 args.stroke_up_ms, args.strokes)
    print(f"{args.spinup_ms / 1000:.1f}s spin-up to {args.spinup_hz:g}Hz + "
          f"{args.strokes} strokes -> {total_ms / 1000:.1f}s "
          f"(record >= {total_ms / 1000 + 5:.1f}s)")

    out_path = os.path.join(args.out_dir, args.out_name)
    schedule = {
        "resolution_ms": 25,
        "initial_freq": 0.0,
        "initial_duty": [50, 50, 50, 50],
        "direction": args.direction.upper(),
        "schedule": events,
    }
    write_experiment(schedule, out_path)
    print(f"wrote {out_path}: {len(events)} events")


if __name__ == "__main__":
    main()
