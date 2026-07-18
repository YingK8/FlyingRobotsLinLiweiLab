#!/usr/bin/env python3
"""Open-loop flight maneuver over serial: drive main_flight.cpp (env:flight)
through takeoff -> hover -> accelerate -> land, logging telemetry to a .log.

Outer-loop stand-in until the camera returns. Reuses ai/serial_comm.py.

Usage: uv run python ai/flight_controller.py [--port /dev/ttyUSB0] [--log flight_run.log]
"""
from __future__ import annotations

import argparse
import time

from serial_comm import SerialComm

# (t_seconds, command). Scaled to the firmware's 30 s spin-up.
MANEUVER = [  # (t_seconds, command), scaled to the 30 s spin-up
    (0.0, "takeoff"),
    (33.0, "throttle=80"),   # hover
    (40.0, "az=45"),         # aim thrust at 45 deg
    (41.0, "mag=0.4"),       # accelerate
    (48.0, "hover"),         # level off
    (53.0, "land"),
]
END_S = 62.0

# events fire in list order, so time must be monotonic
assert all(a[0] <= b[0] for a, b in zip(MANEUVER, MANEUVER[1:])), "maneuver not time-sorted"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--log", default="flight_run.log")
    args = ap.parse_args()

    comm = SerialComm(port=args.port)
    comm.reset_device()          # reboot firmware to IDLE
    time.sleep(1.5)              # wait out the boot banner

    events = list(MANEUVER)
    t0 = time.time()
    with open(args.log, "w") as f:
        while (t := time.time() - t0) < END_S:
            out = ""
            if events and t >= events[0][0]:
                out = events.pop(0)[1]
                print(f"[{t:5.1f}s] -> {out}")
            line = comm.handle_serial_comm(out)  # send (if any), drain one line
            if line:
                f.write(line + "\n")
                f.flush()
                print(line)
            time.sleep(0.01)
        comm.handle_serial_comm("stop")  # de-energize on exit
    comm.close()
    print(f"done -> {args.log}  (plot: uv run python ai/plot_pid_log.py {args.log})")


if __name__ == "__main__":
    main()
