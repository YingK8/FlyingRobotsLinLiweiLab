#!/usr/bin/env python3
"""Reset the ESP32 into main_state_space.cpp, send dir=<cw|ccw> then
rmax=<cap> (both must land during the ARMING window -- main_state_space.cpp
only accepts dir= while phase is ARMING/STOPPED), then log serial for
--log-seconds. Uses pyserial's bulk readline(), matching
tools/trigger_reset_log.py's proven pattern -- byte-by-byte reads (e.g.
SerialComm.handle_serial_comm(), designed for short interactive protocols)
silently drop bytes under this firmware's long, fast-bursting telemetry
line, confirmed on hardware (only trailing fragments of each line survived
with that approach).

Safety: always sends 's' (e-stop) before closing, on every exit path --
harmless no-op here (main_state_space.cpp has no 's' handler) but matches
trigger_reset_log.py's convention.

Usage:
  uv run python tools/run_state_space_directional.py --direction cw --rmax 2.0 \
      --log-seconds 40 --out state_space_cw_2A.log
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_port(explicit):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found -- pass --port")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--direction", choices=["cw", "ccw"], required=True)
    ap.add_argument("--rmax", type=float, required=True)
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dir-cmd-delay", type=float, default=0.5,
                     help="seconds after reset to send dir= (default: %(default)s)")
    ap.add_argument("--rmax-cmd-delay", type=float, default=1.0,
                     help="seconds after reset to send rmax= (default: %(default)s)")
    ap.add_argument("--log-seconds", type=float, default=40.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-matrix", action="store_true",
                     help="suppress the coil-current matrix visualiser window")
    args = ap.parse_args()
    port = find_port(args.port)

    view = None
    if not args.no_matrix:
        from coil_current_matrix import CoilCurrentMatrixView
        view = CoilCurrentMatrixView()

    s = serial.Serial()
    s.port = port
    print(f"[run] using port {port}", flush=True)
    s.baudrate = args.baud
    s.timeout = 0.5
    s.dtr = False  # IO0 stays high -> APP boot, not bootloader
    s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True   # EN low: hold in reset
    time.sleep(0.15)
    s.rts = False  # release -> clean app boot
    t0 = time.time()
    print(f"[run] EN pulse done at {time.strftime('%H:%M:%S')}", flush=True)

    dir_sent = rmax_sent = False
    f = open(args.out, "w")
    try:
        while time.time() - t0 < args.log_seconds:
            elapsed = time.time() - t0
            if not dir_sent and elapsed >= args.dir_cmd_delay:
                s.write((f"dir={args.direction}\n").encode())
                dir_sent = True
                print(f"[run] sent 'dir={args.direction}' at t={elapsed:.2f}s", flush=True)
            elif not rmax_sent and elapsed >= args.rmax_cmd_delay:
                s.write((f"rmax={args.rmax}\n").encode())
                rmax_sent = True
                print(f"[run] sent 'rmax={args.rmax}' at t={elapsed:.2f}s", flush=True)
            line = s.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").rstrip()
            stamped = f"{time.time() - t0:7.2f}s  {text}"
            print(stamped, flush=True)
            f.write(stamped + "\n")
            f.flush()
            if view is not None:
                view.feed_line(text)
                view.refresh()
    except KeyboardInterrupt:
        print("\n[run] stopped early by user", flush=True)
    finally:
        try:
            s.write(b"s\n")
            time.sleep(0.3)
        except Exception as e:
            print(f"[run] WARNING: failed to send e-stop ({e})", flush=True)
        f.close()
        s.close()
        if view is not None:
            view.close()
        print(f"[run] log saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
