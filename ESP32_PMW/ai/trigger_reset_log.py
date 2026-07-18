#!/usr/bin/env python3
"""EN-pulse reset the ESP32 into a clean APP boot, then log its serial output.

This is the reliable way to sync a boot-relative firmware sweep to a PicoScope
recording: start picoscope_record.py in the background, run this ~4 s in, and
the sweep starts fresh inside the capture window.

The pulse: open the port with DTR deasserted (keeps IO0 high -> app mode, not
bootloader), assert RTS 150 ms (EN low = reset), release. Manual RST-button
timing was flaky and gave empty captures; this wasn't.

SAFETY: unconditionally sends 's' (e-stop) before closing the port, on EVERY
exit path (normal completion, Ctrl-C, or error) -- matching run_experiment.py's
`finally: do_stop()`. Firmware with an unbounded HOLD state (main_current_pid.cpp)
will otherwise keep driving the coils forever after this script exits, since
--log-seconds only stops the PC-side log, not the ESP32's own control loop.
Harmless no-op on firmware with no serial command listener (main_coupling_test.cpp).

Usage:
  uv run python ai/trigger_reset_log.py [--port /dev/ttyUSB0] [--delay 4]
      [--log-seconds 60] [--out serial.log]
      [--cmd b --cmd-delay 7]   # optional: send a one-letter serial command
                                # --cmd-delay seconds after the reset (e.g. 'b'
                                # to begin once main_current_pid.cpp's boot/
                                # sanity-check sequence has reached IDLE)
"""
import argparse
import glob
import sys
import time

import serial


def find_port(explicit):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found -- pass --port")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", default=None,
                   help="default: first /dev/ttyUSB*|ttyACM* found (port "
                        "numbering shifts across replugs/other USB devices)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds to wait before pulsing reset")
    p.add_argument("--log-seconds", type=float, default=60.0,
                   help="how long to log serial after the reset")
    p.add_argument("--out", default=None, help="log file (default: stdout only)")
    p.add_argument("--cmd", default=None,
                   help="one-line serial command to send after the reset "
                        "(e.g. 'b' to begin) -- sent once, --cmd-delay seconds in")
    p.add_argument("--cmd-delay", type=float, default=7.0,
                   help="seconds after the reset pulse to send --cmd")
    args = p.parse_args()
    port = find_port(args.port)

    if args.delay > 0:
        time.sleep(args.delay)

    s = serial.Serial()
    s.port = port
    print(f"[trigger] using port {port}", flush=True)
    s.baudrate = args.baud
    s.timeout = 0.5
    s.dtr = False   # IO0 stays high -> APP boot, not bootloader
    s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True    # EN low: hold in reset
    time.sleep(0.15)
    s.rts = False   # release -> clean app boot
    t0 = time.time()
    print(f"[trigger] EN pulse done at {time.strftime('%H:%M:%S')}", flush=True)

    f = open(args.out, "w") if args.out else None
    cmd_sent = args.cmd is None
    try:
        while time.time() - t0 < args.log_seconds:
            if not cmd_sent and time.time() - t0 >= args.cmd_delay:
                s.write((args.cmd + "\n").encode())
                cmd_sent = True
                print(f"[trigger] sent {args.cmd!r} at t={time.time()-t0:.2f}s", flush=True)
            line = s.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").rstrip()
            stamped = f"{time.time() - t0:7.2f}s  {text}"
            print(stamped, flush=True)
            if f:
                f.write(stamped + "\n")
                f.flush()
    except KeyboardInterrupt:
        print("\n[trigger] stopped early by user", flush=True)
    finally:
        # SAFETY: always e-stop before exiting -- never leave coils energized
        # on an unbounded-HOLD firmware just because this script stopped logging.
        try:
            s.write(b"s\n")
            time.sleep(0.3)
            print("[trigger] sent 's' (e-stop) -- coils de-energized", flush=True)
        except Exception as e:
            print(f"[trigger] WARNING: failed to send e-stop ({e}) -- "
                  f"verify coils are off by hand", flush=True)
        if f:
            f.close()
        s.close()


if __name__ == "__main__":
    main()
