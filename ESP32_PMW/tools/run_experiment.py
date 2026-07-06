#!/usr/bin/env python3
"""Serial monitor + controller coordinating the ESP32 tilt sweep with a
PicoScope recording.

Firmware contract (main_tilt.cpp, env:tilt): the board boots ARMED with all
coils at 0% carrier and waits for a single-letter newline command:
    b (begin) -> begin/restart the freq ramp + duty sweep
    s (stop)  -> EMERGENCY e-stop: immediate hard cut of all coils, latched
    e (end)   -> graceful ramp all coils down to 0, latched
After s or e the coils stay down until the next 'b'.

So the scope can be armed BEFORE any current flows: on 'begin' this tool
launches the recorder first, then tells the ESP32 to go, so the CSV covers the
run from t=0.

The recorder (picoscope_record.py) is launched with THIS interpreter, so
`uv run python tools/run_experiment.py` reuses the same venv (picosdk, numpy).

Interactive (default): type  begin | stop | end | quit  (or b | s | e | q).
Hands-free:  --auto SECONDS   -> arm scope, begin, wait, end, exit.
On exit / Ctrl-C it always e-stops (s) first — coils are never left energized.

Console shows only events (acks, errors); the noisy 1 Hz `state=...` telemetry
goes to the log file only. Pass --verbose to echo everything.
"""
from __future__ import annotations

import argparse
import glob
import os
import select
import subprocess
import sys
import termios
import threading
import time
import tty

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial missing — run `uv sync`, then "
             "`uv run python tools/run_experiment.py`")

HERE = os.path.dirname(os.path.abspath(__file__))


def find_port(explicit):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found — pass --port")


def reader_thread(ser, logf, stop, verbose):
    """Log every ESP32 line; echo only events (not `state=` telemetry)."""
    while not stop.is_set():
        try:
            line = ser.readline()
        except Exception:
            break
        if not line:
            continue
        text = line.decode("utf-8", "replace").rstrip()
        logf.write(text + "\n")
        logf.flush()
        if verbose or not text.startswith("state="):
            print(f"[esp32] {text}")


def scope_cmd(seconds, out, extra):
    cmd = [sys.executable, os.path.join(HERE, "picoscope_record.py"),
           "--seconds", str(seconds)]
    if out:
        cmd += ["--out", out]
    return cmd + list(extra)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Coordinate ESP32 tilt sweep + PicoScope recording.")
    p.add_argument("--port",
                   help="ESP32 serial port (default: first ttyUSB/ttyACM)")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--seconds", type=float, default=70.0,
                   help="scope recording length (tilt run ~60s; default 70)")
    p.add_argument("--out",
                   help="scope CSV path (default: recorder's timestamp name)")
    p.add_argument("--log", help="serial log path (default: timestamped)")
    p.add_argument("--auto", type=float, metavar="SECONDS",
                   help="hands-free: start, wait SECONDS, ramp down, exit")
    p.add_argument("--scope-arg", action="append", default=[], metavar="ARG",
                   help="extra arg forwarded to picoscope_record.py (repeat)")
    p.add_argument("--verbose", action="store_true",
                   help="echo all serial lines incl. 1 Hz telemetry")
    p.add_argument("--self-test", action="store_true",
                   help="check wiring, no hardware")
    args = p.parse_args(argv)

    if args.self_test:
        return self_test()

    port = find_port(args.port)
    log_path = args.log or f"esp32_serial_{time.strftime('%Y%m%d_%H%M%S')}.log"
    # Deassert DTR/RTS BEFORE open so we don't reset the ESP32 or knock it into
    # the bootloader — leave the running armed sketch alone.
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = args.baud
    ser.timeout = 0.2
    ser.dtr = False
    ser.rts = False
    ser.open()
    print(f"[pc] serial {port} @ {args.baud}, log -> {log_path}")
    logf = open(log_path, "w")
    stop = threading.Event()
    threading.Thread(target=reader_thread,
                     args=(ser, logf, stop, args.verbose),
                     daemon=True).start()

    # Pin ONE CSV path for the whole session so a stop->begin restart overwrites
    # the previous recording instead of leaving a stale partial file behind. The
    # recorder only writes the CSV when its capture finishes (or is left to run),
    # so a killed/restarted recording never produces a file — the restart's fresh
    # run writes to this same path.
    out_path = args.out or f"picoscope_stream_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    scope: dict = {"proc": None}

    def kill_scope():
        """Terminate a running recorder (discards its in-progress recording)."""
        proc = scope["proc"]
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        scope["proc"] = None

    def do_begin():
        # Restart cleanly: if a recorder is still running (e.g. after a stop),
        # kill it and relaunch against the same out_path so the CSV is overwritten
        # from t=0 rather than refusing with "already running".
        if scope["proc"] and scope["proc"].poll() is None:
            print("[pc] scope already running — restarting (overwriting previous CSV)")
            kill_scope()
        cmd = scope_cmd(args.seconds, out_path, args.scope_arg)
        print("[pc] scope:", " ".join(cmd))
        scope["proc"] = subprocess.Popen(cmd)
        time.sleep(0.5)            # let the scope arm before current flows
        ser.write(b"b\n")
        print("[pc] -> begin")

    def do_stop():
        ser.write(b"s\n")          # immediate e-stop
        print("[pc] -> stop (e-stop)")

    def do_end():
        ser.write(b"e\n")          # graceful ramp-down
        print("[pc] -> end (ramping down)")
        # Let the recorder run to completion so it saves the CSV, then we exit.
        if scope["proc"] and scope["proc"].poll() is None:
            print("[pc] waiting for scope to finish + save CSV...")
            scope["proc"].wait()
            print(f"[pc] CSV saved -> {out_path}")

    try:
        if args.auto is not None:
            do_begin()
            time.sleep(args.auto)
            do_end()
            if scope["proc"]:
                scope["proc"].wait()
        else:
            # Single-KEYPRESS console (no Enter needed): put the tty in cbreak
            # mode and act on each character immediately. -/+ nudge the B & C
            # field trims together (the tilt-axis pair) via the firmware.
            print("[pc] keys: b=begin  s=stop  e=end&exit  q=quit  "
                  "-=AD field down 2%  +=AD field up 2%  t=show trims")
            if not sys.stdin.isatty():
                print("[pc] (stdin is not a tty — falling back to line input)")
            fd = sys.stdin.fileno() if sys.stdin.isatty() else None
            old_tty = termios.tcgetattr(fd) if fd is not None else None
            try:
                if fd is not None:
                    tty.setcbreak(fd)
                while True:
                    if fd is not None:
                        r, _, _ = select.select([sys.stdin], [], [], 0.2)
                        if not r:
                            continue
                        ch = sys.stdin.read(1).lower()
                    else:
                        ch = (input().strip().lower() or " ")[0]
                    if ch == "b":
                        do_begin()
                    elif ch == "s":
                        do_stop()
                    elif ch == "e":
                        do_end()   # ramps down, saves CSV, then exits
                        break
                    elif ch == "q":
                        break
                    elif ch in ("-", "_", "+", "=", "t"):
                        raw = "-" if ch in ("-", "_") else \
                              "+" if ch in ("+", "=") else "t"
                        ser.write((raw + "\n").encode())
                        print(f"[pc] -> {raw}")
            finally:
                if old_tty is not None:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_tty)
    except (KeyboardInterrupt, EOFError):
        print("\n[pc] interrupted")
    finally:
        do_stop()                  # safety: e-stop, never leave coils on
        time.sleep(0.3)
        stop.set()
        if scope["proc"] and scope["proc"].poll() is None:
            print("[pc] waiting for scope to finish recording...")
            scope["proc"].wait()
        ser.close()
        logf.close()
    return 0


def self_test():
    """Command construction + port passthrough, no hardware."""
    assert find_port("/dev/ttyXYZ") == "/dev/ttyXYZ"
    cmd = scope_cmd(70.0, "out.csv", ["--range", "500mv"])
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("picoscope_record.py")
    assert cmd[2:] == ["--seconds", "70.0", "--out", "out.csv",
                       "--range", "500mv"]
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
