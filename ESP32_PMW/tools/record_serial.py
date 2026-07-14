#!/usr/bin/env python3
"""Passive, no-command serial recorder. Opens the port, reads, logs -- never
writes anything to the board (no reset by default, no start/stop/e-stop
commands, no TUI). Pairs with autonomous firmware (main_experiment.cpp,
main_tilt_pi.cpp) that needs no serial handshake to run: flash it separately
(`pio run -t upload`), then run this to capture whatever it streams.

Reuses the same buffered, split-proof line-reading design as
tools/run_experiment.py's SerialSession (non-blocking, accumulates raw bytes
across polls, only ever returns a complete line -- a short poll timeout on a
plain blocking readline() was confirmed on hardware to truncate/split lines,
desyncing anything parsing the output).

Usage:
  uv run python tools/record_serial.py
  uv run python tools/record_serial.py --reset --out data/2026-07-14_tilt/run1.log
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")

_TELEMETRY_RE = re.compile(
    r"I\[A\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)"
    r"(?:\s*\|\s*duty\[%\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+))?"
)
_PHASE_NUM_RE = re.compile(r"\bphase=(\d+)\b")  # only current_pid/pi_profile print this


def find_port(explicit=None):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found -- pass --port")


class PassiveSerialReader:
    """Non-blocking, buffered, split-proof line reader -- read-only, never
    writes to the port. See tools/run_experiment.py's SerialSession for the
    same buffering design (used there alongside command-sending)."""

    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0  # non-blocking; line assembly is buffered below
        # Must be set BEFORE open() -- simply opening the port with
        # whatever the OS's default modem-control-line state is triggers
        # the ESP32 dev board's auto-reset circuit, and can glitch GPIO0
        # low, dropping it into DOWNLOAD_BOOT ("waiting for download")
        # instead of a normal app boot. Confirmed on hardware: this bit
        # was missing here (present in tools/run_experiment.py's
        # SerialSession) and reliably reproduced the download-mode glitch
        # even with --reset never passed.
        self.ser.dtr = False  # IO0 stays high -> APP boot, not bootloader
        self.ser.rts = False
        self.ser.open()
        self._rxbuf = b""

    def reset(self) -> None:
        """Optional EN-pulse reset for a clean boot-relative capture start --
        still sends zero application-level commands, so "purely record"
        holds either way. Re-asserts dtr=False right alongside the RTS
        toggle (not just once at open()) -- confirmed on hardware that a
        single dtr=False set at open() can fail to latch on some CP2102/
        CH340 USB-serial chips, leaving GPIO0 in an unreliable state and
        occasionally dropping the board into DOWNLOAD_BOOT instead of a
        normal app boot (matches esptool.py's own hard-reset sequence,
        which re-touches DTR right before every RTS toggle rather than
        relying on a value set once much earlier)."""
        self.ser.dtr = False  # IO0 stays high -> APP boot, not bootloader
        self.ser.rts = True   # EN low: hold in reset
        time.sleep(0.15)
        self.ser.dtr = False  # re-assert -- see docstring
        self.ser.rts = False  # release -> clean app boot
        time.sleep(0.05)
        self._rxbuf = b""

    def readline(self):
        """Non-blocking. Returns a decoded, stripped complete line, or None
        if none has finished yet -- never drops or splits a line regardless
        of how often/rarely this is called."""
        n = self.ser.in_waiting
        if n:
            self._rxbuf += self.ser.read(n)
        nl = self._rxbuf.find(b"\n")
        if nl == -1:
            return None
        raw, self._rxbuf = self._rxbuf[:nl], self._rxbuf[nl + 1:]
        return raw.decode("utf-8", errors="replace").replace("\x00", "").rstrip("\r")

    def close(self) -> None:
        self.ser.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="ESP32 serial port (default: first ttyUSB/ttyACM)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--reset", action="store_true",
                     help="EN-pulse reset before recording, for a clean boot-relative "
                          "capture start (off by default -- still sends zero commands "
                          "either way, this only pulses the reset line)")
    ap.add_argument("--out", default=None,
                     help="log file path (default: timestamped under data/); a "
                          "same-named .csv with parsed telemetry is written alongside it")
    args = ap.parse_args()

    port = find_port(args.port)
    out_path = args.out or os.path.join(
        REPO_ROOT, "data", f"recorded_{time.strftime('%Y%m%d_%H%M%S')}.log")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.splitext(out_path)[0] + ".csv"

    print(f"[pc] recording {port} @ {args.baud} -> {out_path} (Ctrl-C to stop)")
    reader = PassiveSerialReader(port, args.baud)
    if args.reset:
        reader.reset()
    t0 = time.time()

    try:
        with open(out_path, "w") as log_f, open(csv_path, "w") as csv_f:
            csv_f.write("t_s,phase,A,B,C,D,dutyA,dutyB,dutyC,dutyD\n")
            phase = ""
            while True:
                line = reader.readline()
                if line is None:
                    time.sleep(0.01)  # readline() is non-blocking -- avoid busy-spin
                    continue
                elapsed = time.time() - t0
                stamped = f"{elapsed:7.2f}s  {line}"
                print(stamped, flush=True)
                log_f.write(stamped + "\n")
                log_f.flush()

                m_phase = _PHASE_NUM_RE.search(line)
                if m_phase:
                    phase = m_phase.group(1)

                m = _TELEMETRY_RE.search(line)
                if m:
                    a, b, c, d, da, db, dc, dd = m.groups()
                    da, db, dc, dd = (da or "", db or "", dc or "", dd or "")
                    csv_f.write(f"{elapsed:.3f},{phase},{a},{b},{c},{d},"
                                f"{da},{db},{dc},{dd}\n")
                    csv_f.flush()
    except KeyboardInterrupt:
        print("\n[pc] stopped by user", flush=True)
    except OSError as e:
        print(f"\n[pc] serial port error ({e}) -- the port likely disappeared or the "
              "board reset into a bad state (e.g. stuck in download mode). Check the "
              "board and try again.", flush=True)
        return 1
    finally:
        try:
            reader.close()
        except OSError:
            pass

    print(f"[pc] log -> {out_path}")
    print(f"[pc] csv -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
