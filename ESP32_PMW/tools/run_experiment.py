#!/usr/bin/env python3
"""Unified build+flash+run driver for all three ESP32_PMW firmware variants.
Replaces the old per-firmware scripts (trigger_reset_log.py,
stage_pid_profile.py, run_tilt_pid_experiment.py) with one entrypoint
selected by --fw.

--fw experiment (main_experiment.cpp) is autonomous: ARMING -> RUNNING with
no serial command needed to begin -- upload/reset and it just runs. 's'
e-stops it at any time. --fw current_pid/pi_profile still use the
manual-start contract (see lib/ExperimentPhase): ARMING -> WAITING (coils
latched off, waiting for an explicit start command) -> <running> ->
STOPPED. By default this driver opens a curses TUI (tools/experiment_tui.py)
once serial is open for live telemetry/recording and (for current_pid/
pi_profile) a human-driven start/stop; pass --auto-start for scripted/
unattended use (sweeps, autotune trials), which sends the start command
itself the instant WAITING is seen (a no-op for --fw experiment, which
never waits).

Usage:
  uv run python tools/run_experiment.py --fw experiment --json task_sequences/tilt.json
  uv run python tools/run_experiment.py --fw experiment --json task_sequences/tilt.json \\
      --pi-compensate --pi-profile task_sequences/pi_profile_tilt.json
  uv run python tools/run_experiment.py --fw pi_profile --profile task_sequences/pi_profile_tilt.json
  uv run python tools/run_experiment.py --fw current_pid --dir cw
  uv run python tools/run_experiment.py --fw current_pid --skip-build --auto-start --no-tui \\
      --kp 2.5 --out-dir tools/autotune_runs/trial_003
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
DEFAULT_PIO = os.path.expanduser("~/.platformio/penv/bin/pio")  # system `pio` (apt) is broken -- see project memory
DEFAULT_UPLOAD_SPEED = 115200  # 460800 (platformio default) fails "Unable to verify flash chip" on this rig

# env name == --fw value; whether it needs the SPIFFS uploadfs step first.
FW_NEEDS_SPIFFS = {"experiment": True, "pi_profile": True, "current_pid": False}

# --fw values that still use the manual-start (WAITING) contract --
# main_experiment.cpp is autonomous and never waits.
NEEDS_START_COMMAND = {"experiment": False, "current_pid": True, "pi_profile": True}

# Substrings (checked lowercase) in the firmware's own banners.
BOOT_MARKER = "arming"  # printed once at boot by all three firmwares
WAITING_MARKER = "waiting for start"
ESTOP_MARKER = "estop ->"
RUNNING_MARKERS = ("-> ramping", "-> running experiment")
DONE_MARKERS = {
    "experiment": ("experiment complete", "failed to load"),
    "current_pid": ("coils off",),
    "pi_profile": ("coils off", "failed to load"),
}


def find_port(explicit=None):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found -- pass --port")


def stage_json(json_path: str) -> None:
    dest = os.path.join(REPO_ROOT, "task_sequences", "experiment.json")
    shutil.copyfile(json_path, dest)
    print(f"[pc] staged {json_path} -> {dest}")


def stage_profile(profile_path: str) -> None:
    result = subprocess.run([sys.executable, os.path.join(HERE, "validate_pi_profile.py"),
                              profile_path])
    if result.returncode != 0:
        raise SystemExit(f"{profile_path} failed validation -- fix it before staging "
                          "(see tools/validate_pi_profile.py output above)")
    dest = os.path.join(REPO_ROOT, "task_sequences", "pi_profile.json")
    shutil.copyfile(profile_path, dest)
    print(f"[pc] staged {profile_path} -> {dest}")


def build_cmd(pio: str, env: str, targets: list[str]) -> list[str]:
    cmd = [pio, "run", "-e", env]
    for t in targets:
        cmd += ["-t", t]
    return cmd


def run_pio(pio: str, upload_speed: int, env: str, targets: list[str]) -> None:
    cmd = build_cmd(pio, env, targets)
    envvars = dict(os.environ, PLATFORMIO_UPLOAD_SPEED=str(upload_speed))
    print(f"[pc] running: {' '.join(cmd)} (PLATFORMIO_UPLOAD_SPEED={upload_speed})")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=envvars)
    if result.returncode != 0:
        raise SystemExit(f"pio command failed (exit {result.returncode}): {' '.join(cmd)}")


class SerialSession:
    """EN-pulse reset + buffered non-blocking line reader. Accumulates raw
    bytes across calls and only returns a line once a complete '\\n'-
    terminated line has arrived -- NOT pyserial's blocking readline(with a
    short timeout), which silently TRUNCATES a line if it doesn't finish
    arriving within that timeout (the remaining bytes then get misread as
    the start of the next line). That truncation is exactly the kind of
    byte-loss this project has already hit once with byte-by-byte reads
    (see SerialComm.handle_serial_comm()'s docstring) -- a short poll
    timeout (e.g. the TUI's ~30ms tick) reintroduces the same class of bug
    by splitting a banner like "START -> running experiment" across two
    garbled reads, so the caller's substring matching on either fragment
    never fires and its state (running/phase) never updates. Buffering
    across calls makes line assembly independent of poll cadence."""

    def __init__(self, port: str, baud: int):
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0  # non-blocking; line assembly is buffered below
        self.ser.dtr = False  # IO0 stays high -> APP boot, not bootloader
        self.ser.rts = False
        self.ser.open()
        self._rxbuf = b""

    def reset(self) -> None:
        self.ser.reset_input_buffer()
        self._rxbuf = b""
        self.ser.dtr = False  # re-assert, not just once at __init__ -- confirmed on
        self.ser.rts = True   # hardware that some CP2102/CH340 chips fail to latch a
        time.sleep(0.15)      # single early dtr=False, occasionally dropping the board
        self.ser.dtr = False  # into DOWNLOAD_BOOT instead of a normal app boot (matches
        self.ser.rts = False  # esptool.py's own hard-reset, which re-touches DTR too)
        time.sleep(0.05)

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
        # Strip stray NUL bytes -- occasional line noise around EN-pulse
        # reset, never legitimate in this text telemetry protocol. Left in,
        # they crash curses.addstr() with "embedded null character".
        return raw.decode("utf-8", errors="replace").replace("\x00", "").rstrip("\r")

    def send(self, cmd: str) -> None:
        self.ser.write((cmd + "\n").encode())

    def close(self) -> None:
        self.ser.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fw", choices=("experiment", "current_pid", "pi_profile"),
                     help="which firmware/PlatformIO env to build+flash+run "
                          "(required unless --self-test)")
    ap.add_argument("--json", help="(--fw experiment) JSON to stage as task_sequences/experiment.json")
    ap.add_argument("--profile", help="(--fw pi_profile) profile JSON to validate+stage")
    ap.add_argument("--pi-compensate", action="store_true",
                     help="(--fw experiment) send 'pi=on' once booted, engaging "
                          "closed-loop PI compensation on top of the JSON schedule's "
                          "carrier-duty commands -- requires --pi-profile (same schema "
                          "as --fw pi_profile's --profile)")
    ap.add_argument("--pi-profile",
                     help="(--fw experiment, with --pi-compensate) ratio/gains profile "
                          "JSON to validate+stage as task_sequences/pi_profile.json")
    ap.add_argument("--dir", choices=("cw", "ccw"),
                     help="(--fw current_pid/pi_profile) send dir=<value> once WAITING is seen")
    ap.add_argument("--kp", type=float, help="send kp=<value> once WAITING is seen")
    ap.add_argument("--ki", type=float, help="send ki=<value> once WAITING is seen")
    ap.add_argument("--kd", type=float, help="send kd=<value> once WAITING is seen")
    ap.add_argument("--ramp", type=float, help="send ramp=<value> once WAITING is seen")
    ap.add_argument("--port", help="ESP32 serial port (default: first ttyUSB/ttyACM)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--skip-build", action="store_true",
                     help="don't build/uploadfs/upload -- board is already flashed")
    ap.add_argument("--pio", default=DEFAULT_PIO,
                     help="path to the pio binary (default: %(default)s -- the "
                          "apt-packaged system `pio` is known broken on this rig)")
    ap.add_argument("--upload-speed", type=int, default=DEFAULT_UPLOAD_SPEED)
    ap.add_argument("--out-dir", default=None,
                     help="directory for the session log + recordings "
                          "(default: data/<date>_<fw>/<time>)")
    ap.add_argument("--auto-start", action="store_true",
                     help="send the start command immediately once WAITING is seen, "
                          "instead of waiting for a human keypress in the TUI -- for "
                          "scripted/unattended callers only (sweeps, autotune trials)")
    ap.add_argument("--no-tui", action="store_true",
                     help="plain print()-based capture loop instead of the curses TUI "
                          "(headless/CI use); requires --auto-start, since there is no "
                          "other way to send the start command without the TUI")
    ap.add_argument("--self-test", action="store_true", help="check command construction, no hardware")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.fw:
        raise SystemExit("--fw is required")

    if args.no_tui and not args.auto_start and NEEDS_START_COMMAND[args.fw]:
        raise SystemExit("--no-tui requires --auto-start for --fw current_pid/pi_profile "
                          "(no other way to send the start command without the TUI)")

    if args.pi_compensate and args.fw != "experiment":
        raise SystemExit("--pi-compensate only applies to --fw experiment")

    if args.fw == "experiment":
        if args.json:
            stage_json(args.json)
        elif not args.skip_build:
            raise SystemExit("--json is required for --fw experiment unless --skip-build is given")
        if args.pi_compensate:
            if args.pi_profile:
                stage_profile(args.pi_profile)
            elif not args.skip_build:
                raise SystemExit("--pi-profile is required with --pi-compensate unless --skip-build is given")
    elif args.fw == "pi_profile":
        if args.profile:
            stage_profile(args.profile)
        elif not args.skip_build:
            raise SystemExit("--profile is required for --fw pi_profile unless --skip-build is given")

    if not args.skip_build:
        targets_pre = ["uploadfs"] if FW_NEEDS_SPIFFS[args.fw] else []
        if targets_pre:
            run_pio(args.pio, args.upload_speed, args.fw, targets_pre)
        run_pio(args.pio, args.upload_speed, args.fw, ["upload"])

    pending_commands = []
    if args.dir:
        pending_commands.append(f"dir={args.dir}")
    if args.kp is not None:
        pending_commands.append(f"kp={args.kp}")
    if args.ki is not None:
        pending_commands.append(f"ki={args.ki}")
    if args.kd is not None:
        pending_commands.append(f"kd={args.kd}")
    if args.ramp is not None:
        pending_commands.append(f"ramp={args.ramp}")
    if args.pi_compensate:
        pending_commands.append("pi=on")

    port = find_port(args.port)
    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, "data", f"{time.strftime('%Y-%m-%d')}_{args.fw}", time.strftime("%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "serial.log")

    print(f"[pc] fw={args.fw} port={port} @ {args.baud}, session -> {out_dir}")
    session = SerialSession(port, args.baud)
    session.reset()
    t0 = time.time()

    done_markers = DONE_MARKERS[args.fw]
    needs_start = NEEDS_START_COMMAND[args.fw]
    try:
        with open(log_path, "w") as log_f:
            if args.no_tui:
                seen_done = _plain_loop(session, log_f, t0, needs_start, done_markers, pending_commands)
            else:
                from experiment_tui import run as run_tui
                seen_done = run_tui(session, args.fw, out_dir, log_f, t0, needs_start, done_markers,
                                     pending_commands, auto_start=args.auto_start)
    finally:
        session.close()

    print(f"[pc] session log -> {log_path}")
    return 0 if seen_done else 1


def _plain_loop(session, log_f, t0, needs_start, done_markers, pending_commands) -> bool:
    """--no-tui capture loop: sends pending_commands once booted (plus the
    start command, for firmware that still gates on WAITING), prints/logs
    every line, e-stops unconditionally on exit."""
    sent_pending = False
    trigger_marker = WAITING_MARKER if needs_start else BOOT_MARKER
    trigger_seen = False
    seen_done = False
    try:
        while True:
            line = session.readline()
            if line is None:
                time.sleep(0.01)  # readline() is now non-blocking -- avoid busy-spin
                continue
            stamped = f"{time.time() - t0:7.2f}s  {line}"
            print(stamped, flush=True)
            log_f.write(stamped + "\n")
            log_f.flush()

            low = line.lower()
            if trigger_marker in low and not trigger_seen:
                trigger_seen = True
                if not sent_pending:
                    for cmd in pending_commands:
                        session.send(cmd)
                        time.sleep(0.1)
                    sent_pending = True
                if needs_start:
                    session.send("start")
            if any(m in low for m in done_markers) or ESTOP_MARKER in low:
                seen_done = True
                break
    except KeyboardInterrupt:
        print("\n[pc] stopped early by user", flush=True)
    finally:
        try:
            session.send("s")
            time.sleep(0.3)
        except Exception as e:
            print(f"[pc] WARNING: failed to send e-stop ({e})", flush=True)
    return seen_done


def self_test() -> int:
    """Command construction only, no hardware."""
    assert find_port("/dev/ttyXYZ") == "/dev/ttyXYZ"
    cmd = build_cmd("pio", "experiment", ["uploadfs"])
    assert cmd == ["pio", "run", "-e", "experiment", "-t", "uploadfs"]
    cmd2 = build_cmd("pio", "current_pid", ["upload"])
    assert cmd2 == ["pio", "run", "-e", "current_pid", "-t", "upload"]
    assert FW_NEEDS_SPIFFS["current_pid"] is False
    assert FW_NEEDS_SPIFFS["experiment"] is True
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
