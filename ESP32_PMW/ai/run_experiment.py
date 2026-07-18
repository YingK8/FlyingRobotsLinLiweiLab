#!/usr/bin/env python3
"""Build, flash, and capture a main_experiment.cpp JSON-driven run (env
`experiment`) -- e.g. ai/gen_solo_sweep_experiment.py's or
ai/gen_coupling_experiment.py's output.

This REPLACES this file's old main_tilt.cpp-specific b/s/e single-letter
protocol: main_tilt.cpp no longer exists (platformio.ini only defines the
current_pid/serialcomm_demo/experiment envs), and main_experiment.cpp is
fully autonomous -- ARMING -> RUNNING -> DONE with NO serial commands needed
during a run (see its header comment). So this driver stages the JSON,
(optionally) builds+flashes, EN-pulse resets (ai/serial_comm.py's
SerialComm, same timing as trigger_reset_log.py), and logs serial until the
firmware's own completion banner appears or --timeout-s elapses -- bounded by
the firmware's real state machine, not by sending stop commands it doesn't
have.

Usage:
  uv run python ai/run_experiment.py --json spiffs_data/solo_sweep.json \
      --timeout-s 300 --log solo_sweep_run.log
  uv run python ai/run_experiment.py --skip-build --log rerun.log   # already flashed
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_comm import SerialComm, find_port

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
DEFAULT_PIO = os.path.expanduser("~/.platformio/penv/bin/pio")  # system `pio` (apt) is broken -- see project memory
ENV_NAME = "experiment"
POLL_S = 0.02

# main_experiment.cpp's own terminal banners -- either means the run is over.
DONE_MARKERS = ("experiment complete", "failed to load")


def stage_json(json_path: str) -> None:
    dest = os.path.join(REPO_ROOT, "spiffs_data", "experiment.json")
    shutil.copyfile(json_path, dest)
    print(f"[pc] staged {json_path} -> {dest}")


def build_cmd(pio: str, upload_speed: int, targets: list[str]) -> list[str]:
    cmd = [pio, "run", "-e", ENV_NAME]
    for t in targets:
        cmd += ["-t", t]
    return cmd


def run_pio(pio: str, upload_speed: int, targets: list[str]) -> None:
    cmd = build_cmd(pio, upload_speed, targets)
    env = dict(os.environ, PLATFORMIO_UPLOAD_SPEED=str(upload_speed))
    print(f"[pc] running: {' '.join(cmd)} (PLATFORMIO_UPLOAD_SPEED={upload_speed})")
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"pio command failed (exit {result.returncode}): {' '.join(cmd)}")


def capture(port, baud, timeout_s, log_path) -> bool:
    """EN-reset, then log serial until a DONE_MARKERS line appears or
    timeout_s elapses. Returns True if a completion banner was seen."""
    comm = SerialComm(port=port, baud=baud)
    saw_done = False
    try:
        comm.reset_device()
        t0 = time.time()
        with open(log_path, "w") as f:
            while time.time() - t0 < timeout_s:
                line = comm.handle_serial_comm()
                if line is None:
                    time.sleep(POLL_S)
                    continue
                f.write(f"{time.time() - t0:7.2f}s  {line}\n")
                f.flush()
                print(f"[esp32] {line}")
                if any(marker in line.lower() for marker in DONE_MARKERS):
                    saw_done = True
                    break
    finally:
        comm.close()
    return saw_done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json",
                     help="experiment JSON to stage as spiffs_data/experiment.json "
                          "before building (omit with --skip-build to just capture "
                          "an already-flashed run)")
    ap.add_argument("--port", help="ESP32 serial port (default: first ttyUSB/ttyACM)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--skip-build", action="store_true",
                     help="don't build/uploadfs/upload -- board is already flashed")
    ap.add_argument("--pio", default=DEFAULT_PIO,
                     help="path to the pio binary (default: %(default)s -- the "
                          "apt-packaged system `pio` is known broken on this rig)")
    ap.add_argument("--upload-speed", type=int, default=115200,
                     help="460800 (platformio default) fails 'Unable to verify "
                          "flash chip' on this rig (default: %(default)s)")
    ap.add_argument("--timeout-s", type=float, default=300.0,
                     help="max time to wait for the firmware's completion banner "
                          "-- size this to the sweep generator's printed total "
                          "runtime + margin (default: %(default)s)")
    ap.add_argument("--log", help="serial log path (default: timestamped)")
    ap.add_argument("--self-test", action="store_true", help="check command construction, no hardware")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.json:
        stage_json(args.json)
    elif not args.skip_build:
        raise SystemExit("--json is required unless --skip-build is given "
                          "(nothing to flash otherwise)")

    if not args.skip_build:
        run_pio(args.pio, args.upload_speed, ["uploadfs"])
        run_pio(args.pio, args.upload_speed, ["upload"])

    port = find_port(args.port)
    log_path = args.log or f"experiment_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"[pc] capturing on {port} @ {args.baud}, log -> {log_path} "
          f"(timeout {args.timeout_s:.0f}s)")
    saw_done = capture(port, args.baud, args.timeout_s, log_path)
    if not saw_done:
        print(f"[pc] WARNING: timed out after {args.timeout_s:.0f}s without seeing "
              "a completion banner -- log may be incomplete, or --timeout-s is too short")
    print(f"[pc] log saved -> {log_path}")
    return 0 if saw_done else 1


def self_test() -> int:
    """Command construction only, no hardware."""
    assert find_port("/dev/ttyXYZ") == "/dev/ttyXYZ"
    cmd = build_cmd("pio", 115200, ["uploadfs"])
    assert cmd == ["pio", "run", "-e", "experiment", "-t", "uploadfs"]
    cmd2 = build_cmd("pio", 115200, ["upload"])
    assert cmd2 == ["pio", "run", "-e", "experiment", "-t", "upload"]
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
