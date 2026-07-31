#!/usr/bin/env python3
"""Flash and capture a swim run.

Uploads spiffs_data/ to SPIFFS, flashes [env:swim], EN-pulse resets the board
(ai/serial_comm.py's SerialComm, same timing as trigger_reset_log.py), and logs
serial for --capture-s. There is no schedule to stage: each firmware opens the
filename it was built for, and uploadfs ships every JSON in spiffs_data/ at
once.

main_swim.cpp prints no completion banner -- it just plays the schedule -- so
the capture is bounded by --capture-s. Size it from the runtime
ai/gen_swim_experiment.py prints (it already adds 5s of margin). A schedule
that fails to load IS detected, and exits non-zero.

Usage:
  uv run python ai/run_swim.py --capture-s 40 --log swim_run.log
  uv run python ai/run_swim.py --skip-build --log rerun.log   # already flashed
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_comm import SerialComm, find_port

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
DEFAULT_PIO = os.path.expanduser("~/.platformio/penv/bin/pio")  # system `pio` (apt) is broken -- see project memory
ENV_NAME = "swim"
POLL_S = 0.02

# JsonPhaseSequencer / driveBoot failure banners -- the run is not valid.
FAIL_MARKERS = ("cannot open", "spiffs mount failed")


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


def capture(port, baud, capture_s, log_path) -> bool:
    """EN-reset, then log serial for capture_s. Returns False if the firmware
    reported that it could not load its schedule."""
    comm = SerialComm(port=port, baud=baud)
    loaded = True
    try:
        comm.reset_device()
        t0 = time.time()
        with open(log_path, "w") as f:
            while time.time() - t0 < capture_s:
                line = comm.handle_serial_comm()
                if line is None:
                    time.sleep(POLL_S)
                    continue
                f.write(f"{time.time() - t0:7.2f}s  {line}\n")
                f.flush()
                print(f"[esp32] {line}")
                if any(marker in line.lower() for marker in FAIL_MARKERS):
                    loaded = False
                    break
    finally:
        comm.close()
    return loaded


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
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
    ap.add_argument("--capture-s", type=float, default=40.0,
                     help="how long to log serial after reset -- size this to the "
                          "runtime ai/gen_swim_experiment.py prints, which already "
                          "includes 5s of margin (default: %(default)s)")
    ap.add_argument("--log", help="serial log path (default: timestamped)")
    ap.add_argument("--self-test", action="store_true", help="check command construction, no hardware")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.skip_build:
        run_pio(args.pio, args.upload_speed, ["uploadfs"])
        run_pio(args.pio, args.upload_speed, ["upload"])

    port = find_port(args.port)
    log_path = args.log or f"swim_run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"[pc] capturing on {port} @ {args.baud}, log -> {log_path} "
          f"(for {args.capture_s:.0f}s)")
    loaded = capture(port, args.baud, args.capture_s, log_path)
    if not loaded:
        print("[pc] ERROR: firmware could not load its schedule -- run "
              "`pio run -e swim -t uploadfs` and check spiffs_data/swim.json")
    print(f"[pc] log saved -> {log_path}")
    return 0 if loaded else 1


def self_test() -> int:
    """Command construction only, no hardware."""
    assert find_port("/dev/ttyXYZ") == "/dev/ttyXYZ"
    cmd = build_cmd("pio", 115200, ["uploadfs"])
    assert cmd == ["pio", "run", "-e", "swim", "-t", "uploadfs"]
    cmd2 = build_cmd("pio", 115200, ["upload"])
    assert cmd2 == ["pio", "run", "-e", "swim", "-t", "upload"]
    print("self-test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
