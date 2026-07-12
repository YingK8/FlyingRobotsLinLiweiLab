#!/usr/bin/env python3
"""Orchestrate a full CW+CCW coil-coupling characterization sweep. For each
direction: copy the matching task_sequences/coupling_<dir>.json onto the ESP32's
SPIFFS image (main_experiment.cpp always loads /experiment.json, see
task_sequences/README.md) and flash it, then record a PicoScope capture in sync
with a fresh boot-relative run: start picoscope_record.py first, then
tools/trigger_reset_log.py's EN-pulse reset + labeled-telemetry log a few
seconds later. Reuses the existing reset/record primitives as-is; the JSON
experiment needs no commands sent once armed, so trigger_reset_log.py's plain
reset+log mode is enough (unlike tools/pid_autotune.py, which does send
commands). Regenerate the JSON files first with tools/gen_coupling_experiment.py
for different current levels/timing.

Usage:
  uv run python tools/run_coupling_sweep.py [--port /dev/ttyUSB0]
      [--pio ~/.platformio/penv/bin/pio] [--direction cw ccw]
      [--assume-coils-off]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
task_sequences = os.path.join(REPO_ROOT, "task_sequences")
ARM_MS = 3000       # main_experiment.cpp's fixed ARMING duration
MARGIN_S = 10.0     # extra recording time beyond the computed experiment length


def compute_total_s(json_path: str) -> float:
    """Total experiment duration = ARM_MS + every addWaitTask's duration_ms.
    Read from the JSON itself (not hardcoded) so it stays correct regardless
    of gen_coupling_experiment.py's --levels/--active-ms/--gap-ms settings."""
    with open(json_path) as f:
        events = json.load(f)
    wait_ms = sum(e.get("duration_ms", 0) for e in events
                  if e.get("method") == "addWaitTask")
    return (ARM_MS + wait_ms) / 1000.0


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def flash_experiment(pio: str, direction: str) -> None:
    src = os.path.join(task_sequences, f"coupling_{direction}.json")
    dst = os.path.join(task_sequences, "experiment.json")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run tools/gen_coupling_experiment.py first")
    shutil.copyfile(src, dst)
    print(f"copied {src} -> {dst}")
    run([pio, "run", "-e", "experiment", "-t", "uploadfs"], cwd=REPO_ROOT)
    run([pio, "run", "-e", "experiment", "-t", "upload"], cwd=REPO_ROOT)


def record_direction(direction: str, total_s: float, port, out_dir: str):
    ts = time.strftime("%H%M%S")
    csv_path = os.path.join(out_dir, f"coupling_{direction}_{ts}.csv")
    log_path = os.path.join(out_dir, f"coupling_{direction}_{ts}_serial.log")
    record_s = total_s + MARGIN_S

    scope_cmd = [sys.executable, os.path.join(REPO_ROOT, "tools", "picoscope_record.py"),
                 "--seconds", str(record_s), "--out", csv_path]
    print("+", " ".join(scope_cmd), "(background)")
    scope_proc = subprocess.Popen(scope_cmd)

    trigger_cmd = [sys.executable, os.path.join(REPO_ROOT, "tools", "trigger_reset_log.py"),
                   "--delay", "4", "--log-seconds", str(record_s), "--out", log_path]
    if port:
        trigger_cmd += ["--port", port]
    run(trigger_cmd)

    scope_proc.wait()
    print(f"[{direction}] saved {csv_path} and {log_path}")
    return csv_path, log_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    ap.add_argument("--pio", default=os.path.expanduser("~/.platformio/penv/bin/pio"))
    ap.add_argument("--direction", nargs="+", default=["cw", "ccw"], choices=["cw", "ccw"])
    ap.add_argument("--out-dir", default=None,
                     help="default: data/<today>_coupling-sweep/ (see data/README.md convention)")
    ap.add_argument("--assume-coils-off", action="store_true",
                     help="skip the interactive coil-off confirmation before flashing")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        REPO_ROOT, "data", f"{time.strftime('%Y-%m-%d')}_coupling-sweep")
    os.makedirs(out_dir, exist_ok=True)

    if not args.assume_coils_off:
        resp = input("Flashing resets the board through the bootloader -- "
                      "confirm the coil supply is OFF before continuing [y/N]: ")
        if resp.strip().lower() != "y":
            raise SystemExit("aborted -- turn the coil supply off, then re-run "
                              "(or pass --assume-coils-off)")

    results = []
    for direction in args.direction:
        json_path = os.path.join(task_sequences, f"coupling_{direction}.json")
        total_s = compute_total_s(json_path)
        print(f"== {direction.upper()}: experiment ~{total_s:.1f}s, "
              f"recording {total_s + MARGIN_S:.1f}s ==")
        flash_experiment(args.pio, direction)
        input("Firmware flashed -- power the coils now, then press Enter to "
              "start recording.")
        csv_path, log_path = record_direction(direction, total_s, args.port, out_dir)
        results.append((direction, csv_path, log_path))

    print("\nDone. Results:")
    for direction, csv_path, log_path in results:
        print(f"  {direction}: {csv_path}  {log_path}")
    print(f"\nAdd a session entry to data/README.md for {out_dir} per the "
          "existing convention. Analyze with, e.g.:")
    for direction, csv_path, log_path in results:
        print(f"  uv run python tools/coupling_matrix.py {csv_path} "
              f"--segments-log {log_path} --direction {direction}")


if __name__ == "__main__":
    main()
