#!/bin/zsh
# Finish 90 Hz (3 more) and run 100 Hz (5), thermal gate OFF at the operator's instruction
# (coils measured 36 C, stamp re-anchored to it). ~87 C of predicted heating.
#
# One process per chunk: the single-process sweep segfaulted in the camera-open path after
# ~17 open/close cycles, which skips tilt_sweep's `finally` and parks nothing.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."

run() {
  echo "=== chunk $1 Hz x $2 ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$1" --repeats "$2" \
      --no-cool --force-hot --out "$ROOT" || echo "!! $1 Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
}

run 90 3
run 100 5
echo "=== both chunks attempted ==="
