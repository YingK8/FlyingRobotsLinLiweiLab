#!/bin/zsh
# Remaining campaign, thermal gate OFF and headroom guard overridden at the operator's
# explicit approval (2026-09-10). ~250 C predicted. The operator is the thermometer and
# GPIO14 is the stop.
#
# One process per chunk: the single-process sweep segfaulted in the camera-open path after
# ~17 open/close cycles, which skips tilt_sweep's `finally` and parks nothing. The board is
# parked between chunks in case a chunk died before its own park.
#
# `--repeats` COUNTS NEW REPEATS: run_chunk appends to an existing index and continues the
# numbering, so 100 Hz asks for 3 to reach 5 while the others ask for 5.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."

COOL=${COOL:-900}   # 15 min between chunks, at the operator's instruction (70 C measured)

run() {
  echo "=== cooling ${COOL}s before $1 Hz ==="
  sleep "$COOL"
  echo "=== chunk $1 Hz x $2 new ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$1" --repeats "$2" \
      --no-cool --force-hot --out "$ROOT" || echo "!! $1 Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
}

run 100 1      # repeat 5, after 3 GPIO14 aborts
run 110 5      # new frequency
run 120 5
run 50  5      # retakes: characterise the modal fraction, not the angle
run 60  5
run 70  5
echo "=== all chunks attempted ==="
