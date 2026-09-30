#!/bin/zsh
# Remaining frequencies, ONE PROCESS EACH.
#
# The single-process sweep segfaulted (exit 139) during 50 Hz repeat 3, in the camera-open
# path after ~17 open/close cycles of the AVFoundation capture. A segfault cannot be caught
# in-process and it skips tilt_sweep's `finally`, so it takes the whole campaign with it and
# leaves nothing parked. Running one chunk per process costs a few seconds of startup each
# and bounds the damage to one frequency.
#
# The board is parked between chunks because a crashed chunk may have skipped its own park.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")"

for f in 50 60 70 80 90 100 120 130; do
  echo "=== chunk ${f} Hz ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$f" \
      --repeats 5 --no-cool --out "$ROOT" || echo "!! ${f} Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
done
echo "=== all chunks attempted ==="
