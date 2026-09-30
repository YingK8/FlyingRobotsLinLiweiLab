#!/bin/zsh
# Bring 70/80/90/100 Hz from 5 to 10 repeats -- the last of the "all frequencies to 10
# points" target. Run AFTER high_freq.sh (110/120) has exited.
#
# ~165 C of predicted heating (22.5 + 34.5 + 46.5 + 62), the most expensive set left, so
# 25 min between chunks and the operator is the thermometer: --force-hot is on and nothing
# here reads a temperature. Launch only with a measured coil temp in hand.
#
# `--repeats` counts NEW repeats: run_chunk appends to an existing index and continues the
# numbering, so 5 takes each of these from 5 to 10.
set -u
ROOT=results/alignment_rate/20260909_205843
COOL=${COOL:-1500}
cd "$(dirname "$0")/../.."

run() {
  echo "=== chunk $1 Hz x $2 new ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$1" --repeats "$2" \
      --no-cool --force-hot --out "$ROOT" || echo "!! $1 Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
  echo "=== cooling ${COOL}s ==="
  sleep "$COOL"
}

run 70 5
run 80 5
run 90 5
run 100 5
echo "=== 70-100 top-up done ==="
