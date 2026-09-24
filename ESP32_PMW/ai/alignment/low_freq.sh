#!/bin/zsh
# Low-frequency top-up: 10, 20, 40 Hz to 10 repeats each, and 30 Hz from scratch at 10.
#
# 12.3 C of predicted heating for the whole set -- these are the cheap frequencies (10 Hz is
# 0.06 C a repeat against 13 C at 100 Hz), so 5 min between chunks is ample rather than the
# 15 the high end needed.
#
# `--repeats` counts NEW repeats: run_chunk appends to an existing index and continues the
# numbering, so 5 here takes each of 10/20/40 Hz from 5 to 10. 30 Hz has no prior index and
# gets all 10.
set -u
ROOT=results/alignment_rate/20260909_205843
COOL=${COOL:-300}
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

run 10 5
run 20 5
run 30 10
run 40 5
echo "=== low-frequency set done ==="
