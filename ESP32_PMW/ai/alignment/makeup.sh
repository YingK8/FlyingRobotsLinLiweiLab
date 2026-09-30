#!/bin/zsh
# Makeup ramps: bring 70-120 Hz to 10 repeats each. 32 takes, ~246 C of predicted heating.
#
# ORDERED COLDEST FIRST. 70 Hz books 3.0 C a repeat and 120 Hz books 10.7, so running the
# cheap chunks while the coils are cold and the expensive ones after a long soak is the only
# scheduling that matters -- the campaign is cooling-bound, not run-bound.
#
# 246 C against ~48 C of usable headroom (70 C ceiling, 22 C ambient) is FIVE sessions, not
# one. SESSION is which block to run: pass SESSION=1..5. Nothing here reads a temperature --
# --force-hot is on and the operator is the thermometer. Take a reading before each session
# and re-anchor with:
#     date '+<measured>  %Y-%m-%d %H:%M:%S' > ai/thermal/.last_energised
#
# 120 Hz needs 7 because 3 of its 10 were lost to GPIO14 aborts on 2026-09-10.
set -u
ROOT=results/alignment_rate/20260909_205843
COOL=${COOL:-1500}
SESSION=${SESSION:-0}
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

case "$SESSION" in
  1) run 70 5 ;;                       # 14.8 C
  2) run 80 5 ;;                       # 23.6 C
  3) run 90 5 ;;                       # 34.2 C
  4) run 100 5 ;;                      # 47.4 C
  5) run 110 5 ;;                      # 50.5 C
  6) run 120 4; run 120 3 ;;           # 75.0 C, split so the second half gets a cool
  *) echo "SESSION=1..6 required. 1:70Hz 2:80Hz 3:90Hz 4:100Hz 5:110Hz 6:120Hz(x7)"
     echo "Take a coil temperature first and re-anchor ai/thermal/.last_energised."
     exit 2 ;;
esac
echo "=== session $SESSION done ==="
