#!/bin/zsh
# Re-record 5 takes each at 10 and 20 Hz, after the 40 Hz chunk finishes.
#
# These REPLACE nothing: the earlier takes stay on disk and in index.csv. At 10 and 20 Hz
# the old stride-4 solves are still valid -- the wobble sits at the drive frequency and both
# are below the stride-4 Nyquist of 25.6 Hz, so they were never aliased. They simply cannot
# be re-solved at stride 1, because their video was deleted in error on 2026-09-10. The new
# takes give those two frequencies a stride-1 population as well.
#
# Cheap: 0.03 C/repeat at 10 Hz and 0.15 at 20, so 0.9 C for the pair of chunks. The thermal
# gate is ON and does the pacing.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."

echo "=== waiting for the 40 Hz chunk ==="
while pgrep -f "tilt_run.py" > /dev/null; do sleep 30; done

for f in 10 20; do
  echo "=== chunk $f Hz x 5  $(date +%H:%M) ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$f" --repeats 5 \
      --out "$ROOT" || echo "!! $f Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
done
echo "=== low-frequency re-record done $(date) ==="
