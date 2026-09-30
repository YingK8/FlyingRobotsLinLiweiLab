#!/bin/zsh
# Re-record 5 takes each at 40, 10 and 20 Hz with the ORIGINAL hold (3 s settle + 7 s hold).
#
# Why: 40 Hz reps 1-5 were solved at stride 4, above which the once-per-rev wobble aliases,
# and their video was deleted so they cannot be re-solved. 10 and 20 Hz sit below the
# stride-4 Nyquist and were never aliased, but their video is gone too, so they have no
# stride-1 population. Nothing is replaced -- every earlier take stays on disk and in
# index.csv.
#
# The hold went 7000 -> 500 -> 2000 -> 7000 during 2026-09-10. Reps recorded at the short
# holds are a DIFFERENT CONDITION (the hold sets how settled the robot is at the cut), so
# they are kept but should not be pooled with these. Only takes recorded at 3+7 s are
# like-for-like with the campaign's ~100 originals.
#
# Thermal gate OFF at the operator's instruction. It is not a real risk here: 0.67 C/repeat
# at 40 Hz, 0.15 at 20 and 0.03 at 10, so ~4.3 C for all fifteen takes.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."

for f in 40 10 20; do
  echo "=== chunk $f Hz x 5  $(date +%H:%M) ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$f" --repeats 5 \
      --no-cool --force-hot --out "$ROOT" || echo "!! $f Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
done
echo "=== re-record done $(date) ==="
