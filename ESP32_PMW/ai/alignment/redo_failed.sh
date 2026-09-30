#!/bin/zsh
# Replace the two takes the operator watched fail on 2026-09-10:
#   10 Hz repeat 19  (2026-09-10_074759)
#   20 Hz repeat 11  (2026-09-10_075012)
# Both are marked failed:operator in index.csv and stay on disk -- the host recorded them
# outcome=ok because the SCHEDULE ran to TILT_OFF, which is all it can see. Whether the
# robot actually flew is not observable from the serial log, only from the bench.
#
# Same hold as the campaign (3 s settle + 7 s hold). Thermal gate off; 0.2 C for the pair.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."
echo "=== waiting for the running chunk ==="
while pgrep -f "tilt_run.py" > /dev/null; do sleep 20; done
for f in 10 20; do
  echo "=== redo $f Hz x 1  $(date +%H:%M) ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$f" --repeats 1 \
      --no-cool --force-hot --out "$ROOT" || echo "!! $f Hz redo exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
done
echo "=== redos done $(date) ==="
