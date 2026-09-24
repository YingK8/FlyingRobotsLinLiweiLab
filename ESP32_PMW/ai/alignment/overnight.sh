#!/bin/zsh
# Overnight makeup run, 45 takes. Waits for the stride-1 reprocess to finish first.
#
# THE THERMAL GATE IS ON. Every chunk tonight ran --no-cool --force-hot with the operator
# reading a thermometer; unattended there is no thermometer, so this uses tilt_run's own
# gate, which waits until the model says the next point fits under the 70 C ceiling and
# sizes each gap from that point's own heat (0.03 C a repeat at 10 Hz, 10.7 at 120). It is
# a MODEL, not a sensor -- there is no temperature sensor on this rig and no watchdog. The
# supply's 10 A limit and the GPIO14 button remain the only hard stops.
#
# Sequential, never concurrent with the re-solve: capture writes 2x210 fps and CPU
# starvation shows up as dropped frames, which silently corrupts a take.
#
# Coldest first, because the campaign is cooling-bound: 228 C of heating against ~48 C of
# headroom means the order decides the wall-clock, not the run time.
#
# 10/20/40 Hz are re-run to REPLACE the five takes each whose video was deleted on
# 2026-09-10 -- their stride-4 CSVs survive but can never be re-solved above Nyquist.
# 70-120 Hz are short of the 10-repeat target.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."

echo "=== reprocess already complete ==="
echo "=== reprocess finished, starting hardware $(date) ==="

run() {
  echo "=== chunk $1 Hz x $2  $(date +%H:%M) ==="
  # NO --no-cool and NO --force-hot: the gate does the waiting.
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$1" --repeats "$2" \
      --out "$ROOT" || echo "!! $1 Hz chunk exited $?"
  # Park after every chunk in case the chunk died before its own finally ran.
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
}

for f in 20 40 70 80 90 100 110 120; do
  run $f 5
done
echo "=== overnight done $(date) ==="
uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
    2>/dev/null | grep -a coils
