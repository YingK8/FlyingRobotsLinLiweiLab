#!/bin/zsh
# One frequency of the rim J(f) experiment, as two blocks of N repeats.
#
# There was no driver for this protocol: `ai/make_tilt.py` writes the schedule and
# `controller/camera/record.py` films it, and the host has to time the stop, close the
# take and park the board itself. Nothing here reads a temperature, so the coils are the
# operator's job -- GPIO14 is the only kill `main_tilt` can hear.
#
#   zsh ai/alignment/run_rim_blocks.sh <freq_hz> <ramp_ms> [n_per_block] [pause_s]
#
# RAMP is rate-based, matching `controller/control/tilt_schedule.py`: 3.5 Hz/s below
# 60 Hz, 2.8 Hz/s at or above it, so ramp_ms = round((f - 1) / rate * 1000). At 90 Hz
# that is 31786 ms; at 150 Hz, 53214 ms. The ramp is not a warm-up -- all four coils sit
# at 100% carrier while it sweeps up, so it is the largest single heat term in a repeat.
#
# TWO BLOCKS, NOT ONE OF 10. Measured heating in this protocol is ~9 C a repeat at 90 Hz
# rising to ~14 C at 150 Hz, against an all-off window of 21.5 s that sheds ~0.4-0.7 C.
# Ten back-to-back repeats therefore climb to 100-145 C from ambient. The pause between
# blocks does NOT lower that peak -- only the block SIZE does -- so keep N at 5 (peak
# ~66 C at 90 Hz, ~93 C at 150 Hz) or drop to 3 for the top of the range (~64 C).
#
# The note string drives `record.take_suffix`, so each take lands as
# `results/flights/<stamp>_whole<freq>/`; blk1/blk2 go into meta.json for provenance.
set -u
F=${1:?usage: run_rim_blocks.sh <freq_hz> <ramp_ms> [n_per_block] [pause_s]}
RAMP=${2:?give the ramp in ms, e.g. 31786 for 90 Hz at 2.8 Hz/s}
N=${3:-5}
PAUSE=${4:-600}
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
REC=""
cleanup() {
  echo '*** INTERRUPTED: closing the take and parking the board ***'
  [ -n "$REC" ] && { kill -INT $REC 2>/dev/null; wait $REC 2>/dev/null; }
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" 2>&1 | tail -1
  exit 130
}
trap cleanup INT TERM

HOLD=5000
[ "$F" -gt 100 ] && HOLD=3000          # matches make_tilt's default
period=$(( (RAMP + HOLD + 15000 + 1000 + 21500) / 1000 ))   # post=15000, reset=1000, off=21500
total=$(( period * N ))

for K in 1 2; do
  echo "===== ${F} Hz block ${K}/2  ramp=${RAMP}ms  n=${N}  post=15s  schedule ${total}s ====="
  uv run python ai/make_tilt.py --hz "$F" --n "$N" whole --ramp-ms "$RAMP" --post-ms 15000 || break
  pio run -e tilt -t uploadfs 2>&1 | tail -2 || break
  $PY controller/camera/record.py --mode 640x400 --start \
      --note "whole ring ${F}Hz x${N} without cardan post15 blk${K}" > /tmp/rim_${F}_b${K}_rec.log 2>&1 &
  REC=$!
  ( sleep $((total + 20)); kill -INT $REC 2>/dev/null ) &
  KILLER=$!
  wait $REC 2>/dev/null
  kill $KILLER 2>/dev/null
  REC=""
  tail -3 /tmp/rim_${F}_b${K}_rec.log
  T=$(ls -dt results/flights/*_whole${F} 2>/dev/null | head -1)
  [ -n "$T" ] && echo "   take=$T  frames=$(( $(wc -l < "$T/frames.csv") - 1 ))"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" 2>&1 | tail -1
  [ "$K" -eq 1 ] && { echo "===== cooling ${PAUSE} s before block 2 ====="; sleep "$PAUSE"; }
done
echo "===== ${F} Hz DONE: 2 blocks of ${N} ====="
