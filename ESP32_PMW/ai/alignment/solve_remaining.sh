#!/bin/zsh
# Solve the whole-ring blocks that have no axis.csv yet, two at a time.
#
# WHY A LIMIT AND NOT ONE BIG BATCH. `disc_axis.solve` runs at ~170% CPU by itself, so on this
# 8-core machine two jobs is ~3.5 cores and leaves room for the desktop. Five at once
# oversubscribes and every one of them slows down; the wall time barely improves.
#
# Each take goes through `process_block.py`, so each is validated and uploaded on its own:
# one bad take cannot stop the others, and the ones that pass are on the USB before the next
# pair starts.
#
# NOT run while a recorder is live. The cameras and the drive share a USB bus, and a solve or
# a copy alongside a take shows up as dropped frames.
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python

TAKES=(
  2026-09-24_042238_whole90
  2026-09-24_043944_whole90
  2026-09-24_045651_whole100
  2026-09-24_051416_whole100
  2026-09-24_053143_whole110
)

# Wait for any solve already running (the 120 Hz block, if the sweep was stopped mid-process).
while pgrep -f disc_axis > /dev/null; do
  echo "[queue] waiting for a solve already running"
  sleep 60
done

MAX_JOBS=${MAX_JOBS:-1}

# Count live solves from an EXPLICIT PID LIST, kept by this shell.
#
# Two earlier attempts at this throttle both silently failed to engage:
#   `pgrep -fc 'ai/alignment/process_block.py'` -- the name is truncated in the process table,
#      so it matched nothing and returned an empty string, making `[ "" -ge 2 ]` false;
#   `jobs -p` -- zsh does not track jobs in a non-interactive script unless MONITOR is set.
# Both launched every solve at once. Five at once put load at 21 on 8 cores and drove swap to
# 9.5 of 10.2 GB; three killed by hand, and two takes were left with unusable partial CSVs.
# A list of PIDs this shell started has no such ambiguity, and is the whole point of a throttle.
pids=()
alive() {   # rewrite pids[] with the ones still running; return the count
  local out=()
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && out+=("$p"); done
  pids=("${out[@]}")
  echo "${#pids[@]}"
}

failed=()
for take in "${TAKES[@]}"; do
  out="results/rim/$take"
  if [ -f "$out/axis.csv" ]; then
    echo "[skip] $take already solved"
    continue
  fi
  while [ "$(alive)" -ge "$MAX_JOBS" ]; do sleep 15; done
  echo "[solve] $take  (~23 min)"
  $PY ai/alignment/process_block.py "results/flights/$take" > "/tmp/solve_${take}.log" 2>&1 &
  pids+=($!)
done
while [ "$(alive)" -gt 0 ]; do sleep 15; done

echo "=== results ==="
for take in "${TAKES[@]}"; do
  log="/tmp/solve_${take}.log"
  verdict=$(grep -o 'VERDICT: [A-Z ]*' "$log" 2>/dev/null | tail -1)
  up=$(grep -c 'uploaded ->' "$log" 2>/dev/null)
  echo "  $take  ${verdict:-no verdict}  uploaded=${up}"
done
