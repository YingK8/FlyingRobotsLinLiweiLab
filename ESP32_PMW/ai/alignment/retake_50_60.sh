#!/bin/zsh
# Retakes at 50 and 60 Hz, appending to the existing chunks (5 -> 10 repeats each).
#
# The angle at these frequencies is already well determined -- the modal repeats sit at
# 47.9 and 48.6 deg with SD 2.7 and 3.5. What is NOT determined is how OFTEN the robot
# reaches it: 3/5 at both, and five Bernoulli trials put that anywhere from 15% to 95%.
# Ten repeats is what separates "usually works" from "coin flip".
#
# Coils measured 63 C at the start, so only ~7 C of headroom: --force-hot is required and
# the operator is watching. 15 min of cooling between the two chunks.
set -u
ROOT=results/alignment_rate/20260909_205843
cd "$(dirname "$0")/../.."

run() {
  echo "=== chunk $1 Hz x $2 new ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$1" --repeats "$2" \
      --no-cool --force-hot --out "$ROOT" || echo "!! $1 Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
}

run 50 5
echo "=== cooling 900s before 60 Hz ==="
sleep 900
run 60 5
echo "=== retakes done ==="
