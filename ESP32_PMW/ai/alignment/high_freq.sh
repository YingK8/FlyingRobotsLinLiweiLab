#!/bin/zsh
# The top of the range: 110 Hz x 5 then 120 Hz x 5. Neither has ever been run.
#
# The open question at the top end is not the angle -- every frequency from 10 to 100 lands
# at 40-53 deg -- but whether the rig completes the point at all. The failure rate climbs with
# frequency (9/10 at 10 Hz, 9/9 at 20, 5/5 at 40, 8/10 at 50, 5/9 at 60, 2/4 at 70, and 100 Hz
# needed three attempts before one stuck), so five repeats is the right size to find out
# before committing ten repeats of heat.
#
# ~119 C for the pair with the shortened high-frequency hold, against a 48 C headroom from
# ambient, so --force-hot and 25 min of cooling before each chunk. The operator is the
# thermometer; GPIO14 is the stop.
#
# 110 Hz ramps 44 s and 120 Hz ramps 46 s, both close to MAX_RAMP_S = 55, so the high-f ramp
# cannot be slowed much further to buy heat back.
set -u
ROOT=results/alignment_rate/20260909_205843
COOL=${COOL:-1500}
cd "$(dirname "$0")/../.."

run() {
  echo "=== cooling ${COOL}s before $1 Hz ==="
  sleep "$COOL"
  echo "=== chunk $1 Hz x $2 new ==="
  uv run python -u controller/control/tilt_run.py --sweep --freqs "$1" --repeats "$2" \
      --no-cool --force-hot --out "$ROOT" || echo "!! $1 Hz chunk exited $?"
  uv run python -c "from controller.control.tilt_sweep import park; park('/dev/cu.SLAB_USBtoUART')" \
      2>/dev/null | grep -a coils
}

run 110 4      # 1 already done before the disk guard stopped it
run 120 5
echo "=== high-frequency set done ==="
