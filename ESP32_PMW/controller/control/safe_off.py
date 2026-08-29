#!/usr/bin/env python3
"""
De-energise the coils, and prove it from the firmware's own telemetry.

    uv run python controller/control/safe_off.py

`controller_loop` already sends `stop` from a `finally`, but "the command was sent" and
"the coils are off" are different claims, and only the second one matters when walking
away from a rig with a 12 V 10 A supply behind it. A `finally` can be cut short, a serial
write can be buffered, and a board that rebooted mid-run comes back in IDLE having never
seen the command at all.

So this sends `stop` and then *reads back* `driveTelemetry` (`src/drive_common.h:39-63`),
which reports per-coil current and duty at 2 Hz in every state, and exits non-zero unless
they are all zero. Run it after any session that energised the coils, and any time the
state of the rig is not certain.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

# "t=17644 freq=0.0 | I[A]: A=0.00 B=0.00 C=0.00 D=0.00 | duty[%]: A=0.0 ... "
_I_RE = re.compile(r"I\[A\]:\s*(.*?)\s*\|")
_DUTY_RE = re.compile(r"duty\[%\]:\s*(.*?)(?:\||$)")
_VAL_RE = re.compile(r"[A-D]=(-?[\d.]+)")

I_EPS = 0.02  # A, below the sense noise floor
DUTY_EPS = 0.5  # %


def _values(line, head):
    m = head.search(line)
    return [float(v) for v in _VAL_RE.findall(m.group(1))] if m else []


def coils_off(port=None, settle_s=3.0, need=2):
    """
    Send `stop`, then wait for `need` consecutive all-zero telemetry lines.

        Returns ``(ok, last_line)``. Several lines rather than one because the drive
        ramps down rather than stopping dead, and a single sample can land mid-ramp
        and read zero by luck at a current zero-crossing.
    """

    from link import SerialComm

    comm = SerialComm(port=port)
    try:
        for _ in range(3):  # cheap redundancy: `stop` is idempotent
            comm.handle_serial_comm("stop")
            time.sleep(0.05)

        clean, last, deadline = 0, "", time.monotonic() + settle_s
        while time.monotonic() < deadline:
            line = comm.handle_serial_comm()
            if line is None:
                time.sleep(0.01)
                continue
            if "I[A]" not in line:
                continue
            last = line
            amps, duty = _values(line, _I_RE), _values(line, _DUTY_RE)
            if not amps:
                continue
            quiet = (all(abs(a) <= I_EPS for a in amps)
                     and all(abs(d) <= DUTY_EPS for d in duty))
            clean = clean + 1 if quiet else 0
            if clean >= need:
                return True, last
        return False, last
    finally:
        comm.close()


def _self_check():
    line = ("t=17644 freq=0.0 | I[A]: A=0.00 B=0.00 C=0.00 D=0.00 | "
            "duty[%]: A=0.0 B=0.0 C=0.0 D=0.0 | spread=0.000 bal=0 trip=0")
    assert _values(line, _I_RE) == [0.0, 0.0, 0.0, 0.0]
    assert _values(line, _DUTY_RE) == [0.0, 0.0, 0.0, 0.0]

    live = line.replace("A=0.00 B=0.00", "A=0.27 B=0.28").replace(
        "duty[%]: A=0.0", "duty[%]: A=100.0")
    amps = _values(live, _I_RE)
    assert amps[0] == 0.27, amps
    assert not all(abs(a) <= I_EPS for a in amps)
    # A line with current at zero but the bridges still driving is NOT off.
    assert max(_values(live, _DUTY_RE)) > DUTY_EPS

    assert _values("garbage", _I_RE) == []       # never read "off" out of noise
    print("safe_off: parses zero and live telemetry, rejects noise\n  ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
        raise SystemExit(0)
    ok, last = coils_off()
    print(last or "no telemetry seen")
    print("COILS OFF -- confirmed by telemetry" if ok else
          "NOT CONFIRMED OFF: cut the supply and check the rig by hand")
    raise SystemExit(0 if ok else 1)
