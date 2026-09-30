#!/usr/bin/env python3
"""
De-energise the coils, and prove it from the firmware's own telemetry.

    uv run python controller/control/safe_off.py

"The command was sent" and "the coils are off" are different claims, and only the second
one matters when walking away from a rig with a 12 V 10 A supply behind it.

This sends `stop`, then reads back `driveTelemetry` -- per-coil current and duty at 2 Hz in
every state -- and exits non-zero unless they are all zero. Run it after any session that
energised the coils, and any time the state of the rig is not certain.
"""

from __future__ import annotations

import sys
import time

from controller.control.link import SerialComm, parse_telemetry

I_EPS = 0.02  # A, below the sense noise floor
DUTY_EPS = 0.5  # %


def coils_off(port=None, settle_s=3.0, need=2):
    """
    Send `stop`, then wait for `need` consecutive all-zero telemetry lines.

        Returns ``(ok, last_line)``. Several lines rather than one because the drive
        ramps down rather than stopping dead, and a single sample can land mid-ramp
        and read zero by luck at a current zero-crossing.
    """

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
            t = parse_telemetry(line)
            if not t.amps:
                continue
            quiet = (all(abs(a) <= I_EPS for a in t.amps)
                     and all(abs(d) <= DUTY_EPS for d in t.duty))
            clean = clean + 1 if quiet else 0
            if clean >= need:
                return True, last
        return False, last
    finally:
        comm.close()


def _self_check():
    """The parser itself is checked in link.demo(); this checks the off/not-off verdict."""

    off = ("t=17644 freq=0.0 | I[A]: A=0.00 B=0.00 C=0.00 D=0.00 | "
           "duty[%]: A=0.0 B=0.0 C=0.0 D=0.0 | spread=0.000 bal=0 trip=0")

    def quiet(line):
        t = parse_telemetry(line)
        return bool(t.amps) and (all(abs(a) <= I_EPS for a in t.amps)
                                 and all(abs(d) <= DUTY_EPS for d in t.duty))

    assert quiet(off)
    assert not quiet(off.replace("A=0.00 B=0.00", "A=0.27 B=0.28"))
    # Current at zero with the bridges still driving is NOT off.
    assert not quiet(off.replace("duty[%]: A=0.0", "duty[%]: A=100.0"))
    assert not quiet("garbage")  # never read "off" out of noise
    print("safe_off: calls zero telemetry off, live current and live duty not off\n  ok")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
        raise SystemExit(0)
    ok, last = coils_off()
    print(last or "no telemetry seen")
    print("COILS OFF -- confirmed by telemetry" if ok else
          "NOT CONFIRMED OFF: cut the supply and check the rig by hand")
    raise SystemExit(0 if ok else 1)
