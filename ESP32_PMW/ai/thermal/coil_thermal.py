#!/usr/bin/env python3
"""Coil temperature bookkeeping. One place, because the last version lived inside
`fly()` and every ad-hoc script that drove the coils directly bypassed it --
the estimate read 52 C while the coils were actually at 75 C.

All three numbers are measured on the rig (2026-08-29), not assumed:

    HEAT_C_PER_S   +1 C per 2 s of drive, by hand across several runs.
    TAU_COOL_S     Newton fit to 80 -> 55 C over 13.2 min at ~22 C ambient gives ~22 min.
                   Cooling is not a clean exponential -- convective h goes as dT^0.25 and
                   radiation is comparable at these temperatures, so tau LENGTHENS as it
                   cools and a hot-fit under-predicts the time to reach a low temperature.
                   25 min is used deliberately: erring long errs safe.
    T_CEILING_C    Working ceiling, under the 80 C actually reached. Enamel survives more,
                   but a neodymium rotor magnet loses flux irreversibly near it.

There is no temperature sensor. Coil current at a fixed frequency is the only physical
check available (copper gains ~0.39 %/degC and |Z| ~ R near the ~174 Hz resonance), so
compare the same frequency bin across runs to sanity-check this model.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

HEAT_C_PER_S = 0.5
TAU_COOL_S = 1500.0
T_CEILING_C = 70.0
T_AMBIENT_C = 22.0
# Beside this module, NOT under `results/`: a results dir is meant to be regenerated,
# and `temp_now` reads a missing stamp as ambient, so tidying one reset a hot coil.
# Here, "missing" honestly means "never run".
STAMP = Path(__file__).resolve().parent / ".last_energised"


def temp_now(stamp: Path | None = None) -> float:
    """Estimated coil temperature, Newton-cooled from the last recorded value."""

    stamp = stamp or STAMP
    if not stamp.exists():
        return T_AMBIENT_C
    try:
        t_end = float(stamp.read_text().split()[0])
    except (ValueError, IndexError, OSError):
        # An unreadable stamp must NOT read as cold: that once cleared a 69 C coil to
        # run immediately, which is the failure this whole module exists to prevent.
        return T_CEILING_C
    idle = time.time() - stamp.stat().st_mtime
    return T_AMBIENT_C + (t_end - T_AMBIENT_C) * math.exp(-idle / TAU_COOL_S)


def add_energised(seconds: float, stamp: Path | None = None) -> float:
    """Record `seconds` of drive: cool forward to now, add the rise, write the stamp."""

    stamp = stamp or STAMP
    t = temp_now(stamp) + HEAT_C_PER_S * max(seconds, 0.0)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"{t:.1f}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return t


def wait_until_safe(energised_s: float, stamp: Path | None = None, verbose=True) -> float:
    """Block until a run of `energised_s` would finish under the ceiling."""

    need = T_CEILING_C - HEAT_C_PER_S * energised_s
    t = temp_now(stamp)
    if t > need:
        wait = TAU_COOL_S * math.log((t - T_AMBIENT_C) / max(need - T_AMBIENT_C, 0.5))
        if verbose:
            print(f"coils ~{t:.0f}C; a {energised_s:.0f}s run adds "
                  f"{HEAT_C_PER_S * energised_s:.0f}C and the ceiling is {T_CEILING_C:.0f}C "
                  f"-- waiting {wait / 60:.1f} min", flush=True)
        time.sleep(max(wait, 0.0))
        t = temp_now(stamp)
    return t


def demo():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        s = Path(d) / "stamp"
        assert abs(temp_now(s) - T_AMBIENT_C) < 1e-9, "no stamp must read ambient"
        s.write_text("not-a-number")
        assert temp_now(s) == T_CEILING_C, "an unreadable stamp must fail SAFE, not cold"
        s.unlink()                       # back to a clean slate for the heat test
        t = add_energised(20.0, s)
        assert abs(t - (T_AMBIENT_C + 10.0)) < 0.2, t
        # cooling: backdate one tau and check it falls by 1/e of the excess
        import os
        os.utime(s, (time.time() - TAU_COOL_S, time.time() - TAU_COOL_S))
        cooled = temp_now(s)
        assert abs((cooled - T_AMBIENT_C) - (t - T_AMBIENT_C) / math.e) < 0.3, cooled
    print("self-check PASS: ambient default, fail-safe on garbage, heat and cool exact")


if __name__ == "__main__":
    demo()
