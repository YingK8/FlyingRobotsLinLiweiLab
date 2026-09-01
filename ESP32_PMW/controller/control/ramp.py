#!/usr/bin/env python3
"""The takeoff frequency profile: one definition, one encoder, one validator.

A profile is a tuple of segments, each `(from_hz, to_hz, seconds, mode, k)`, which is
exactly the argument list of `PwmSequencer::addRampTask`. The host builds the shape and
sends it as data; `lib/PwmSequencer` already takes arbitrary ramp tasks, so a new shape
never needs a reflash. The numbers themselves live in `constants.py` -- this module owns
the SHAPE and its wire encoding, nothing else.

This is the only place a ramp is defined. Do not add a second path.

    uv run python controller/control/ramp.py
"""

from __future__ import annotations

from controller.control import constants as C

# Re-exported so a caller writing a profile does not have to import two modules to name a
# curve. `POLY` with k=1 is linear, k=2 quadratic; `EASE` is the symmetric sigmoid.
POLY, EASE, EXP = C.RAMP_POLYNOMIAL, C.RAMP_EASE, C.RAMP_EXPONENTIAL
_MODE_NAME = {POLY: "POLY", EASE: "EASE", EXP: "EXP"}

# The flight profile. HOVER ACHIEVED with this 2026-08-29: the robot held its own weight
# and stayed up when the takeoff rod was pulled away. EASE k=2 and NOT linear, because a
# linear ramp crosses the ~4.9 Hz pull-in window in 0.2 s and never captures the rotor,
# where EASE leaves rest at zero slope and spends ~1.5 s down there (`theory.md` 18.3).
DEFAULT = ((C.RAMP_START_HZ, C.RAMP_TARGET_HZ, C.RAMP_S, EASE, C.RAMP_K),)


def duration_s(segs) -> float:
    """Total drive time of the ramp, in seconds. What the thermal gate budgets for."""

    return float(sum(seg[2] for seg in segs))


def peak_rate(segs) -> list[float]:
    """Steepest slope of each segment, Hz/s.

    For EASE the sigmoid's derivative at its midpoint is exactly k times the average
    (`theory.md` 18.2), so the peak rate is k*df/T and it lands halfway through -- which
    is why a "gentle" 30 s ramp can still hit 14 Hz/s in the middle. Other modes are not
    symmetric, so this is the EASE figure and an approximation elsewhere.
    """

    return [seg[4] * abs(seg[1] - seg[0]) / seg[2] for seg in segs]


def label(segs) -> str:
    """One-line description, stamped into every flight CSV so an attempt records its own
    ramp. Without it `takeoff_report.compare()` can only label curves by clock time."""

    return " + ".join(
        f"{f0:g}->{f1:g}Hz/{s:g}s/{_MODE_NAME.get(mode, mode)}/k{k:g}"
        for f0, f1, s, mode, k in segs
    )


def check(segs) -> None:
    """Raise ValueError unless the profile is safe to send. Refuses, never clamps: the
    segments reach the firmware verbatim, so a value trimmed here would only mislead."""

    if not segs:
        raise ValueError("empty ramp profile: nothing would spin the rotor up")
    for f0, f1, s, mode, k in segs:
        if s <= 0:
            raise ValueError(f"segment {f0:g}->{f1:g} Hz has duration {s:g}s; must be > 0")
        if mode not in _MODE_NAME:
            raise ValueError(f"unknown mode {mode!r}; use POLY, EASE or EXP")
    # `seq=go` compiles the queue at the FIRST segment's start only, so the firmware never
    # re-datums between segments: a gap here is a commanded step in frequency, which is
    # the one thing guaranteed to break sync. Nothing else checks this.
    for a, b in zip(segs, segs[1:]):
        if a[1] != b[0]:
            raise ValueError(
                f"gap between segments: {a[1]:g} Hz -> {b[0]:g} Hz is a commanded step. "
                f"Segments must be continuous -- the firmware does not ramp across them."
            )
    total = duration_s(segs)
    if total > C.MAX_RAMP_S:
        raise ValueError(
            f"profile totals {total:.0f}s of drive, over the {C.MAX_RAMP_S:.0f}s cap. "
            f"Shorten a segment -- the cap cannot be applied for you, because the "
            f"firmware is sent the segments verbatim."
        )


def seq_lines(segs) -> list[str]:
    """The profile as serial lines for `main_flight.cpp`. Validated first, so a bad shape
    fails on the host rather than half-loading into the sequencer."""

    check(segs)
    return (["seq=clear"]
            + [f"seq=ramp:{f0:g}:{f1:g}:{s * 1e3:.0f}:{mode:g}:{k:g}"
               for f0, f1, s, mode, k in segs]
            + ["seq=go"])


def demo() -> None:
    assert seq_lines(DEFAULT) == [
        "seq=clear", "seq=ramp:2:210:30000:1:2", "seq=go"], seq_lines(DEFAULT)
    assert label(DEFAULT) == "2->210Hz/30s/EASE/k2", label(DEFAULT)
    assert duration_s(DEFAULT) == C.RAMP_S

    # theory.md 18.2: EASE peaks at k*df/T. 2*(210-2)/30 = 13.87 Hz/s, mid-ramp.
    assert abs(peak_rate(DEFAULT)[0] - 13.867) < 1e-3, peak_rate(DEFAULT)

    two = ((2.0, 6.0, 8.0, EASE, 2.0), (6.0, 198.0, 14.0, EASE, 2.0))
    check(two)
    assert duration_s(two) == 22.0
    assert label(two) == "2->6Hz/8s/EASE/k2 + 6->198Hz/14s/EASE/k2", label(two)

    for bad, why in (
        ((), "empty"),
        (((2.0, 6.0, 0.0, EASE, 2.0),), "zero duration"),
        (((2.0, 6.0, 8.0, 9, 2.0),), "unknown mode"),
        (((2.0, 6.0, 8.0, EASE, 2.0), (50.0, 198.0, 14.0, EASE, 2.0)), "gap 6->50"),
        (((2.0, 210.0, C.MAX_RAMP_S + 1, EASE, 2.0),), "over the cap"),
    ):
        try:
            check(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"check() accepted a profile it must refuse: {why}")

    print(f"ramp: DEFAULT {label(DEFAULT)}, peak {peak_rate(DEFAULT)[0]:.1f} Hz/s\n"
          f"  {' / '.join(seq_lines(DEFAULT))}\n  ok")


if __name__ == "__main__":
    demo()
