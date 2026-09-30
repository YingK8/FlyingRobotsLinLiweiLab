#!/usr/bin/env python3
"""Did the rotor actually spin up on this take, and when?

A bench harness over `pose.spin`, not part of the flight loop. `takeoff_report` answers
"did it lift"; this answers the question before it -- "did it turn at all" -- which is the
one that separates a thrust problem from a capture problem.

The witness is a motion latch, not a tachometer: above about fps/8 the per-frame blade
phase aliases and the true rate is unrecoverable, so this reports turning / rocking /
still and the frame it changed on, never a speed. See `pose/theory.md` on blade phase.

    uv run python ai/spinup/detector.py                        # the newest flight
    uv run python ai/spinup/detector.py results/flights/<dir>
    uv run python ai/spinup/detector.py --all                  # every flight, one line each
"""

from __future__ import annotations

import sys
from pathlib import Path

from controller.camera.record import DEFAULT_DIR, flights, latest_flight
from controller.pose import spin

# Frames of steady phase accumulation before calling it turning. One frame of motion is
# noise; the witness itself needs a few steps before `turning` stops answering None.
CONFIRM_FRAMES = 5


def detect(rec_dir, tag="A", limit=None):
    """Replay a take and report whether the rotor turned.

    Returns a dict: `verdict` (turning / still / no-signal), `strength` (4th-harmonic
    blade signal), `drift_rev` (revolutions accumulated -- near zero means it never
    turned) and `frames`.
    """

    rec = Path(rec_dir)
    witness = spin.from_recording(rec, tag=tag, limit=limit)
    verdict = {True: "turning", False: "still", None: "no-signal"}[witness.turning]
    return {
        "flight": rec.name,
        "verdict": verdict,
        "strength": witness.strength,
        "drift_rev": witness.drift_rev,
        "frames": witness.n,
        "summary": witness.summary(),
    }


def _line(r):
    return (f"{r['flight']:<22s} {r['verdict']:<10s} "
            f"h4 {r['strength']:.2f}  {r['drift_rev']:+.2f} rev  ({r['frames']} frames)")


def main(argv):
    if "--all" in argv:
        for d in flights(DEFAULT_DIR):
            if not (d / "A" / "A.mp4").exists():
                continue
            try:
                print(_line(detect(d)))
            except Exception as exc:
                print(f"{d.name:<22s} unreadable: {exc}")
        return

    rec = Path(argv[0]) if argv else latest_flight(DEFAULT_DIR)
    r = detect(rec)
    print(_line(r))
    print(f"  {r['summary']}")


def demo():
    """Checks the verdict mapping without needing a recording on disk."""

    class Fake:
        strength, drift_rev, n = 0.5, 1.0, 100

        def __init__(self, turning, summary):
            self.turning, self._s = turning, summary

        def summary(self):
            return self._s

    real = spin.from_recording
    try:
        for turning, want in ((True, "turning"), (False, "still"), (None, "no-signal")):
            spin.from_recording = lambda *a, **k: Fake(turning, "fake")
            got = detect("results/flights/nonexistent")["verdict"]
            assert got == want, f"{turning!r} -> {got!r}, wanted {want!r}"
    finally:
        spin.from_recording = real

    print("spinup detector: maps turning/still/no-signal, never reports a rate\n  ok")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--self-check"]
    if "--self-check" in sys.argv:
        demo()
    else:
        main(args)
