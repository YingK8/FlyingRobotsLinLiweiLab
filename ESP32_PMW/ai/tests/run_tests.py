"""Run every `test_*.py` suite in this package and summarise.

    uv run python controller/pose/run_tests.py [-v] [name ...]

Each suite stays its own file. They are not merged into one, and that is a
choice: they are ~2000 lines covering six unrelated subjects (conic algebra,
datum arithmetic, tilt calibration, Kalman behaviour, stereo geometry,
Cramer-Rao bounds), and a single file would only make the failure you are
chasing harder to find. What was actually missing was one command, which is
this.

Run as **subprocesses**, not imports, for two reasons. Each suite does its work
under ``if __name__ == "__main__"`` and reports by exit code, so importing it
runs nothing; and two of them build a GL context, which pyglet permits only once
per process (`validation/render.py`), so importing both into one interpreter
would fail on the second.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


#: Ordered cheapest-and-most-fundamental first, so a broken conic solve is the
#: first thing you see rather than the twentieth. `conic` underpins everything
#: else here; if it fails, no other result in the package means anything.
SUITES = [
    ("conic", "geometry: circle <-> ellipse round-trip, back-projection"),
    ("zeroing", "the datum: build, apply, invert  [needs a display]"),
    ("calibration", "the fitted tilt correction"),
    ("filter", "constant-velocity Kalman: velocity, coasting, dropout"),
    ("bounds", "Cramer-Rao floors vs Monte Carlo"),
    ("stereo", "two-view solver against analytic geometry  [needs a display]"),
    ("appearance", "rig appearance: bright-on-dark and black-on-white (mono)"),
    ("backgrounds", "synthetic background generators"),
    ("elp_captures", "the dark appearance on the real ELP frames"),
]


def run(name, verbose=False):
    path = HERE / f"test_{name}.py"
    if not path.exists():
        return name, None, 0.0, f"missing: {path.name}"
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(path)], capture_output=not verbose,
                          text=True)
    dt = time.monotonic() - t0
    tail = ""
    if proc.returncode != 0 and not verbose:
        out = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        tail = "\n".join("      " + ln for ln in out[-12:])
    return name, proc.returncode, dt, tail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="suites to run (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream each suite's own output instead of capturing it")
    args = ap.parse_args(argv)

    wanted = args.names or [n for n, _ in SUITES]
    known = dict(SUITES)
    unknown = [n for n in wanted if n not in known]
    if unknown:
        ap.error(f"unknown suite(s): {', '.join(unknown)}. "
                 f"choose from: {', '.join(known)}")

    print(f"running {len(wanted)} suite(s)\n")
    failed, skipped, total = [], [], 0.0
    for name in wanted:
        print(f"  {name:<14} {known[name]}", flush=True)
        name, code, dt, tail = run(name, args.verbose)
        total += dt
        if code is None:
            skipped.append(name)
            print(f"  {'':14} SKIP  {tail}\n")
            continue
        print(f"  {'':14} {'PASS' if code == 0 else 'FAIL'}  {dt:5.1f}s\n")
        if code != 0:
            failed.append(name)
            if tail:
                print(tail + "\n")

    print("=" * 68)
    print(f"{len(wanted) - len(failed) - len(skipped)}/{len(wanted)} passed "
          f"in {total:.1f}s")
    if skipped:
        print(f"skipped: {', '.join(skipped)}")
    if failed:
        print(f"FAILED:  {', '.join(failed)}")
        print(f"rerun one with:  uv run python {Path(__file__).name} "
              f"-v {failed[0]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
