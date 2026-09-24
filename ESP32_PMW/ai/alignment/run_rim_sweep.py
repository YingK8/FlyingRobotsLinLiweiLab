#!/usr/bin/env python3
"""Run the rim J(f) experiment autonomously: every frequency, two blocks of N repeats.

    uv run python ai/alignment/run_rim_sweep.py 100 110 120 130 140 150

WHY PYTHON AND NOT A SHELL LOOP
------------------------------
The host has to time the stop itself -- nothing in this protocol reads a temperature and
`main_tilt` parses no serial. A shell `( sleep N; kill -INT $PID ) &` did NOT fire on
2026-09-24: the 100 Hz block-1 recorder ran **84 minutes** instead of 6 and produced a
1.1 GB take, filling the disk. `subprocess` plus a direct SIGINT is deterministic, and
the `finally` closes the take even on Ctrl-C.

PACING IS MODEL-BASED
---------------------
No temperature is read anywhere on this path, so the pacing comes from the `tilt_run`
current table: ~9 C a repeat at 90 Hz rising to ~14 C at 150 Hz, against an all-off
window of 21.5 s that sheds only ~0.4-0.7 C. Ten back-to-back repeats would climb to
100-145 C from ambient, so each frequency is TWO blocks of N with a pause between them.
The pause does not lower the peak -- the block SIZE does. Keep N at 5; drop to 3 for the
top of the range if a measured reading ever says the model is wrong in the hot direction.

The ramp is rate-based, matching `controller/control/tilt_schedule.py`: 3.5 Hz/s below
60 Hz, 2.8 Hz/s at or above it. The ramp is not a warm-up -- all four coils sit at 100%
carrier while it sweeps up, so it is the largest single heat term in a repeat.

Cuts channels [0, 2] (coils A and C), which is what the whole analysis chain assumes.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "bin" / "python"
PORT = "/dev/cu.SLAB_USBtoUART"
RATE_HI, RATE_LO, RATE_SPLIT_HZ = 2.8, 3.5, 60.0
HOLD_HI_MS, HOLD_LO_MS, POST_MS, RESET_MS, OFF_MS = 3000, 5000, 15000, 1000, 21500
N_PER_BLOCK, PAUSE_S, MARGIN_S = 5, 600, 25
RAMP_FIX_EPOCH = datetime(2026, 9, 24, 4, 20).timestamp()
CAP_MODE, CAP_FPS, PREVIEW = "640x400", "max", False
CAP_TAG = "capmax-np"

sys.path.insert(0, str(ROOT))
from controller.control import tilt_schedule as _ts  # noqa: E402


def ramp_ms(f):
    """Total up-ramp duration, in ms. 90 Hz -> 35357 ms; 150 Hz -> 56786 ms."""

    return _ts.ramp_ms(f)


def period_s(f):
    hold = HOLD_LO_MS if float(f) <= 100 else HOLD_HI_MS
    return (ramp_ms(f) + hold + POST_MS + RESET_MS + OFF_MS) / 1000.0


def block_note(f, n, blk):
    """A take's note. ONE definition -- `block_done` matches on exactly this string.

    Built in one place because a note written in two places drifts, and that drift is
    silent: the matcher then either never matches, or matches a take from a different
    protocol. `CAP_TAG` is what distinguishes this series from the pre-capture-rate runs.
    """

    return f"whole ring {f:g}Hz x{n} without cardan post15 {CAP_TAG} blk{blk}"


def sh(cmd, capture=False):
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=capture, text=True)


def park():
    """Park the board, unless the serial device is gone -- then there is nothing to park."""

    if not Path(PORT).exists():
        print("  (no serial device -- skipping park)", flush=True)
        return
    sh(
        [
            "uv",
            "run",
            "python",
            "-c",
            f"from controller.control.tilt_sweep import park; park('{PORT}')",
        ]
    )


def newest_take(f):
    found = sorted((ROOT / "results" / "flights").glob(f"*_whole{int(f)}"))
    return found[-1] if found else None


def serial_present():
    """True if the board's USB bridge is there.

    A missing device is NOT a per-block failure, it is the hardware being unplugged. On
    2026-09-24 the bridge vanished at 05:38 mid-sweep and the driver went on sleeping
    through nine 600 s pauses with every `uploadfs` failing -- two hours of nothing. Stop
    instead. The schedule on SPIFFS is left alone; whatever is flashed runs on next boot.
    """

    if Path(PORT).exists():
        return True
    print(
        f"  !! {PORT} does not exist: the board is not connected.\n"
        f"     Aborting rather than idling through the remaining pauses.",
        flush=True,
    )
    return False


def block_done(f, n, blk):
    """Name of an existing take that already holds this exact block, else None.

    Makes a re-run idempotent: a sweep cut short by a dropped USB link is restarted with
    the same command and records only what is missing, matching on the note string (which
    carries the frequency, the block and the post delay) plus a loose frame-count floor.

    The note does NOT carry the ramp, so the pre-fix run's takes share this note string.
    They are excluded by the clock: nothing recorded before `RAMP_FIX_EPOCH` used the
    two-segment capture ramp, so it cannot stand in for one of these blocks however good
    its frame count looks. Without this, a re-run silently skipped the corrected 90 Hz
    blocks in favour of the old single-segment takes (found 2026-09-24).
    """

    note = block_note(f, n, blk)
    want = int(period_s(f) * n * 120)  # loose: the real capture rate is ~185 fps
    for d in sorted((ROOT / "results" / "flights").glob(f"*_whole{int(f)}")):
        meta = d / "meta.json"
        if not meta.exists():
            continue
        try:
            started = datetime.strptime(d.name.split("_whole")[0], "%Y-%m-%d_%H%M%S")
        except ValueError:
            continue
        if started.timestamp() < RAMP_FIX_EPOCH:
            continue
        try:
            if json.loads(meta.read_text()).get("note") != note:
                continue
        except Exception:
            continue
        if sum(1 for _ in open(d / "frames.csv")) - 1 >= want:
            return d.name
    return None


def _restore_sigint():
    """Undo the ``SIG_IGN`` a non-interactive shell gives to background jobs.

    POSIX: an asynchronous command started by a NON-INTERACTIVE shell inherits SIGINT as
    SIG_IGN, and CPython deliberately does not install its KeyboardInterrupt handler when
    it starts with SIGINT already ignored. So a recorder launched with `&` from a script
    can never be interrupted -- `kill -INT` is silently a no-op.

    That is exactly what happened on 2026-09-24: the 100 Hz block-1 recorder ignored every
    SIGINT and ran for 84 minutes, producing a 1.1 GB take and filling the disk. Setting
    SIG_DFL in the forked child, before exec, makes the child Python install its handler
    normally, so the stop is reliable however this driver was started.
    """

    signal.signal(signal.SIGINT, signal.SIG_DFL)


def one_block(f, n, blk):
    """Schedule, flash, film, stop, park. Returns True if the take looks complete."""

    total = period_s(f) * n
    print(
        f"\n===== {f:g} Hz block {blk}/2  ramp={ramp_ms(f)}ms  n={n}  "
        f"schedule {total:.0f}s =====",
        flush=True,
    )
    if sh(
        [
            "uv",
            "run",
            "python",
            "ai/make_tilt.py",
            "--hz",
            str(f),
            "--n",
            str(n),
            "whole",
            "--post-ms",
            str(POST_MS),
        ]
    ).returncode:
        print("  !! make_tilt failed", flush=True)
        return False
    if not serial_present():
        raise SystemExit(2)
    r = sh(["pio", "run", "-e", "tilt", "-t", "uploadfs"], capture=True)
    tail = (r.stdout or "").strip().splitlines()
    print(f"  uploadfs: {tail[-1] if tail else 'no output'}", flush=True)
    if r.returncode:
        print("  !! uploadfs failed", flush=True)
        return False

    log = Path(f"/tmp/rim_{int(f)}_b{blk}_rec.log")
    cmd = [
        str(PY),
        "controller/camera/record.py",
        "--mode",
        CAP_MODE,
        "--start",
        "--cap-fps",
        CAP_FPS,
        "--note",
        block_note(f, n, blk),
    ]
    if not PREVIEW:
        cmd.append("--no-preview")
    with open(log, "w") as fh:
        rec = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=fh,
            stderr=subprocess.STDOUT,
            preexec_fn=_restore_sigint,
        )
    try:
        time.sleep(total + MARGIN_S)
    finally:
        if rec.poll() is None:
            rec.send_signal(signal.SIGINT)
        try:
            rec.wait(timeout=240)
        except subprocess.TimeoutExpired:
            print("  !! recorder ignored SIGINT -- killing", flush=True)
            rec.kill()
            rec.wait()
    print("  " + " | ".join(log.read_text().strip().splitlines()[-3:]), flush=True)
    park()

    take = newest_take(f)
    if take is None:
        print("  !! no take directory found", flush=True)
        return False
    n_frames = sum(1 for _ in open(take / "frames.csv")) - 1
    want = int(total * 150)
    print(f"  take={take.name}  frames={n_frames}", flush=True)
    if n_frames < want:
        print(
            f"  !! only {n_frames} frames for a {total:.0f}s schedule -- short take",
            flush=True,
        )
        return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("freqs", nargs="+", type=float)
    ap.add_argument("--n", type=int, default=N_PER_BLOCK, help="repeats per block")
    ap.add_argument(
        "--pause", type=float, default=PAUSE_S, help="seconds between blocks"
    )
    a = ap.parse_args(argv)

    signal.signal(signal.SIGINT, signal.default_int_handler)

    done, failed = [], []
    try:
        for f in a.freqs:
            print(
                f"\n############ {f:g} Hz  (period {period_s(f):.1f} s, "
                f"{2 * a.n} repeats in 2 blocks) ############",
                flush=True,
            )
            for blk in (1, 2):
                already = block_done(f, a.n, blk)
                if already:
                    print(
                        f"  {f:g} Hz block {blk}/2 already on disk ({already}) -- skipping",
                        flush=True,
                    )
                    done.append((f, blk))
                    continue
                ok = one_block(f, a.n, blk)
                (done if ok else failed).append((f, blk))
                if blk == 1 and f != a.freqs[-1]:
                    print(f"  cooling {a.pause:.0f} s before block 2", flush=True)
                    time.sleep(a.pause)
            if f != a.freqs[-1]:
                print(
                    f"  cooling {a.pause:.0f} s before the next frequency", flush=True
                )
                time.sleep(a.pause)
    except KeyboardInterrupt:
        print("\n*** interrupted -- parking the board ***", flush=True)
        park()
        raise SystemExit(130)
    finally:
        park()

    print(
        f"\n===== sweep finished: {len(done)} block(s) ok, {len(failed)} failed =====",
        flush=True,
    )
    if failed:
        print(
            f"  FAILED: {failed}  -- re-run with: {' '.join(str(f) for f, _ in failed)}",
            flush=True,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
