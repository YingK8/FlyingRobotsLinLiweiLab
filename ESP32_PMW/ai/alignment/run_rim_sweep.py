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
#: Pause to use when `--process` is on. The solve measured 23 min on a 435 s take (170% CPU,
#: 89007 frames, stride 1), so 600 s cannot hold it -- and the ordering only works if it does.
PROCESS_PAUSE_S = 2100.0
RAMP_FIX_EPOCH = datetime(2026, 9, 24, 4, 20).timestamp()
CAP_MODE, CAP_FPS, PREVIEW = "640x400", "max", False
#: Empty on purpose: the note must match the takes ALREADY on disk (90-110 Hz, 2026-09-24) or
#: `block_done` reports them missing and the sweep re-records five good blocks. The capture
#: rate is documented in `results/rim/frame_rates.txt` instead of in the note, so a rate change
#: no longer silently invalidates every finished block.
CAP_TAG = ""
#: Where the per-take frame-rate record is written. See `write_frame_rates`.
FRAME_RATES = ROOT / "results" / "rim" / "frame_rates.txt"

sys.path.insert(0, str(ROOT))
from controller.control import tilt_schedule as _ts  # noqa: E402


def ramp_ms(f):
    """Total up-ramp duration, in ms. 90 Hz -> 35357 ms; 150 Hz -> 56786 ms."""

    return _ts.ramp_ms(f)


def period_s(f):
    hold = HOLD_LO_MS if float(f) <= 100 else HOLD_HI_MS
    return (ramp_ms(f) + hold + POST_MS + RESET_MS + OFF_MS) / 1000.0


def block_note(f, n, blk):
    """A take's note. ONE definition -- `block_done` matches on exactly this string."""

    tag = f" {CAP_TAG}" if CAP_TAG else ""
    return f"whole ring {f:g}Hz x{n} without cardan post15{tag} blk{blk}"


def _fmt(value, width=0):
    """A rate for the table, or ``-`` when the take predates it being recorded."""

    return f"{float(value):.1f}" if value else "-"


def write_frame_rates():
    """Regenerate `results/rim/frame_rates.txt`: the frame rate every take actually got.

    WHY A FILE AND NOT JUST meta.json. The rate is the one number in this protocol that cannot
    be recovered from a take's name or note, and it varies: the 2026-09-24 blocks were recorded
    at the sensor's own default (~185-192 fps) and everything after `cap_fps` reached the driver
    asks for the clamp (~194 measured). Nothing downstream records which is which, and the
    analysis gate `spin_check` gets easier as the rate rises, so "which blocks were fast" is a
    question a later reader will have and cannot answer from the directory listing.

    `reduced` is the honest number: `fps_measured`, not what the driver granted and not what the
    mp4 declares. The declared rate is what the file PLAYS at; the measured rate is what was
    CAPTURED, and only the second one describes the data.
    """

    rows = []
    for d in sorted((ROOT / "results" / "flights").glob("*_whole*")):
        meta = d / "meta.json"
        if not meta.exists() or not (d / "frames.csv").exists():
            continue
        try:
            m = json.loads(meta.read_text())
        except Exception:
            continue
        note = str(m.get("note") or "")
        if "whole ring" not in note:
            continue
        freq = note.split("whole ring", 1)[1].strip().split("Hz", 1)[0].strip()
        blk = note.rsplit("blk", 1)[-1].strip() if "blk" in note else "-"
        granted = m.get("cap_fps_granted") or []
        rows.append({
            "take": d.name,
            "hz": freq,
            "blk": blk,
            "frames": sum(1 for _ in open(d / "frames.csv")) - 1,
            "measured_fps": m.get("fps_measured"),
            "declared_fps": m.get("fps"),
            "requested_fps": m.get("cap_fps_requested"),
            "granted_fps": "/".join(f"{float(g):g}" for g in granted) if granted else "-",
            "dropped": m.get("dropped"),
        })
    if not rows:
        return None
    # Mark which takes the analysis should USE. Same source of truth as `block_done`, so a take
    # this file calls `use` is exactly the one a re-run would skip -- the superseded ones
    # (wrong ramp, or a partial take from an interrupted run) sit on disk and must not be
    # silently averaged with the good ones.
    chosen = {(f, b): block_done(f, N_PER_BLOCK, b)
              for f in (90, 100, 110, 120, 130, 140, 150) for b in (1, 2)}
    for r in rows:
        try:
            r["status"] = "use" if chosen.get((float(r["hz"]), int(r["blk"]))) == r["take"] \
                else "superseded"
        except (TypeError, ValueError):
            r["status"] = "superseded"
    FRAME_RATES.parent.mkdir(parents=True, exist_ok=True)
    w = max(len(r["take"]) for r in rows)
    with open(FRAME_RATES, "w") as fh:
        fh.write(
            "Frame rates, rim J(f) experiment (whole ring), 640x400\n"
            "===================================================\n\n"
            "`measured` is the capture rate actually achieved: the number to quote, and the one\n"
            "that sets the practical limit for resolving the spin line (measured/2 Hz).\n"
            "`declared` is what the mp4 plays at; `dropped` is frames the encoder queue lost.\n"
            "Only rows marked `use` belong in the analysis; `superseded` takes are kept on disk\n"
            "from interrupted runs or the pre-2026-09-24 single-segment ramp.\n\n"
            "Rates are recorded PER TAKE rather than fixed across the series, because a higher\n"
            "request did not deliver a higher rate: the pipeline is encoder-bound, not\n"
            "sensor-bound. The 2026-09-24 blocks were filmed with no rate requested, so the\n"
            "sensor kept its own default; from `cap_fps` onward the driver is asked for its\n"
            "clamp. See the notes below.\n\n"
            f"{'take'.ljust(w)}  {'hz':>4} {'blk':>3} {'frames':>7} {'measured':>9} "
            f"{'declared':>9} {'asked':>8} {'granted':>9} {'dropped':>7}  status\n"
        )
        for r in sorted(rows, key=lambda r: (float(r["hz"]), r["blk"], r["take"])):
            asked = f"{r['requested_fps']:.1f}" if r["requested_fps"] else "-"
            fh.write(
                f"{r['take'].ljust(w)}  {r['hz']:>4} {r['blk']:>3} {r['frames']:>7} "
                f"{_fmt(r['measured_fps']):>9} {_fmt(r['declared_fps']):>9} "
                f"{asked:>8} {r['granted_fps']:>9} {str(r['dropped']):>7}  {r['status']}\n"
            )
        fh.write(
            "\nnotes\n-----\n"
            "* `measured` is the rate to quote. It fell to ~194 fps with the encoder running\n"
            "  against ~209 fps for the bare source, so the pipeline is encoder-bound, not\n"
            "  sensor-bound: asking for more does not deliver more.\n"
            "* 271.3 fps is the SENSOR-only figure from `modes.py`. Over `open_source` the\n"
            "  AVFoundation driver clamps a higher request to 210.0 (measured 2026-09-27), and\n"
            "  `record.CAP_FPS_CEILING` now asks for 210 so `granted` reads back honestly.\n"
            "* `declared` above `measured` means the mp4 plays slightly fast. The analysis is\n"
            "  unaffected: every time is taken from `frames.csv`, which stamps real capture\n"
            "  times, not from frame indices divided by the declared rate.\n"
        )
    print(f"  frame rates -> {FRAME_RATES.relative_to(ROOT)} ({len(rows)} takes)", flush=True)
    return FRAME_RATES


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
        # -u is not cosmetic: Python buffers stdout when it is a FILE, and record.py only prints
        # at open and close, so the log stayed EMPTY for the whole 7-minute block on 2026-09-27
        # and a healthy run looked dead. Unbuffered, the log shows the take directory as soon as
        # the cameras are open.
        "-u",
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
    # Write the frame-rate record after every block, not at the end: a sweep that is stopped
    # part way must still document what it recorded, and that is the whole point of the file.
    write_frame_rates()
    want = int(total * 150)
    print(f"  take={take.name}  frames={n_frames}", flush=True)
    if n_frames < want:
        print(
            f"  !! only {n_frames} frames for a {total:.0f}s schedule -- short take",
            flush=True,
        )
        return False
    return True


def process_take(take):
    """Solve, validate and upload one take, inside the pause that follows its block.

    The solve runs at ~170% CPU for ~23 minutes on a 435 s take, which is LONGER than the 600 s
    default pause. So `--process` only fits if the pause is also raised; `main` does that
    automatically, because a pause that overruns is a solve running alongside the next
    recording, competing with it for the USB bus and the CPU. That is the one thing the
    record-then-process ordering exists to avoid.
    """

    print(f"\n  ----- processing {Path(take).name} (solve -> validate -> upload) -----",
          flush=True)
    r = sh(["uv", "run", "python", "ai/alignment/process_block.py", str(take)])
    if r.returncode:
        print("  !! processing reported a problem; the take stays on disk and was not "
              "uploaded if validation failed", flush=True)
    return r.returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("freqs", nargs="+", type=float)
    ap.add_argument("--n", type=int, default=N_PER_BLOCK, help="repeats per block")
    ap.add_argument(
        "--pause", type=float, default=PAUSE_S, help="seconds between blocks"
    )
    ap.add_argument(
        "--process",
        action="store_true",
        help="after each block, run process_block.py: solve to CSV, validate, upload to USB. "
        "Runs INSIDE the pause, never during a recording -- the cameras and the drive share a "
        "USB bus, so a solve or a copy alongside a take shows up as dropped frames.",
    )
    ap.add_argument(
        "--only-block",
        type=int,
        default=None,
        choices=(1, 2),
        help="record only this block of each frequency, then stop, with no pauses. For "
        "running one block at a time and checking the take before spending time on the next.",
    )
    a = ap.parse_args(argv)

    # The solve is longer than the default pause, so --process needs a longer one. Bump it
    # rather than let the solve overlap the next recording.
    if a.process and a.pause < PROCESS_PAUSE_S:
        print(f"  --process: raising the pause {a.pause:.0f} s -> {PROCESS_PAUSE_S:.0f} s so the "
              f"solve fits inside it", flush=True)
        a.pause = PROCESS_PAUSE_S

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
                if a.only_block is not None and blk != a.only_block:
                    continue
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
                if a.process and ok:
                    take = newest_take(f)
                    if take is not None:
                        process_take(take)
                    else:
                        print("  !! no take found to process", flush=True)
                if a.only_block is not None:
                    continue
                if blk == 1 and f != a.freqs[-1]:
                    print(f"  cooling {a.pause:.0f} s before block 2", flush=True)
                    time.sleep(a.pause)
            if a.only_block is not None:
                continue
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
