#!/usr/bin/env python3
"""Supervisor for `pmw_fly`: owns arming, the thermal stamp, and the child's life.

    uv run python controller/control/fly_native.py --seconds 10          # no coils
    uv run python controller/control/fly_native.py --port /dev/cu.usbserial-... --seconds 30

WHY A SUPERVISOR AT ALL
-----------------------
`CLAUDE.md` makes one rule about the coils: **every drive path goes through the one
chokepoint that stamps energised seconds into `ai/thermal/coil_thermal.py`.** Coils reach
80 C after four ramps, there is no temperature sensor, and the model is the only thing
standing between a session and a de-magnetised rotor. A native binary that opened its own
serial port would bypass it.

Three alternatives were considered and rejected:

  * **The binary writes the stamp.** That duplicates `HEAT_C_PER_S`, `TAU_COOL_S` and
    `T_AMBIENT_C` in C++ -- three measured numbers with two homes, which is exactly what
    `pmw.h` forbids for the pose core and is a worse idea here because being wrong burns
    hardware. It also races: two writers, one file.
  * **The binary shells out to Python.** ~200 ms per spawn, and it still needs a parent to
    hold `wait_until_safe()`, which BLOCKS for minutes.
  * **Nothing at all.** That is the status quo the rule exists to prevent.

So: the parent gates and stamps, the child ticks and owns the serial port exclusively (one
owner, no DTR/RTS race). Four rules make it a chokepoint rather than a claim:

  1. The child REFUSES to arm without `--armed-token`, which only this file issues, and
     only after `wait_until_safe()` returns. There is no way to run `pmw_fly` by hand and
     energise a coil.
  2. The child prints `DRIVE on` / `DRIVE off <s>` / `ENERGISED <s>`, mirroring
     `link.SerialComm._note_drive`. The parent parses it.
  3. If the child dies without reporting, the parent charges WALL CLOCK from the first
     `DRIVE on`. That over-charges heat, which is the safe direction -- `coil_thermal`'s
     own unreadable-stamp branch fails to the ceiling for the same reason.
  4. The child holds this process's stdout as its stdin. EOF means the supervisor is gone
     and the child cuts the coils: a host-side liveness backstop for free.

**What none of this covers: the USB cable.** With no firmware watchdog (an operating
decision, `theory.md` 4.0) the host is the only thing that can stop the coils, so an
unplug leaves them driven until the bench supply's limit or the GPIO14 button. Said once,
here and in CLAUDE.md, and not pretended away.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from controller.control import native_config
from controller.control import ramp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BINARY = ROOT / "build" / "fly" / "pmw_fly"

#: Seconds of drive to reserve against the thermal model before a run is allowed to start.
#: The ramp plus a settle; `SETTLE_S` is a guess and is labelled one in `constants.py`.
SETTLE_S = 12.0


def _thermal():
    """The model, or a refusal. `ai/` is gitignored, so a fresh clone does not have it."""

    try:
        from ai.thermal import coil_thermal
    except ImportError:
        sys.exit(
            "REFUSING to arm: ai/thermal/coil_thermal.py is missing.\n"
            "It is the only estimate of coil temperature this rig has -- there is no\n"
            "sensor -- and `ai/` is gitignored, so a fresh clone does not have it.\n"
            "Restore it before driving the coils.")
    return coil_thermal


def fly(port=None, seconds=30.0, csv=None, segments=None, binary=BINARY, dry_run=False,
        cfg_path=None, wait=True):
    """Run one native flight. Returns the child's exit code."""

    binary = Path(binary)
    if not binary.exists():
        sys.exit(f"{binary} not built. Run:\n"
                 f"  cmake -S controller/native -B build/fly -DPMW_FLY=ON "
                 f"-DPMW_MODULE=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build/fly")

    # The profile is validated HERE, before anything is armed: `ramp.check` refuses and
    # never clamps, because the segments reach the firmware verbatim.
    segments = ramp.DEFAULT if segments is None else segments
    ramp.check(segments)

    cfg_path = Path(cfg_path or (ROOT / "build" / "fly" / "fly.cfg"))
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    native_config.write(cfg_path)

    thermal = _thermal()
    budget = ramp.total_s(segments) + SETTLE_S if hasattr(ramp, "total_s") else seconds
    if port and not dry_run and wait:
        # BLOCKS, possibly for minutes. Printed before, not after, so an operator watching
        # the terminal knows why nothing is happening.
        print(f"[coils] estimated {thermal.temp_now():.0f} C; "
              f"waiting for headroom for {budget:.0f}s of drive")
        thermal.wait_until_safe(budget)

    token = f"supervised-{int(time.time())}"
    cmd = [str(binary), "--config", str(cfg_path), "--armed-token", token,
           "--seconds", str(seconds)]
    if port and not dry_run:
        cmd += ["--port", port]
    if csv:
        cmd += ["--csv", str(csv)]

    t_drive_on = None
    drive_s = 0.0
    reported = None
    started = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=None, text=True, bufsize=1)
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            s = line.strip()
            if s.startswith("DRIVE on"):
                t_drive_on = t_drive_on or time.monotonic()
            elif s.startswith("DRIVE off"):
                if t_drive_on is not None:
                    drive_s += time.monotonic() - t_drive_on
                    t_drive_on = None
            elif s.startswith("ENERGISED"):
                try:
                    reported = float(s.split()[1])
                except (IndexError, ValueError):
                    pass
    except KeyboardInterrupt:
        # Closing stdin is the child's cue to send `stop` and exit -- the same path a
        # supervisor crash takes, exercised deliberately here rather than only in anger.
        print("\n[supervisor] interrupt: closing the child's stdin to cut the coils")
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    if t_drive_on is not None:            # still open: the child died mid-drive
        drive_s += time.monotonic() - t_drive_on
    if reported is None and drive_s == 0.0 and port and not dry_run:
        # The child never said anything and we cannot prove it did not drive. Charge the
        # whole wall clock: over-charging heat is the safe direction.
        drive_s = time.monotonic() - started
        print(f"[coils] child reported no drive interval -- charging wall clock "
              f"{drive_s:.0f}s, which over-counts on purpose")
    secs = reported if reported is not None else drive_s
    if secs > 0.5:
        t = thermal.add_energised(secs)
        print(f"[coils] {secs:.0f}s energised -> ~{t:.0f} C estimated")
    return proc.returncode


def demo():
    """Self-check, no coils and no binary needed for the parts that matter."""

    # 1. The binary refuses to arm without a token. This is the whole chokepoint: if it
    #    ever stops refusing, every ad-hoc script becomes an unaccounted drive path.
    if BINARY.exists():
        r = subprocess.run([str(BINARY), "--config", "/dev/null"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        assert r.returncode != 0, "pmw_fly accepted a run with no --armed-token"
        cfg = ROOT / "build" / "fly" / "selfcheck.cfg"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        native_config.write(cfg)
        r = subprocess.run([str(BINARY), "--config", str(cfg)],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        assert r.returncode == 3 and "REFUSING to arm" in r.stderr, (r.returncode, r.stderr)

        # 2. With a token it runs -- and stops on stdin EOF, which is the liveness
        #    backstop standing in for a supervisor that died.
        t0 = time.monotonic()
        r = subprocess.run([str(BINARY), "--config", str(cfg), "--armed-token", "t",
                            "--seconds", "30"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=20)
        assert r.returncode == 0, r.stderr
        assert time.monotonic() - t0 < 10, "stdin EOF did not stop the child"
        assert "clock:" in r.stdout, r.stdout
        print("fly_native: binary refuses without a token, and stops on stdin EOF")
    else:
        print("fly_native: pmw_fly not built -- skipped the arming checks")

    # 3. A missing thermal model must REFUSE, not proceed. `ai/` is gitignored and a fresh
    #    clone genuinely does not have it, so this path is reachable in practice.
    import builtins
    real_import = builtins.__import__

    def no_thermal(name, *a, **k):
        if name.startswith("ai.thermal"):
            raise ImportError("simulated missing model")
        return real_import(name, *a, **k)

    builtins.__import__ = no_thermal
    try:
        _thermal()
    except SystemExit as e:
        assert "REFUSING to arm" in str(e), e
    else:
        raise AssertionError("a missing thermal model was not refused")
    finally:
        builtins.__import__ = real_import
    print("fly_native: refuses to arm with no thermal model\n  ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        demo()
        return 0
    return fly(port=a.port, seconds=a.seconds, csv=a.csv, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
