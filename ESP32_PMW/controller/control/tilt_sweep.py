#!/usr/bin/env python3
"""Run the channel-0 amplitude sweep, film it, and cut one still per duty step.

`spiffs_data/tilt.json` holds coils B/C/D at 100% carrier and steps channel A through
20/40/60/80/100%, holding 2.5 s at each. `main_tilt.cpp` runs that schedule off SPIFFS
and parses no serial at all, so this host does three things and only three:

1.  resets the board, which is what starts the schedule;
2.  films the stereo pair and logs the firmware's `label=` lines **in one process**;
3.  afterwards, pulls the frame that belongs to each duty step out of the video.

THE SYNC IS FREE HERE FOR THE SAME REASON IT IS IN `sync.py`
------------------------------------------------------------
Frames and serial lines are stamped with `time.monotonic()` in this one loop, so there
is no offset to fit and none is fitted -- `sync.Run` does the lookup unchanged. What is
different from a `fly()` run is only the marker: a flight has `<> csv t=0` and a run CSV;
a schedule has neither, because nothing here commands anything. The `label=` line printed
by the firmware on every label change IS the event, and it is the only one. If the
labels stop arriving, this produces no stills rather than stills at guessed times.

    uv run python controller/control/tilt_sweep.py            # self-check, no hardware
    uv run python controller/control/tilt_sweep.py --run      # fly it (coils energise)
    uv run python controller/control/tilt_sweep.py --stills results/flights/<take>

Photos land in `results/tilt_sweep/<stamp>/ch0_<pct>pct_<cam>.png`, zero-padded so a
directory listing is the sweep in order, plus an `index.csv` saying which frame each
still came from and how far it sat from the label.

FIRST STEP IS THE HARSHEST STEP. The sweep runs 20% upward as asked, so the largest
asymmetry is commanded 3 s after spin-up and every later step reduces it. Reversing the
five `CH0_DUTY_*` blocks in `tilt.json` walks in gently instead; nothing here cares
about the order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: `label=FREQ_040HZ` -> 40. The schedule's other labels (DOWN_*, TILT_OFF) are kept as
#: timeline markers -- DOWN_<f>HZ is what bounds the hold -- but photograph nothing.
POINT_RE = re.compile(r"^FREQ_(\d+)HZ$")
#: A label can arrive glued to the boot banner rather than alone on its line: the ESP32
#: prints its ROM banner at a different baud, so the first `label=` of a run lands at the
#: tail of a line of mojibake. Searched for anywhere in the text, not anchored -- the
#: anchored version silently lost the first point of every run (measured 2026-09-01).
LABEL_RE = re.compile(r"label=([A-Z0-9_]+)")

OUT_DIR = ROOT / "results" / "tilt_sweep"
#: Photograph this long BEFORE the next label, not after the current one: the rotor is
#: still swinging into the new lean for most of a 2.5 s hold, and the settled attitude
#: is the measurement. Backed off from the boundary so a late frame cannot cross it.
SETTLE_BACK_S = 0.3


# ---- the run ------------------------------------------------------------------------
#: Drive seconds this schedule spends energised. The up-ramp scales with the target
#: the target and is tuned per point by the operator (25 s at 20 Hz, 23 s at 40, rising
#: to 40 s at 160), which is why this is a measured total and not a formula: 268 s of
#: up-ramp + 32 s of down + 40 s of holds. Re-measure it from `tilt.json` after any edit.
#: Passed to `coil_thermal.wait_until_safe`, which needs the figure up front -- it gates
#: on where the run ENDS, not on where it starts.
#:
#: `timeout_s` must exceed the schedule's WALL time, which is longer than this: the
#: coils-off gaps count against the clock but not against the heat. The 8-point sweep is
#: 420.5 s wall against 340 s of drive, and a 180 s timeout once cut it off at 140 Hz.
SCHEDULE_DRIVE_S = 340.5


def run(port=None, out_dir=None, width=640, height=400, fps=210.0, rotate180=True,
        indices=None, timeout_s=560.0, ignore_thermal=False, drone=None):
    """Reset the board, film the schedule, stop when it says `TILT_OFF`.

    Returns ``(flight_dir, log_path)``. Both are needed by `stills`, and both are
    printed, because a take whose folder nobody wrote down is a take nobody can cut.

    **640x400 at ~210 fps, the flight pipeline's mode, at the operator's instruction.**
    Both modes were measured on this rig 2026-09-01: 207.7 fps at 640x400, 119.7 fps at
    1280x800 native. The trade is pixels for frames -- native is 4x the pixels on a
    sub-centimetre rotor, which is what reading a lean angle off a still wants, while
    640x400 puts ~1040 frames inside a 5 s hold instead of ~600. Frames won. Changing
    `width`/`height`/`fps` here is the only thing that switches it, and no re-cut
    recovers resolution that was never captured -- so decide before the run, not after.
    """

    from controller.camera import identify, record, sources
    from controller.control.hover_controller_runner import CommandLink

    # Same gate as `fly()`, for the same reason: ~95 s of drive per sweep and no
    # temperature sensor anywhere on the rig. Missing model = refuse, never assume cold.
    from ai.thermal import coil_thermal
    if ignore_thermal:
        # Deliberate operator override. Printed, not silent, and the heat is still
        # stamped afterwards -- skipping the WAIT must not also skip the BOOKKEEPING,
        # or the next run reads a cold coil that is not cold.
        print(f"[coils] THERMAL GATE BYPASSED at the operator's instruction: "
              f"~{coil_thermal.temp_now():.0f}C now, this run adds "
              f"~{coil_thermal.HEAT_C_PER_S * SCHEDULE_DRIVE_S:.0f}C, ceiling is "
              f"{coil_thermal.T_CEILING_C:.0f}C. Watch the coils.")
    else:
        coil_thermal.wait_until_safe(SCHEDULE_DRIVE_S)

    idx = (identify.elp_indices() if indices is None else list(indices))
    tags = "AB"[:len(idx)]
    src = sources.open_stereo([f"camera:{i}" for i in idx], max_skew_s=None,
                              width=width, height=height, fps=fps, grayscale=True,
                              rotate180=rotate180)

    # WHICH ROBOT is not recoverable from anything else in the take: the schedule, the
    # firmware and the rig are identical between drones, so two sweeps differ only by
    # clock time. It goes in the folder name, in every filename, in `index.csv` and in
    # the flight's own `meta.json`, because a photo that gets moved out of its folder
    # must still say what it is a photo of.
    stamp = time.strftime("%Y%m%d_%H%M%S") + (f"_drone{drone}" if drone else "")
    log_path = OUT_DIR / stamp / "sweep.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fw = record.FlightWriter(record.DEFAULT_DIR, tags, fps,
                             meta={"source": "control.tilt_sweep", "schedule": "tilt.json",
                                   "drone": drone,
                                   "camera_indices": idx, "rotate180": bool(rotate180)})
    print(f"filming -> {fw.dir}\nlog      -> {log_path}")

    # Constructing the link resets the board, and the reset IS the start command --
    # there is nothing else to send. It then sleeps out the boot banner, so ~1.5 s of
    # footage is missed; that window is `driveBoot`'s own delay plus the schedule
    # compile, during which the coils are held low by `forceAllGatesLow`.
    link = CommandLink(port, dry_run=False, log_path=str(log_path))
    # The coils are now live with nothing having gone through `handle_serial_comm`.
    link.comm.note_external_drive(True)

    # `drain` already stamps every line into the log; this hook only watches for the
    # schedule's last label, so the parse is not duplicated and the log is not re-read.
    done, blocked = [], []
    def on_line(line, sent=False):
        if sent:
            return
        if line.startswith("label="):
            print(f"  {line}")
            if line == "label=TILT_OFF":
                done.append(line)
        elif "[block]" in line:
            # GPIO14, the only kill this firmware has. The firmware parks in a while
            # loop and stops printing, so waiting for TILT_OFF after this means waiting
            # out the whole timeout -- 180 s of nothing, measured 2026-09-01.
            print(f"  {line.strip()}")
            blocked.append(line)
    link.on_line = on_line

    t_start = time.monotonic()
    try:
        while not done and not blocked:
            item = src.read()
            if item is not None:
                t_cap, frames = item
                fw.add(t_cap, frames, getattr(src, "last_stamps", None),
                       getattr(src, "last_skew", 0.0) or 0.0)
            link.drain()
            if time.monotonic() - t_start > timeout_s:
                print(f"timeout after {timeout_s:.0f}s with no TILT_OFF -- stopping")
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        # Order matters: stop counting heat before anything slow, then close the link
        # (which stamps the interval), then finalise the video.
        link.comm.note_external_drive(False)
        link.close()
        src.close()
        flight = fw.close(src.skew_stats() if hasattr(src, "skew_stats") else {})
        # ALWAYS park, success included. Closing the port above resets the board, and
        # this firmware starts its schedule on boot -- so a run that reached TILT_OFF
        # cleanly is immediately followed by the WHOLE SCHEDULE RUNNING AGAIN, coils and
        # all, with nothing watching it. Measured 2026-09-01: the first successful sweep
        # left the rig re-ramping unattended. There is no `stop` this firmware can hear,
        # so parking the chip out of the app is the only software kill.
        park()
    print("schedule finished" if done else
          "schedule did NOT finish; board parked in the bootloader, coils off")
    return flight, log_path


def park(port="/dev/cu.SLAB_USBtoUART"):
    """Reset the ESP32 into the ROM bootloader and leave it there: app off, coils off.

    `main_tilt` parses no serial, so `safe_off.py` cannot reach it and every port
    open/close resets it straight back into the schedule. This is the software kill.
    """

    import subprocess

    home = Path.home()
    cmd = [str(home / ".platformio/penv/bin/python"),
           str(home / ".platformio/packages/tool-esptoolpy/esptool.py"),
           "--chip", "esp32", "--port", port,
           "--before", "default_reset", "--after", "no_reset", "chip_id"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = "Staying in bootloader" in r.stdout
    print("[coils] board parked in the bootloader, app not running" if ok else
          f"[coils] PARK FAILED -- press GPIO14 NOW.\n{r.stdout[-400:]}{r.stderr[-400:]}")
    return ok


# ---- the stills ---------------------------------------------------------------------
def labels(log_path):
    """``[(t_monotonic, label)]`` from a sweep log, in order.

    Reuses `sync.read_log`, so an unstamped line is dropped here exactly as it is there:
    a label with no clock cannot be placed on the video, and inventing one for it puts a
    confident wrong percentage on a photo.
    """

    from controller.control import sync

    got, _ = sync.read_log(log_path)
    out = []
    for t, d, text, _ in got:
        m = LABEL_RE.search(text)
        if d == "<-" and m:
            out.append((t, m.group(1)))
    return out


def stills(flight_dir, log_path, out_dir=None, drone=None):
    """Cut one still per camera per duty step. Returns the list of files written.

    Decoded sequentially rather than by `CAP_PROP_POS_FRAMES`: seeking H.264 by index
    lands on the nearest keyframe, which at a 2.5 s hold is easily the wrong duty step.
    A 45 s take is a few thousand frames and decodes in seconds.
    """

    import cv2

    from controller.camera import record
    from controller.control import sync

    flight_dir, log_path = Path(flight_dir), Path(log_path)
    out_dir = Path(out_dir) if out_dir else log_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fall back to the drone the take recorded, so re-cutting an old flight cannot
    # silently relabel it as an untagged one.
    if drone is None and (flight_dir / "meta.json").exists():
        drone = json.loads((flight_dir / "meta.json").read_text()).get("drone")
    tag_p = f"drone{drone}_" if drone else ""

    marks = labels(log_path)
    if not marks:
        raise SystemExit(f"{log_path} carries no `label=` lines, so nothing can be "
                         f"placed on the video. Is this firmware main_tilt.cpp?")
    run_ = sync.Run(log_path=log_path, flight_dir=flight_dir)
    if not run_.frames:
        raise SystemExit(f"{flight_dir} has no frames.csv, so no frame has a time.")

    # A dropped frame is written to `frames.csv` but never to the mp4 (FlightWriter
    # drops on a full encoder queue and counts it), so row n and decoded frame n stop
    # being the same frame from the first drop onward. Every still after that point
    # would be a real photo of the wrong duty step, which nothing downstream could
    # detect. Refuse: the take is unusable for cutting, even though it replays fine.
    dropped = json.loads((flight_dir / "meta.json").read_text()).get("dropped", 0) \
        if (flight_dir / "meta.json").exists() else 0
    if dropped:
        raise SystemExit(
            f"{flight_dir} dropped {dropped} frame(s), so frames.csv row n is no longer "
            f"mp4 frame n and every still past the first drop would carry the wrong "
            f"duty cycle. Re-shoot, or replay the video by hand against frames.csv.")

    # target instant per point: just before the point ends, see SETTLE_BACK_S.
    #
    # A point is photographed ONLY when its own DOWN_<f>HZ label is present. That label
    # is what proves the hold actually happened: it is emitted after the ramp has
    # finished and the 5 s hold has elapsed. Without it the point was cut short -- the
    # operator's block button, a timeout, a crash -- and there is no settled attitude to
    # photograph. The previous fallback (`t + MAX_HOLD_S` when no next label existed)
    # produced exactly the failure this module's docstring warns about: on
    # 2026-09-01 21:31 a run stopped 10 s in still wrote `f020hz`, and the frame it chose
    # was taken at 4.3 Hz with channel 0 at 100%. Wrong frequency, wrong duty, confident
    # filename. Skipping a point is recoverable; mislabelling one is not.
    wanted, skipped = {}, []
    ends = {name[len("DOWN_"):]: t for t, name in marks if name.startswith("DOWN_")}
    for i, (t, name) in enumerate(marks):
        m = POINT_RE.match(name)
        if not m:
            continue
        t_end = ends.get(f"{int(m.group(1)):03d}HZ")
        if t_end is None or t_end <= t:
            skipped.append(int(m.group(1)))
            continue
        hit = run_.frame_at(t_end - SETTLE_BACK_S)
        if hit is None:
            skipped.append(int(m.group(1)))
            continue
        idx, t_frame, off = hit
        wanted[idx] = (int(m.group(1)), t, t_frame, off)

    caps, _ = record.open_recording(flight_dir)
    tags = [p.parent.name for p in sorted(flight_dir.glob("*/*.mp4"))] or \
           [str(i) for i in range(len(caps))]
    written, rows, n = [], [], 0
    while wanted and any(c.isOpened() for c in caps):
        frames = [c.read() for c in caps]
        if not all(ok for ok, _ in frames):
            break
        if n in wanted:
            pct, t_label, t_frame, off = wanted.pop(n)
            for tag, (_, img) in zip(tags, frames):
                # Zero-padded so `ls` is the sweep in order, and the frequency is in the
                # filename because that is the one thing a photo cannot show.
                path = out_dir / f"{tag_p}f{pct:03d}hz_{tag}.png"
                cv2.imwrite(str(path), img)
                written.append(path)
            rows.append((pct, n, f"{t_label:.6f}", f"{t_frame:.6f}", f"{off * 1e3:+.1f}"))
        n += 1
    for c in caps:
        c.release()

    with open(out_dir / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["drone", "freq_hz", "frame", "t_label", "t_frame",
                    "frame_minus_target_ms"])
        w.writerows([(drone or "", *r) for r in sorted(rows)])
    print(f"{len(written)} still(s) -> {out_dir}")
    for pct, frame, *_ in sorted(rows):
        print(f"  {pct:3d} Hz  frame {frame}")
    if wanted:
        print(f"  MISSING {len(wanted)} point(s): the video ended before them")
    if skipped:
        print(f"  SKIPPED {', '.join(f'{f} Hz' for f in skipped)}: no DOWN_<f>HZ label, so "
              f"the hold never completed and there is no settled attitude to photograph")
    return written


# ---- self-check ---------------------------------------------------------------------
def _self_check():
    """Round-trip a synthetic take: does 40% get the frame from inside the 40% hold?

    Asserts the naming and the frame choice, not the parsing. A still cut from the
    neighbouring hold is a plausible-looking photo of the wrong duty cycle, and that is
    invisible in anything that merely checks the files exist -- so each synthetic frame
    is painted with its own duty step and the pixel is read back.
    """

    import shutil
    import tempfile

    import cv2
    import numpy as np

    from controller.camera import record

    tmp = Path(tempfile.mkdtemp(prefix="tilt-sweep-"))
    try:
        base, dt, hold = 1000.0, 1.0 / 20.0, 2.5
        steps = [20, 40, 60, 80, 100, 120, 140, 160]
        # Frame brightness == the duty step in force when it was captured.
        def point_at(t):
            k = int((t - base) // hold)
            return steps[k] if 0 <= k < len(steps) else 0

        fw = record.FlightWriter(tmp, tags="AB", fps=20.0)
        for i in range(int(len(steps) * hold / dt) + 20):
            t = base + i * dt
            img = np.full((32, 40), point_at(t), np.uint8)
            fw.add(t, [img, img], (t, t + 0.002), 0.002)
            if i % 32 == 31:
                time.sleep(0.005)   # let the encoder drain; a drop here fails the guard
        flight = fw.close()
        assert fw.dropped == 0, f"{fw.dropped} dropped -- stills would misalign"

        log = tmp / "sweep.log"
        # The first label is glued to a boot banner, as it is on the real rig.
        log.write_text(
            f"[{base:.3f}] <- \ufffd\ufffd garbage \ufffd label=FREQ_{steps[0]:03d}HZ\n"
            + "".join(f"[{base + k * hold:.3f}] <- label=FREQ_{p:03d}HZ\n"
                      for k, p in enumerate(steps) if k)
            + f"[{base + len(steps) * hold:.3f}] <- label=TILT_OFF\n"
            + "a line with no stamp at all\n")
        # Each point's hold is bounded by its own DOWN label, as on the rig: the next
        # FREQ label is not a substitute, and the last point has no next label at all.
        lines = log.read_text().splitlines(keepends=True)
        out_lines = []
        for k, p in enumerate(steps):
            out_lines.append(lines[k])
            out_lines.append(f"[{base + (k + 1) * hold - 0.01:.3f}] <- label=DOWN_{p:03d}HZ\n")
        log.write_text("".join(out_lines) + "".join(lines[len(steps):]))

        got = labels(log)
        assert [p for _, p in got][-1] == "TILT_OFF", got
        # The banner-glued first label must survive; anchoring the parse lost it.
        assert got[0][1] == f"FREQ_{steps[0]:03d}HZ", got[0]
        assert len(got) == 2 * len(steps) + 1, got

        out = tmp / "out"
        written = stills(flight, log, out, drone="7")
        assert len(written) == 2 * len(steps), written
        for p in steps:
            for tag in "AB":
                f = out / f"drone7_f{p:03d}hz_{tag}.png"
                assert f.exists(), f
                # The pixel says which hold the frame really came from. Nearest
                # step, not equality: the take is H.264 and a flat 80 decodes as 79.
                # The holds are 20 shades apart, so a neighbour is still unambiguous.
                shade = int(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)[0, 0])
                came_from = min(steps, key=lambda q: abs(q - shade))
                assert came_from == p, f"{f.name} came from the {came_from}% hold"
        assert sorted(x.name for x in out.glob("*.png"))[0].startswith("drone7_f020hz"), \
            "zero padding lost: the listing no longer reads in sweep order"
        # A point whose DOWN_<f>HZ never arrived must produce NO file. This is the
        # 2026-09-01 21:31 regression: the run stopped mid-ramp and still wrote a
        # confidently-named still of the wrong frequency.
        cut = tmp / "cut.log"
        cut.write_text(f"[{base:.3f}] <- label=FREQ_{steps[0]:03d}HZ\n")
        cut_out = tmp / "cut_out"
        assert stills(flight, cut, cut_out, drone="7") == [], "unfinished point was cut"
        assert not list(cut_out.glob("*.png")), list(cut_out.glob("*.png"))

        print(f"tilt_sweep: self-check passed ({len(written)} stills, "
              f"{', '.join(f'{p}Hz' for p in steps)}; unfinished point correctly skipped)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true", help="fly the schedule (ENERGISES COILS)")
    ap.add_argument("--stills", metavar="FLIGHT_DIR",
                    help="cut stills from an existing take instead of flying")
    ap.add_argument("--log", help="sweep log for --stills (default: newest under "
                                  "results/tilt_sweep)")
    ap.add_argument("--port", default=None)
    ap.add_argument("--drone", default=None,
                    help="which robot is on the rig; tags the folder, files and meta")
    ap.add_argument("--ignore-thermal", action="store_true",
                    help="skip the cool-down wait (the heat is still stamped)")
    args = ap.parse_args()

    if args.run:
        flight, log = run(port=args.port, ignore_thermal=args.ignore_thermal,
                          drone=args.drone)
        stills(flight, log, drone=args.drone)
    elif args.stills:
        log = args.log or max(OUT_DIR.glob("*/sweep.log"), key=lambda p: p.stat().st_mtime)
        stills(args.stills, log, drone=args.drone)
    else:
        _self_check()
