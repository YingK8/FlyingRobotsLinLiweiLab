#!/usr/bin/env python3
"""Run one frequency chunk of the alignment-rate sweep: N repeats, cooling between.

    uv run python controller/control/tilt_run.py                             # self-check
    uv run python controller/control/tilt_run.py --dry-run --freq 10         # plan only
    uv run python controller/control/tilt_run.py --freq 10 --repeats 5       # DRIVES COILS

Flash the matching schedule first, and remember that flashing starts a run:

    uv run python controller/control/tilt_schedule.py --freq 10 --write
    pio run -e tilt -t uploadfs && pio run -e tilt -t upload

This wraps `tilt_sweep.run()` and adds only what a chunk needs: the repeat loop, a thermal
gate that knows about frequency, and the GPIO14 restart. Everything else -- opening the
cameras, resetting the board (which IS the start command), `note_external_drive`, the
`sweep.log`, and the unconditional `park()` in the `finally` -- already lives there.

WHY THE FLAT THERMAL MODEL IS REPLACED HERE
-------------------------------------------
`coil_thermal` charges 0.5 C/s for every energised second and knows nothing about frequency.
Heating goes as I^2, and coil current is strongly frequency-dependent below resonance -- from
2406 telemetry samples across 13 existing `sweep.log`s, the commutated sum runs 2.37 A at
20 Hz against 13.82 A at 100 Hz. So the flat rate over-charges a 10 Hz point by about 30x,
and a five-repeat campaign that is genuinely ~5 hours of cooling would be gated as 23.

`I2_ANCHOR_A` is where the flat rate is right: the ~14 A plateau, which is also where the
bench supply's 100 W (10 V x 10 A) ceiling binds. Everything below scales by (I / anchor)^2.

This runs with `ignore_thermal=True` and does its own waiting. That is a deliberate override
of the documented chokepoint and it is printed every time, with the numbers. The flat model's
stamp still accumulates underneath (`link.close()` books wall-seconds of drive regardless),
so nothing is lost -- a later `fly()` or `tilt_sweep` still sees the conservative book.

**It is still a model, not a thermometer.** There is no temperature sensor on this rig. Coils A
AND C read a negative offset -- measured on the first design-B take, 2026-09-10, and it holds at
0% duty -- so a driven channel reading negative is charged the mean of the channels that read
sensibly (`_drive_sum`). Watch the coils.

GPIO14
------
The firmware prints two different lines and they are different events:

    press 1  "[block] button pressed -- gates off, press again to restart"
    press 2  "[block] button pressed -- restarting"

Press 1 kills the coils in under 30 ms, in firmware, before this process learns anything --
that property is why the firmware is left alone. This runner then discards the repeat and
starts it again from the beginning, which is what "press the button to restart it" means
operationally. The board is parked between the two, so the restart is a fresh boot into a
fresh take.

The aborted repeat is DISCARDED, never analysed. A restarted run appended to the same take
would give its `sweep.log` two `FREQ_<f>HZ` labels, and `alignment_rate.timeline()` builds one
point per `FREQ_` label -- so the analysis would silently gain a duplicate point whose first
half is a truncated ramp. `tilt_sweep.stills()` refuses incomplete points for exactly this
reason.

`RESTART_DELAY_S` is the escape hatch: Ctrl-C during it abandons the chunk instead of
restarting. Without it a press would be un-cancellable, since a parked board reports nothing.
"""

from __future__ import annotations

import argparse
import csv
import re
import select
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from controller.control import tilt_schedule, tilt_sweep
from controller.control.link import parse_telemetry

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results" / "alignment_rate"

#: Commutated-sum current where the flat 0.5 C/s was calibrated, and where the 100 W supply
#: ceiling binds. Heat scales as (I / this)^2.
I2_ANCHOR_A = 13.94
#: Seconds after a GPIO14 abort before the repeat restarts. Ctrl-C here abandons the chunk.
RESTART_DELAY_S = 10.0
#: Give up on a point after this many GPIO14 aborts -- something is wrong with the rig, and
#: retrying it forever just heats the coils.
MAX_ABORTS = 5
#: Between-repeat pause when the heat gate is OFF, per predicted degree of the point's own
#: heating. 20 s flat was the old behaviour and it is far too short where it matters: a
#: 120 Hz repeat books ~13 C, so this gives it ~275 s while 10 Hz (0.07 C) still falls to
#: the floor below. Added 2026-09-10 at the operator's instruction after 110/120 Hz chunks
#: ran back to back. It is a PAUSE, not a thermal guarantee -- with `--no-cool` nothing
#: here reads a temperature and the operator is still the thermometer.
#:
#: Raised 10 -> 20 on 2026-09-10. Where the number comes from: Newton cooling sheds
#: (T - T_AMBIENT) / TAU_COOL_S degrees a second, so at ~60 C that is 38/1500 = 0.025 C/s,
#: and giving back one degree costs ~40 s. 20 s/C is deliberately HALF of honest -- it keeps
#: the campaign near 5 h instead of 7.6 h, and the operator's measurement at the prompt is
#: what covers the difference. It is a compromise, not a guarantee; see `control/theory.md`
#: 24.12.
NO_COOL_S_PER_C = 20.0

#: Floor on the between-repeat wait. Even a thermally free point wants the rotor stopped and
#: the rig still before the next ramp starts.
MIN_COOL_S = 20.0
#: Ceiling on ONE wait, so a hot point cannot stall the campaign indefinitely. At 20 s/C this
#: binds above 30 C of predicted heat per repeat, which nothing in the 10-120 Hz sweep reaches.
MAX_COOL_S = 600.0
#: Floor on the wait BETWEEN frequencies. The next chunk flashes, reboots and re-opens the
#: cameras, so this is on top of ~20 s of dead time either way.
MIN_CHUNK_COOL_S = 180.0

#: Above this frequency the campaign takes fewer repeats: the points are hot and slow, and
#: `control/theory.md` 24.5 argues five is the minimum that can see the non-modal response
#: at all ("five repeats is the minimum, not a luxury"). Ten below it buys a real median.
HIGH_REPEAT_F_HZ = 80.0
HIGH_REPEATS = 5

#: Prompt the operator on any wait longer than this. Shorter ones just sleep, so a ten-repeat
#: 10 Hz chunk does not ask ten times for a measurement that cannot matter -- the prompt then
#: appears exactly where the heat is, which is the high frequencies and between chunks.
PROMPT_ABOVE_S = 60.0
#: The out-of-process way to answer that prompt: write a temperature, or "go", into this file.
#: Beside the thermal stamp it re-anchors, and gitignored with the rest of `ai/`.
OPERATOR_FILE = ROOT / "ai" / "thermal" / "operator_reading"
#: How often a waiting prompt looks for an answer. Also bounds how late an early release is.
OPERATOR_POLL_S = 5.0

#: Disk budget per take, megabytes. MEASURED, not fitted: `results/flights/Archive.zip` is
#: 2.92 GB of video over 139 design-A takes, ~18 MB each, and design-B takes are longer
#: (DROP_MS went to 15 s). Rounded up.
MB_PER_TAKE = 25
#: Headroom multiplier on the pre-flight space check. A run that dies at 90% full loses the
#: campaign, and the analysis writes ~2 MB of solved CSV per take on top of the video.
SPACE_MARGIN = 1.3


def heat_c(log_path, fallback_s=0.0):
    """Degrees this run added, integrating I^2 dt over the telemetry it actually printed.

    Returns ``(degrees, n_samples)``. With no telemetry it falls back to the flat model on
    ``fallback_s`` -- erring toward the conservative number rather than toward zero.
    """

    from ai.thermal import coil_thermal

    rows = []
    for raw in Path(log_path).read_text(errors="replace").splitlines():
        m = re.match(r"^\[(\d+\.\d+)\]", raw)
        if not m:
            continue
        tel = parse_telemetry(raw)
        if tel is None or not tel.amps:
            continue
        # Gate on COMMANDED DUTY, not on the measured current. Coils A and C read -1.28 and
        # -1.04 A with every carrier at 0% -- a known sense fault -- so summing |I| unfiltered
        # bills a parked board. It charged ~1 C to a 10 Hz run that drew almost nothing
        # (2026-09-10). Zero duty is zero drive by definition, whatever the ADC says.
        if not (tel.duty and max(tel.duty) > 0.0):
            continue
        rows.append((float(m.group(1)), _drive_sum(tel.amps, tel.duty)))
    if len(rows) < 2:
        return coil_thermal.HEAT_C_PER_S * fallback_s, 0

    # Trapezoid on I^2, so a run that ramps is charged for the ramp and not for its peak.
    total = 0.0
    for (t0, i0), (t1, i1) in zip(rows, rows[1:]):
        dt = t1 - t0
        if 0.0 < dt < 5.0:                     # a longer gap is a stall, not drive
            total += 0.5 * (i0 ** 2 + i1 ** 2) * dt
    return coil_thermal.HEAT_C_PER_S * total / I2_ANCHOR_A ** 2, len(rows)


def _drive_sum(amps, duty):
    """Commutated current sum for one telemetry row, counting only what can be real.

    A channel at 0% duty draws nothing, whatever it reads. A NEGATIVE reading is not a
    magnitude at all: coils A and C both sit at -1.8 to -2.2 A through a whole design-B take
    and HOLD -2.24 / -1.84 with their own duty at 0% and the drive at 0 Hz (2026-09-10,
    `tilt_sweep/20260910_231937_droneB`). That is a sense offset, and summing its magnitude
    billed a 10 Hz take 1.66 C against ~0.07 real -- 24x, all of it from two phantom amps.

    A driven channel with an untrustworthy reading is charged the mean of the driven channels
    that read sensibly, since all four see the same drive. If none do, fall back to the raw
    magnitudes: over-charging is the safe direction when there is nothing better.
    """

    live = [(a, d) for a, d in zip(amps, duty) if d > 0.0]
    good = [a for a, _ in live if a >= 0.0]
    if not good:
        return sum(abs(a) for a, _ in live)
    return sum(good) + (len(live) - len(good)) * (sum(good) / len(good))


def cool_for(temp_c, next_point_c):
    """Seconds to wait so the next point still lands under the ceiling. Newton cooling."""

    import math

    from ai.thermal import coil_thermal

    need = coil_thermal.T_CEILING_C - next_point_c
    if temp_c <= need:
        return MIN_COOL_S
    amb = coil_thermal.T_AMBIENT_C
    wait = coil_thermal.TAU_COOL_S * math.log(max(temp_c - amb, 1e-6) / max(need - amb, 0.5))
    return max(wait, MIN_COOL_S)


def check_space(out_root, n_takes):
    """Refuse to start a campaign that cannot fit on disk. Refuses, never warns.

    Same posture as `ramp.check`: a five-hour unattended run that fills the volume at take 80
    loses the takes it was still writing AND the ones it never got to, and no amount of
    watching afterwards recovers them. `results/flights` is gitignored, so a lost take is
    lost for good.
    """

    root = Path(out_root) if out_root else record_dir()
    root.mkdir(parents=True, exist_ok=True)
    need_mb = MB_PER_TAKE * n_takes * SPACE_MARGIN
    free_mb = shutil.disk_usage(root).free / (1 << 20)
    print(f"  disk: {free_mb:.0f} MB free on {root}, "
          f"{n_takes} takes need ~{need_mb:.0f} MB")
    if free_mb < need_mb:
        raise SystemExit(
            f"  REFUSED: {free_mb:.0f} MB free on {root} against ~{need_mb:.0f} MB needed "
            f"for {n_takes} takes at ~{MB_PER_TAKE} MB each.\n"
            f"  Point --flights-root at a larger volume, or cut the sweep.")


def record_dir():
    """Where takes land by default. Imported lazily -- `record` pulls in OpenCV."""

    from controller.camera import record

    return record.DEFAULT_DIR


def repeats_for(f_hz, base):
    """Repeats for one frequency. See `HIGH_REPEAT_F_HZ`."""

    return HIGH_REPEATS if float(f_hz) > HIGH_REPEAT_F_HZ else int(base)


def wait_s(heat_c, mult=1.0, floor=MIN_COOL_S):
    """Deterministic, frequency-scaled pause, in seconds.

    Reads no temperature -- that is the point. The thermal model's running total is a worst
    case that compounds (`run_chunk`'s docstring has the 2026-09-10 numbers), so the campaign
    waits on a schedule instead and asks the operator for the one real reading.
    """

    return min(max(floor, NO_COOL_S_PER_C * float(heat_c) * mult), MAX_COOL_S)


def wait_or_prompt(seconds, why=""):
    """Sleep, but let the operator end it early by measuring the coils.

    There is no temperature sensor on this rig, so the operator IS the thermometer. A reading
    is written to `coil_thermal.STAMP` -- the only real temperature the model ever gets, and
    what makes the headroom line `run_chunk` prints mean anything. A bare "go" releases the
    wait without claiming a measurement.

    Two ways in, because the campaign is usually run detached from any terminal:

    * a tty: type the temperature (or nothing) and Enter;
    * `OPERATOR_FILE`: write the temperature or ``go`` into it. Checked every few seconds,
      consumed on read. This is how a reading relayed from outside the process gets in.

    Unattended it waits out the full time: the prompt is an accelerator, never a barrier, so
    the campaign still completes with nobody watching.

    A number at or above the ceiling is RECORDED and does not release the wait. "Safe" is the
    operator's judgement to make, but it is not made by typing a bigger number.

    `KeyboardInterrupt` propagates, so the callers' existing handlers still abandon the chunk.
    """

    seconds = max(float(seconds), 0.0)
    if seconds <= PROMPT_ABOVE_S:
        time.sleep(seconds)
        return

    from ai.thermal import coil_thermal

    tty = sys.stdin.isatty()
    deadline, last_note = time.monotonic() + seconds, 0.0
    # Fixed marker text: a log watcher keys off "MEASURE THE COILS".
    print(f"  {why}waiting {seconds:.0f} s. MEASURE THE COILS -- "
          + ("type the temperature in C and Enter (bare Enter continues), or " if tty else "")
          + f"write it (or 'go') to {OPERATOR_FILE}. Ctrl-C abandons.", flush=True)
    while True:
        left = deadline - time.monotonic()
        if left <= 0.0:
            print("    wait complete", flush=True)
            return
        line = _operator_line(min(left, OPERATOR_POLL_S))
        if line is None:
            if time.monotonic() - last_note >= 60.0:
                print(f"    {left:.0f} s left", flush=True)
                last_note = time.monotonic()
            continue
        if line.lower() == "go":
            print("    continuing (no measurement recorded)", flush=True)
            return
        try:
            temp = float(line)
        except ValueError:
            print(f"    {line!r} is not a number or 'go' -- still waiting", flush=True)
            continue
        # Re-anchor the model to the measurement, whichever way the decision goes: a reading
        # ABOVE the ceiling is the most valuable one there is and must not be thrown away.
        _stamp(temp)
        if temp >= coil_thermal.T_CEILING_C:
            print(f"    {temp:.0f} C recorded -- at or over the "
                  f"{coil_thermal.T_CEILING_C:.0f} C ceiling, so still waiting", flush=True)
            continue
        print(f"    {temp:.0f} C recorded; continuing early", flush=True)
        return


def _operator_line(timeout):
    """One answer from the operator within ``timeout`` s, or None.

    A tty line if there is a terminal, else (or also) the contents of `OPERATOR_FILE`,
    consumed on read so one reading is one answer.
    """

    if sys.stdin.isatty():
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.readline().strip() or "go"
    else:
        time.sleep(timeout)
    if OPERATOR_FILE.exists():
        line = OPERATOR_FILE.read_text().strip() or "go"
        OPERATOR_FILE.unlink()
        return line
    return None


def _stamp(temp_c):
    """Write a MEASURED temperature into the thermal stamp -- the model's only real anchor."""

    from ai.thermal import coil_thermal

    coil_thermal.STAMP.parent.mkdir(parents=True, exist_ok=True)
    coil_thermal.STAMP.write_text(f"{temp_c:.1f}  {time.strftime('%Y-%m-%d %H:%M:%S')}")


#: A reading that opened the gate this recently is reused, so the flash before a chunk and the
#: take right after it do not ask twice for the same measurement.
GATE_FRESH_S = 120.0
_gate_opened_at = [float("-inf")]


def gate_on_reading(max_c, why=""):
    """Block until the operator reports the coils below ``max_c``. No timeout.

    Added 2026-09-11, after clock waits sized from `_predicted_c` let the coils reach a
    measured 100 C while the model read 63-75 C (`control/theory.md` 24.12). The model
    under-counts heat ~2x from ~60 Hz up, so anything derived from it -- including a longer
    clock -- inherits the error. This asks the one instrument that has been right.

    Unattended, the campaign PAUSES here. That is the intended failure mode: a take that
    waits for a person costs time, and a take that does not cost the coils.

    ``go`` does not open it. The gate exists for a measurement, and a release without one is
    exactly what the clock already offered.
    """

    if time.monotonic() - _gate_opened_at[0] < GATE_FRESH_S:
        return
    print(f"  {why}GATE: MEASURE THE COILS -- a reading below {max_c:.0f} C starts the next "
          f"take. Write it to {OPERATOR_FILE}"
          + (" or type it here" if sys.stdin.isatty() else "")
          + ". No timeout; Ctrl-C abandons.", flush=True)
    last_note = time.monotonic()
    while True:
        line = _operator_line(OPERATOR_POLL_S)
        if line is None:
            if time.monotonic() - last_note >= 300.0:
                print("    gate still waiting for a coil reading", flush=True)
                last_note = time.monotonic()
            continue
        try:
            temp = float(line)
        except ValueError:
            print(f"    {line!r}: the gate needs a temperature in C", flush=True)
            continue
        _stamp(temp)
        if temp >= max_c:
            print(f"    {temp:.0f} C recorded -- not below {max_c:.0f} C, gate stays shut",
                  flush=True)
            continue
        print(f"    {temp:.0f} C recorded; gate open", flush=True)
        _gate_opened_at[0] = time.monotonic()
        return


def _gated(freq, gate):
    return gate is not None and float(freq) >= gate[0]


def plan(freq, repeats):
    """What the chunk will cost, before anything is energised."""

    drive = tilt_schedule.drive_s(freq)
    per = _predicted_c(freq, drive)
    return {"freq": freq, "repeats": repeats, "ramp_s": tilt_schedule.ramp_ms(freq) / 1000.0,
            "dwell_s": tilt_schedule.dwell_s(freq), "drive_s": drive,
            "heat_per_repeat_c": per, "heat_chunk_c": per * repeats}


def _predicted_c(freq, drive_s):
    """Rough heat for one point, from the current-vs-frequency table. A prior, not a fit."""

    from ai.thermal import coil_thermal

    # Measured commutated sum |I| against drive frequency (13 sweep.logs, 2406 samples).
    tbl = [(20, 2.37), (40, 4.61), (60, 6.68), (80, 10.41), (100, 13.82), (170, 13.94)]
    f = float(freq)
    if f <= tbl[0][0]:
        i = tbl[0][1] * f / tbl[0][0]
    else:
        i = tbl[-1][1]
        for (f0, i0), (f1, i1) in zip(tbl, tbl[1:]):
            if f0 <= f <= f1:
                i = i0 + (i1 - i0) * (f - f0) / (f1 - f0)
                break
    ts = tilt_schedule
    # Weight each phase by the current it actually draws. The ramp sweeps 2 Hz -> f, so it
    # spends most of its time well below I(f); I^2 averaged over a roughly linear sweep is
    # I(f)^2 / 3. During the drop two of the four coils are off, so call it half.
    # Read the hold from the schedule generator, not from its module constants: above
    # HIGH_F_HZ the hold is shortened and the ramp slowed, and hard-coding the constants here
    # charged the old 10 s hold against the new 5 s schedule -- which made the change look
    # like it INCREASED heat when it reduces it.
    settle_ms, hold_msec = ts.hold_ms(f)
    equiv_s = (ts.ramp_ms(f) / 1000.0 / 3.0
               + (settle_ms + hold_msec) / 1000.0
               + ts.DROP_MS / 1000.0 / 2.0
               + ts.DOWN_MS / 1000.0 / 3.0)
    return coil_thermal.HEAT_C_PER_S * (i ** 2) / I2_ANCHOR_A ** 2 * equiv_s


def run_chunk(freq, repeats=5, port=None, out_dir=None, dry_run=False, cool=True,
              force_hot=False, gate=None, **kw):
    """N repeats of one frequency, cooling between. Returns the index rows.

    ``cool=False`` runs them back to back. That is an operating decision, taken by the
    operator on 2026-09-10, and it removes the last thermal protection on a rig that has no
    temperature sensor: the 10 A supply limit and the GPIO14 button are what remain. The
    measured heat is still recorded per repeat, so the record exists even when nothing acts
    on it.

    Its immediate justification: `coil_thermal`'s flat book had reached 98 C after a run of
    95 s timeouts that drew almost no current, and was demanding 11 minutes of cooling for a
    10 Hz point that the I^2 integral measures at 0.46 C. Gating on a number that wrong is
    not safety, it is just delay.
    """

    from ai.thermal import coil_thermal

    p = plan(freq, repeats)
    print(f"\n=== {freq:g} Hz x {repeats} ===")
    print(f"  ramp {p['ramp_s']:.0f} s (dwell {p['dwell_s']:.1f} s below pull-in), "
          f"{p['drive_s']:.0f} s energised per repeat")
    print(f"  predicted heat {p['heat_per_repeat_c']:.2f} C/repeat, "
          f"{p['heat_chunk_c']:.1f} C for the chunk, "
          f"against {coil_thermal.T_CEILING_C - coil_thermal.temp_now():.0f} C of headroom now")
    for lbl in tilt_schedule.schedule(freq):
        if lbl["method"] == "label":
            print(f"    label={lbl['value']}")
    # The guard that 2026-09-10 needed and did not have. `--no-cool` skips the WAIT; it must
    # not also skip the arithmetic. Seven chunks ran that night with cooling off and booked
    # 105.9 C of heating, and the operator measured the coils at 100 C -- the model was right
    # and was overridden every run. The cost is superlinear and the cheap chunks come first,
    # so by the time the gate mattered it had been off for an hour. `control/theory.md` 24.4.
    headroom = coil_thermal.T_CEILING_C - coil_thermal.temp_now()
    if p["heat_chunk_c"] > headroom and not dry_run:
        msg = (f"  {freq:g} Hz x {repeats} is predicted at {p['heat_chunk_c']:.1f} C against "
               f"{headroom:.1f} C of headroom.")
        if not force_hot:
            raise SystemExit(
                msg + f"\n  REFUSED. Cool the coils, drop to "
                f"{max(int(headroom / max(p['heat_per_repeat_c'], 1e-6)), 0)} repeats, or pass "
                f"--force-hot if you are watching them.")
        print(msg + "  --force-hot: proceeding anyway. WATCH THE COILS.")
    if dry_run:
        print("  --dry-run: nothing energised")
        return []

    out = Path(out_dir or (OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
                           / f"{int(freq):03d}hz"))
    out.mkdir(parents=True, exist_ok=True)
    # Continue an existing index rather than replacing it. Re-running a chunk to ADD repeats
    # is the normal case (90 Hz was finished in two sittings on 2026-09-10), and overwriting
    # silently dropped the earlier takes from the index while leaving them on disk -- the
    # analysis then pooled 3 repeats and reported it as the full set.
    prior = []
    idx_path = out / "index.csv"
    if idx_path.exists():
        with open(idx_path) as fh:
            prior = [r for r in csv.DictReader(fh)]
        # DictReader yields strings. Rows appended later in this run hold real numbers, so
        # anything that sums across both must see one type -- `sum(r["heat_c"] ...)` raised
        # TypeError on the closing summary after a resumed chunk, AFTER the run had already
        # completed and been written.
        for r in prior:
            try:
                r["heat_c"] = float(r.get("heat_c") or 0.0)
                r["repeat"] = int(r["repeat"])
                r["telemetry_samples"] = int(r.get("telemetry_samples") or 0)
            except (TypeError, ValueError):
                r["heat_c"] = 0.0
        if prior:
            print(f"  {len(prior)} existing repeat(s) in {idx_path.name}; appending")
    rows, aborts = list(prior), 0
    rep = max((int(r["repeat"]) for r in prior), default=0) + 1
    last = rep + repeats - 1
    while rep <= last:
        print(f"\n-- {freq:g} Hz repeat {rep}/{last} "
              f"(model says ~{coil_thermal.temp_now():.0f} C) --")
        if _gated(freq, gate):
            gate_on_reading(gate[1], f"{freq:g} Hz repeat {rep}: ")
        t_before = coil_thermal.temp_now()
        flight, log, outcome = tilt_sweep.run(
            port=port, drive_s=p["drive_s"], ignore_thermal=True,
            timeout_s=p["drive_s"] + tilt_schedule.OFF_MS / 1000.0 + 60.0, **kw)
        c, n = heat_c(log, p["drive_s"])
        # Re-anchor the stamp to the MEASURED heat. `tilt_sweep.run` stamps through
        # SerialComm's flat 0.5 C/s model, which bills every energised second at the
        # resonance current whatever the drive frequency -- and coil current is strongly
        # frequency dependent (2.4 A summed at 20 Hz against 13.9 at 150+). On 2026-09-10 a
        # 10 Hz chunk that measured 5.1 C stamped 85.6 C, roughly 40x over, and the gate
        # then refused all eight remaining chunks of an unattended overnight run. The I^2
        # integral is the measurement; the flat model is a worst case, and using a worst
        # case as a running total compounds it.
        if n > 0:
            coil_thermal.STAMP.parent.mkdir(parents=True, exist_ok=True)
            coil_thermal.STAMP.write_text(
                f"{t_before + c:.1f}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        rows.append({"freq_hz": freq, "repeat": rep, "outcome": outcome,
                     "flight": str(flight), "log": str(log),
                     "heat_c": round(c, 3), "telemetry_samples": n})
        _write_index(out / "index.csv", rows)
        print(f"  outcome={outcome}  measured heat {c:.2f} C from {n} telemetry samples")

        if outcome.startswith("gpio14"):
            aborts += 1
            if aborts > MAX_ABORTS:
                print(f"  {aborts} GPIO14 aborts on this chunk -- stopping. "
                      f"Something is wrong with the rig, not with the schedule.")
                break
            print(f"  repeat {rep} DISCARDED (it will not be analysed). "
                  f"Restarting it in {RESTART_DELAY_S:.0f} s -- Ctrl-C to abandon the chunk.")
            try:
                time.sleep(RESTART_DELAY_S)
            except KeyboardInterrupt:
                print("\n  chunk abandoned by the operator")
                break
            continue                              # same repeat number, fresh take

        if rep < last and not _gated(freq, gate):      # a gated take waits at its gate
            wait = (cool_for(coil_thermal.temp_now(), p["heat_per_repeat_c"]) if cool
                    else wait_s(p["heat_per_repeat_c"]))
            print(f"  {'cooling' if cool else 'pausing (heat gate OFF)'} "
                  f"{wait:.0f} s before repeat {rep + 1}")
            try:
                wait_or_prompt(wait, f"before {freq:g} Hz repeat {rep + 1}: ")
            except KeyboardInterrupt:
                print("\n  chunk abandoned by the operator during cooldown")
                break
        rep += 1

    ok = sum(r["outcome"] == "ok" for r in rows)
    print(f"\n{freq:g} Hz: {ok}/{len(rows)} good, "
          f"{sum(r['heat_c'] for r in rows):.1f} C added -> {out}")
    return rows


def flash(freq, verbose=True):
    """Generate the schedule for one frequency, upload it, and park the board.

    The park matters: `uploadfs` resets the chip, and `main_tilt` starts its schedule on
    boot -- so without it the rig runs a whole unfilmed point before the camera is open.
    """

    import subprocess

    tilt_schedule.write(freq)
    r = subprocess.run(["pio", "run", "-e", "tilt", "-t", "uploadfs"],
                       cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"uploadfs failed for {freq:g} Hz:\n{r.stdout[-800:]}{r.stderr[-800:]}")
    if verbose:
        print(f"  flashed {freq:g} Hz")
    tilt_sweep.park()
    return True


def sweep(freqs=None, repeats=5, port=None, out_dir=None, cool=True, force_hot=False,
          gate=None, **kw):
    """The whole campaign: every frequency, `repeats` each, flashing between.

    One schedule per frequency has to be uploaded because `main_tilt` parses no serial --
    the JSON on SPIFFS *is* the experiment, and the only way to change the target frequency
    is to write a new one and reboot.
    """

    freqs = list(tilt_schedule.FREQS if freqs is None else freqs)
    root = Path(out_dir or (OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")))
    root.mkdir(parents=True, exist_ok=True)
    reps = {f: repeats_for(f, repeats) for f in freqs}
    # RESUMABLE: count the good takes already in each chunk and run only the shortfall. A
    # five-hour campaign gets interrupted -- a port drops, a kernel dies, the operator stops
    # it -- and re-running the same command with the same --out must finish the job, not
    # stack a second set of repeats on top of the first. Only `ok` counts: a timeout or a
    # GPIO14 abort is a missing take, and it is re-recorded.
    have = {f: [r for r in _read_index(root / f"{int(f):03d}hz" / "index.csv")
                if r.get("outcome") == "ok"] for f in freqs}
    need = {f: max(reps[f] - len(have[f]), 0) for f in freqs}
    n_takes = sum(need.values())
    print(f"sweep: {[(f, reps[f]) for f in freqs]} -> {root}")
    print(f"  {sum(len(v) for v in have.values())} good takes already on disk, "
          f"{n_takes} to record")
    if not kw.get("dry_run") and n_takes:
        check_space(kw.get("out_root"), n_takes)
    allrows = []
    for f in freqs:
        chunk = root / f"{int(f):03d}hz"
        if need[f] == 0:
            print(f"\n=== {f:g} Hz: {len(have[f])}/{reps[f]} good takes already -- done ===")
            allrows += _read_index(chunk / "index.csv")
            continue
        try:
            # --dry-run must not touch the hardware. `flash` uploads AND resets the board,
            # and `main_tilt` starts its schedule on boot, so a "dry" run was energising the
            # coils -- it only failed here because another chunk held the serial port.
            if not kw.get("dry_run"):
                # Gate BEFORE the flash: `uploadfs` resets the board, and `main_tilt` energises
                # on boot until `flash` parks it. Seconds of drive, but not on hot coils.
                if _gated(f, gate):
                    gate_on_reading(gate[1], f"before flashing {f:g} Hz: ")
                flash(f)
                # The schedule that ran, beside the takes it produced. `tilt.json` on SPIFFS
                # is overwritten by the next frequency, so without this copy the only record
                # of what a chunk flew is the generator's code at some later commit.
                chunk.mkdir(parents=True, exist_ok=True)
                shutil.copy(tilt_schedule.SPIFFS, chunk / "tilt.json")
            rows = run_chunk(f, need[f], port=port, out_dir=chunk,
                             cool=cool, force_hot=force_hot, gate=gate, **kw)
        except SystemExit as e:
            print(f"  {f:g} Hz SKIPPED: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\nsweep interrupted during {f:g} Hz")
            break
        allrows += rows
        if allrows:                       # a dry run produces none
            _write_index(root / "sweep_index.csv", allrows)
        # Cool BETWEEN chunks as well as between repeats. `sweep` had no inter-chunk wait at
        # all until 2026-09-10, so a ten-repeat chunk ended and the next frequency's ramp
        # started ~20 s later with every one of those degrees still in the coils. Twice the
        # per-repeat wait, because a chunk is worth many repeats of heat.
        nxt = freqs[freqs.index(f) + 1] if f != freqs[-1] else None
        if nxt is not None and rows and not kw.get("dry_run") and not _gated(nxt, gate):
            wait = wait_s(plan(f, 1)["heat_per_repeat_c"], mult=2.0, floor=MIN_CHUNK_COOL_S)
            print(f"\n  between chunks: {wait:.0f} s before the next frequency")
            try:
                wait_or_prompt(wait, f"after the {f:g} Hz chunk: ")
            except KeyboardInterrupt:
                print("\nsweep abandoned by the operator between chunks")
                break
    ok = sum(r["outcome"] == "ok" for r in allrows)
    heat = sum(float(r.get("heat_c") or 0.0) for r in allrows)
    print(f"\nsweep done: {ok}/{len(allrows)} good takes, {heat:.1f} C measured -> {root}")
    short = {f: reps[f] - sum(r["outcome"] == "ok" and float(r["freq_hz"]) == f
                              for r in allrows) for f in freqs}
    short = {f: n for f, n in short.items() if n > 0}
    if short and not kw.get("dry_run"):
        print(f"  STILL SHORT: {short}. Re-run the same command to record them.")
    return allrows


def _read_index(path):
    """Rows of an `index.csv`, or none if it does not exist yet."""

    if not Path(path).exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _write_index(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _self_check():
    import tempfile

    # the heat integral: a constant anchor current for 10 s must cost the flat rate
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "sweep.log"
        lines = []
        for k in range(21):
            t = 1000.0 + 0.5 * k
            a = I2_ANCHOR_A / 4.0                 # four channels summing to the anchor
            lines.append(f"[{t:.3f}] <- t=1 freq=100.0 | I[A]: A={a:.2f} B={a:.2f} "
                         f"C={a:.2f} D={a:.2f} | duty[%]: A=100.0 B=100.0 C=100.0 "
                         f"D=100.0 | spread=0.0 bal=0 trip=0")
        log.write_text("\n".join(lines))
        from ai.thermal import coil_thermal
        c, n = heat_c(log, 0.0)
        assert n == 21, n
        assert abs(c - coil_thermal.HEAT_C_PER_S * 10.0) < 0.05, c
        # half the current is a quarter of the heat
        half = "\n".join(x.replace(f"{I2_ANCHOR_A / 4:.2f}", f"{I2_ANCHOR_A / 8:.2f}")
                         for x in lines)
        log.write_text(half)
        c2, _ = heat_c(log, 0.0)
        assert abs(c2 - c / 4.0) < 0.02, (c, c2)
        # a parked board reads current on the faulty channels but commands zero duty, and
        # must be billed nothing. This is the 2026-09-10 bug: ~1 C charged to an idle rig.
        idle = "\n".join(
            f"[{1000.0 + 0.5 * k:.3f}] <- t=1 freq=0.0 | I[A]: A=-1.28 B=0.00 C=-1.04 "
            f"D=0.00 | duty[%]: A=0.0 B=0.0 C=0.0 D=0.0 | spread=1.28 bal=0 trip=0"
            for k in range(21))
        log.write_text(idle)
        c4, n4 = heat_c(log, 0.0)
        assert n4 == 0 and c4 == 0.0, (c4, n4)

        # the design-B take: A and C read a negative offset whether driven or not. Driven,
        # they are charged what B and D measure; at 0% duty they are charged nothing.
        assert abs(_drive_sum([-2.24, 0.40, -1.84, 0.40], [100.0] * 4) - 1.6) < 1e-9
        assert abs(_drive_sum([-2.24, 0.40, -1.84, 0.40], [0.0, 100.0, 0.0, 100.0])
                   - 0.8) < 1e-9
        # nothing trustworthy at all: the raw magnitudes, erring hot
        assert abs(_drive_sum([-1.0, -1.0, -1.0, -1.0], [100.0] * 4) - 4.0) < 1e-9

        # no telemetry falls back to the flat model rather than to zero
        log.write_text("nothing useful here\n")
        c3, n3 = heat_c(log, 40.0)
        assert n3 == 0 and abs(c3 - coil_thermal.HEAT_C_PER_S * 40.0) < 1e-9, (c3, n3)

    # cooling. 65 C with a 5 C point needs NO wait -- it lands exactly on the 70 C ceiling,
    # which is what `need = ceiling - next_point` means and was worth getting wrong once.
    assert cool_for(25.0, 5.0) == MIN_COOL_S
    assert cool_for(65.0, 5.0) == MIN_COOL_S
    assert cool_for(69.0, 10.0) > 300.0
    # a more expensive next point demands more cooling from the same temperature
    assert cool_for(69.0, 20.0) > cool_for(69.0, 10.0)

    # the plan must rank the sweep the way the physics does: 130 Hz costs far more than 10
    lo, hi = plan(10, 5), plan(130, 5)
    assert hi["heat_per_repeat_c"] > 20 * lo["heat_per_repeat_c"], (lo, hi)
    assert hi["ramp_s"] > lo["ramp_s"]

    # the repeat split: 80 Hz is still a low point, 90 is a high one
    assert repeats_for(80, 10) == 10 and repeats_for(90, 10) == HIGH_REPEATS
    # time-based waits clamp at both ends and scale in between
    assert wait_s(0.0) == MIN_COOL_S and wait_s(1e6) == MAX_COOL_S
    assert abs(wait_s(5.0, mult=2.0) - 2.0 * wait_s(5.0)) < 1e-9
    assert wait_s(0.0, floor=MIN_CHUNK_COOL_S) == MIN_CHUNK_COOL_S
    # a short wait never prompts and returns on time
    t0 = time.monotonic()
    wait_or_prompt(0.05)
    assert time.monotonic() - t0 < 1.0

    # the gate: a reading below the threshold opens it and is reused while fresh; the check
    # runs on temp paths so it cannot touch the real stamp or a real operator reading
    global OPERATOR_FILE
    from ai.thermal import coil_thermal as _ct
    real_op, real_stamp = OPERATOR_FILE, _ct.STAMP
    with tempfile.TemporaryDirectory() as d:
        OPERATOR_FILE, _ct.STAMP = Path(d) / "op", Path(d) / "stamp"
        try:
            assert _gated(70, (70.0, 45.0)) and not _gated(60, (70.0, 45.0))
            assert not _gated(120, None)
            _gate_opened_at[0] = float("-inf")
            OPERATOR_FILE.write_text("30")
            gate_on_reading(45.0)                              # opens on 30 < 45
            assert not OPERATOR_FILE.exists(), "a reading must be consumed"
            assert _ct.STAMP.read_text().startswith("30.0"), "a reading must be stamped"
            t0 = time.monotonic()
            gate_on_reading(45.0)                              # fresh: no second ask
            assert time.monotonic() - t0 < 0.5
        finally:
            OPERATOR_FILE, _ct.STAMP = real_op, real_stamp
            _gate_opened_at[0] = float("-inf")

    print("tilt_run: self-check passed (I^2 integral, scaling, fallback, cooling, plan, "
          "repeats, waits, gate)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="every frequency, flashing between")
    ap.add_argument("--freqs", default=None, help="comma list, with --sweep")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-cool", action="store_true",
                    help="skip the thermal wait between repeats (operator override)")
    ap.add_argument("--force-hot", action="store_true",
                    help="run a chunk predicted to exceed the remaining headroom")
    ap.add_argument("--repeats-high", type=int, default=HIGH_REPEATS,
                    help=f"repeats above {HIGH_REPEAT_F_HZ:g} Hz (default {HIGH_REPEATS})")
    ap.add_argument("--drone", default=None,
                    help="which robot, e.g. B. Goes in the take folder name and meta.json")
    ap.add_argument("--flights-root", default=None,
                    help="where takes are filmed, e.g. /Volumes/<DRIVE>/flights")
    ap.add_argument("--seg2-rate", type=float, default=None,
                    help="force one segment-2 ramp rate, Hz/s (rate-swap control, 25.14)")
    ap.add_argument("--gate-above-hz", type=float, default=None,
                    help="at or above this drive, no take starts without a coil reading")
    ap.add_argument("--gate-below-c", type=float, default=45.0,
                    help="the reading that opens the gate must be below this (default 45)")
    a = ap.parse_args()
    tilt_schedule.SEG2_RATE_OVERRIDE = a.seg2_rate
    gate = (a.gate_above_hz, a.gate_below_c) if a.gate_above_hz is not None else None
    # Module-level rebind: `repeats_for` reads the global, and threading one more argument
    # through `sweep` and `run_chunk` to reach it would be plumbing for a knob nobody turns.
    HIGH_REPEATS = a.repeats_high
    if a.sweep:
        fr = [float(x) for x in a.freqs.split(",")] if a.freqs else None
        sweep(fr, a.repeats, port=a.port, out_dir=a.out, cool=not a.no_cool,
              force_hot=a.force_hot, dry_run=a.dry_run, drone=a.drone,
              out_root=a.flights_root, gate=gate)
        sys.exit()
    if a.freq is None:
        _self_check()
        sys.exit()
    run_chunk(a.freq, a.repeats, port=a.port, out_dir=a.out, dry_run=a.dry_run,
              cool=not a.no_cool, force_hot=a.force_hot, drone=a.drone,
              out_root=a.flights_root, gate=gate)
