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

**It is still a model, not a thermometer.** There is no temperature sensor on this rig. Coil C
reads negative -- a known hardware fault -- so the current here is a sum of magnitudes. Watch
the coils.

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
#: 120 Hz repeat books ~13 C, so this gives it ~130 s while 10 Hz (0.06 C) still falls to
#: the floor below. Added 2026-09-10 at the operator's instruction after 110/120 Hz chunks
#: ran back to back. It is a PAUSE, not a thermal guarantee -- with `--no-cool` nothing
#: here reads a temperature and the operator is still the thermometer.
NO_COOL_S_PER_C = 10.0

#: Floor on the between-repeat wait. Even a thermally free point wants the rotor stopped and
#: the rig still before the next ramp starts.
MIN_COOL_S = 20.0


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
        rows.append((float(m.group(1)), sum(abs(a) for a in tel.amps)))
    if len(rows) < 2:
        return coil_thermal.HEAT_C_PER_S * fallback_s, 0

    # Trapezoid on I^2, so a run that ramps is charged for the ramp and not for its peak.
    total = 0.0
    for (t0, i0), (t1, i1) in zip(rows, rows[1:]):
        dt = t1 - t0
        if 0.0 < dt < 5.0:                     # a longer gap is a stall, not drive
            total += 0.5 * (i0 ** 2 + i1 ** 2) * dt
    return coil_thermal.HEAT_C_PER_S * total / I2_ANCHOR_A ** 2, len(rows)


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
              force_hot=False, **kw):
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
        flight, log, outcome = tilt_sweep.run(
            port=port, drive_s=p["drive_s"], ignore_thermal=True,
            timeout_s=p["drive_s"] + tilt_schedule.OFF_MS / 1000.0 + 60.0, **kw)
        c, n = heat_c(log, p["drive_s"])
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

        if rep < last:
            wait = (cool_for(coil_thermal.temp_now(), p["heat_per_repeat_c"]) if cool
                    else max(MIN_COOL_S, NO_COOL_S_PER_C * p["heat_per_repeat_c"]))
            print(f"  {'cooling' if cool else 'pausing (heat gate OFF)'} "
                  f"{wait:.0f} s before repeat {rep + 1}")
            try:
                time.sleep(wait)
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


def sweep(freqs=None, repeats=5, port=None, out_dir=None, cool=True, force_hot=False, **kw):
    """The whole campaign: every frequency, `repeats` each, flashing between.

    One schedule per frequency has to be uploaded because `main_tilt` parses no serial --
    the JSON on SPIFFS *is* the experiment, and the only way to change the target frequency
    is to write a new one and reboot.
    """

    freqs = list(tilt_schedule.FREQS if freqs is None else freqs)
    root = Path(out_dir or (OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")))
    root.mkdir(parents=True, exist_ok=True)
    print(f"sweep: {freqs} x {repeats} -> {root}")
    allrows = []
    for f in freqs:
        try:
            # --dry-run must not touch the hardware. `flash` uploads AND resets the board,
            # and `main_tilt` starts its schedule on boot, so a "dry" run was energising the
            # coils -- it only failed here because another chunk held the serial port.
            if not kw.get("dry_run"):
                flash(f)
            rows = run_chunk(f, repeats, port=port, out_dir=root / f"{int(f):03d}hz",
                             cool=cool, force_hot=force_hot, **kw)
        except SystemExit as e:
            print(f"  {f:g} Hz SKIPPED: {e}")
            continue
        except KeyboardInterrupt:
            print(f"\nsweep interrupted during {f:g} Hz")
            break
        allrows += rows
        if allrows:                       # a dry run produces none
            _write_index(root / "sweep_index.csv", allrows)
    ok = sum(r["outcome"] == "ok" for r in allrows)
    print(f"\nsweep done: {ok}/{len(allrows)} good takes, "
          f"{sum(r['heat_c'] for r in allrows):.1f} C measured -> {root}")
    return allrows


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

    print("tilt_run: self-check passed (I^2 integral, scaling, fallback, cooling, plan)")


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
    a = ap.parse_args()
    if a.sweep:
        fr = [float(x) for x in a.freqs.split(",")] if a.freqs else None
        sweep(fr, a.repeats, port=a.port, out_dir=a.out, cool=not a.no_cool,
              force_hot=a.force_hot, dry_run=a.dry_run)
        sys.exit()
    if a.freq is None:
        _self_check()
        sys.exit()
    run_chunk(a.freq, a.repeats, port=a.port, out_dir=a.out, dry_run=a.dry_run,
              cool=not a.no_cool, force_hot=a.force_hot)
