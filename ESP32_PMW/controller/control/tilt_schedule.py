#!/usr/bin/env python3
"""The alignment-rate schedule: one frequency point, generated, not hand-written.

    uv run python controller/control/tilt_schedule.py                 # self-check
    uv run python controller/control/tilt_schedule.py --freq 10       # print it
    uv run python controller/control/tilt_schedule.py --freq 10 --write

Writes `spiffs_data/tilt.json`, which `src/main_tilt.cpp` reads off SPIFFS on boot. That
firmware parses no serial, so this file IS the experiment: nothing the host sends can change
it once flashed.

ONE POINT PER SCHEDULE
----------------------
The sweep repeats each frequency five times with the coils cooling in between, and the
cooldown belongs to the host (`tilt_run.py`), not to an `addWaitTask`. A schedule holding all
five repeats would film ~25 minutes of an idle rig per chunk and give the host no place to
put a thermal gate between them. One point per flash costs no extra `uploadfs` -- there is one
per frequency either way -- and yields one short take per repeat.

THE RAMP DURATION RULE
----------------------
`duration_ms = RAMP_BASE_MS + RAMP_PER_HZ_MS * f`, EASE k=2 from 2.0 Hz. This is the policy
from the refined schedule (merge-base `01f3354`, still on `micro-robot-controller`), which
fitted it to hold the sub-pull-in DWELL roughly constant.

Following the ramp is not the constraint: `control/theory.md` 18.3 measures 40-60x margin at
every point on the curve. Capture at t=0 is, and capture needs time below the ~4.9 Hz pull-in
crossing. The dwell *fraction* of an EASE ramp falls as the target rises -- 43% of the ramp at
a 10 Hz target against 11% at 210 Hz -- so a fixed duration would starve the high points and
waste time on the low ones. `dwell_s()` below computes it in closed form, and the self-check
holds the whole sweep to a floor.

FOUR RULES, ALL PAID FOR ALREADY
--------------------------------
1. `activateChannels mask 15 value 100` in EVERY block. Mask 15 includes channel 0, so it
   silently resets a previous drop back to 100%. On 2026-09-01 only the first point of an
   8-point sweep actually ran tilted because of this.
2. The drop goes AFTER the ramp. Ramping up already asymmetric is how a capture is lost.
3. The down-ramp runs BEFORE the carrier is cut. 0 Hz with the carrier still on is not "off",
   it is DC into a stationary field -- the hottest thing this rig can do, and it reads as idle
   in the telemetry.
4. `label TILT_OFF` last. `tilt_sweep.run()` ends on that label; without it (the state of
   `tilt.json` on this branch) every run instead sits out the 560 s timeout.

The queue does NOT de-energise when it runs out -- `PwmSequencer::run()` returns on an
exhausted queue and the last commanded state persists indefinitely. The closing
`activateChannels ... 0.0` is what makes the terminal state safe. It is not a substitute for
`park()`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPIFFS = ROOT / "spiffs_data" / "tilt.json"

#: The frequencies this campaign sweeps, lowest first -- which is also cheapest first, since
#: coil current and therefore heating rise steeply with frequency (`control/theory.md`).
FREQS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130)

#: Ramp start. 2.0 Hz, not 1.0: under the ~4.9 Hz pull-in crossing with margin, and low
#: enough that the rotor is stationary when the field starts. `theory.md` 18.3.
RAMP_FROM_HZ = 2.0
#: TWO segments, because capture and acceleration want opposite things.
#:
#: The refined schedule used one segment at `24000 + 100 * f` ms. That rule was fitted over
#: 20-160 Hz targets, where the span is 18-158 Hz, and it is almost all base term. Extrapolated
#: to a 10 Hz target it gives 25 s to cover 8 Hz -- 0.32 Hz/s average, 0.64 Hz/s peak. Run on
#: the rig 2026-09-10: the rotor tracked it but the whole ramp crawled.
#:
#: Segment 1 exists to be slow: capture needs TIME below the ~4.9 Hz pull-in crossing, and at
#: a 10 Hz target a single segment spends its whole length in that band by construction.
#: Segment 2 then accelerates at a fixed rate. `theory.md` 18.3 measures ḟ_max at 77-127 Hz/s
#: through the 6-10 Hz band, so 3.5 Hz/s carries >10x margin -- the slope was never the limit.
CAPTURE_TO_HZ = 5.0
CAPTURE_MS = 5000
SEG2_RATE_HZ_S = 3.5
#: Floor on segment 2, so a target barely above CAPTURE_TO_HZ still gets a real segment.
SEG2_MIN_MS = 1000
#: Pull-in crossing. The ramp must spend real time below this or the rotor is never caught.
F_PULL_IN_HZ = 4.9
#: Floor on that dwell. 3 s is what `ramp.DEFAULT` achieves at 210 Hz; the refined schedule
#: runs ~1.6x more conservative and this checks we stay above the floor, not at it.
MIN_DWELL_S = 3.0

#: Settle at the commanded frequency before anything is measured. An EASE ramp ends with
#: sigma'(1) = 0 so the commanded frequency is already flat, but the ROTOR is not -- it is
#: still catching up to the field. Nothing reads this window; it exists to be thrown away.
#: Raised 1 s -> 3 s on 2026-09-10: the operator wanted longer at frequency before the drop,
#: to let the robot stabilise. Total time at target before the kill is SETTLE + HOLD.
SETTLE_MS = 3000
#: Above this frequency the hold is shortened and the ramp slowed, to buy heat back.
#:
#: The hold sits at the target frequency, where the current is highest: at 100 Hz a second of
#: hold costs 0.49 C against roughly a third of that for a ramp second, which is spent
#: climbing through lower frequencies and lower current. Trading hold for ramp is therefore
#: net cooler for the same experiment -- ~1.9 C a repeat at 100 Hz, ~19 C over a ten-repeat
#: chunk -- and a slower ramp may also help the capture reliability that falls off above
#: 50 Hz (24.5).
#:
#: The hold still has to outlast what the analysis reads from it: `alignment_rate.PRE_S` is
#: 1 s, so HIGH_HOLD_MS cannot go below about 2 s without starving the baseline.
HIGH_F_HZ = 60.0
HIGH_SETTLE_MS = 2000
HIGH_HOLD_MS = 3000
HIGH_SEG2_RATE_HZ_S = 2.8
#: Level hold after the settle. Briefly cut to 500 then 2000 ms on 2026-09-10 for
#: thermal reasons, then RESTORED to the original 7000 (3000 above HIGH_F_HZ) at the
#: operator's instruction: hold sets how settled the robot is when the coils are cut,
#: so a take recorded at a different hold is a different condition, and the campaign's
#: 100+ existing takes all used the original. Consistency beat the ~6 s of coil heat.
#: Original note follows -- the arithmetic in it still holds for the shortened case:
#: instruction: at 120 Hz this is the hottest second of the schedule, and 6.5 s of it per
#: repeat bought nothing. The 1 s baseline `alignment_rate` needs (`theta_from` is the
#: median of the 1 s before the kill) is NOT lost -- SETTLE_MS/HIGH_SETTLE_MS is also at
#: the target frequency, so the pre-kill dwell is still 3.5 s (2.5 s above HIGH_F_HZ) and
#: the median keeps margin either side. Do not cut the settle to match without redoing
#: that arithmetic.
HOLD_MS = 7000
#: How long the robot is left leaning. The measured transient is ~3 s (2026-09-08 take).
#:
#: Raised 5000 -> 15000 on 2026-09-10 at the operator's instruction, and it fixes a
#: measurement bug at the same time. `alignment_rate` reads the settled axis over
#: POST_FROM_S..POST_TO_S (4-6 s after the kill) but CLAMPS that window to the DOWN_ label
#: -- `t_stop = min(tk + POST_TO_S, p["t_end"])` at alignment_rate.py:2677. At 5000 the label
#: landed at kill+5 s, so every settled measurement the campaign has ever taken was truncated
#: to 4-5 s, and `theory.md` 25.5 says so in as many words. 15 s un-clamps it.
#:
#: CONSEQUENCE FOR COMPARISONS: takes recorded before 2026-09-10 measured the truncated
#: window and takes after it measure the full one. A design-A/design-B comparison of any
#: `settle_*` or `cone_final` column must say which it is quoting.
DROP_MS = 15000
#: Spin-down. Not a capture -- nothing has to be caught on the way back to zero.
#:
#: Cut 4000 -> 100 on 2026-09-10 at the operator's instruction: a near-step stop rather than
#: a ramp. Rule 3 still holds in letter -- this still precedes the carrier cut, which is what
#: keeps 0 Hz-with-carrier-on from happening -- and the rest window opens 4.9 s after the cut
#: instead of 1.0 s, so `datum` gets MORE settling time, not less.
#:
#: The risk it takes: stopping a 120 Hz rotor in 100 ms can set the robot swinging on its
#: wire, and `datum` refuses a rest window over DATUM_MAX_SPREAD_DEG (5 deg). Watch the
#: skipped-take count on the first chunk; if it climbs, this number is why.
DOWN_MS = 100
#: Coils-off tail. This is the rest window `alignment_rate.datum` takes its zero from, so it
#: has to outlast REST_TO_S there.
OFF_MS = 15000
#: A short step for the closing TILT_OFF label to attach to. A `label` entry "updates the
#: current label without pushing a queue entry of its own" (JsonPwmSequencer.cpp:216), so a
#: label placed LAST tags nothing, never becomes `stepLabel()`, and the firmware prints a bare
#: `label=`. Measured on the rig 2026-09-10: two 10 Hz takes ran the whole schedule correctly
#: and still timed out, because `tilt_sweep` waits for exactly `label=TILT_OFF`.
END_MS = 500

#: Channels cut, and to what. [0, 2] = coils A and C. As an ARRAY so both change in the same
#: queue step; two single-channel calls land a compile tick apart.
DROP_CHANNELS = [0, 2]
DROP_PCT = 0.0

RESOLUTION_MS = 25
DIRECTION = "CW"          #: the labels are inverted on this rig; CW here spins CCW
INITIAL_DUTY = [50, 50, 50, 50]


def hold_ms(f_hz):
    """``(settle_ms, hold_ms)`` for a target frequency. See `HIGH_F_HZ`."""

    if float(f_hz) >= HIGH_F_HZ:
        return HIGH_SETTLE_MS, HIGH_HOLD_MS
    return SETTLE_MS, HOLD_MS


#: Force ONE segment-2 rate at every target, Hz/s. None = the two-tier rule above.
#:
#: Exists for the rate-swap control (`control/theory.md` 25.14). The two-tier rule changes
#: rate at exactly 60 Hz, which is exactly where the pre-cut cone collapses 8.4 -> 1.9 deg on
#: the design-A campaign -- so rate and frequency are confounded there. 50 Hz run at the
#: high-f rate and 60 Hz at the low-f rate separate them. Set from `tilt_run --seg2-rate`.
SEG2_RATE_OVERRIDE = None


def seg2_rate(f_hz):
    """Segment-2 climb rate for a target frequency, Hz/s. See `HIGH_F_HZ`."""

    if SEG2_RATE_OVERRIDE is not None:
        return float(SEG2_RATE_OVERRIDE)
    return HIGH_SEG2_RATE_HZ_S if float(f_hz) >= HIGH_F_HZ else SEG2_RATE_HZ_S


def ramp_tasks(f_hz):
    """The up-ramp, as one or two EASE tasks. Segment 2 is dropped if the target is low.

    The segments abut exactly -- segment 1 ends where segment 2 begins -- because a gap
    between them is a commanded frequency STEP, which is the one thing guaranteed to break
    sync (`ramp.check` refuses the same thing on the `seq=` path).
    """

    f = float(f_hz)
    if f <= CAPTURE_TO_HZ:
        return [{"method": "addEaseRampTask", "from": RAMP_FROM_HZ, "to": f,
                 "duration_ms": CAPTURE_MS}]
    ms2 = max(int(round(1000.0 * (f - CAPTURE_TO_HZ) / seg2_rate(f))), SEG2_MIN_MS)
    return [{"method": "addEaseRampTask", "from": RAMP_FROM_HZ, "to": CAPTURE_TO_HZ,
             "duration_ms": CAPTURE_MS},
            {"method": "addEaseRampTask", "from": CAPTURE_TO_HZ, "to": f,
             "duration_ms": ms2}]


def ramp_ms(f_hz):
    """Total up-ramp duration for a target frequency, in ms."""

    return sum(t["duration_ms"] for t in ramp_tasks(f_hz))


def dwell_s(f_hz, ramp_s=None, f0=RAMP_FROM_HZ, k=2.0):
    """Seconds the ramp spends below the pull-in crossing -- segment 1's whole job.

    With the two-segment ramp this is computed over segment 1 alone (2 -> CAPTURE_TO_HZ),
    since segment 2 starts at or above the crossing and contributes nothing.

    EASE is ``sigma(u) = u^k / (u^k + (1-u)^k)``, so the fraction of the ramp elapsed when
    the frequency reaches ``F_PULL_IN_HZ`` inverts in closed form: with
    ``s = (f_pull - f0) / (f - f0)`` and ``x = (s / (1 - s)) ** (1/k)``, that fraction is
    ``x / (1 + x)``.
    """

    f_top = min(float(f_hz), CAPTURE_TO_HZ)
    total = ramp_s if ramp_s is not None else (
        CAPTURE_MS / 1000.0 if float(f_hz) > CAPTURE_TO_HZ else ramp_ms(f_hz) / 1000.0)
    span = f_top - f0
    if span <= 0:
        return float("inf")
    s = (F_PULL_IN_HZ - f0) / span
    if s <= 0:
        return 0.0
    if s >= 1:
        return float(total)
    x = (s / (1.0 - s)) ** (1.0 / k)
    return total * x / (1.0 + x)


def drive_s(f_hz):
    """Energised seconds for one point. The coils-off tail is not drive."""

    st, hd = hold_ms(f_hz)
    return (ramp_ms(f_hz) + st + hd + DROP_MS + DOWN_MS) / 1000.0


def schedule(f_hz):
    """The task list for one point. Labels are what the host and the analysis key off."""

    f = float(f_hz)
    tag = f"{int(round(f)):03d}HZ"
    return [
        {"method": "label", "value": f"FREQ_{tag}"},
        # Rule 1: this line resets every channel in the mask, so it must open every block.
        {"method": "activateChannels", "mask": 15, "value": 100.0},
        *ramp_tasks(f),
        {"method": "label", "value": f"SETTLE_{tag}"},
        {"method": "addWaitTask", "duration_ms": hold_ms(f)[0]},
        {"method": "label", "value": f"HOLD_{tag}"},
        {"method": "addWaitTask", "duration_ms": hold_ms(f)[1]},
        {"method": "label", "value": f"KILL_{tag}"},
        # Rule 2: after the ramp, never before.
        {"method": "addCarrierDutyCycleTask", "channels": list(DROP_CHANNELS),
         "value": float(DROP_PCT)},
        {"method": "addWaitTask", "duration_ms": DROP_MS},
        {"method": "label", "value": f"DOWN_{tag}"},
        # Rule 3: spin down first, cut the carrier second.
        {"method": "addEaseRampTask", "from": f, "to": 0.0, "duration_ms": DOWN_MS},
        {"method": "activateChannels", "mask": 15, "value": 0.0},
        # The coils-off rest window. `alignment_rate.datum` takes the zero from here, so
        # TILT_OFF must come AFTER it -- the host stops filming on that label.
        {"method": "addWaitTask", "duration_ms": OFF_MS},
        # Rule 4: without this the host waits out its whole timeout instead of ending here.
        # The label must PRECEDE a real step or it tags nothing; see END_MS.
        {"method": "label", "value": "TILT_OFF"},
        {"method": "addWaitTask", "duration_ms": END_MS},
    ]


HEADER = """\
{{
  // GENERATED by controller/control/tilt_schedule.py -- do not hand-edit, regenerate.
  //
  // ONE point of the alignment-rate sweep: ramp to {f:g} Hz, settle {settle:g} s, hold level
  // {hold:g} s, cut coils A and C (channels {chans}) to {drop:g}%, hold {dropped:g} s, spin
  // down and turn off. The settle is discarded; the hold is the measured baseline.
  // `tilt_run.py` flashes this, runs it once per repeat, and owns the cooldown between.
  //
  // Ramp {ramp:g} s from {f0:g} Hz, EASE k=2, segment 2 at {rate:g} Hz/s. Leaves {dwell:.1f} s below the ~{fpull:g} Hz
  // pull-in crossing -- capture at t=0 is the tight constraint, not following the curve
  // (`control/theory.md` 18.3 measures 40-60x margin on the slope everywhere).
  //
  // Energised: {drive:.0f} s. Heating goes as I^2 and coil current rises steeply with
  // frequency, so this point is far cheaper at 10 Hz than at 130. `tilt_run.py` integrates
  // the telemetry current rather than charging every second alike.
  //
  // `main_tilt` runs PASSTHROUGH (no current balancer). Required, not incidental: closing
  // the balance loop makes a dropped channel a "tilt follower" and drags the other three
  // down toward its mis-read negative current. `control/theory.md` 23.1.
  //
  // THE SCHEDULE END DOES NOT DE-ENERGISE. `PwmSequencer::run()` returns on an exhausted
  // queue and the last state persists forever; the closing `activateChannels ... 0.0` is
  // what makes that terminal state safe. The kill is still GPIO14 or `tilt_sweep.park()`.
"""


def render(f_hz):
    """The full JSON text, comments included (ARDUINOJSON_ENABLE_COMMENTS=1)."""

    f = float(f_hz)
    head = HEADER.format(f=f, hold=hold_ms(f)[1] / 1000, settle=hold_ms(f)[0] / 1000,
                         chans=DROP_CHANNELS, drop=DROP_PCT,
                         dropped=DROP_MS / 1000, ramp=ramp_ms(f) / 1000, f0=RAMP_FROM_HZ,
                         dwell=dwell_s(f), fpull=F_PULL_IN_HZ, drive=drive_s(f), rate=seg2_rate(f))
    body = {"resolution_ms": RESOLUTION_MS, "initial_freq": 0.0,
            "initial_duty": INITIAL_DUTY, "direction": DIRECTION, "schedule": schedule(f)}
    lines = [f'  "{k}": {json.dumps(v)},' for k, v in list(body.items())[:-1]]
    lines.append('  "schedule": [')
    lines += [f"    {json.dumps(s)}," for s in body["schedule"]]
    lines[-1] = lines[-1].rstrip(",")
    lines.append("  ]")
    return head + "\n".join(lines) + "\n}\n"


def write(f_hz, path=SPIFFS):
    path = Path(path)
    path.write_text(render(f_hz))
    print(f"{path}: {f_hz:g} Hz, ramp {ramp_ms(f_hz) / 1000:g} s "
          f"(dwell {dwell_s(f_hz):.1f} s), {drive_s(f_hz):.0f} s energised")
    print("   flash with:  pio run -e tilt -t uploadfs && pio run -e tilt -t upload")
    print("   NOTE flashing starts the run.")
    return path


def _self_check():
    # segment 2 runs at the commanded rate, and the segments abut without a gap
    for f in (10, 40, 130):
        t = ramp_tasks(f)
        assert len(t) == 2, (f, t)
        assert t[0]["to"] == t[1]["from"] == CAPTURE_TO_HZ, t
        rate = (t[1]["to"] - t[1]["from"]) / (t[1]["duration_ms"] / 1000.0)
        assert (abs(rate - seg2_rate(f)) < 0.1
                or t[1]["duration_ms"] == SEG2_MIN_MS), (f, rate, seg2_rate(f))
    # a target at or below the capture frequency collapses to one segment
    assert len(ramp_tasks(4.0)) == 1
    # the failure this replaced: 10 Hz must no longer take 25 s to cover 8 Hz
    assert ramp_ms(10) < 9000, ramp_ms(10)
    # and the high end stays near the durations that were actually validated on the rig
    # Longer since the high-frequency ramp was slowed to buy heat back (HIGH_SEG2_RATE_HZ_S),
    # but it still has to fit inside MAX_RAMP_S.
    assert 30000 < ramp_ms(130) < 52000, ramp_ms(130)
    from controller.control import constants as _C
    assert ramp_ms(130) / 1000.0 <= _C.MAX_RAMP_S, ramp_ms(130)

    # every point in the sweep clears the capture floor, and none exceeds MAX_RAMP_S
    from controller.control import constants as C
    for f in FREQS:
        d = dwell_s(f)
        assert d >= MIN_DWELL_S, f"{f} Hz: only {d:.2f} s below pull-in"
        assert ramp_ms(f) / 1000.0 <= C.MAX_RAMP_S, f"{f} Hz ramp exceeds MAX_RAMP_S"
    # the dwell FRACTION falls with target -- the reason the rule is not a constant
    assert dwell_s(10) / (ramp_ms(10) / 1000) > 3 * dwell_s(130) / (ramp_ms(130) / 1000)

    s = schedule(20)
    names = [x["value"] for x in s if x["method"] == "label"]
    assert names == ["FREQ_020HZ", "SETTLE_020HZ", "HOLD_020HZ", "KILL_020HZ",
                     "DOWN_020HZ", "TILT_OFF"], names
    # The pre-kill dwell must outlast what the analysis takes its median from. The dwell is
    # SETTLE + HOLD, not HOLD alone -- both wait at the target frequency -- which is what
    # let HOLD_MS drop to 500 ms on 2026-09-10 without touching the baseline. Asserting on
    # HOLD_MS alone was the stricter test but the wrong one, and it would have blocked a
    # change that costs the measurement nothing.
    from controller.control import alignment_rate as _ar
    for _f in (20.0, 100.0):
        _st, _ho = hold_ms(_f)
        assert (_st + _ho) / 1000.0 > _ar.PRE_S + 1.0, (_f, _st, _ho, _ar.PRE_S)
    assert hold_ms(100)[1] == HIGH_HOLD_MS and hold_ms(40)[1] == HOLD_MS
    assert seg2_rate(100) < seg2_rate(40)
    # the rate-swap override wins at every target, and clearing it restores the rule
    global SEG2_RATE_OVERRIDE
    SEG2_RATE_OVERRIDE = 2.8
    assert seg2_rate(50) == seg2_rate(60) == 2.8 and "2.8 Hz/s" in render(50)
    SEG2_RATE_OVERRIDE = None
    assert seg2_rate(50) == SEG2_RATE_HZ_S

    # rule 1: the block opens by resetting every channel to 100
    assert s[1] == {"method": "activateChannels", "mask": 15, "value": 100.0}, s[1]
    # rule 2: the drop is after the ramp, not before
    i_ramp = next(i for i, x in enumerate(s) if x["method"] == "addEaseRampTask")
    i_drop = next(i for i, x in enumerate(s) if x["method"] == "addCarrierDutyCycleTask")
    assert i_drop > i_ramp, (i_ramp, i_drop)
    # ... and it takes both channels in ONE task, so they change in the same queue step
    assert s[i_drop]["channels"] == [0, 2], s[i_drop]
    # rule 3: down-ramp precedes the carrier cut
    i_down = next(i for i, x in enumerate(s)
                  if x["method"] == "addEaseRampTask" and x["to"] == 0.0)
    i_off = next(i for i, x in enumerate(s)
                 if x["method"] == "activateChannels" and x["value"] == 0.0)
    assert i_down < i_off, (i_down, i_off)
    # rule 4: every label must be followed by an entry that actually pushes a queue step,
    # or it never becomes stepLabel() and the firmware prints a bare `label=`.
    for i, x in enumerate(s):
        if x["method"] == "label":
            rest = [y["method"] for y in s[i + 1:] if y["method"] != "label"]
            assert rest, f"label {x['value']} tags nothing -- it will never be printed"
    assert s[-2]["value"] == "TILT_OFF" and s[-1]["method"] == "addWaitTask", s[-2:]

    # the rest window must outlast what the analysis reads from it. Measured from the DOWN_
    # label, so the down-ramp counts toward it -- which matters now that DOWN_MS is 100 ms
    # and OFF_MS alone is the looser of the two bounds.
    from controller.control import alignment_rate as ar
    assert (DOWN_MS + OFF_MS) / 1000.0 > ar.REST_TO_S, (DOWN_MS, OFF_MS, ar.REST_TO_S)
    # the settled window must no longer be CLAMPED by the DOWN_ label. This is the whole
    # reason DROP_MS went to 15 s: at 5000 `alignment_rate` silently truncated its own
    # 4-6 s settled window to 4-5 s (alignment_rate.py:2677, theory.md 25.5).
    assert DROP_MS / 1000.0 >= ar.POST_TO_S, (DROP_MS, ar.POST_TO_S)

    # it parses, comments and all, as the firmware's ArduinoJson would
    txt = render(20)
    stripped = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("//"))
    got = json.loads(stripped)
    assert got["schedule"] == s and got["direction"] == DIRECTION

    print("tilt_schedule: self-check passed "
          f"(duration rule, dwell floor on all {len(FREQS)} points, 4 rules, round-trip)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freq", type=float, default=None)
    ap.add_argument("--write", action="store_true", help=f"write {SPIFFS}")
    ap.add_argument("--table", action="store_true", help="the whole sweep, as a table")
    a = ap.parse_args()
    if a.table:
        print(f"{'f':>5} {'ramp s':>7} {'dwell s':>8} {'drive s':>8}")
        for f in FREQS:
            print(f"{f:5d} {ramp_ms(f) / 1000:7.0f} {dwell_s(f):8.1f} {drive_s(f):8.0f}")
    elif a.freq is None:
        _self_check()
    elif a.write:
        write(a.freq)
    else:
        print(render(a.freq))
