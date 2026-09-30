#!/usr/bin/env python3
"""
Real-time hover controller: closes the position loop around src/main_flight.cpp.

One tick loop, one state machine. The firmware ramps itself to FLIGHT on the `seq=` profile
the host sends (`ramp.py`); until it gets there the loop drains telemetry and feeds the viewer
but commands nothing. In FLIGHT z is z_track.ZTracker and lateral is polar az/mag.
Details: theory.md 6.5.

Every tick is logged to `<csv_dir>/<stamp>.csv` from the first tick, ramp included -- the ramp
is the measurement, see takeoff_report.py.

Where:
    ticks     iterable of live_viz.Tick: xyz_mm is the datum frame in mm, +z up, None if lost
    ctrl      (x, y) DiscreteHoverController pair, one per lateral axis, K row 0 each
    loop      metres and seconds internally; viz.setpoint is mm; serial is deg / 0..1 / Hz
    armed     nothing is commanded until viz.armed; viz.mag_max starts 0, so lateral is OFF
    latches   viz.estop/takeoff/land are one-shot, cleared on read. Read ONCE per tick into a
              local, at the top of the loop, and dispatch on the locals -- a second read is a
              press silently thrown away. That bug cost a whole session: the takeoff latch was
              consumed behind an `and link.state != FLIGHT` and the coils never enabled.
    safety    NO automatic backstops. The viser stop button is the only software kill; the
              only other de-energising paths are SIGINT, an exception, and `finally`.
              Removed 2026-08-29 on the operator's instruction: they report the firmware's
              500 ms silence watchdog did not work, and asked for every cap and backstop
              stripped. The host-side lands (no-fix, step-out, never-reached-FLIGHT) went
              with it -- not because they were shown harmful, but because that was the
              instruction. The case for reinstating them is in theory.md 4.0 and still
              stands.  Usage: fly(dry_run=True) for the stub rehearsal.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from controller.control import attitude
from controller.control import predictor
from controller.control import ramp
from controller.control import z_track
from controller.control import constants as C
from controller.control.link import SerialComm, parse_telemetry
from controller.control.reference_profiles import Profile
from controller.control.simulate_hover import DiscreteHoverController, Scenario, simulate


# Consecutive good fixes required before an automatic takeoff commands the ramp. At the
# measured ~24 fps of the stereo pipeline this is well under a second, and the fix rate
# once the datum exists is 100%, so it costs nothing on a healthy rig and blocks a blind
# ramp on a sick one.
PRIME_FIXES = 15

# Capture trigger. Capture happens at 2-4 Hz, which is the ONE band where the blade
# witness gives a confident answer: `SpinWitness.turning` returns False only while the
# commanded field is under its alias limit (fps/8), because above that a strobed rotor
# can imitate a standstill. So the ramp can check its own premise for free, and abort
# instead of driving a stationary rotor for the next 15-45 s.
# Measured 2026-08-29: a linear ramp crosses the ~4.2 Hz pull-in window in 0.2 s and
# fails to spin at all, where EASE k=2 spends 1.5 s there. This is what catches that.
# `turning` is NOT the test. It is a motion latch, and a rotor rocking a few degrees in
# the field is moving -- it returned "turning" 240 times for a rotor that never span.
# Rotation is distinguished from rocking by ACCUMULATED phase: a turning rotor racks up
# revolutions monotonically, a rocking one oscillates about zero and nets out. So compare
# `drift_rev` against the revolutions the field actually commanded over the same window.
CAPTURE_MIN_REV = 1.0        # commanded revolutions needed before the call means anything
CAPTURE_REV_FRAC = 0.30      # measured/commanded below this = never caught
CAPTURE_MIN_FRAMES = 20      # blade-signal frames needed before judging at all

# Coil heat lives in `coil_thermal`, shared with `link.SerialComm` so that ANY script
# driving the coils is accounted for -- not just flights through fly(). The one-minute
# cap on a single ramp is enforced by `ramp.check`, next to the profile it applies to.
#
# Drive that continues past the ramp: the FLIGHT hold plus landing, before the operator
# stops. A GUESS -- nothing measures it. `SerialComm.energised_s` prints the true total at
# close, so if that consistently exceeds ramp+SETTLE_S, raise this.
SETTLE_S = 12.0


def _anchor(ctrl, ztrk, f_reached):
    """Re-anchor the controller onto the frequency the ramp actually reached.

    The gains carry `f_hover` from their design point and the law commands that plus a
    correction, so without this the FIRST closed-loop command is the design frequency --
    not where the rotor is. Ramp to 170 and arm, and the drive is walked down toward 160
    at the full slew rate the instant `armed` is ticked. The failure was observed before
    in the other direction and is recorded in the git history: "steady at 60, then
    straight to 200".

    The TRIM ONLY moves. freq_min/freq_max stay at the gain file's own limits, wide.
    """

    if not f_reached or f_reached <= 0:
        return
    for c in ctrl:
        c.f_hover = f_reached
        c.prev_f_field = f_reached      # slew reference, or the first step is a jump
    if ztrk is not None:
        ztrk.f_hat = min(max(f_reached, ztrk.f_lo), ztrk.f_hi)
    print(f"controller trim anchored at the measured {f_reached:.1f} Hz "
          f"(band left at [{ctrl[0].freq_min:.0f}, {ctrl[0].freq_max:.0f}] Hz)")


def _thermal_gate(cfg) -> float:
    """Validate the ramp and block until the coils can take the run.

    Returns the drive length budgeted for, in seconds. Dry runs neither wait nor stamp.
    An over-cap profile is REFUSED by `ramp.check`, never clamped -- the segments reach
    the firmware verbatim, so a value trimmed here would cap a number nobody reads.
    """

    ramp.check(cfg.segments)
    ramp_s = ramp.duration_s(cfg.segments)
    if cfg.dry_run:
        return ramp_s

    try:
        from ai.thermal import coil_thermal
    except ImportError as exc:
        raise SystemExit(
            "refusing to arm: ai/thermal/coil_thermal.py is missing, so nothing is "
            "tracking coil heat. The coils reach 80 C after four ramps and there is no "
            "temperature sensor. Restore it before flying."
        ) from exc
    # +SETTLE_S: coils stay energised past the ramp through FLIGHT and landing. A GUESS
    # at that tail, not a measurement -- `SerialComm.energised_s` prints the real figure
    # at close, so compare the two.
    t = coil_thermal.wait_until_safe(ramp_s + SETTLE_S)
    print(f"coils ~{t:.0f}C at start (ceiling {coil_thermal.T_CEILING_C:.0f}C)")
    return ramp_s


# Send only what the coils would notice. Below these a resend is latency spent to say
# nothing, and ungated at 200 Hz three commands a step is 4.4-5.8 kB/s. With the deadbands
# in, the wire carries a few hundred B/s. Bandwidth stopped being the argument when the
# link went to 921600 (`link.SerialComm.BAUD`); these now exist for the LATENCY of a line
# -- 0.31 ms, against 2.5 ms at the old 115200 -- and because there is no ack, so every
# byte not sent is a byte that cannot arrive corrupted. Sized to what the actuator
# resolves: 1 deg of field angle,
# 0.005 of the 0..1 magnitude, and 0.05 Hz against a 200 Hz/s slew ceiling (1.0 Hz per
# 200 Hz step), so a real transient still sends every step and a hover does not.
AZ_DEADBAND_DEG = 1.0
MAG_DEADBAND = 0.005
FREQ_DEADBAND_HZ = 0.05
# ...but never silence for longer than this. There is no firmware watchdog and no ack, so
# a corrupted line would otherwise stand until the next value crossed a deadband -- which
# in a steady hover is never. This is what bounds that, not a heartbeat for its own sake.
RESEND_S = 0.5

# The two "nothing was sent" log lines are the disarmed and `enable_freq_cmd=False` paths,
# i.e. the DEFAULT ones, and they were written every tick into a line-buffered file: 200
# flushes a second at 200 Hz and 500 inside a 2 ms budget at 500. They exist so a quiet
# log is distinguishable from a hung loop, which one line every half second says just as
# well. Deliberately the same period as RESEND_S: both answer "is this thing still alive".
WITHHELD_LOG_S = 0.5

SPINUP, FLIGHT = 1, 2  # main_flight.cpp State: IDLE=0 SPINUP=1 FLIGHT=2 LANDING=3 OFF=4

# Seconds the coils may stay driven with no pose fix before the host stops them.
#
# THE ONE AUTOMATIC LAND, put back deliberately on 2026-09-01 at the operator's request
# after a run drove the coils for 10 084 ticks (~50 s) with zero fixes: the robot left the
# tracked volume sideways at ~104 Hz and nothing noticed, because every other backstop had
# been stripped on 2026-08-29. The argument for it is theory.md 4.0.
#
# 1.0 s, not tighter: the pose pipeline delivers ~65 Hz and `predictor` bridges ordinary
# dropouts, and the healthy part of that same run held a 100 % fix rate below 100 Hz. So a
# whole second without a fix is not a hiccup, it is the robot gone.
LOST_LAND_S = 1.0

CSV_HEADER = C.CSV_COLUMNS

def _spin_state(spin):
    """`turning` / `stopped` / blank, straight off `pose/spin.SpinWitness.turning`.

    Not a rate. The witness is a motion latch and exposes none: above fps/8 the
    per-frame phase step aliases and the true rate is unrecoverable, so it answers
    True/False/None instead of putting a confident wrong number beside `f_hz`.

    The asymmetry is what makes the column worth a slot. `True` is valid at ANY speed
    -- motion is provable whatever the strobe does -- so `turning` at 50 Hz proves the
    rotor is moving at 50 Hz. `False` is gated on the field being below the alias
    limit, so it is decisive only down in the pull-in region, which is where the tight
    constraint is anyway: pull_in_hz(tau_max(3 Hz)) = 4.94 Hz against a 3 Hz starting
    slip, a 1.65x margin. Blank means the witness declined and proves nothing -- never
    write a `stopped` where it returned None.
    """

    turning = getattr(spin, "turning", None) if spin is not None else None
    return "" if turning is None else ("turning" if turning else "stopped")


def _live_viz():
    """Lazy import, so a missing viser breaks only the camera runs that need the panel."""

    from controller.viz import live_viz

    return live_viz


def _pace(t0: float, when: float) -> None:
    """Sleep until `when` seconds after `t0`, for the real-time playback sources."""

    while (wait := t0 + when - time.monotonic()) > 0.0:
        time.sleep(min(wait, 0.005))


def stub_ticks(gains, viz, duration: float = 10.0):
    """Replay a simulate_hover run in real time as Ticks. Open loop: commands do not feed back."""

    Tick = _live_viz().Tick
    out = simulate(Scenario("stub", x0_mm=10, z0_mm=10, duration=duration), gains)
    t0 = time.monotonic()
    for i, t in enumerate(out["t"]):
        _pace(t0, float(t))
        xyz = np.array([out["x"][i] * 1e3, 0.0, out["z"][i] * 1e3])
        yield Tick(t=time.monotonic(), xyz_mm=xyz, pose=None, frames=None, lost=0, viz=viz)


class _PoseFeed:
    """The pose source, newest tick only, optionally on its own thread.

    Drop-oldest single slot, the same shape as `sources.MonoCamera`: for feedback a
    stale pose is worse than no pose, so a queue that buffers backlog would be actively
    harmful. The COUNTER, not the slot, is what says "new" -- the control loop still
    wants the last tick's viz and frames on the steps where nothing arrived.

    Threaded for the live pipeline, where `stereo_frames` is a generator and segmenting
    a pair costs ~15 ms, so running it inline caps the control loop at the pose rate.
    Synchronous for the stub, where one tick per step is the contract `test_panel`
    asserts on.
    """

    def __init__(self, ticks, threaded=True):
        self._it = iter(ticks)
        self.tick, self.n, self.exc, self.done = None, 0, None, False
        self._threaded = threaded
        if not threaded:
            return
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="pose", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            for tick in self._it:
                with self._lock:      # nothing but the assignment under it
                    self.tick, self.n = tick, self.n + 1
        except BaseException as e:    # noqa: BLE001 -- the control loop must see it
            self.exc = e
        finally:
            self.done = True
            # Here, not in `close()`: the generator is suspended at this point, so this
            # is legal. Calling it from the consumer while the producer is inside it
            # raises "generator already executing". This is what runs `stereo_frames`'
            # own finally -- cameras released, viewer closed.
            close = getattr(self._it, "close", None)
            if close:
                close()

    def read(self):
        """``(tick, counter)``. Compare the counter with the last one to test freshness."""

        if not self._threaded:
            try:
                self.tick, self.n = next(self._it), self.n + 1
            except StopIteration:
                self.done = True
            except BaseException as e:  # noqa: BLE001
                self.exc, self.done = e, True
            return self.tick, self.n
        with self._lock:
            return self.tick, self.n

    def close(self):
        self.done = True
        if not self._threaded:
            close = getattr(self._it, "close", None)
            if close:
                close()
            return
        # MonoCamera.read blocks up to its 2 s timeout, so give it room. The thread is a
        # daemon: a camera wedged past this cannot hold the process open.
        self._thread.join(timeout=3.0)


class CommandLink:
    """Serial link to main_flight.cpp, or a printing stand-in for dry_run."""

    def __init__(self, port: str | None, dry_run: bool, log_path: str,
                 takeoff_cmd: list | None = None):
        # The sequencer profile: seq=clear / seq=ramp:... / seq=go. `PwmSequencer` takes
        # arbitrary ramp tasks, so a profile is data the host sends, not a shape compiled
        # into firmware behind a reflash -- and since 2026-08-31 it is the ONLY way to
        # spin the coils up, the firmware's own `takeoff` command having been deleted.
        # None means the default profile, so a link built directly is still armable.
        self.takeoff_cmd = (list(ramp.seq_lines(ramp.DEFAULT))
                            if takeoff_cmd is None else takeoff_cmd)
        # Set by the runner to Viz.log_line, so the Link panel shows the serial
        # traffic live. A spin-up stalls in about a second; reading the log
        # afterwards means knowing to look, watching it means noticing.
        self.on_line = None
        self.dry = dry_run
        # Line buffered: the serial port cannot be opened twice, so this log is the
        # only way to watch the link live. Block buffering makes `tail -f` lag by
        # thousands of telemetry lines, which is the same as not having it.
        self.log = open(log_path, "w", buffering=1)
        self.freq: float | None = None  # last frequency the firmware reported, Hz
        self.state: int | None = None  # last firmware State it reported
        self.currents: tuple | None = None  # last (A, B, C, D) coil currents, amps
        if dry_run:
            self.comm = None
            # Nothing will ever report a state, and a loop that never sees FLIGHT
            # commands nothing -- which would make the rehearsal rehearse silence.
            self.state = FLIGHT
        else:
            self.comm = SerialComm(port=port)
            self.comm.reset_device()  # reboot firmware to IDLE
            time.sleep(1.5)  # wait out the boot banner

    def send_takeoff(self) -> None:
        """Send every line of the ramp profile, in order.

        All of it, or the sequencer is left half-loaded and `seq=go` ramps a shape nobody
        chose. `test_panel` asserts the whole profile goes out on one tick for that reason.
        """

        for c in self.takeoff_cmd:
            self.send(c)

    def send(self, cmd: str) -> None:
        stamp = f"[{time.monotonic():.3f}] -> {cmd}"
        print(stamp)
        self.log.write(stamp + "\n")
        if self.on_line:
            self.on_line(cmd, sent=True)
        if self.comm:
            self.comm.handle_serial_comm(cmd)

    def drain(self) -> None:
        """Log pending telemetry, keeping the last freq=, state= and I[A]: seen."""

        if not self.comm:
            return
        # A non-match leaves the last good value standing: lines arrive truncated
        # and interleaved, and a dropped field is not a reading of zero.
        while (line := self.comm.handle_serial_comm()) is not None:
            self.log.write(f"[{time.monotonic():.3f}] <- {line}\n")
            if self.on_line:
                self.on_line(line)
            t = parse_telemetry(line)
            if t.state is not None:
                self.state = t.state
            if t.freq is not None:
                self.freq = t.freq
            if len(t.amps) == 4:
                self.currents = t.amps

    def arm(self) -> None:
        """Reboot the firmware to IDLE and send the ramp profile again.

        The firmware allows one flight per boot, and all three `seq=` verbs are accepted
        only from IDLE, so re-arming after a stop or a failed ramp means resetting the
        board -- not just re-sending the profile. Costs 1.5 s and makes the button work
        from any state, OFF included.
        """

        if self.comm:
            self.comm.reset_device()
            time.sleep(1.5)          # wait out the boot banner, as at construction
        self.state = None if self.comm else FLIGHT   # dry run: keep flying the stub
        self.freq = None
        self.send_takeoff()

    def close(self) -> None:
        self.log.close()
        if self.comm:
            self.comm.close()


def controller_loop(ticks, link: CommandLink, ctrl, args, ztrk=None, rows=None,
                    threaded=False) -> None:
    """Fly `ticks` on a fixed ts cadence. `ctrl` is the (x, y) controller pair; ztrk owns z.

    `rows` is a csv.writer for CSV_HEADER, or None. One row per tick from the first tick.
    """

    landed = False

    def land(reason: str):
        nonlocal landed
        if not landed:
            print(f"landing: {reason}")
            link.send("hover")
            link.send("land")
            landed = True

    sigints = 0

    def on_sig(_sig, _frm):
        nonlocal sigints
        sigints += 1
        if sigints == 1:
            land("SIGINT")
        else:
            link.send("stop")
            sys.exit(1)

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    ctrl_x, ctrl_y = ctrl
    ts = ctrl_x.ts
    # Bound before the try, so a failure to start the pose thread still reaches a
    # `finally` that can de-energise instead of a NameError that hides it.
    feed = None
    try:
        # `takeoff=True` does NOT command the ramp here. The datum is not set until the
        # estimator has held a still robot for a moment, and until it is, `tick.xyz_mm`
        # is None -- so a ramp commanded at t=0 spends its first seconds unmeasured.
        # That is precisely the window the experiment is about: capture happens in the
        # first second at RAMP_START_HZ, and it is the only part of the ramp the spin
        # witness can adjudicate (theory.md 18.4). Measured: a 35 s rehearsal commanded
        # the ramp before the first fix and logged no usable ramp samples at all.
        # So the ramp waits, inside the one loop, for PRIME_FIXES consecutive fixes.
        pending_takeoff = bool(args.takeoff)
        fixes = 0
        pending_capture, cap_rev_cmd, cap_t0, cap_drift0, cap_n0 = True, 0.0, None, 0.0, 0

        print("loop running; waiting for the datum before the ramp is commanded"
              if pending_takeoff else
              "loop running; the firmware ramps to FLIGHT, nothing is commanded before that")
        feed = _PoseFeed(ticks, threaded=threaded)
        t_start = time.monotonic()
        # The run CSV's `t` is relative to this instant and nothing else records it, so
        # the log is where the three artefacts are tied together: video frames, serial
        # traffic and CSV rows all reduce to one monotonic clock through this line.
        # `sync.py` reads it.
        link.log.write(f"[{t_start:.3f}] <> csv t=0\n")
        pred = predictor.StatePredictor()
        # What the coils were last told, and the current hover estimate. Passing one for
        # both would make every prediction a constant-velocity coast.
        f_cmd_last = f_hat_last = ztrk.f_hat if ztrk is not None else ctrl[0].f_hover
        last_step, next_step = t_start - ts, t_start
        last_tick = t_start
        prev_az = prev_mag = prev_freq = None
        t_resend = t_withheld = t_start
        t_meas = None            # shutter stamp of the last fix, for the rate interval
        # Clock health. The fall-behind branch below used to resync in silence, so a loop
        # that never once made its period looked exactly like one that always did. At
        # 500 Hz the period is 2 ms against a macOS sleep granularity of ~1 ms, which is
        # the whole question, so it gets counted and reported rather than assumed.
        n_ticks = n_overrun = 0
        dts = deque(maxlen=20000)
        flying = False
        t_flight = None
        auto_armed = False
        mag_cmd = az_cmd = ""     # last commanded, last-value-stands as for freq/state
        t_pred = t_start          # where the predictor has been advanced to
        t_fix = t_start           # wall time of the last pose fix, for the tracking land
        trim_sent = False         # the tilt trim is commanded once, above trim_at_hz
        # Attitude. The estimator runs ALWAYS -- it is the measurement, and it costs a
        # savgol over a 0.25 s ring buffer -- but the controller only acts when
        # `attitude_closed` is set AND a rotation has been measured. Flying it open is how
        # the rotation gets identified in the first place (theory.md 21.2).
        tv = attitude.ThrustVector()
        tilt_ctl = attitude.TiltController(gain=args.attitude_gain,
                                           rot_deg=args.attitude_rot_deg)
        cmd_tilt = (0.0, 0.0)
        id_phase = -1             # which dither azimuth is commanded
        lost_land_s = getattr(args, "lost_land_s", LOST_LAND_S)
        n_seen = 0
        while True:
            if landed:
                break
            # THE CLOCK. The loop steps on `ts`, not on frame arrival: pose costs ~15 ms
            # a pair and the controller wants to command every `ts`. A grid, not a
            # "ts since last step" gate -- against a source running at exactly ts that
            # aliases and the loop silently runs at 19 Hz.
            now = time.monotonic()
            if now < next_step:
                # Capped so a long `ts` still notices `landed` and a dead producer
                # promptly. The cap is a QUARTER of the period, not a fixed 2 ms: at
                # 500 Hz a 2 ms cap is the entire period, so the sleep overshot the next
                # step every time and the grid below resynced on every tick. macOS
                # sleep granularity is ~1 ms, so this still lands late at 500 Hz -- that
                # is what `n_overrun` is for, and a busy-wait is the next rung if the
                # measurement asks for one.
                time.sleep(min(next_step - now, ts / 4.0))
                continue
            dts.append(now - last_tick)
            last_tick = now
            n_ticks += 1
            next_step += ts
            if next_step < now:
                next_step = now + ts  # fell behind: resync rather than a catch-up burst
                n_overrun += 1

            tick, n_tick = feed.read()
            if feed.exc is not None:
                raise feed.exc        # the producer died: fall into `finally`, coils off
            if tick is None:
                if feed.done:
                    break             # source exhausted before it ever yielded
                continue              # not warm yet
            # `fresh` is the whole gating rule: anything that CONSUMES a frame is gated
            # on it, anything that COMMANDS the coils runs on the clock. Ungating the
            # frame consumers is not a tidiness point -- one held fix would satisfy
            # PRIME_FIXES in 75 ms and ramp on a single frame.
            fresh = n_tick != n_seen
            n_seen = n_tick
            if feed.done and not fresh:
                break                 # replayed or stubbed source ran out
            link.drain()
            viz = tick.viz
            if link.on_line is None:
                # See `fly`: on the camera path this is the only place the viewer exists.
                link.on_line = getattr(viz, "log_line", None)

            # THE ONE READ. estop/takeoff/land are one-shot latches cleared on read;
            # everything below dispatches on these locals and never touches viz.* again.
            estop, takeoff_req, land_req = viz.estop, viz.takeoff, viz.land
            armed, mag_max, gain = viz.armed, viz.mag_max, viz.gain_scale
            # Fixed tilt trim, commanded once the rotor is CAPTURED and held after that.
            # The rotor normal sits ~1.6 deg off-axis at rest and ~3 deg under thrust in a
            # fixed azimuth (theory.md 18.15). That tilt costs almost no lift --
            # cos(4.6 deg) = 0.997 -- it costs flight TIME, throwing the robot out of the
            # tracked volume at 0.08 g before thrust accumulates. A constant counter-tilt
            # needs no bandwidth, so unlike the closed loop it is not bound by the
            # 0.16-0.78 Hz poles.
            #
            # GATED, and not sent with the ramp: an ungated `az=315 mag=0.40` weakens TWO
            # coils (max(0,cos) catches both A and D at 0.707) and distorted the rotating
            # field through the whole capture window -- the rotor never span, 2026-09-01.
            # `az=0` weakens one and survived it, which is why the flaw hid for a run.
            # Capture happens at 2-5 Hz, so 20 Hz is clear of it and far below any tilt.
            if args.trim_mag and not trim_sent and link.freq and \
                    link.freq >= args.trim_at_hz:
                trim_sent = True
                print(f"tilt trim: az={args.trim_az:.0f} mag={args.trim_mag:.2f} "
                      f"engaged at {link.freq:.1f} Hz")
                link.send(f"az={args.trim_az:.0f}")
                link.send(f"mag={args.trim_mag:.3f}")

            # IDENTIFICATION DITHER. Steps the commanded weak direction through a set of
            # azimuths so `attitude.fit_rotation` has a known input to regress the tilt
            # response against. It runs on the same clock as the trim and REPLACES its
            # azimuth, so the two never fight over `az=`.
            #
            # This is the only excitation available: the thrust sensor reads nothing while
            # the robot is seated (the pad cancels lateral acceleration), so the dither has
            # to happen inside an airborne window that is currently 1-2 s long. If it is too
            # short, `fit_rotation` refuses rather than fitting noise -- which is the
            # designed outcome, not a failure of the run.
            # Gated on FREQUENCY, not on the trim having fired: identification has to be
            # runnable with no trim at all, and an earlier version that keyed off
            # `trim_sent` silently did nothing whenever `trim_mag` was 0.
            if args.id_azimuths and link.freq and link.freq >= args.trim_at_hz:
                k = int((now - t_start) / args.id_dwell_s) % len(args.id_azimuths)
                if k != id_phase:
                    id_phase = k
                    link.send(f"az={args.id_azimuths[k]:.0f}")

            # Arm on the RAMP, not just on FLIGHT. Measured 2026-09-01: across five runs the
            # lateral loop never armed once, because arming waited for the firmware to
            # report FLIGHT and the robot departed before `seq.isDone()` every time. The
            # restriction was host-side only -- `main_flight.cpp` runs `applyMixer()` in
            # `case SPINUP` as well as `case FLIGHT`, so az=/mag= are honoured mid-ramp.
            # The thrust axis is tilted ~4.6 deg (theory.md 18.14) and the resulting drift
            # is under 2 Hz, so the controller has to be in the loop BEFORE the robot
            # leaves the pad, which is what this arms in time for.
            if args.arm_at_hz is not None and link.freq and link.freq >= args.arm_at_hz:
                if not auto_armed:
                    auto_armed = True
                    print(f"auto-arm: closed loop engaged on the ramp at "
                          f"{link.freq:.1f} Hz (mag_max {args.auto_mag_max:.2f})")
                armed = True
                mag_max = max(mag_max, args.auto_mag_max)
            elif args.auto_arm_s is not None and t_flight is not None:
                if now - t_flight >= args.auto_arm_s:
                    if not auto_armed:
                        auto_armed = True
                        print(f"auto-arm: closed loop engaged {args.auto_arm_s:.1f}s "
                              f"after FLIGHT (mag_max {args.auto_mag_max:.2f})")
                    armed = True
                    mag_max = max(mag_max, args.auto_mag_max)
            sp_x, sp_y, sp_z = (v * 1e-3 for v in viz.setpoint)  # viz is mm, the loop is m
            if args.hover_z_mm is not None:
                # An explicit target overrides the slider. The slider defaults to 60 mm,
                # which is far past anything this rig has reached (~1 mm), so leaving it
                # in charge would command full climb authority from the first tick.
                sp_z = args.hover_z_mm * 1e-3

            if estop:
                # Not `land`: landing is a ramp, and a stop is not. The button has
                # already disarmed, so this is the part that de-energises the coils.
                # It also eats a takeoff pressed on the same tick -- a stop and a start
                # arriving together must not start the coils.
                print("STOP: coils off. 'takeoff' re-arms the firmware.")
                link.send("stop")
                link.state, flying, takeoff_req = None, False, False
                t_flight, auto_armed = None, False
                # And one already counting down: `pending_takeoff` fires `send_takeoff()`
                # at PRIME_FIXES, so leaving it set ramped the coils a few frames AFTER
                # an explicit stop.
                pending_takeoff, fixes = False, 0
            if land_req:
                land("viz land button")
                break
            if takeoff_req:
                print("takeoff requested from the viewer")
                link.arm()               # reset -> IDLE -> seq=clear/ramp:/go
                flying, prev_az, pending_takeoff, fixes = False, None, False, 0
                prev_mag = prev_freq = None   # re-arm resends everything
                t_flight, auto_armed = None, False
                pending_capture, cap_rev_cmd, cap_t0, cap_drift0, cap_n0 = True, 0.0, None, 0.0, 0

            if pending_takeoff and fresh:
                # `fresh`, or the count is of control steps rather than fixes: at 200 Hz
                # one held pose satisfies PRIME_FIXES in 75 ms and ramps on one frame.
                # Consecutive, not cumulative: one fix followed by a dropout means the
                # estimator has not settled, and the ramp is 40 s of coils either way.
                fixes = fixes + 1 if tick.xyz_mm is not None else 0
                if fixes >= PRIME_FIXES:
                    # The constructor already reset the board, so it is IDLE: no arm(),
                    # which would spend 1.5 s rebooting a board that just booted.
                    print(f"datum holding ({PRIME_FIXES} consecutive fixes) -- ramping")
                    link.send_takeoff()
                    pending_takeoff = False
                    pending_capture, cap_rev_cmd, cap_t0, cap_drift0, cap_n0 = True, 0.0, None, 0.0, 0

            # --- tracking-loss land, the one automatic backstop (see LOST_LAND_S) -----
            # WALL time, not tick.t. The hazard includes the frame producer stalling, and
            # a stalled producer freezes tick.t -- a timer on frame time would then never
            # fire, which is precisely the case this exists to catch.
            if fresh and tick.xyz_mm is not None:
                t_fix = now
            if (lost_land_s and not link.dry and not pending_takeoff
                    and link.state in (SPINUP, FLIGHT) and now - t_fix > lost_land_s):
                print(f"TRACKING LOST for {now - t_fix:.1f}s with the coils driven at "
                      f"{link.freq or float('nan'):.1f} Hz -- stopping. The robot is not "
                      f"where the cameras can see it; driving on is blind.")
                link.send("stop")
                break

            if (args.capture_check and pending_capture and fresh
                    and tick.spin is not None and link.freq):
                # Only below the alias limit, where accumulated phase is physical: above
                # it the unwrap picks the smallest step and the sign is not even reliable.
                if link.freq <= tick.spin.alias_limit_hz:
                    # Integrate on tick.t, the FRAME timestamp, not wall time: drift_rev
                    # is accumulated per frame, so both sides of the comparison have to
                    # be on the camera's clock or the ratio is meaningless.
                    if cap_t0 is None:
                        cap_t0, cap_drift0 = tick.t, tick.spin.drift_rev
                        cap_n0 = tick.spin.n
                    else:
                        cap_rev_cmd += link.freq * max(tick.t - cap_t0, 0.0)
                        cap_t0 = tick.t
                elif cap_rev_cmd >= CAPTURE_MIN_REV:
                    pending_capture = False
                    seen = tick.spin.n - cap_n0
                    if seen < CAPTURE_MIN_FRAMES:
                        # No blade signal is NOT evidence of a stalled rotor -- the robot
                        # may simply be unreadable. Saying so beats aborting a good run.
                        print(f"capture unverified: only {seen} frames of blade signal "
                              f"under {tick.spin.alias_limit_hz:.1f} Hz -- continuing")
                        continue
                    got = abs(tick.spin.drift_rev - cap_drift0)
                    frac = got / max(cap_rev_cmd, 1e-6)
                    if frac >= CAPTURE_REV_FRAC:
                        print(f"capture OK: rotor turned {got:.1f} rev against "
                              f"{cap_rev_cmd:.1f} commanded ({frac * 100:.0f}%)")
                    else:
                        print(f"CAPTURE FAILED: rotor turned only {got:.1f} rev against "
                              f"{cap_rev_cmd:.1f} commanded ({frac * 100:.0f}%). It is "
                              f"rocking in the field, not rotating -- aborting rather "
                              f"than driving a stationary rotor through the climb.")
                        link.send("stop")
                        break
                elif link.freq > 4.0 * tick.spin.alias_limit_hz:
                    pending_capture = False
                    print(f"capture unverified: only {cap_rev_cmd:.1f} commanded rev "
                          f"under {tick.spin.alias_limit_hz:.1f} Hz -- continuing")

            if tick.spin is not None and link.freq:
                # What unlocks the confident `stopped`. SpinWitness will not call a
                # still phase stopped unless it knows the field is under its own alias
                # limit, and stereo_frames -- which owns the witness -- never sees a
                # command, so nothing else can tell it. Without this the column is
                # turning-or-blank and a rotor that never captured looks identical to
                # one the strobe cannot resolve. link.freq is the firmware's own
                # reported field rate, not our intent.
                tick.spin.field_hz = link.freq

            if rows is not None:
                x, y, z = ([f"{v:.3f}" for v in tick.xyz_mm]
                           if tick.xyz_mm is not None else ("", "", ""))
                # The rotor normal, which the estimator has always produced and nothing
                # ever recorded: tilt away from the datum axis, and the azimuth it points
                # along. Blank when there is no pose, exactly as x/y/z are -- a zero here
                # would read as "measured, and level", which is the opposite of unknown.
                tilt, tilt_az = (
                    (f"{tick.pose.theta_deg:.2f}", f"{tick.pose.phi_deg:.1f}")
                    if tick.pose is not None else ("", ""))
                i = link.currents or ("", "", "", "")
                # Blank, not zero, when the thrust sensor declines: a 0.00 here would read
                # as "measured, and level", which is the opposite of "not airborne yet".
                at = (("", "") if tv.tilt_xy is None
                      else (f"{tv.tilt_xy[0]:.3f}", f"{tv.tilt_xy[1]:.3f}"))
                rows.writerow([f"{now - t_start:.4f}", link.state if link.state is not None
                               else "", link.freq if link.freq is not None else "",
                               x, y, z, tilt, tilt_az, at[0], at[1],
                               f"{cmd_tilt[0]:.4f}", f"{cmd_tilt[1]:.4f}",
                               mag_cmd, az_cmd, int(armed),
                               _spin_state(tick.spin), tick.lost, *i])

            if link.state != FLIGHT:
                # Drain, log, draw -- but command nothing. The firmware owns the ramp.
                viz.push(tick.pose, u=(0.0, 0.0, link.freq or 0.0, args.throttle, 0.0),
                         t=tick.t, frames=tick.frames, lost=tick.lost, state=link.state,
                         spin=tick.spin)
                continue
            if not flying:
                flying = True
                t_flight = now
                print(f"FLIGHT at {link.freq or float('nan'):.1f} Hz")
                link.send(f"throttle={args.throttle:.0f}")
                _anchor(ctrl, ztrk, link.freq)

            # Push the miss too, or the trace interpolates straight over a dropout.
            # The interval the MEASUREMENT moved over, for the rate estimate: shutter
            # stamp to shutter stamp, not control period to control period. None on a
            # tick with no new fix, where the position came from the predictor and
            # re-differencing it would return the velocity that produced it. See
            # `simulate_hover.VelocityEstimator.update` and `theory.md` 19.8.
            dt_meas = None
            if fresh and tick.xyz_mm is not None:
                if t_meas is not None and tick.t > t_meas:
                    dt_meas = tick.t - t_meas
                t_meas = tick.t
                xyz_mm = pred.update(tick.xyz_mm, t=tick.t)
                tv.update(tick.xyz_mm, tick.t)
                # From the SHUTTER, not from now: `tick.t` is the capture stamp, so the
                # next step propagates this pose forward by its full pipeline age. That
                # is the latency compensation `filter.predict_ahead` was written for and
                # never wired to, and here it costs nothing.
                t_pred = tick.t
            else:
                if fresh:
                    viz.push(tick.pose, t=tick.t, frames=tick.frames, lost=tick.lost,
                             spin=tick.spin)
                if not pred.initialised:
                    continue         # nothing to propagate from yet
                xyz_mm = pred.predict(
                    predictor.Command(f_cmd_last, f_hat_last), max(now - t_pred, 0.0)
                )
                t_pred = now
            dt = min(now - last_step, 5.0 * ts)
            last_step, t = now, now - t_start

            x_m, y_m, z_m = np.asarray(xyz_mm, dtype=float) * 1e-3
            # The sliders move the origin and the profile moves relative to it, so the
            # controller is fed the offset measurement.
            ux, f_lqr = ctrl_x.step(t, x_m - sp_x, z_m - sp_z, dt_meas)
            uy, _ = ctrl_y.step(t, y_m - sp_y, z_m - sp_z, dt_meas)
            ux, uy = gain[0] * ux, gain[0] * uy

            # THE 5-DOF SEAM. Position gives (ux, uy); attitude gives another Cartesian
            # term in the same frame and units; they sum here, and one polar conversion
            # below serves both. Remaining degrees of freedom join at this same point.
            cmd_tilt = tilt_ctl.step(tv.tilt_xy, t)
            if args.attitude_closed:
                ux, uy = ux + cmd_tilt[0], uy + cmd_tilt[1]

            # Polar, because the actuator is: one azimuth and one magnitude, never a signed
            # magnitude with a 180 degree flip hiding in it.
            az = math.degrees(math.atan2(uy, ux)) % 360.0
            mag = min(math.hypot(ux, uy), mag_max)

            if ztrk is not None:
                f_z = ztrk.step(t, z_m, dt, z_target=sp_z)
                f_trim = ztrk.f_hat
            else:
                f_z, f_trim = f_lqr, ctrl_x.f_hover
            f_field = f_trim + gain[1] * (f_z - f_trim)  # scales authority, never trim
            f_cmd_last, f_hat_last = f_field, f_trim

            if armed:
                due = now - t_resend >= RESEND_S
                if due:
                    t_resend = now
                if (prev_az is None or due
                        or abs((az - prev_az + 180.0) % 360.0 - 180.0) > AZ_DEADBAND_DEG):
                    link.send(f"az={az:.0f}")
                    prev_az = az
                if prev_mag is None or due or abs(mag - prev_mag) > MAG_DEADBAND:
                    link.send(f"mag={mag:.3f}")
                    prev_mag = mag
                mag_cmd, az_cmd = f"{mag:.3f}", f"{az:.0f}"
                if args.enable_freq_cmd:
                    if (prev_freq is None or due
                            or abs(f_field - prev_freq) > FREQ_DEADBAND_HZ):
                        link.send(f"freq={f_field:.2f}")
                        prev_freq = f_field
                elif now - t_withheld >= WITHHELD_LOG_S:
                    t_withheld = now
                    link.log.write(
                        f"[{now:.3f}] (freq={f_field:.2f} not sent, "
                        f"pass enable_freq_cmd to close the altitude loop)\n"
                    )
            elif now - t_withheld >= WITHHELD_LOG_S:
                t_withheld = now
                link.log.write(
                    f"[{now:.3f}] (disarmed: mag={mag:.3f} az={az:.0f} "
                    f"freq={f_field:.2f} withheld)\n"
                )

            # After the sends, never before: the command reaching the coils is the only thing
            # on this path with a deadline.
            ref_p, _, _ = ctrl_x.profile.eval(t)
            viz.push(
                tick.pose,
                ref=(ref_p[0] + sp_x, ref_p[1] + sp_z),
                u=(mag, az, f_field, args.throttle, f_trim),
                t=tick.t,
                frames=tick.frames,
                lost=tick.lost,
                spin=tick.spin,
            )
    except Exception as e:  # noqa: BLE001 -- de-energize on ANY failure
        print(f"error: {e}", file=sys.stderr)
        link.send("stop")
        raise
    finally:
        land("loop exit")
        link.send("stop")
        # Clock health, printed unconditionally: the design rate is a claim, and this is
        # the only place that says whether the loop kept it. Reported AFTER the coils are
        # off -- nothing here is allowed to delay that.
        if n_ticks:
            d = np.sort(np.asarray(dts)[1:]) * 1e3      # drop the first, seeded at t_start
            if len(d):
                print(f"clock: {n_ticks} ticks, design {1.0 / ts:.0f} Hz, "
                      f"achieved {1e3 / d.mean():.1f} Hz | dt ms "
                      f"med {np.median(d):.2f} p95 {d[int(0.95 * len(d))]:.2f} "
                      f"max {d.max():.2f} | {n_overrun} overrun "
                      f"({n_overrun / n_ticks:.1%})")
        # Run the source's own finally now: that is where stereo_frames closes its cameras
        # and its viewer, and a traceback holding a reference would otherwise keep them
        # open. AFTER the stop above, never before -- the coils come off first, and only
        # then do we join a thread that may be blocked in a 2 s camera read.
        close = getattr(feed, "close", None) or getattr(ticks, "close", None)
        if close:
            close()


@dataclass
class RunConfig:
    """Everything `controller_loop` reads, in one place."""

    source: str = "stub"  # "stub" | "camera"
    camera: str = "camera:0,camera:1"  # stereo pair for live_viz.stereo_frames
    # 640x400 at 210 fps: a true 0.5x of the calibrated 1280x800, so the rig rescales
    # exactly. 42 -> 65 Hz measured, for the 0.119 mm the theory already priced.
    width: int = 640
    height: int = 400
    fps: int = 210
    rig: str = None  # stereo rig calibration; None takes the saved one
    axes: tuple = ("x", "y", "z")  # datum-frame axis names, for the viz labels
    profile: str = None  # reference profile JSON; None holds at the viz setpoint
    port: str = None
    # None pairs the serial log with the run's CSV as `<csv_dir>/<stamp>.log`. A fixed
    # path is honoured, but is overwritten every run -- see `fly`.
    log: str | None = None
    csv_dir: str = "results/takeoff"  # one CSV per attempt, read by takeoff_report.py
    takeoff: bool = True
    # THE ramp. `((from_hz, to_hz, seconds, mode, k), ...)`, straight onto
    # `PwmSequencer::addRampTask` -- see `ramp.py`, which owns the shape, the validation
    # and the `seq=` encoding. `ramp.DEFAULT` is the profile the robot flew on
    # 2026-08-29. Pass a tuple to try another; `ramp.check` refuses gaps and over-cap
    # totals, and `ramp.label` stamps whatever was sent into the run's CSV.
    segments: tuple = ramp.DEFAULT
    # Hand over to the closed loop automatically, this many seconds after FLIGHT.
    # None keeps the manual `armed` checkbox as the only way in. Auto-arming exists
    # because liftoff is over in a second or two and a human tick is too slow.
    auto_arm_s: float | None = None
    # Arm the lateral loop DURING the ramp, as soon as the firmware reports this field
    # frequency. Takes precedence over `auto_arm_s`, which waits for FLIGHT -- a wait the
    # robot lost five times out of five on 2026-09-01 by departing mid-ramp. Set it below
    # the liftoff frequency so control is live before the robot leaves the pad. None keeps
    # the old FLIGHT-only behaviour. See theory.md 18.14.
    arm_at_hz: float | None = None
    # Constant tilt trim, commanded once with the ramp and held. `trim_mag` 0 disables it.
    # Operator observation 2026-09-01: with coil A dropped the robot "flew off in a much
    # more stable manner", and A at 0 deg is ~34 deg from the direction that opposes the
    # measured ~146-155 deg tilt. NOT yet a calibrated trim -- the mixer sign is still
    # unverified (18.16), so treat the direction as a hypothesis under test.
    trim_az: float = 0.0
    trim_mag: float = 0.0
    # Hold the trim back until the drive reaches this. Capture happens at 2-5 Hz and a
    # trim applied through it distorts the rotating field enough to stop the rotor being
    # caught at all -- measured 2026-09-01 with az=315, which never span.
    trim_at_hz: float = 20.0
    # Attitude loop (theory.md 21). The estimator always runs and is always logged; these
    # only decide whether it ACTS. `attitude_rot_deg` is the mixer rotation and has no
    # default -- 20.5 allows only 69.4 deg of error before the loop is uncertifiable, and
    # 12.8 measured 72 deg on a related model, so it is measured by `attitude.fit_rotation`
    # from an open-loop run before anything closes. `attitude_closed` stays False until then.
    attitude_rot_deg: float | None = None
    attitude_closed: bool = False
    attitude_gain: float = 0.02
    # Identification: cycle the commanded weak direction through these while flying, so
    # `attitude.fit_rotation` has a known input. Empty disables it. Dwell must exceed the
    # estimator's 0.25 s window or every sample straddles two commands.
    id_azimuths: tuple = ()
    id_dwell_s: float = 0.4
    # Lateral authority when auto-armed. STAYS 0 until the az sweep is run: `applyMixer`
    # steers by COIL_AZ, which main_flight.cpp labels a seed guess, and a wrong azimuth
    # map pushes lateral the wrong way -- the same class of bug as the inverted
    # PHASES_CW/CCW labels. Altitude (freq=) needs no such geometry, so it is safe first.
    auto_mag_max: float = 0.0
    # Constant hover target in mm, used when no waypoint file is given. ZTracker holds
    # frequency until z reaches this, then backs it down to the hover equilibrium.
    hover_z_mm: float | None = None
    # Abort the ramp when the rotor is rocking rather than turning. OFF by default: the
    # check has to commit before the field passes the witness's ~3 Hz alias limit, which
    # is barely a second in, and a rotor that captures later is failed by it. Observed
    # 2026-08-29: aborted at 4% of commanded revolutions on a run that then span up fine.
    # The logic is sound and tested; the WINDOW is too narrow to trust.
    capture_check: bool = False
    # Stop the coils after this many seconds with no pose fix, once the ramp is running.
    # The only automatic land on the host -- see LOST_LAND_S for why it came back. Set to
    # 0 or None to fly blind, which is what every run before 2026-09-01 did.
    lost_land_s: float | None = LOST_LAND_S

    throttle: float = 100.0
    enable_freq_cmd: bool = False  # False = open loop, telemetry-only dress rehearsal
    dry_run: bool = False  # print commands instead of opening serial
    gains: str = None  # defaults to hover_controller.json beside this file
    waypoints: str = None  # z waypoints; None reverts z to the LQR
    viz: bool = False  # live 3-D view; stub only -- source='camera' always builds one
    record: object = None  # film the run: a flight dir, or True for results/flights/
    viz_port: int = 8080

    def __post_init__(self):
        here = os.path.dirname(__file__)
        if self.gains is None:
            self.gains = os.path.join(here, "hover_controller.json")


def fly(cfg=None, **kw):
    """Run the hover controller. ``fly(dry_run=True)`` is the stub-source dress rehearsal."""

    cfg = cfg or RunConfig(**kw)
    os.makedirs(cfg.csv_dir, exist_ok=True)
    _thermal_gate(cfg)

    with open(cfg.gains) as f:
        gains = json.load(f)
    profile = Profile.from_json(cfg.profile) if cfg.profile else Profile.hold()
    ctrl = (DiscreteHoverController(gains, profile), DiscreteHoverController(gains, profile))

    lv = _live_viz()
    own_viz = None
    if cfg.source == "camera":
        # stereo_frames builds its own viewer and hands it out on every Tick: it needs one
        # during datum priming, before this loop exists.
        ticks = lv.stereo_frames(
            specs=cfg.camera, rig_path=cfg.rig, width=cfg.width, height=cfg.height,
            fps=cfg.fps,
            port=cfg.viz_port, axes=tuple(cfg.axes), label="hover",
            record=cfg.record,
        )
    else:
        own_viz = lv.make_viz(
            enabled=cfg.viz, port=cfg.viz_port, axes=tuple(cfg.axes), label="hover"
        )
        ticks = stub_ticks(gains, own_viz)

    ztrk = None
    if cfg.hover_z_mm is not None and not cfg.waypoints:
        ztrk = z_track.ZTracker(np.array([0.0, 1e9]),
                                np.array([cfg.hover_z_mm * 1e-3] * 2))
        print(f"altitude: holding {cfg.hover_z_mm:.0f} mm, f_hat seeded "
              f"{ztrk.f_hat:.0f} Hz over [{ztrk.f_lo:.0f}, {ztrk.f_hi:.0f}] Hz")
    if cfg.waypoints:
        times, heights = z_track.load_waypoints(cfg.waypoints)
        # Constructing ZTracker computes f_ceiling, which raises if the torque calibration
        # leaves no headroom at f_hover. Fail on the bench, not with the robot spinning.
        ztrk = z_track.ZTracker(times, heights)
        print(
            f"altitude: {len(times)} waypoints from {cfg.waypoints}, "
            f"f_hat seeded at {ztrk.f_hat:.0f} Hz (SEED GUESS until liftoff is measured), "
            f"f_ceiling={ztrk.f_ceil:.1f} Hz, a_max={ztrk.a_ceiling:.2f} m/s^2"
        )

    # One stamp for both artefacts, so an attempt's CSV and its serial log are named
    # alike and neither can be attributed to the wrong run.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(cfg.csv_dir, stamp + ".csv")
    # `cfg.log` defaults to None, which means "beside the CSV". The old fixed
    # `hover_run.log` was opened "w" every run, so each attempt silently destroyed the
    # previous attempt's serial trace -- fine when the log was a live tail, useless once
    # `sync.py` aligns it to a flight that is kept. An explicit path still wins, and
    # still clobbers, which is the caller's choice.
    log_path = cfg.log or os.path.join(cfg.csv_dir, stamp + ".log")
    link = CommandLink(cfg.port, cfg.dry_run, log_path,
                       takeoff_cmd=ramp.seq_lines(cfg.segments))
    # The viewer the camera path built is reachable through the first tick; for the
    # stub path it is own_viz. On the camera path `stereo_frames` builds its viewer
    # inside the generator, so there is no reference here -- `controller_loop` wires it
    # off the first tick instead. Without that the Link panel read "no serial traffic
    # yet" for every real flight, which is the only path it matters on.
    if own_viz is not None:
        link.on_line = own_viz.log_line

    # Block-buffered, not line-buffered. A row a tick at 500 Hz is 500 flushes a second
    # inside a 2 ms budget; the default 8 kB buffer is ~40 rows, i.e. ~80 ms of flight,
    # and `fh.close()` in the `finally` below flushes the tail on every exit that runs
    # Python at all -- SIGINT and exceptions included. A hard-killed kernel loses that
    # 80 ms, which is the same failure mode that already leaves the coils energised.
    fh = open(csv_path, "w", newline="")
    rows = csv.writer(fh)
    fh.write(f"# ramp: {ramp.label(cfg.segments)}\n")
    # The trim is the other thing that changes between attempts, so it is stamped for the
    # same reason the ramp is: without it a comparison rests on remembering what was typed.
    _coil = {0: "A", 90: "B", 180: "C", 270: "D"}.get(int(cfg.trim_az) % 360)
    fh.write(f"# trim: resultant weak direction {cfg.trim_az:g} deg"
             + (f" (coil {_coil})" if _coil else "")
             + f", strength {cfg.trim_mag:.2f}"
             + (f", from {cfg.trim_at_hz:g} Hz" if cfg.trim_mag else " (off)") + "\n")
    rows.writerow(CSV_HEADER)
    try:
        controller_loop(ticks, link, ctrl, cfg, ztrk, rows,
                        threaded=cfg.source == "camera")
    finally:
        fh.close()
        link.close()
        if own_viz is not None:
            own_viz.close()
        print(f"done -> {log_path}, {csv_path}")
        # A missing or broken report must never mask the flight's own exception.
        try:
            from controller.control import takeoff_report

            takeoff_report.report(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"no takeoff report ({type(exc).__name__}: {exc})")
    return log_path
