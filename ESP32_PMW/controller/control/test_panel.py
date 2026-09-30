#!/usr/bin/env python3
"""Does the viser panel actually reach the coils? One-shot latches, read once per tick.

viz.estop/takeoff/land are cleared on read. Reading one twice in a tick throws a press away,
and every symptom of the old runner -- "coils will not start", "coils restart on their own",
"stop does nothing during the ramp" -- was that single bug. The fake viz here answers each
queued press EXACTLY once, like the real widgets; a sticky-True fake would pass either way.

Run: uv run python controller/control/test_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from controller.control import hover_controller_runner as R
from controller.control import ramp
from controller.control import z_track
from controller.viz.live_viz import LiveViz, NullViz, Tick
from controller.control.reference_profiles import Profile
from controller.control.simulate_hover import DiscreteHoverController

SPINUP, FLIGHT = 1, 2   # main_flight.cpp State
PRESS_AT = 10           # frame the button is pressed on
N = 40


class PanelViz(NullViz):
    """Armed NullViz whose latches fire once per queued press, as the real properties do."""

    armed, mag_max = True, 0.8
    setpoint, gain_scale = (0.0, 0.0, 60.0), (1.0, 1.0)

    def __init__(self, presses):
        self.presses = presses          # {frame: (name, ...)}
        self._estop = self._takeoff = self._land = False

    def click(self, i):
        """The render thread's job: set the latch, never clear it."""

        for name in self.presses.get(i, ()):
            setattr(self, "_" + name, True)

    # Borrowed from LiveViz, not reimplemented: the fake then exercises the REAL latch
    # protocol. It matters that a press answers EXACTLY once -- a sticky-True fake would
    # pass whether or not the runner reads each latch once, which is the whole point of
    # the file. (NullViz answers these as plain False class attributes, which is right
    # for a dead viewer and useless for testing a latch.)
    _latch = LiveViz._latch
    estop, takeoff, land = LiveViz.estop, LiveViz.takeoff, LiveViz.land


def run(presses, state, takeoff_cfg, log_dir):
    """Fly N frames with `presses`, holding the firmware in `state`. Returns [(frame, cmd)]."""

    gains = json.load(open(HERE / "hover_controller.json"))
    cfg = R.RunConfig(dry_run=True, takeoff=takeoff_cfg, enable_freq_cmd=True,
                      log=str(Path(log_dir) / "panel.log"))
    ctrl = tuple(DiscreteHoverController(gains, Profile.hold()) for _ in range(2))
    ztrk = z_track.ZTracker(np.array([0.0, 1e9]), np.array([0.06, 0.06]))
    link = R.CommandLink(None, True, cfg.log, takeoff_cmd=ramp.seq_lines(cfg.segments))
    viz, frame, sent = PanelViz(presses), [0], []
    link.send = lambda c: sent.append((frame[0], c))

    def ticks():
        # Unpaced: the asserts are about WHICH tick a command lands on, not when in
        # wall time, and 4 scenarios x N frames at 60 fps is 3 s of nothing.
        for i in range(N):
            frame[0] = i
            viz.click(i)
            link.state = state       # arm() clears it; the firmware would re-report it
            yield Tick(t=i / 60.0, xyz_mm=np.array([0.0, 0.0, 60.0]), pose=None,
                       frames=None, lost=0, viz=viz)

    try:
        R.controller_loop(ticks(), link, ctrl, cfg, ztrk)
    finally:
        link.close()
    return sent


def run_coast(gap_frames, log_dir):
    """Fly with `gap_frames` consecutive lost fixes mid-flight. Returns [(frame, cmd)].

    The half of the deleted `test_dropout.py` that is still live behaviour: a short gap
    must keep commanding off `predictor.StatePredictor`, not go quiet. `pred.stale` is
    deliberately ignored now, so nothing else guards this path.
    """

    gains = json.load(open(HERE / "hover_controller.json"))
    cfg = R.RunConfig(dry_run=True, takeoff=False, enable_freq_cmd=True,
                      log=str(Path(log_dir) / "coast.log"))
    ctrl = tuple(DiscreteHoverController(gains, Profile.hold()) for _ in range(2))
    ztrk = z_track.ZTracker(np.array([0.0, 1e9]), np.array([0.06, 0.06]))
    link = R.CommandLink(None, True, cfg.log, takeoff_cmd=ramp.seq_lines(cfg.segments))
    viz, frame, sent = PanelViz({}), [0], []
    link.send = lambda c: sent.append((frame[0], c))
    lost_at = set(range(N // 2, N // 2 + gap_frames))

    def ticks():
        for i in range(N):
            frame[0] = i
            link.state = FLIGHT
            yield Tick(t=i / 60.0, frames=None, lost=0, viz=viz, pose=None,
                       xyz_mm=None if i in lost_at
                       else np.array([0.0, 0.0, 60.0 + 0.05 * i]))

    try:
        R.controller_loop(ticks(), link, ctrl, cfg, ztrk)
    finally:
        link.close()
    return sent, lost_at


def run_lost(gap_frames, log_dir, lost_land_s=0.05):
    """Fly with `gap_frames` consecutive lost fixes while the coils are driven.

    Unlike `run_coast`, this link is NOT dry: the tracking-loss land is disabled on a dry
    link, because a rehearsal drives no coils and a land there would only break the stub.
    """

    gains = json.load(open(HERE / "hover_controller.json"))
    cfg = R.RunConfig(dry_run=True, takeoff=False, enable_freq_cmd=False,
                      lost_land_s=lost_land_s,
                      log=str(Path(log_dir) / "lost.log"))
    ctrl = tuple(DiscreteHoverController(gains, Profile.hold()) for _ in range(2))
    ztrk = z_track.ZTracker(np.array([0.0, 1e9]), np.array([0.06, 0.06]))
    link = R.CommandLink(None, True, cfg.log, takeoff_cmd=ramp.seq_lines(cfg.segments))
    link.dry = False          # a live link, without opening a serial port
    link.freq = 104.0         # the frequency the real run departed at
    viz, frame, sent = PanelViz({}), [0], []
    link.send = lambda c: sent.append((frame[0], c))
    lost_at = set(range(4, 4 + gap_frames))

    def ticks():
        for i in range(N):
            frame[0] = i
            link.state = R.SPINUP        # coils driven, mid-ramp
            yield Tick(t=i / 60.0, frames=None, lost=0, viz=viz, pose=None,
                       xyz_mm=None if i in lost_at else np.array([0.0, 0.0, 60.0]))

    try:
        R.controller_loop(ticks(), link, ctrl, cfg, ztrk)
    finally:
        link.close()
    return sent


def run_capture(verdict, log_dir, n_low=200):
    """Ramp with the witness returning `verdict` under the alias limit.

    `verdict` True = turning, False = stopped, None = declines. The firmware is held in
    SPINUP and `link.freq` walks from 2 Hz up past the alias limit, which is what makes
    the trigger commit. Returns the commands sent.
    """

    gains = json.load(open(HERE / "hover_controller.json"))
    # capture_check is OFF in production (its window is too narrow to trust), but the
    # logic still has to be correct for when it is switched on deliberately.
    cfg = R.RunConfig(dry_run=True, takeoff=False, enable_freq_cmd=False,
                      capture_check=True, log=str(Path(log_dir) / "cap.log"))
    ctrl = tuple(DiscreteHoverController(gains, Profile.hold()) for _ in range(2))
    link = R.CommandLink(None, True, cfg.log, takeoff_cmd=ramp.seq_lines(cfg.segments))
    sent = []
    link.send = lambda c: sent.append(c)

    class W:                       # stands in for SpinWitness
        alias_limit_hz = 3.0
        field_hz = None
        turning = True             # ALWAYS true: a rocking rotor is "moving" too, which
                                   # is exactly why the trigger must not key off this
        def __init__(self, rev, n):
            self.drift_rev, self.n = rev, n

    viz = PanelViz({})

    def ticks():
        rev = 0.0
        for i in range(n_low + 30):
            # freq under the alias limit for n_low frames, then above it
            link.freq = 2.0 if i < n_low else 12.0
            link.state = 1         # SPINUP: the ramp phase
            if i < n_low and verdict is not None:
                # `verdict` = did the rotor actually turn. True accumulates revolutions at
                # the commanded rate; False rocks and nets out to nothing. None means the
                # witness never got blade signal at all, so `n` never advances either.
                rev += (2.0 / 60.0) if verdict else 0.0
            n = 0 if verdict is None else i
            yield Tick(t=i / 60.0, xyz_mm=np.array([0.0, 0.0, 60.0]), pose=None,
                       frames=None, lost=0, viz=viz, spin=W(rev, n))

    try:
        R.controller_loop(ticks(), link, ctrl, cfg, None)
    finally:
        link.close()
    return sent


def demo(log_dir="/tmp"):
    # The profile is three lines; `seq=go` is the one that starts the coils turning,
    # so it is what "the takeoff was commanded exactly once" means on the wire.
    is_takeoff = lambda c: c == "seq=go"

    sent = run({PRESS_AT: ("estop",)}, SPINUP, True, log_dir)
    stops = [i for i, c in sent if c == "stop"]
    assert stops and stops[0] == PRESS_AT, stops
    assert not [c for _, c in sent if is_takeoff(c)], sent
    print(f"stop during spin-up : sent on frame {stops[0]}, no takeoff after it")

    sent = run({PRESS_AT: ("estop", "takeoff")}, SPINUP, False, log_dir)
    assert any(c == "stop" for i, c in sent if i == PRESS_AT), sent
    assert not [c for _, c in sent if is_takeoff(c)], sent
    print("stop + takeoff      : stop sent, takeoff dropped, none on any later frame")

    sent = run({PRESS_AT: ("takeoff",)}, FLIGHT, False, log_dir)
    assert [i for i, c in sent if is_takeoff(c)] == [PRESS_AT], sent
    # ...and the whole profile went out ahead of it, in order, on that same frame. A
    # partial send would leave the sequencer half-loaded and `seq=go` would then ramp a
    # shape nobody chose -- which no assertion on `seq=go` alone can see.
    profile = [c for i, c in sent if i == PRESS_AT and c.startswith("seq=")]
    assert profile == ramp.seq_lines(ramp.DEFAULT), profile
    print(f"takeoff in FLIGHT   : honoured, all {len(profile)} profile lines on one frame")

    # 4. Nothing pressed, nothing started.
    sent = run({}, SPINUP, False, log_dir)
    assert not [c for _, c in sent if is_takeoff(c)], sent
    print("no presses          : zero takeoff commands")

    # 5. The CSV's spin column. A witness that declines (aliased, or no blade signal)
    # must leave the cell BLANK: a manufactured `stopped` there reads as a rotor
    # confirmed still, which is the opposite conclusion and the one the whole capture
    # test turns on. `turning` is valid at any speed; `stopped` only below fps/8.
    witness = lambda t: type("W", (), {"turning": t})()
    assert R._spin_state(None) == ""
    assert R._spin_state(witness(None)) == ""
    assert R._spin_state(witness(True)) == "turning"
    assert R._spin_state(witness(False)) == "stopped"
    print("spin column         : blank when the witness declines, never a false stopped")

    # 6. The capture trigger. Capture happens at 2-4 Hz, under the witness's alias limit,
    # which is the one band where a `stopped` verdict is trustworthy. A ramp whose rotor
    # was never caught must ABORT rather than drive a stationary rotor for another 15-45 s
    # of coil heat -- but a ramp that did catch, and one the witness could not judge,
    # must both be left alone. These three must not collapse into each other.
    # `finally` always sends one `stop`, so presence alone proves nothing -- an abort is
    # an EXTRA stop sent while still in SPINUP. Count them.
    stopped_run = run_capture(False, log_dir)   # rotor rocks: drift stays ~0
    assert stopped_run.count("stop") == 2, ("rocking rotor must abort", stopped_run)
    turning_run = run_capture(True, log_dir)     # rotor turns: drift tracks command
    assert turning_run.count("stop") == 1, ("a turning rotor must NOT abort", turning_run)
    silent_run = run_capture(None, log_dir)      # witness declines: no accumulation seen
    assert silent_run.count("stop") == 1, ("unknown must not abort", silent_run)
    print("capture trigger     : aborts when phase does not accumulate, not on a motion latch")

    # 7. Short pose gaps must still command. Salvaged from `test_dropout.py`, whose
    # long-gap half tested a land that was deliberately removed -- this half tests
    # behaviour that is still live and, with `pred.stale` now ignored, unguarded.
    for gap in (1, 3, 5):
        sent, lost_at = run_coast(gap, log_dir)
        during = [i for i, c in sent if c.startswith("freq=") and i in lost_at]
        assert len(during) == gap, (gap, during, sorted(lost_at))
    print("coast on dropout    : freq= keeps flowing through 1, 3 and 5 lost frames")

    # 8. The pose feed, threaded. The live path runs `stereo_frames` on its own thread so
    # the control clock is not held to the pose rate; newest-only is the contract, and a
    # producer that dies has to surface rather than leave the loop spinning on a stale
    # pose forever.
    import time
    feed = R._PoseFeed(iter(range(1000)), threaded=True)
    for _ in range(200):
        if feed.done:
            break
        time.sleep(0.005)
    assert feed.done, "producer never finished"
    assert feed.read() == (999, 1000), feed.read()   # newest only, all counted

    def angry():
        yield 1
        raise RuntimeError("producer died")

    feed = R._PoseFeed(angry(), threaded=True)
    for _ in range(200):
        if feed.exc is not None:
            break
        time.sleep(0.005)
    assert isinstance(feed.exc, RuntimeError), feed.exc
    feed.close()
    print("pose feed           : newest-only, counted, and a dead producer surfaces")
    # 7. The tracking-loss land. Added 2026-09-01 after a run drove the coils for 10 084
    # ticks (~50 s) with zero fixes -- the robot departed the tracked volume sideways at
    # ~104 Hz and nothing noticed, every other backstop having been stripped. This is the
    # only automatic land on the host, so it gets the only test that can catch its loss.
    sent = run_lost(N, log_dir)                       # never regains a fix
    stops = [i for i, c in sent if c == "stop"]
    assert stops, f"coils driven blind and NOTHING stopped them: {sent}"
    print(f"tracking-loss land  : stopped on frame {stops[0]} after the fixes stopped")

    # ...and it must not fire on a rig that is tracking. A land that trips on a healthy
    # run gets disabled by the operator, and then it is not a backstop at all.
    sent = run_lost(0, log_dir)                       # every frame has a fix
    # A clean exit always sends hover -> land -> stop, so "no stop at all" would be the
    # wrong assertion. What must not happen is a stop BEFORE that shutdown.
    cmds = [c for _, c in sent]
    shutdown = cmds.index("land") if "land" in cmds else len(cmds)
    assert "stop" not in cmds[:shutdown], sent
    print("tracking-loss land  : silent while fixes keep arriving")

    print("self-check PASS: every latch read exactly once per tick")


if __name__ == "__main__":
    demo(*sys.argv[1:])
