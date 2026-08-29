#!/usr/bin/env python3
"""
Real-time hover controller: closes the position loop around src/main_flight.cpp.
Takeoff identifies f_hover from the firmware's spin-up ramp, then sweeps upward in FLIGHT if
the robot is still down; z is then z_track.ZTracker and lateral is polar az/mag. Details and
the sensitivity argument: theory.md 6.5.  Usage: fly(dry_run=True) for the stub rehearsal.

Where:
    ticks     iterable of live_viz.Tick: xyz_mm is the datum frame in mm, +z up, None if lost
    ctrl      (x, y) DiscreteHoverController pair, one per lateral axis, K row 0 each
    loop      metres and seconds internally; viz.setpoint is mm; serial is deg / 0..1 / Hz
    armed     nothing is commanded until viz.armed; viz.mag_max starts 0, so lateral is OFF
    safety    watchdog lands after `timeout` with no fix; SIGINT lands then stops; stop on
              any exception; land + stop in `finally`
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import predictor
import z_track
from reference_profiles import Profile
from simulate_hover import DiscreteHoverController, Scenario, simulate

VIZ_DIR = Path(__file__).resolve().parent.parent / "viz"

# Pad datum: the median z before the robot leaves the ground, so the setpoint is measured
# from where it actually sat rather than from the datum's zero.
PAD_SAMPLES = 20

AZ_DEADBAND_DEG = 1.0        # below this a resend is latency spent to say nothing

FLIGHT = 2  # main_flight.cpp State: IDLE=0 SPINUP=1 FLIGHT=2 LANDING=3 OFF=4

# Anchored to line start or space/pipe so "!freq=163.00", a rejection, is not read as a
# frequency the coils are running at.
_FREQ_RE = re.compile(r"(?:^|[ |])freq=(-?\d+(?:\.\d+)?)")
_STATE_RE = re.compile(r"(?:^|[ |])state=(\d)")


def _live_viz():
    """Import controller/viz/live_viz by path. Lazy, so a missing viser breaks only camera runs."""

    if str(VIZ_DIR) not in sys.path:
        sys.path[:0] = [str(VIZ_DIR)]
    import live_viz

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


def replay_ticks(csv_path: str, viz):
    """CSV playback as Ticks. Columns t,x,z in s and m; y is 0 if absent (pre-stereo logs)."""

    Tick = _live_viz().Tick
    with open(csv_path) as f:
        rows = [
            (float(r["t"]), float(r["x"]), float(r.get("y") or 0.0), float(r["z"]))
            for r in csv.DictReader(f)
        ]
    t0 = time.monotonic()
    for t, x, y, z in rows:
        _pace(t0, t)
        xyz = np.array([x * 1e3, y * 1e3, z * 1e3])
        yield Tick(t=time.monotonic(), xyz_mm=xyz, pose=None, frames=None, lost=0, viz=viz)


class CommandLink:
    """Serial link to main_flight.cpp, or a printing stand-in for dry_run."""

    def __init__(self, port: str | None, dry_run: bool, log_path: str):
        self.dry = dry_run
        self.log = open(log_path, "w")
        self.freq: float | None = None  # last frequency the firmware reported, Hz
        self.state: int | None = None  # last firmware State it reported
        if dry_run:
            self.comm = None
        else:
            from link import SerialComm  # local import: pyserial only if needed

            self.comm = SerialComm(port=port)
            self.comm.reset_device()  # reboot firmware to IDLE
            time.sleep(1.5)  # wait out the boot banner

    def send(self, cmd: str) -> None:
        stamp = f"[{time.monotonic():.3f}] -> {cmd}"
        print(stamp)
        self.log.write(stamp + "\n")
        if self.comm:
            self.comm.handle_serial_comm(cmd)

    def drain(self) -> None:
        """Log pending telemetry, keeping the last freq= (Hz) and state= seen; unmatched is kept."""

        if not self.comm:
            return
        # driveTelemetry prints "t=<ms> freq=<hz> | I[A]: ..." at 2 Hz in every state. Lines
        # arrive truncated and interleaved, so a non-match leaves the last good value standing.
        while (line := self.comm.handle_serial_comm()) is not None:
            self.log.write(line + "\n")
            if m := _STATE_RE.search(line):
                self.state = int(m.group(1))
            if m := _FREQ_RE.search(line):
                self.freq = float(m.group(1))

    def close(self) -> None:
        self.log.close()
        if self.comm:
            self.comm.close()


def takeoff_to_flight(ticks, link: CommandLink, args) -> tuple[float | None, bool]:
    """Ramp the firmware to FLIGHT and return (pad_z_mm, ok); ok False must stop the flight.

    No liftoff search. On the pad the ground pins z, so the altitude error stays at its
    maximum and ZTracker's f_hat ramps at gamma*e until the robot lifts, then settles at the
    true f_hover as e goes to zero. The estimator is the identification: see theory.md 6.6.
    """

    link.send("takeoff")
    if link.comm is None:
        print("dry run: no firmware to ramp")
        return None, True

    pad: list[float] = []
    deadline = time.monotonic() + args.spinup_s + args.search_s
    print(f"spin-up: {args.spinup_s:.0f}s firmware ramp, then the loop arms on the pad")

    for tick in ticks:
        link.drain()
        tick.viz.push(
            tick.pose, u=(0.0, 0.0, link.freq or 0.0, args.throttle, 0.0),
            t=tick.t, frames=tick.frames, lost=tick.lost, state=link.state,
        )
        if tick.xyz_mm is not None and len(pad) < PAD_SAMPLES:
            pad.append(float(tick.xyz_mm[2]))
        if link.state == FLIGHT:
            pad_z = float(np.median(pad)) if pad else None
            print(f"FLIGHT at {link.freq or float('nan'):.1f} Hz, pad z = "
                  + ("unknown" if pad_z is None else f"{pad_z:.1f} mm"))
            return pad_z, True
        if time.monotonic() > deadline:
            print(f"never reached FLIGHT in {args.spinup_s + args.search_s:.0f}s")
            return None, False
    print("the position stream stopped before FLIGHT")
    return None, False


def controller_loop(ticks, link: CommandLink, ctrl, args, ztrk=None) -> None:
    """Fly `ticks` on a fixed ts cadence. `ctrl` is the (x, y) controller pair; ztrk owns z."""

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
    try:
        if args.takeoff:
            pad_z, ok = takeoff_to_flight(ticks, link, args)
            if not ok:
                land("never reached FLIGHT")
                return
            if pad_z is not None:
                args.pad_z_mm = pad_z
            link.send(f"throttle={args.throttle:.0f}")
            if ztrk is not None:
                print(
                    f"altitude: f_hat starts at {ztrk.f_hat:.1f} Hz and adapts over "
                    f"[{ztrk.f_lo:.0f}, {ztrk.f_hi:.0f}] Hz; the pad holds the error up "
                    "until it lifts"
                )

        print("closed loop engaged")
        t_start = time.monotonic()
        last_fix = t_start
        pred = predictor.StatePredictor()
        # What the coils were last told, and the current hover estimate. Passing one for
        # both would make every prediction a constant-velocity coast.
        f_cmd_last = f_hat_last = ztrk.f_hat if ztrk is not None else ctrl[0].f_hover
        last_step, next_step = t_start - ts, t_start
        prev_az = None
        for tick in ticks:
            if landed:
                break
            link.drain()
            viz, now = tick.viz, time.monotonic()
            # Polled once per iteration, as live_viz's own loops read viz.thresh. `land` is a
            # one-shot latch, so it must be read exactly once.
            armed, mag_max, gain = viz.armed, viz.mag_max, viz.gain_scale
            sp_x, sp_y, sp_z = (v * 1e-3 for v in viz.setpoint)  # viz is mm, the loop is m
            if viz.land:
                land("viz land button")
                break

            # Push the miss too, or the trace interpolates straight over a dropout.
            if tick.xyz_mm is None:
                viz.push(tick.pose, t=tick.t, frames=tick.frames, lost=tick.lost)
                if not pred.initialised or pred.stale or now - last_fix > args.timeout:
                    land(f"no position fix for {now - last_fix:.2f}s")
                    continue
                # Propagate on the frequency we commanded. A kinematic coast costs about
                # twice the error over the same gap: predictor.py's table, theory.md 6.5.
                xyz_mm = pred.predict(
                    predictor.Command(f_cmd_last, f_hat_last), now - last_step
                )
            else:
                xyz_mm = pred.update(tick.xyz_mm, t=tick.t)
                last_fix = now
            if now < next_step:
                viz.push(tick.pose, t=tick.t, frames=tick.frames, lost=tick.lost)
                continue
            # A grid, not a "ts since last step" gate: against a source running at exactly ts
            # that gate aliases and the loop silently runs at 19 Hz. K was designed at ts and
            # the cameras deliver 60 fps, so a per-frame step would double every gain.
            next_step += ts
            if next_step < now:
                next_step = now + ts  # fell behind: resync rather than fire a catch-up burst
            dt = min(now - last_step, 5.0 * ts)
            last_step, t = now, now - t_start

            x_m, y_m, z_m = np.asarray(xyz_mm, dtype=float) * 1e-3
            # The sliders move the origin and the profile moves relative to it, so the
            # controller is fed the offset measurement.
            ux, f_lqr = ctrl_x.step(t, x_m - sp_x, z_m - sp_z)
            uy, _ = ctrl_y.step(t, y_m - sp_y, z_m - sp_z)
            ux, uy = gain[0] * ux, gain[0] * uy

            # Polar, because the actuator is: one azimuth and one magnitude, never a signed
            # magnitude with a 180 degree flip hiding in it.
            az = math.degrees(math.atan2(uy, ux)) % 360.0
            mag = min(math.hypot(ux, uy), mag_max)

            if ztrk is not None:
                f_z = ztrk.step(t, z_m, dt, z_target=sp_z)
                f_trim = ztrk.f_hat
                if ztrk.stepout:
                    land("step-out: all available torque commanded while z falls")
            else:
                f_z, f_trim = f_lqr, ctrl_x.f_hover
            f_field = f_trim + gain[1] * (f_z - f_trim)  # scales authority, never trim
            f_cmd_last, f_hat_last = f_field, f_trim

            if armed:
                if prev_az is None or abs((az - prev_az + 180.0) % 360.0 - 180.0) > AZ_DEADBAND_DEG:
                    link.send(f"az={az:.0f}")
                    prev_az = az
                link.send(f"mag={mag:.3f}")
                if args.enable_freq_cmd:
                    link.send(f"freq={f_field:.2f}")
                else:
                    link.log.write(
                        f"[{now:.3f}] (freq={f_field:.2f} not sent, "
                        f"pass enable_freq_cmd to close the altitude loop)\n"
                    )
            else:
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
            )
    except Exception as e:  # noqa: BLE001 -- de-energize on ANY failure
        print(f"error: {e}", file=sys.stderr)
        link.send("stop")
        raise
    finally:
        land("loop exit")
        link.send("stop")
        # Run the generator's own finally now: that is where stereo_frames closes its cameras
        # and its viewer, and a traceback holding a reference would otherwise keep them open.
        close = getattr(ticks, "close", None)
        if close:
            close()


@dataclass
class RunConfig:
    """Everything `controller_loop` reads, in one place."""

    source: str = "stub"  # "stub" | "replay" | "camera"
    camera: str = "camera:0,camera:1"  # stereo pair for live_viz.stereo_frames
    width: int = 1280
    height: int = 800
    rig: str = None  # stereo rig calibration; None takes the saved one
    axes: tuple = ("x", "y", "z")  # datum-frame axis names, for the viz labels
    replay_csv: str = None
    profile: str = None  # reference profile JSON; None holds at the viz setpoint
    port: str = None
    log: str = "hover_run.log"
    timeout: float = 0.5  # watchdog: land after this long without a fix
    takeoff: bool = True
    spinup_s: float = 33.0  # firmware SPINUP_MS + margin
    search_s: float = 60.0  # grace window past the ramp before FLIGHT is called lost
    pad_z_mm: float = 0.0   # measured at takeoff; setpoints are relative to it
    throttle: float = 80.0
    enable_freq_cmd: bool = False  # False = telemetry-only dress rehearsal
    dry_run: bool = False  # print commands instead of opening serial
    gains: str = None  # defaults to hover_controller.json beside this file
    waypoints: str = None  # z waypoints; None reverts z to the LQR
    viz: bool = False  # live 3-D view at http://localhost:<viz_port>
    viz_port: int = 8080

    def __post_init__(self):
        here = os.path.dirname(__file__)
        if self.gains is None:
            self.gains = os.path.join(here, "hover_controller.json")


def fly(cfg=None, **kw):
    """Run the hover controller. ``fly(dry_run=True)`` is the stub-source dress rehearsal."""

    cfg = cfg or RunConfig(**kw)

    with open(cfg.gains) as f:
        gains = json.load(f)
    profile = Profile.from_json(cfg.profile) if cfg.profile else Profile.hold()
    # One instance per lateral axis, each with its own velocity estimator and error
    # integrator, which is what K row 0 was designed against. ctrl_y's vertical half is
    # computed and dropped: two matrix rows against a second control law.
    ctrl = (DiscreteHoverController(gains, profile), DiscreteHoverController(gains, profile))

    lv = _live_viz()
    own_viz = None
    if cfg.source == "camera":
        # stereo_frames builds its own viewer and hands it out on every Tick: it needs one
        # during datum priming, before this loop exists.
        ticks = lv.stereo_frames(
            specs=cfg.camera, rig_path=cfg.rig, width=cfg.width, height=cfg.height,
            port=cfg.viz_port, axes=tuple(cfg.axes), label="hover",
        )
    else:
        own_viz = lv.make_viz(
            enabled=cfg.viz, port=cfg.viz_port, axes=tuple(cfg.axes), label="hover"
        )
        if cfg.source == "replay":
            if not cfg.replay_csv:
                raise ValueError("source='replay' needs replay_csv")
            ticks = replay_ticks(cfg.replay_csv, own_viz)
        else:
            ticks = stub_ticks(gains, own_viz)

    ztrk = None
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

    link = CommandLink(cfg.port, cfg.dry_run, cfg.log)
    try:
        controller_loop(ticks, link, ctrl, cfg, ztrk)
    finally:
        link.close()
        if own_viz is not None:
            own_viz.close()
        print(f"done -> {cfg.log}")
    return cfg.log
