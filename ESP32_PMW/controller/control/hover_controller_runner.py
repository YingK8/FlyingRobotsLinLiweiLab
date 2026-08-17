#!/usr/bin/env python3
"""
Real-time hover-controller runner -- closes the position loop around
src/main_flight.cpp using the state-space controller designed by
controller/control/design_hover_lqr.py (gains: controller/control/hover_controller.json).

Replaces the open-loop controller/control/flight_controller.py.
Position sources are pluggable behind PositionSource: a simulation replay
(--source stub), CSV playback (--source replay), or the live camera
(--source camera), which runs the whole stage 1-3 pipeline behind
CameraSource.read() and hands back (t, x_m, z_m).

Control law per frame (same DiscreteHoverController code path that passed
the simulate_hover.py scenarios):
    u = u_trim + u_ff(t) - K [x_hat - x_ref(t) ; q]
Lateral output: az=<axis> (u>=0) or az=<axis+180> (u<0), mag=|u_lat|.
Vertical output: f_field is computed and LOGGED always, but only SENT with
--enable-freq-cmd -- main_flight.cpp has no freq= command yet (deferred
firmware step; until then the vertical channel is telemetry-only).

Safety: measurement watchdog (no fix for --timeout s -> "hover" then
"land"); first SIGINT -> "land", second -> "stop"; "stop" also sent on any
unhandled exception.

Usage:
  hover_controller_runner.fly(dry_run=True)                    # stub source, no serial
  hover_controller_runner.fly(source="replay", replay_csv="run.csv",
                              port="/dev/ttyUSB0", log="hover_run.log")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import math
import os
import signal
import sys
from pathlib import Path
import time
from abc import ABC, abstractmethod

import numpy as np

import z_track
from reference_profiles import Profile
from simulate_hover import DiscreteHoverController, Scenario, simulate


class PositionSource(ABC):
    """
    Provides drone position fixes. read() is non-blocking: returns
        (t_monotonic_s, x_m, z_m) when a new fix is available, else None.
    """

    @abstractmethod
    def read(self) -> tuple[float, float, float] | None: ...

    def close(self) -> None:
        pass


class StubSource(PositionSource):
    """
    Replays the measurement stream of a simulate_hover run in real time
        (open loop -- commands do not feed back). Lets the full runner pipeline
        execute end-to-end before the camera exists.
    """

    def __init__(self, gains: dict, duration: float = 10.0):
        sc = Scenario("stub", x0_mm=10, z0_mm=10, duration=duration)
        out = simulate(sc, gains)
        self._t = out["t"]
        self._x, self._z = out["x"], out["z"]
        self._i = 0
        self._t0 = time.monotonic()

    def read(self):
        if self._i >= len(self._t):
            return None  # stream exhausted -> watchdog will land
        now = time.monotonic()
        if now - self._t0 < self._t[self._i]:
            return None
        i, self._i = self._i, self._i + 1
        return now, float(self._x[i]), float(self._z[i])


class ReplaySource(PositionSource):
    """
    CSV playback (columns: t,x,z in seconds/meters), real-time paced.
    """

    def __init__(self, path: str):
        with open(path) as f:
            rows = [
                (float(r["t"]), float(r["x"]), float(r["z"])) for r in csv.DictReader(f)
            ]
        self._rows = rows
        self._i = 0
        self._t0 = time.monotonic()

    def read(self):
        if self._i >= len(self._rows):
            return None
        now = time.monotonic()
        t, x, z = self._rows[self._i]
        if now - self._t0 < t:
            return None
        self._i += 1
        return now, x, z


class CameraSource(PositionSource):
    """
    Live position from the camera: the seam where vision meets control.

        The whole of stages 1-3 sits behind this one method. `camera.sources` opens
        the device and grabs on its own thread, `pose.estimator` turns each frame
        into a 5-DOF pose, and `pose.filter` smooths position and supplies velocity.
        This class does nothing but adapt that to `PositionSource`'s contract, and
        the adaptation is two specific things, both easy to get silently wrong:

        **Units.** The contract is metres. The pose stack is millimetres throughout
        -- `RADIUS_MM`, `xyz_mm`, `SIGMA_LATERAL_MM`. The conversion happens here,
        once, at the boundary. A missed factor of 1000 does not raise; it produces a
        controller that thinks the robot is a kilometre away and commands
        accordingly.

        **Which axes.** The controller is planar: it wants `x` (lateral) and `z`
        (height). The camera's frame is `x` right, `y` down, `z` along the optical
        axis, so *the camera's z is depth, not height*. For a side-on camera looking
        horizontally at the robot, height is **-y**: negated because the image y axis
        points down and height points up. `axes=` makes that explicit rather than
        assumed, because the right mapping depends on where the camera is mounted and
        getting it wrong yields a control loop that is stable and flying the wrong
        axis.

        **Non-blocking.** `read()` must return rather than wait. A frame that has not
        arrived, and a frame the segmenter could not use, both return ``None`` -- the
        caller's watchdog treats a gap the same either way, and returning a stale
        repeat instead would hide a lost robot as a stationary one.
    """

    def __init__(
        self,
        source="camera:0",
        width=1280,
        height=800,
        axes=("x", "-y"),
        intrinsics=None,
        use_filter=True,
        timeout=0.0,
    ):
        HERE = Path(__file__).resolve().parent
        sys.path[:0] = [
            str(HERE.parent / "pose"),
            str(HERE.parent / "calib"),
            str(HERE.parent / "camera"),
        ]
        import sources
        from estimator import PoseEstimator, load_intrinsics
        from filter import PoseFilter

        K, dist = load_intrinsics(intrinsics) if intrinsics else load_intrinsics()
        self._src = sources.open_source(source, width=width, height=height)
        self._est = PoseEstimator(camera_matrix=K, dist_coeffs=dist)
        self._filt = PoseFilter() if use_filter else None
        self._axes = axes
        self._timeout = timeout
        self.n_lost = 0

    def _component(self, xyz_mm, spec):
        i = {"x": 0, "y": 1, "z": 2}[spec[-1]]
        v = float(xyz_mm[i]) / 1000.0  # mm -> m, once, here
        return -v if spec.startswith("-") else v

    def read(self):
        item = self._src.read(self._timeout) if self._timeout else self._src.read()
        if item is None:
            return None
        t_cap, frame = item
        pose = self._est.update(frame, t=t_cap)
        if pose is None:
            self.n_lost += 1
            return None
        xyz = pose.xyz_mm
        if self._filt is not None:
            fused = self._filt.update(pose, t=t_cap)
            if fused is not None:
                xyz = fused[0]
        return (
            time.monotonic(),
            self._component(xyz, self._axes[0]),
            self._component(xyz, self._axes[1]),
        )

    def close(self):
        self._src.close()


class CommandLink:
    """
    Serial link to main_flight.cpp, or a printing stand-in for --dry-run.
    """

    def __init__(self, port: str | None, dry_run: bool, log_path: str):
        self.dry = dry_run
        self.log = open(log_path, "w")
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
        """
        Pull pending telemetry lines into the log (non-blocking).
        """

        if not self.comm:
            return
        while (line := self.comm.handle_serial_comm()) is not None:
            self.log.write(line + "\n")

    def close(self) -> None:
        self.log.close()
        if self.comm:
            self.comm.close()


def controller_loop(
    src: PositionSource,
    link: CommandLink,
    ctrl: DiscreteHoverController,
    args,
    ztrk: "z_track.ZTracker | None" = None,
) -> None:
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

    if args.takeoff:
        link.send("takeoff")
        print(f"spin-up: waiting {args.spinup_s:.0f}s (firmware ramp)...")
        t_end = time.monotonic() + args.spinup_s
        while time.monotonic() < t_end:
            link.drain()
            time.sleep(0.05)
        link.send(f"throttle={args.throttle:.0f}")

    print("closed loop engaged")
    t_start = time.monotonic()
    last_fix = t_start
    prev_az_flip = None
    try:
        while not landed:
            link.drain()
            fix = src.read()
            now = time.monotonic()
            if fix is None:
                if now - last_fix > args.timeout:
                    land(f"no position fix for {args.timeout}s")
                time.sleep(0.002)
                continue
            _, x_m, z_m = fix
            mag, f_lqr = ctrl.step(now - t_start, x_m, z_m)

            # Lateral (mag) stays on the LQR. The vertical channel goes through
            # ai/z_track.py instead: exact lift inversion + torque-ratio clamps,
            # tracking the z waypoint file. See its module docstring for why the
            # LQR's linearized 2g/f_hover gain is the wrong tool at takeoff.
            if ztrk is not None:
                f_field = ztrk.step(now - t_start, z_m, max(now - last_fix, 1e-3))
            else:
                f_field = f_lqr
            last_fix = now

            az = args.az_axis_deg if mag >= 0 else (args.az_axis_deg + 180.0) % 360.0
            if az != prev_az_flip:
                link.send(f"az={az:.0f}")
                prev_az_flip = az
            link.send(f"mag={abs(mag):.3f}")
            if args.enable_freq_cmd:
                link.send(f"freq={f_field:.2f}")
            else:
                link.log.write(
                    f"[{now:.3f}] (freq={f_field:.2f} not sent -- "
                    f"pass --enable-freq-cmd to close the loop)\n"
                )
    except Exception as e:  # noqa: BLE001 -- de-energize on ANY failure
        print(f"error: {e}", file=sys.stderr)
        link.send("stop")
        raise
    finally:
        land("loop exit")
        link.send("stop")


@dataclass
class RunConfig:
    """
    Everything `controller_loop` reads, in one place.

        A dataclass rather than loose keyword arguments because the loop takes it
        whole and passes it down; a namespace assembled ad hoc is how a field gets
        read that was never set.
    """

    source: str = "stub"  # "stub" | "replay" | "camera"
    camera: str = "camera:0"
    width: int = 1280
    height: int = 800
    axes: tuple = ("x", "-y")  # which pose axes are (lateral, height)
    replay_csv: str = None
    profile: str = None  # reference profile JSON; None holds at origin
    port: str = None
    log: str = "hover_run.log"
    az_axis_deg: float = 0.0  # lab azimuth of the controlled lateral axis
    timeout: float = 0.5  # watchdog: land after this long without a fix
    takeoff: bool = True
    spinup_s: float = 33.0  # firmware SPINUP_MS + margin
    throttle: float = 80.0
    enable_freq_cmd: bool = False  # False = telemetry-only dress rehearsal
    dry_run: bool = False  # print commands instead of opening serial
    gains: str = None  # defaults to hover_controller.json beside this file
    waypoints: str = None  # z waypoints; None reverts z to the LQR

    def __post_init__(self):
        here = os.path.dirname(__file__)
        if self.gains is None:
            self.gains = os.path.join(here, "hover_controller.json")


def fly(cfg=None, **kw):
    """
    Run the hover controller. ``fly()`` is the stub-source dress rehearsal.

        Pass a `RunConfig`, or keyword overrides of its fields. Defaults are the safe
        ones: the stub source, no frequency commands, and the watchdog armed.
    """

    cfg = cfg or RunConfig(**kw)

    with open(cfg.gains) as f:
        gains = json.load(f)
    profile = Profile.from_json(cfg.profile) if cfg.profile else Profile.hold()
    ctrl = DiscreteHoverController(gains, profile)

    if cfg.source == "stub":
        src: PositionSource = StubSource(gains)
    elif cfg.source == "replay":
        if not cfg.replay_csv:
            raise ValueError("source='replay' needs replay_csv")
        src = ReplaySource(cfg.replay_csv)
    else:
        src = CameraSource(cfg.camera, cfg.width, cfg.height, axes=tuple(cfg.axes))

    ztrk = None
    if cfg.waypoints:
        times, heights = z_track.load_waypoints(cfg.waypoints)
        # Constructing ZTracker computes f_ceiling, which raises if the torque
        # calibration leaves no headroom at f_hover. Fail here, on the bench,
        # rather than after the robot is already spinning.
        ztrk = z_track.ZTracker(times, heights)
        print(
            f"altitude: {len(times)} waypoints from {cfg.waypoints}, "
            f"f_hover={ztrk.lim.f_hover:.0f} Hz, "
            f"f_ceiling={ztrk.lim.f_ceiling():.1f} Hz, "
            f"a_max={ztrk.a_ceiling:.2f} m/s^2"
        )

    link = CommandLink(cfg.port, cfg.dry_run, cfg.log)
    try:
        controller_loop(src, link, ctrl, cfg, ztrk)
    finally:
        src.close()
        link.close()
        print(f"done -> {cfg.log}")
    return cfg.log
