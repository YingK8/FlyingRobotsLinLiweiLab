#!/usr/bin/env python3
"""Real-time hover-controller runner -- closes the position loop around
src/main_flight.cpp using the state-space controller designed by
ai/design_hover_lqr.py (gains: ai/hover_controller.json).

Replaces the open-loop ai/flight_controller.py once the camera exists.
Position sources are pluggable behind PositionSource: today a simulation
replay (--source stub) or CSV playback (--source replay); the camera team
implements CameraSource.read() and everything else stays untouched.

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
  uv run python ai/hover_controller_runner.py --source stub --dry-run
  uv run python ai/hover_controller_runner.py --source replay --replay-csv run.csv \
      --port /dev/ttyUSB0 --log hover_run.log
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from abc import ABC, abstractmethod

import numpy as np

from reference_profiles import Profile
from simulate_hover import DiscreteHoverController, Scenario, simulate


class PositionSource(ABC):
    """Provides drone position fixes. read() is non-blocking: returns
    (t_monotonic_s, x_m, z_m) when a new fix is available, else None."""

    @abstractmethod
    def read(self) -> tuple[float, float, float] | None: ...

    def close(self) -> None:
        pass


class StubSource(PositionSource):
    """Replays the measurement stream of a simulate_hover run in real time
    (open loop -- commands do not feed back). Lets the full runner pipeline
    execute end-to-end before the camera exists."""

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
    """CSV playback (columns: t,x,z in seconds/meters), real-time paced."""

    def __init__(self, path: str):
        with open(path) as f:
            rows = [(float(r["t"]), float(r["x"]), float(r["z"]))
                    for r in csv.DictReader(f)]
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
    """TODO(camera team): return (time.monotonic(), x_m, z_m) per frame.
    See docs/pose_localization_project_context.md for the pose pipeline."""

    def __init__(self):
        raise NotImplementedError("camera position source not implemented yet")

    def read(self):
        raise NotImplementedError


class CommandLink:
    """Serial link to main_flight.cpp, or a printing stand-in for --dry-run."""

    def __init__(self, port: str | None, dry_run: bool, log_path: str):
        self.dry = dry_run
        self.log = open(log_path, "w")
        if dry_run:
            self.comm = None
        else:
            from serial_comm import SerialComm  # local import: pyserial only if needed
            self.comm = SerialComm(port=port)
            self.comm.reset_device()  # reboot firmware to IDLE
            time.sleep(1.5)           # wait out the boot banner

    def send(self, cmd: str) -> None:
        stamp = f"[{time.monotonic():.3f}] -> {cmd}"
        print(stamp)
        self.log.write(stamp + "\n")
        if self.comm:
            self.comm.handle_serial_comm(cmd)

    def drain(self) -> None:
        """Pull pending telemetry lines into the log (non-blocking)."""
        if not self.comm:
            return
        while (line := self.comm.handle_serial_comm()) is not None:
            self.log.write(line + "\n")

    def close(self) -> None:
        self.log.close()
        if self.comm:
            self.comm.close()


def controller_loop(src: PositionSource, link: CommandLink,
                    ctrl: DiscreteHoverController, args) -> None:
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
            last_fix = now
            _, x_m, z_m = fix
            mag, f_field = ctrl.step(now - t_start, x_m, z_m)

            az = args.az_axis_deg if mag >= 0 else (args.az_axis_deg + 180.0) % 360.0
            if az != prev_az_flip:
                link.send(f"az={az:.0f}")
                prev_az_flip = az
            link.send(f"mag={abs(mag):.3f}")
            if args.enable_freq_cmd:
                link.send(f"freq={f_field:.2f}")
            else:
                link.log.write(f"[{now:.3f}] (freq={f_field:.2f} not sent -- "
                               f"no firmware freq= command)\n")
    except Exception as e:  # noqa: BLE001 -- de-energize on ANY failure
        print(f"error: {e}", file=sys.stderr)
        link.send("stop")
        raise
    finally:
        land("loop exit")
        link.send("stop")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["stub", "replay", "camera"], default="stub")
    ap.add_argument("--replay-csv", help="CSV (t,x,z) for --source replay")
    ap.add_argument("--gains", default=os.path.join(os.path.dirname(__file__),
                                                    "hover_controller.json"))
    ap.add_argument("--profile", help="reference profile JSON "
                                      "(reference_profiles.py schema); default: hold at origin")
    ap.add_argument("--port", default=None)
    ap.add_argument("--log", default="hover_run.log")
    ap.add_argument("--az-axis-deg", type=float, default=0.0,
                    help="lab-frame azimuth of the controlled lateral axis")
    ap.add_argument("--timeout", type=float, default=0.5,
                    help="watchdog: land after this long without a fix [s]")
    ap.add_argument("--takeoff", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--spinup-s", type=float, default=33.0,
                    help="firmware SPINUP_MS + margin")
    ap.add_argument("--throttle", type=float, default=80.0)
    ap.add_argument("--enable-freq-cmd", action="store_true",
                    help="send freq= (requires the deferred firmware command)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands instead of opening serial")
    args = ap.parse_args()

    with open(args.gains) as f:
        gains = json.load(f)
    profile = Profile.from_json(args.profile) if args.profile else Profile.hold()
    ctrl = DiscreteHoverController(gains, profile)

    if args.source == "stub":
        src: PositionSource = StubSource(gains)
    elif args.source == "replay":
        if not args.replay_csv:
            sys.exit("--source replay requires --replay-csv")
        src = ReplaySource(args.replay_csv)
    else:
        src = CameraSource()

    link = CommandLink(args.port, args.dry_run, args.log)
    try:
        controller_loop(src, link, ctrl, args)
    finally:
        src.close()
        link.close()
        print(f"done -> {args.log}")


if __name__ == "__main__":
    main()
