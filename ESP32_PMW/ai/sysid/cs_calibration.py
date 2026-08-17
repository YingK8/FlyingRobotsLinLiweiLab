#!/usr/bin/env python3
"""
Calibrate the VNH5019 current-sense chain against the FNIRSI DPS-150 as a
reference ammeter. Produces corrected SENS[] values for src/drive_common.h.

Under test: i_meas[i] = SENS[i] * (mV_i - zeroMv_i) / 1000
(lib/PwmController/src/current_sense.cpp:35) -- zero offset, gain, linearity.

-------------------------------------------------------------------------------
WIRING -- read this before running anything
-------------------------------------------------------------------------------
1. JUMPER THE SERIES CAP HEADER (J4 Ch1 / J3 Ch2). The cap is in the main
   current path and BLOCKS DC by design (docs/PCB_Design_Documentation.md
   sec.6), so a DC calibration through the bridge reads zero once the cap
   charges (tau = R*C ~ 3ms). This is why a DC capture with the cap installed
   is not measurable.
2. Replace the coil with a NON-INDUCTIVE power resistor across the XT30
   (J10 Ch1 / J5 Ch2). ~2.7 ohm / 100 W gives ~4.4 A at 12 V, under the
   DPS-150's 5 A ceiling. Its exact value does not matter -- the DPS-150 is the
   reference and the CS chain is the device under test; the resistor only sets
   the operating current.
3. DPS-150 powers the VM/VBAT rail ONLY. Logic 3.3 V must come from the ESP32
   or a separate supply, or logic current lands in the reference reading.
4. Flash main_dc.cpp (passthrough, no balance) with a dc_calibration.json whose
   hold window is longer than the sweep. Drive ONE channel at a time: the
   DPS-150 measures total board current, so a second active channel is
   indistinguishable from the one being calibrated.

-------------------------------------------------------------------------------
MODES
-------------------------------------------------------------------------------
zero-drift   Test 1. Coils off, log raw CS for --duration. _adcZeroMv is seeded
             ONCE at boot from a floating baseline (current_sense.cpp:16); this
             checks whether that seed is still valid later. At SENS ~ 15 A/V,
             1 mV of drift is 0.015 A -- small-looking offsets are amps.

voltage      Test 2 (primary). Carrier parked at 100% = static HIGH
             (PwmController.cpp:521), so the load current is genuinely DC with
             no chopping. Sweeps SUPPLY VOLTAGE instead of duty, which keeps
             every switching artifact out of the gain measurement. The slope of
             i_meas vs I_dps is the SENS correction factor.

duty         Test 3. Supply voltage fixed, carrier duty stepped by the firmware
             schedule. The DPS-150 reads true average current either way, so a
             gain that disagrees with `voltage` mode means the ADC is aliasing
             the chopped CS waveform against the 50 ms EMA. This matters because
             carrier duty is exactly what the balance loop modulates -- a
             duty-dependent gain error makes the actuator corrupt its own sensor.
             Segments on the duty[%] field in telemetry, so no firmware change
             is needed.

All modes also record Test 4 for free: every channel's i_meas is logged, so the
inactive channels show any mux crosstalk (the throwaway-read hack at
current_sense.cpp:29).

Usage:
  uv run python ai/cs_calibration.py --mode zero-drift --duration 900 \
      --out data/cs_cal/zero_drift.json
  uv run python ai/cs_calibration.py --mode voltage --channel A \
      --v-max 12.0 --out data/cs_cal/gain_A.json
  uv run python ai/cs_calibration.py --mode duty --channel A --voltage 12.0 \
      --out data/cs_cal/duty_A.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

import numpy as np
import serial

try:
    from dps150 import DPS150, ProtectionState
except ImportError:
    sys.exit(
        "dps150 package not found. Install the driver library:\n"
        "  pip install -e ../../DPS-150-python-library"
    )

CHANNELS = ["A", "B", "C", "D"]

# Matches driveTelemetry() in src/drive_common.h. Same shape ai/record_serial.py
# and ai/pid_metrics.py parse, so logs stay compatible with the existing tools.
_TELEMETRY_RE = re.compile(
    r"I\[A\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)"
    r"(?:\s*\|\s*duty\[%\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+))?"
)


class TelemetryReader:
    """
    Non-blocking line reader. Accumulates raw bytes across polls and only
        returns complete lines -- a blocking readline() with a short timeout was
        confirmed on this hardware to split lines and desync the parser (see
        ai/record_serial.py).
    """

    def __init__(self, port: str, baud: int = 115200):
        self.ser = serial.Serial(port, baud, timeout=0)
        self._buf = b""
        self.latest: dict | None = None

    def poll(self) -> None:
        try:
            data = self.ser.read(4096)
        except serial.SerialException:
            return
        if not data:
            return
        self._buf += data
        while b"\n" in self._buf:
            raw, self._buf = self._buf.split(b"\n", 1)
            line = raw.decode("utf-8", errors="replace").strip()
            m = _TELEMETRY_RE.search(line)
            if not m:
                continue
            g = m.groups()
            self.latest = {
                "i_meas": {c: float(g[k]) for k, c in enumerate(CHANNELS)},
                "duty": (
                    {c: float(g[4 + k]) for k, c in enumerate(CHANNELS)}
                    if g[4] is not None
                    else None
                ),
                "line": line,
            }

    def drain(self, seconds: float) -> None:
        """
        Discard buffered telemetry so a sample reflects the new setpoint
                rather than whatever was in flight from the previous one.
        """

        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.poll()
            time.sleep(0.02)

    def close(self) -> None:
        self.ser.close()


async def sample(dev: DPS150, tel: TelemetryReader, n: int, gap: float) -> dict:
    """
    Average n paired (DPS, telemetry) reads. get_all() sleeps 100ms
        internally, giving ~10Hz -- well matched to the 2Hz telemetry and to the
        50ms CS filter.
    """

    dps_i, dps_v, dps_p = [], [], []
    meas = {c: [] for c in CHANNELS}
    duty = {c: [] for c in CHANNELS}
    mode = None
    for _ in range(n):
        st = await dev.get_all()
        if st.protection_state != ProtectionState.NORMAL:
            raise RuntimeError(f"supply tripped: {st.protection_state.value}")
        tel.poll()
        dps_i.append(st.output_current)
        dps_v.append(st.output_voltage)
        dps_p.append(st.output_power)
        mode = st.mode
        if tel.latest:
            for c in CHANNELS:
                meas[c].append(tel.latest["i_meas"][c])
                if tel.latest["duty"]:
                    duty[c].append(tel.latest["duty"][c])
        await asyncio.sleep(gap)

    if not any(meas[c] for c in CHANNELS):
        raise RuntimeError(
            "no telemetry parsed -- is main_dc.cpp flashed and its schedule "
            "still inside its hold window?"
        )
    return {
        "i_dps": float(np.mean(dps_i)),
        "i_dps_std": float(np.std(dps_i)),
        "v_dps": float(np.mean(dps_v)),
        "p_dps": float(np.mean(dps_p)),
        "mode": mode.value if mode else None,
        "i_meas": {c: float(np.mean(meas[c])) if meas[c] else None for c in CHANNELS},
        "duty": {c: float(np.mean(duty[c])) if duty[c] else None for c in CHANNELS},
    }


async def measure_quiescent(dev: DPS150, tel: TelemetryReader, args) -> float:
    """
    Board current with every channel off. The DPS-150 sees TOTAL board draw,
        so this baseline must come off every reading before it can be compared to a
        single channel's i_meas.
    """

    await dev.set_voltage(args.quiescent_voltage)
    await asyncio.sleep(args.dwell)
    s = await sample(dev, tel, args.samples, args.sample_gap)
    print(f"quiescent @ {s['v_dps']:.2f}V: {s['i_dps']:.4f} A")
    if s["i_dps"] > 0.5:
        print(
            "  ! quiescent > 0.5A -- is a channel still driven, or is logic 3.3V "
            "being fed from this supply? Both corrupt the reference.",
            file=sys.stderr,
        )
    return s["i_dps"]


async def run_zero_drift(dev: DPS150, tel: TelemetryReader, args) -> dict:
    """
    Test 1: is the boot-time zero seed still valid minutes later?
    """

    await dev.set_voltage(args.quiescent_voltage)
    await dev.enable_output()
    print(f"logging zero drift for {args.duration:.0f}s (coils must be OFF)")
    t0 = time.monotonic()
    trace = []
    while time.monotonic() - t0 < args.duration:
        s = await sample(dev, tel, args.samples, args.sample_gap)
        s["t"] = time.monotonic() - t0
        trace.append(s)
        print(
            f"  t={s['t']:6.1f}s  "
            + "  ".join(f"{c}={s['i_meas'][c]:+.3f}A" for c in CHANNELS)
        )
        await asyncio.sleep(args.drift_interval)

    drift = {}
    for c in CHANNELS:
        vals = [p["i_meas"][c] for p in trace if p["i_meas"][c] is not None]
        if len(vals) >= 2:
            drift[c] = {
                "start_a": vals[0],
                "end_a": vals[-1],
                "range_a": float(max(vals) - min(vals)),
                "std_a": float(np.std(vals)),
            }
    return {"mode": "zero-drift", "trace": trace, "drift": drift}


async def run_voltage_sweep(dev: DPS150, tel: TelemetryReader, args) -> dict:
    """
    Test 2: pure-DC gain and linearity. Carrier is parked at 100% by the
        firmware, so nothing is chopping and the only variable is supply voltage.
    """

    await dev.enable_output()
    quiescent = await measure_quiescent(dev, tel, args)

    setpoints = np.linspace(args.v_min, args.v_max, args.steps)
    points = []
    for direction, seq in (("up", setpoints), ("down", setpoints[::-1])):
        if direction == "down" and not args.updown:
            break
        for v_set in seq:
            await dev.set_voltage(float(v_set))
            await asyncio.sleep(args.dwell)
            tel.drain(args.drain)
            s = await sample(dev, tel, args.samples, args.sample_gap)
            s["v_set"] = float(v_set)
            s["direction"] = direction
            s["i_true"] = s["i_dps"] - quiescent
            points.append(s)
            act = s["i_meas"][args.channel]
            print(
                f"  V={s['v_dps']:5.2f}  I_dps={s['i_true']:6.3f}A  "
                f"i_meas[{args.channel}]={act:+6.3f}A  "
                f"err={act - s['i_true']:+.3f}A"
            )
    return {
        "mode": "voltage",
        "channel": args.channel,
        "quiescent_a": quiescent,
        "points": points,
    }


async def run_duty_sweep(dev: DPS150, tel: TelemetryReader, args) -> dict:
    """
    Test 3: fixed supply voltage, firmware steps the carrier duty. Segments
        on the reported duty[%] rather than on wall-clock, so no firmware change and
        no clock alignment is needed.
    """

    await dev.set_voltage(args.voltage)
    await dev.enable_output()
    quiescent = await measure_quiescent(dev, tel, args)

    print(
        f"recording {args.duration:.0f}s at {args.voltage}V -- the firmware "
        f"schedule should be stepping carrier duty on channel {args.channel}"
    )
    t0 = time.monotonic()
    raw = []
    while time.monotonic() - t0 < args.duration:
        s = await sample(dev, tel, args.samples, args.sample_gap)
        s["t"] = time.monotonic() - t0
        s["i_true"] = s["i_dps"] - quiescent
        raw.append(s)

    # Bin by observed duty. Duty is a commanded staircase so the readings
    # cluster tightly; rounding to --duty-bin recovers the steps without
    # needing to know the schedule.
    bins: dict[float, list[dict]] = {}
    for s in raw:
        d = s["duty"][args.channel]
        if d is None:
            continue
        bins.setdefault(round(d / args.duty_bin) * args.duty_bin, []).append(s)

    points = []
    for d in sorted(bins):
        grp = bins[d]
        if len(grp) < args.min_per_bin:
            continue  # transient between steps, not a settled plateau
        keep = grp[len(grp) // 2 :]  # drop the leading edge of each plateau
        points.append(
            {
                "duty": d,
                "n": len(keep),
                "i_true": float(np.mean([p["i_true"] for p in keep])),
                "v_dps": float(np.mean([p["v_dps"] for p in keep])),
                "i_meas": {
                    c: float(np.mean([p["i_meas"][c] for p in keep])) for c in CHANNELS
                },
            }
        )
        print(
            f"  duty={d:5.1f}%  I_dps={points[-1]['i_true']:6.3f}A  "
            f"i_meas[{args.channel}]={points[-1]['i_meas'][args.channel]:+6.3f}A"
        )
    return {
        "mode": "duty",
        "channel": args.channel,
        "voltage": args.voltage,
        "quiescent_a": quiescent,
        "points": points,
    }


def fit_gain(points: list[dict], channel: str, i_min: float) -> dict | None:
    """
    Fit i_meas = a*I_true + b. The firmware already applies SENS[channel], so
        a == 1.0 means the existing calibration is correct and the corrected value is
        SENS_old / a. b is residual zero-offset error in amps.
    """

    xs = [
        p["i_true"]
        for p in points
        if p["i_meas"][channel] is not None and p["i_true"] >= i_min
    ]
    ys = [
        p["i_meas"][channel]
        for p in points
        if p["i_meas"][channel] is not None and p["i_true"] >= i_min
    ]
    if len(xs) < 2:
        return None
    x, y = np.asarray(xs), np.asarray(ys)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "gain": float(a),
        "offset_a": float(b),
        "r2": float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else None,
        "max_resid_a": float(np.max(np.abs(resid))),
        "n": len(xs),
    }


def crosstalk(points: list[dict], driven: str) -> dict:
    """
    Test 4: max |i_meas| on every undriven channel. Should be ~0; anything
        that tracks the driven channel is mux settling or shared-ground pickup.
    """

    out = {}
    for c in CHANNELS:
        if c == driven:
            continue
        vals = [abs(p["i_meas"][c]) for p in points if p["i_meas"][c] is not None]
        out[c] = float(max(vals)) if vals else None
    return out


def report(result: dict, args) -> None:
    if result["mode"] == "zero-drift":
        print("\n--- zero-offset drift ---")
        worst = 0.0
        for c, d in result["drift"].items():
            print(
                f"  {c}: {d['start_a']:+.3f} -> {d['end_a']:+.3f} A  "
                f"range {d['range_a']:.3f} A  std {d['std_a']:.3f} A"
            )
            worst = max(worst, d["range_a"])
        if worst > args.drift_tol:
            print(
                f"\n  FAIL: {worst:.3f} A drift exceeds --drift-tol {args.drift_tol} A.\n"
                f"  The boot-time seed is not valid for a full run. Call\n"
                f"  CurrentSense::recalibrateZero() during pre-run idle, or re-seed\n"
                f"  after the board reaches thermal steady state."
            )
        else:
            print(f"\n  PASS: worst drift {worst:.3f} A within {args.drift_tol} A.")
        return

    ch = result["channel"]
    pts = result["points"]
    f = fit_gain(pts, ch, args.fit_from)
    result["fit"] = f
    result["crosstalk_a"] = crosstalk(pts, ch)

    if not f:
        print("\nnot enough settled points to fit", file=sys.stderr)
        return

    print(f"\n--- channel {ch} ({result['mode']} sweep) ---")
    print(f"  gain      {f['gain']:.4f}   (1.0000 = existing SENS correct)")
    print(f"  offset    {f['offset_a']:+.4f} A")
    r2 = "n/a" if f["r2"] is None else f"{f['r2']:.5f}"
    print(f"  r2        {r2}   max resid {f['max_resid_a']:.4f} A   n={f['n']}")

    idx = CHANNELS.index(ch)
    print(
        f"\n  SENS[{idx}] ({ch}): {args.sens_old:.2f} -> "
        f"{args.sens_old / f['gain']:.2f} A/V   "
        f"(src/drive_common.h:19)"
    )
    result["sens_corrected"] = args.sens_old / f["gain"]

    if f["r2"] is not None and f["r2"] < 0.999:
        print(
            "  ! nonlinear: the CS chain is not a single gain. Check ESP32 ADC "
            "INL near the rails before trusting a one-number SENS."
        )
    if abs(f["offset_a"]) > args.drift_tol:
        print(
            f"  ! offset {f['offset_a']:+.3f} A is large -- the boot zero seed was "
            f"taken against a baseline that did not hold. Run --mode zero-drift."
        )
    xt = result["crosstalk_a"]
    worst_c = max((v for v in xt.values() if v is not None), default=0.0)
    if worst_c > args.crosstalk_tol:
        print(f"  ! crosstalk on undriven channels up to {worst_c:.3f} A: {xt}")


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", required=True, choices=["zero-drift", "voltage", "duty"])
    ap.add_argument(
        "--channel",
        default="A",
        choices=CHANNELS,
        help="the ONE channel being driven (others must be off)",
    )
    ap.add_argument("--dps-port", help="DPS-150 serial port (auto-detect if omitted)")
    ap.add_argument("--esp-port", required=True, help="ESP32 serial port")
    ap.add_argument(
        "--sens-old",
        type=float,
        default=15.26,
        help="the SENS[] value currently flashed for this channel "
        "(src/drive_common.h:19), used to report the correction",
    )

    ap.add_argument("--v-min", type=float, default=2.0, help="voltage mode: min V")
    ap.add_argument("--v-max", type=float, default=12.0, help="voltage mode: max V")
    ap.add_argument("--steps", type=int, default=10, help="voltage mode: setpoints")
    ap.add_argument("--updown", action="store_true", default=True)
    ap.add_argument("--no-updown", dest="updown", action="store_false")

    ap.add_argument("--voltage", type=float, default=12.0, help="duty mode: fixed V")
    ap.add_argument(
        "--duty-bin", type=float, default=2.5, help="duty mode: bin width, %%"
    )
    ap.add_argument(
        "--min-per-bin",
        type=int,
        default=3,
        help="duty mode: samples needed before a bin counts as settled",
    )

    ap.add_argument(
        "--duration",
        type=float,
        default=900.0,
        help="zero-drift/duty mode: seconds to log",
    )
    ap.add_argument(
        "--drift-interval",
        type=float,
        default=20.0,
        help="zero-drift mode: seconds between samples",
    )

    ap.add_argument(
        "--quiescent-voltage",
        type=float,
        default=12.0,
        help="voltage at which board quiescent current is measured",
    )
    ap.add_argument(
        "--dwell", type=float, default=1.5, help="settle time per setpoint, s"
    )
    ap.add_argument(
        "--drain",
        type=float,
        default=0.6,
        help="telemetry discarded after a setpoint change, s "
        "(must exceed the 50ms CS filter and the 500ms print period)",
    )
    ap.add_argument("--samples", type=int, default=8, help="paired reads averaged")
    ap.add_argument(
        "--sample-gap", type=float, default=0.05, help="delay between reads, s"
    )

    ap.add_argument(
        "--fit-from", type=float, default=0.0, help="ignore below this current, A"
    )
    ap.add_argument(
        "--drift-tol", type=float, default=0.15, help="zero-drift pass threshold, A"
    )
    ap.add_argument(
        "--crosstalk-tol", type=float, default=0.10, help="crosstalk threshold, A"
    )
    ap.add_argument(
        "--i-limit",
        type=float,
        default=4.5,
        help="DPS-150 current limit, A (hardware max 5A)",
    )
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    if args.i_limit > 5.0:
        print("refusing --i-limit above the DPS-150's 5A ceiling", file=sys.stderr)
        return 2

    tel = TelemetryReader(args.esp_port)
    try:
        async with DPS150(port=args.dps_port) as dev:
            info = await dev.get_info()
            print(f"supply: {info.model_name} fw{info.firmware_version}")

            await dev.set_ovp(max(args.v_max, args.voltage) + 2.0)
            await dev.set_ocp(args.i_limit + 0.3)
            await dev.set_opp(max(args.v_max, args.voltage) * (args.i_limit + 0.3))
            await dev.set_current(args.i_limit)

            try:
                if args.mode == "zero-drift":
                    result = await run_zero_drift(dev, tel, args)
                elif args.mode == "voltage":
                    result = await run_voltage_sweep(dev, tel, args)
                else:
                    result = await run_duty_sweep(dev, tel, args)
            finally:
                await dev.disable_output()
                print("output off")
    finally:
        tel.close()

    result["config"] = vars(args)
    report(result, args)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
