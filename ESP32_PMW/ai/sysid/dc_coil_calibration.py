#!/usr/bin/env python3
"""
Per-coil DC characterization with the FNIRSI DPS-150: series resistance R_i
and (optionally) the field-per-amp coefficient B/I from a gaussmeter.

WIRING: unplug the coil from the PCB at its XT30 (J10 Ch1 / J5 Ch2) and wire it
DIRECTLY to the DPS-150. Do NOT run this through the H-bridge -- the series coil
capacitor (docs/PCB_Design_Documentation.md sec.6) is in the main current path and
blocks DC by design, and shorting it puts uncentred DC magnetization through the
coil. Driving the bare coil from the supply removes the bridge drop, the cap, and
the firmware from the measurement entirely.

WHY DC: at steady-state DC there is no dI/dt, so the k~0.24 magnetic coupling
between coils (data/README.md, 2026-07-04a) contributes nothing. This is the only
per-channel measurement in the stack that is coupling-free. It gives:

  * R_i -- pins the parameter fit_rlc_model.py fits worst. That joint fit only
    exposes R cleanly at resonance (see its docstring); feed R_i in as fixed and
    L_i/C_i become much better conditioned.
  * B/I per coil -- the "geometric (field-per-amp)" asymmetry left open in
    data/README.md (2026-07-04g). Nothing else in the stack measures field: the
    VNH5019 CS pin reads SUPPLY-sourced current, not coil current.
  * B(I) linearity -- sweeping up then back down exposes any ferromagnetic
    material in the flux path as a non-overlapping hysteresis loop.

Usage:
  uv run python ai/dc_coil_calibration.py --coil A --out ai/dc_cal_A.json
  uv run python ai/dc_coil_calibration.py --coil A --gauss-prompt --i-max 1.5
  uv run python ai/dc_coil_calibration.py --port /dev/tty.usbserial-10 --coil B

Keep the gaussmeter probe in a RIGID fixture and do not move it between coils --
B falls off steeply with distance, so probe repositioning error will swamp the
per-coil differences you are trying to resolve.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import numpy as np

try:
    from dps150 import DPS150, Mode, ProtectionState
except ImportError:
    sys.exit(
        "dps150 package not found. Install the driver library:\n"
        "  pip install -e ../../DPS-150-python-library"
    )


async def read_settled(dev: DPS150, n: int, gap_s: float) -> dict:
    """
    Average n state reads. get_all() already sleeps 100ms internally, so the
        effective sample rate is ~10Hz -- ample for a DC measurement.
    """

    volts, amps, watts = [], [], []
    mode = None
    for _ in range(n):
        st = await dev.get_all()
        if st.protection_state != ProtectionState.NORMAL:
            raise RuntimeError(f"supply tripped: {st.protection_state.value}")
        volts.append(st.output_voltage)
        amps.append(st.output_current)
        watts.append(st.output_power)
        mode = st.mode
        await asyncio.sleep(gap_s)
    return {
        "v": float(np.mean(volts)),
        "v_std": float(np.std(volts)),
        "i": float(np.mean(amps)),
        "i_std": float(np.std(amps)),
        "p": float(np.mean(watts)),
        "mode": mode.value if mode else None,
    }


async def sweep(dev: DPS150, setpoints, args) -> list[dict]:
    points = []
    for direction, seq in (("up", setpoints), ("down", setpoints[::-1])):
        if direction == "down" and not args.updown:
            break
        for i_set in seq:
            await dev.set_current(float(i_set))
            await asyncio.sleep(args.dwell)
            s = await read_settled(dev, args.samples, args.sample_gap)

            # In CV the load is drawing less than the setpoint, so V/I is not a
            # clean resistance reading -- flag it rather than silently fitting it.
            if s["mode"] != "CC":
                print(
                    f"  ! I_set={i_set:.2f}A settled in {s['mode']}, not CC "
                    f"(raise --compliance above I*R)",
                    file=sys.stderr,
                )

            s.update(i_set=float(i_set), direction=direction)
            s["r_point"] = s["v"] / s["i"] if s["i"] > 1e-3 else None

            if args.gauss_prompt:
                raw = input(
                    f"  I={s['i']:.3f}A V={s['v']:.3f}V -> gaussmeter reading "
                    f"(blank to skip): "
                ).strip()
                s["b"] = float(raw) if raw else None
            else:
                s["b"] = None
                print(
                    f"  I={s['i']:.3f}A V={s['v']:.3f}V P={s['p']:.2f}W "
                    f"R={s['r_point']:.4f}ohm {s['mode']}"
                    if s["r_point"]
                    else f"  I={s['i']:.3f}A V={s['v']:.3f}V (below fit threshold)"
                )
            points.append(s)
    return points


def fit(points: list[dict], key: str, i_min: float) -> dict | None:
    """
    Least-squares y = m*I + c over points with a usable value for `key`.
        The intercept is reported, not forced to zero: a non-zero c on the R fit is
        lead/contact resistance, and on the B fit it is probe offset or ambient
        field. Both are diagnostics worth seeing.
    """

    xs = [p["i"] for p in points if p.get(key) is not None and p["i"] >= i_min]
    ys = [p[key] for p in points if p.get(key) is not None and p["i"] >= i_min]
    if len(xs) < 2:
        return None
    x, y = np.asarray(xs), np.asarray(ys)
    m, c = np.polyfit(x, y, 1)
    resid = y - (m * x + c)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "slope": float(m),
        "intercept": float(c),
        "r2": float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else None,
        "max_resid": float(np.max(np.abs(resid))),
        "n": len(xs),
    }


def fmt_r2(value: float | None) -> str:
    """
    fit() reports r2 = None when every sample shared one y value (ss_tot == 0),
        e.g. a flat supply readback or a gaussmeter pegged/read to too few digits.
        Formatting that as a float would crash the summary before the JSON is
        written, losing the whole sweep.
    """

    return "n/a" if value is None else f"{value:.5f}"


def hysteresis(points: list[dict], key: str) -> float | None:
    """
    Max |up - down| at matched setpoints. On B this is ferromagnetic material
        in the flux path; on V it is coil heating during the sweep (R rising with T).
    """

    up = {
        p["i_set"]: p[key]
        for p in points
        if p["direction"] == "up" and p.get(key) is not None
    }
    dn = {
        p["i_set"]: p[key]
        for p in points
        if p["direction"] == "down" and p.get(key) is not None
    }
    shared = set(up) & set(dn)
    if not shared:
        return None
    return float(max(abs(up[k] - dn[k]) for k in shared))


async def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--port", help="DPS-150 serial port (auto-detect if omitted)")
    ap.add_argument("--coil", required=True, help="coil label, e.g. A B C D")
    ap.add_argument(
        "--i-max",
        type=float,
        default=1.5,
        help="max sweep current, A (default 1.5; coils are rated ~2A RMS "
        "and this is DC with no forced cooling)",
    )
    ap.add_argument("--i-min", type=float, default=0.2, help="min sweep current, A")
    ap.add_argument(
        "--steps", type=int, default=8, help="setpoints per sweep direction"
    )
    ap.add_argument(
        "--compliance",
        type=float,
        default=12.0,
        help="CV voltage limit, V -- must exceed i_max*R to stay in CC",
    )
    ap.add_argument(
        "--dwell", type=float, default=1.5, help="settle time per setpoint, s"
    )
    ap.add_argument(
        "--samples", type=int, default=8, help="reads averaged per setpoint"
    )
    ap.add_argument(
        "--sample-gap", type=float, default=0.05, help="delay between reads, s"
    )
    ap.add_argument(
        "--updown",
        action="store_true",
        default=True,
        help="sweep up then back down (default; exposes hysteresis)",
    )
    ap.add_argument("--no-updown", dest="updown", action="store_false")
    ap.add_argument(
        "--gauss-prompt",
        action="store_true",
        help="prompt for a gaussmeter reading at each setpoint",
    )
    ap.add_argument(
        "--fit-from",
        type=float,
        default=0.0,
        help="ignore points below this current in the fits, A",
    )
    ap.add_argument("--out", help="write results JSON here")
    args = ap.parse_args()

    if args.i_max > 2.5:
        print(
            f"refusing --i-max {args.i_max}A: coils are rated ~2A RMS "
            f"(docs/PCB_Design_Documentation.md sec.6)",
            file=sys.stderr,
        )
        return 2

    setpoints = np.linspace(args.i_min, args.i_max, args.steps)

    async with DPS150(port=args.port) as dev:
        info = await dev.get_info()
        print(f"connected: {info.model_name} fw{info.firmware_version}")

        # Guard rails before anything is energized. OPP is the one that matters:
        # a shorted or mis-wired coil hits it long before OCP at these currents.
        await dev.set_ovp(args.compliance + 2.0)
        await dev.set_ocp(args.i_max + 0.5)
        await dev.set_opp(args.compliance * (args.i_max + 0.5))
        await dev.set_voltage(args.compliance)
        await dev.set_current(float(setpoints[0]))

        print(
            f"coil {args.coil}: {args.steps} setpoints "
            f"{args.i_min}..{args.i_max}A, compliance {args.compliance}V"
        )
        await dev.enable_output()
        try:
            points = await sweep(dev, setpoints, args)
        finally:
            await dev.disable_output()
            print("output off")

    result = {
        "coil": args.coil,
        "config": vars(args),
        "points": points,
        # V = I*R + c, so the slope IS the resistance; using the fit rather than
        # per-point V/I averages out the supply's readback quantization.
        "r_fit_ohm": fit(points, "v", args.fit_from),
        "b_fit": fit(points, "b", args.fit_from),
        "v_hysteresis": hysteresis(points, "v"),
        "b_hysteresis": hysteresis(points, "b"),
    }

    if result["r_fit_ohm"]:
        r = result["r_fit_ohm"]
        print(
            f"\nR = {r['slope']:.4f} ohm  (intercept {r['intercept']:+.4f} V, "
            f"r2={fmt_r2(r['r2'])}, max resid {r['max_resid']*1e3:.1f} mV)"
        )
        if r["r2"] is not None and r["r2"] < 0.999:
            print(
                "  ! nonlinear V(I) -- most likely coil heating during the sweep; "
                "shorten --dwell or let it cool between runs"
            )
    if result["b_fit"]:
        b = result["b_fit"]
        print(
            f"B/I = {b['slope']:.4f} /A  (intercept {b['intercept']:+.4f}, "
            f"r2={fmt_r2(b['r2'])})"
        )
        if b["r2"] is not None and b["r2"] < 0.999:
            print("  ! B is not linear in I -- ferromagnetic material in the flux path")
    if result["b_hysteresis"]:
        print(
            f"B up/down hysteresis: {result['b_hysteresis']:.4f} "
            f"(non-zero => ferromagnetic material)"
        )
    if result["v_hysteresis"]:
        print(
            f"V up/down hysteresis: {result['v_hysteresis']*1e3:.1f} mV "
            f"(=> R drifted with temperature during the sweep)"
        )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
