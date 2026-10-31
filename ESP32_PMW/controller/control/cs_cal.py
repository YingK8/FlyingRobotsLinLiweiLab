#!/usr/bin/env python3
"""VNH5019 current-sense gain K, one channel at a time, from DC into a resistive load.

    V_CS = offset + I_load * R_s / K        I_load = (V_p - V_n) / R_load

`../pico/PROGRESS.md` 5 got K ~ 25 000 against a datasheet 4670-10110 by comparing AC CS
readings with the PSU's DC average. Here the bridge is held in DC (`F0` on `-e servo`), so
CS mirrors one steady high-side current, and the current is read across the load in the
same scope block as CS. Both legs are measured -- `P<ch>,0` holds one high side on,
`P<ch>,180` the other -- because each high-side switch has its own sense mirror and AC
drive uses both, half a cycle each.

SETUP, per channel X
  * coil X and its series cap OUT; the load straight across OUTA_X / OUTB_X.
  * EN (header pin 2) jumpered to GND on the other three bridges: the ESP32 has no EN pin,
    so this is the only way to disable them. Check unpowered that EN reaches its rail
    through a resistor (> 1 kohm), not a direct tie, before grounding it.
  * scope X -> CS_X, scope MATE[X] -> the board-mate's CS (A<->C, B<->D share a board and
    its ground, the likeliest place for an anomaly), the other two -> OUTA_X / OUTB_X,
    single-ended. EVERY ground clip on PGND at bridge X: the scope ground is USB ground,
    and a clip on an OUT node shorts that leg. The load voltage is the DIFFERENCE of the
    two because GND1/PGND1 have been measured ~400 mV apart.
  * PSU OFF until `record` reports the ESP32 idle (opening/closing the port resets it, and
    the carrier pins float until `driveBoot()`); then on at 0 V, limit 10 A, raised slowly.
    CV, not CC: CC would pull the bus under the VNH5019's ~5.5 V UVLO at
    low setpoints, and a setpoint is worth nothing when the scope measures the current.
  * `pio run -e servo -t upload`. Kill: `safe_off.py` (servo now hears `stop`) or GPIO14.

    uv run python controller/control/cs_cal.py record --ch A        # volts only
    uv run python controller/control/cs_cal.py analyse results/cs_cal/* \
        --rs A=2532,B=2530,C=2542,D=2540 --rload 2.031              # ohms go in here
    uv run python controller/control/cs_cal.py          # self-check, no hardware
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from controller.control.link import SerialComm, parse_telemetry

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_DIR = ROOT / "results" / "cs_cal"

CH = "ABCD"
MATE = {"A": "C", "C": "A", "B": "D", "D": "B"}

#: The PicoScope 7 app bundles libps5000a; no separate PicoSDK install on this Mac.
PICO_LIB = "/Applications/PicoScope 7 T&M.app/Contents/Resources"
#: 12-bit timebase 65 = (65-3)/62.5 MHz = 0.992 us; 20 000 samples = 20 ms of DC.
TIMEBASE = 65
BLOCK_N = 20_000

#: driveTelemetry prints every 500 ms, so a 1 s pulse carries >= 1 line from every channel.
PULSE_S = 1.0
#: Guess, not measured: VNH5019 CS settling plus host latency. blocks.npz shows if not.
SETTLE_S = 0.1
#: ~13 W average at 16 V into 2 ohm; keeps the resistor's R near its cold reading.
REST_S = 10 * PULSE_S

#: The VNH5019 drives CS to V_SENSEH (~5 V) on a fault; real readings sit far below.
FAULT_V = 3.5
#: Anomaly thresholds on the idle channels. Guesses: neither noise floor is measured.
ANOM_V = 0.02   # scope, board-mate CS over its zero
ANOM_A = 0.2    # ESP32 telemetry, any idle channel over its zero


def firmware_sens():
    """`SENS` from `src/drive_common.h`, so telemetry amps convert back without a 2nd copy."""
    text = (ROOT / "src" / "drive_common.h").read_text()
    m = re.search(r"SENS\[NUM_CHANNELS\]\s*=\s*\{([^}]*)\}", text)
    return [float(v.strip().rstrip("f")) for v in m.group(1).split(",")]


class Scope:
    def __init__(self, ranges):
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", PICO_LIB)
        from picosdk.functions import adc2mV, assert_pico_ok
        from picosdk.ps5000a import ps5000a as ps
        self.ps, self._mv, ok = ps, adc2mV, assert_pico_ok
        self.h = ctypes.c_int16()
        rc = ps.ps5000aOpenUnit(ctypes.byref(self.h), None,
                                ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_12BIT"])
        if rc in (282, 286):  # no external PSU / not a USB3 port: accept and carry on
            rc = ps.ps5000aChangePowerSource(self.h, rc)
        ok(rc)
        self.max_adc = ctypes.c_int16()
        ok(ps.ps5000aMaximumValue(self.h, ctypes.byref(self.max_adc)))
        self.ranges = {c: ps.PS5000A_RANGE[f"PS5000A_{r}"] for c, r in ranges.items()}
        self.bufs = {}
        for c, rng in self.ranges.items():
            src = ps.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{c}"]
            ok(ps.ps5000aSetChannel(self.h, src, 1, ps.PS5000A_COUPLING["PS5000A_DC"], rng, 0.0))
            self.bufs[c] = (ctypes.c_int16 * BLOCK_N)()
            ok(ps.ps5000aSetDataBuffer(self.h, src, ctypes.byref(self.bufs[c]), BLOCK_N, 0, 0))
        interval, nmax = ctypes.c_float(), ctypes.c_int32()
        ok(ps.ps5000aGetTimebase2(self.h, TIMEBASE, BLOCK_N, ctypes.byref(interval),
                                  ctypes.byref(nmax), 0))
        self.fs = 1e9 / interval.value
        self._ok = ok

    def capture(self):
        """One block -> ({channel: volts}, overflowed channels)."""
        ps = self.ps
        self._ok(ps.ps5000aRunBlock(self.h, 0, BLOCK_N, TIMEBASE, None, 0, None, None))
        ready, t0 = ctypes.c_int16(0), time.monotonic()
        while not ready.value:
            self._ok(ps.ps5000aIsReady(self.h, ctypes.byref(ready)))
            if time.monotonic() - t0 > 2.0:
                raise TimeoutError("scope block never completed")
            time.sleep(0.001)
        n, ovf = ctypes.c_int32(BLOCK_N), ctypes.c_int16()
        self._ok(ps.ps5000aGetValues(self.h, 0, ctypes.byref(n), 0, 0, 0, ctypes.byref(ovf)))
        volts = {c: np.array(self._mv(b, self.ranges[c], self.max_adc)[:n.value]) / 1000.0
                 for c, b in self.bufs.items()}
        return volts, [c for c in self.ranges if ovf.value & (1 << CH.index(c))]

    def close(self):
        try:
            self.ps.ps5000aStop(self.h)
        finally:
            self.ps.ps5000aCloseUnit(self.h)


def _hold(comm, seconds):
    """Drain serial for `seconds`; return every line, so the caller can pick by duty."""
    lines, end = [], time.monotonic() + seconds
    while time.monotonic() < end:
        line = comm.handle_serial_comm()
        if line is None:
            time.sleep(0.002)
        else:
            lines.append(line)
    return lines


def wait_idle(comm, timeout_s=5.0):
    """Block until telemetry shows every carrier at 0: the reset from opening the port is over."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        line = comm.handle_serial_comm()
        if line is None:
            time.sleep(0.005)
            continue
        t = parse_telemetry(line)
        if len(t.duty) == 4 and not any(t.duty):
            return
    raise RuntimeError("no idle telemetry from the ESP32 -- leave the PSU OFF")


def esp_amps(lines, duty_x, i):
    """Last telemetry currents taken while channel i sat at `duty_x`: older lines are stale."""
    hit = None
    for line in lines:
        t = parse_telemetry(line)
        if len(t.amps) == 4 and len(t.duty) == 4 and abs(t.duty[i] - duty_x) < 0.5:
            hit = t.amps
    return hit


def is_fault(v_cs, overflowed, lines):
    return v_cs > FAULT_V or bool(overflowed) or any("trip=1" in l for l in lines)


FIELDS = ["ch", "pol", "v_set", "v_cs", "std_cs", "v_mate", "v_p", "v_n", "v_load",
          "esp_A", "esp_B", "esp_C", "esp_D", "fault", "anomaly"]


def record(args):
    x = args.ch
    i, mate = CH.index(x), MATE[x]
    spare = [c for c in CH if c not in (x, mate)]
    vp, vn = args.vp or spare[0], args.vn or spare[1]
    if len({x, mate, vp, vn}) != 4:
        sys.exit(f"scope channels must be distinct: cs={x} mate={mate} vp={vp} vn={vn}")
    lo, hi, step = (float(v) for v in args.volts.split(":"))
    volts = np.arange(lo, hi + step / 2, step)

    out = OUT_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{x}"
    out.mkdir(parents=True)
    print(f"cs={x} mate={mate} vp={vp} vn={vn} -> {out}")

    scope = Scope({x: "5V", mate: "5V", vp: "20V", vn: "20V"})
    # opening the port resets the ESP32, and its carrier pins float until driveBoot():
    # the PSU stays off until the board reports idle (incident 2026-10-30)
    input("PSU OUTPUT OFF? Enter to open the port ")
    comm = SerialComm(args.port, raw_log_path=out / "serial.jsonl")
    raw = {}
    f = (out / "points.csv").open("w", newline="")
    f.write(f"# ch={x} mate={mate} vp={vp} vn={vn} "
            f"fs={scope.fs:.0f} sens={','.join(map(str, firmware_sens()))}\n")
    w = csv.DictWriter(f, FIELDS)
    w.writeheader()
    try:
        wait_idle(comm)
        comm.handle_serial_comm("S")
        comm.handle_serial_comm("F0")  # DC: the phase now picks a leg, it does not rotate
        input(f"ESP32 up, all carriers 0. PSU ON now: CV, limit 10 A, start at 0 V and raise "
              f"slowly to {volts[0]:.2f} V. Enter ")
        for n, v in enumerate(volts):
            if n:
                input(f"set PSU to {v:.2f} V, Enter ")
            zero = None
            for pol in ("zero", 0, 180):
                if pol != "zero":
                    for k in range(4):  # only X's carrier is ever raised
                        if k != i:
                            comm.handle_serial_comm(f"A{k},0")
                    comm.handle_serial_comm(f"P{i},{pol}")
                    comm.note_external_drive(True)
                    comm.handle_serial_comm(f"A{i},100")
                t0 = time.monotonic()
                lines = _hold(comm, SETTLE_S)
                blk, ovf = scope.capture()
                lines += _hold(comm, PULSE_S - (time.monotonic() - t0))
                if pol != "zero":
                    comm.handle_serial_comm("S")
                    comm.note_external_drive(False)

                m = {c: float(blk[c].mean()) for c in blk}
                esp = esp_amps(lines, 0.0 if pol == "zero" else 100.0, i)
                row = dict(ch=x, pol=pol, v_set=v, v_cs=m[x], std_cs=float(blk[x].std()),
                           v_mate=m[mate], v_p=m[vp], v_n=m[vn],
                           v_load=m[vp] - m[vn],
                           fault=int(is_fault(m[x], ovf, lines)), anomaly=0)
                for k, c in enumerate(CH):
                    row[f"esp_{c}"] = esp[k] if esp else ""
                if pol == "zero":
                    zero = row
                else:
                    idle_esp = [] if not (esp and zero["esp_A"] != "") else [
                        esp[k] - zero[f"esp_{c}"] for k, c in enumerate(CH) if k != i]
                    row["anomaly"] = int(abs(row["v_mate"] - zero["v_mate"]) > ANOM_V
                                         or any(abs(d) > ANOM_A for d in idle_esp))
                w.writerow(row)
                f.flush()
                raw[f"{v:.2f}V_{pol}"] = np.stack([blk[c] for c in (x, mate, vp, vn)])
                print(f"  {v:5.2f} V {str(pol):>4}: V_load={row['v_load']:6.3f} V  "
                      f"CS={row['v_cs'] * 1e3:7.1f} mV  mate={row['v_mate'] * 1e3:6.1f} mV"
                      f"{'  FAULT' if row['fault'] else ''}"
                      f"{'  ANOMALY' if row['anomaly'] else ''}")
                if pol != "zero":
                    _hold(comm, REST_S)
    finally:
        try:
            comm.handle_serial_comm("S")
            comm.note_external_drive(False)
            try:
                input("drive stopped. PSU OUTPUT OFF now -- closing the port resets the ESP32. "
                      "Enter ")
            except (EOFError, KeyboardInterrupt):
                print("\n!! closing the port with the PSU possibly ON -- turn it off NOW")
            comm.close()
        finally:
            scope.close()
            f.close()
            # rows (x, mate, vp, vn) per capture
            np.savez(out / "blocks.npz", **raw)


def read_points(path):
    lines = Path(path).read_text().splitlines()
    meta = dict(kv.split("=", 1) for kv in lines[0].lstrip("# ").split())
    return meta, list(csv.DictReader(lines[1:]))


def fit(i_load, v_cs, rs):
    """Least squares V = off + I*rs/K, with a Theil-Sen slope beside it to expose outliers."""
    from scipy.stats import theilslopes
    i_load, v_cs = np.asarray(i_load, float), np.asarray(v_cs, float)
    slope, off = np.polyfit(i_load, v_cs, 1)
    resid = v_cs - (off + slope * i_load)
    r2 = 1 - (resid ** 2).sum() / ((v_cs - v_cs.mean()) ** 2).sum()
    ts = theilslopes(v_cs, i_load)[0]
    return dict(K=rs / slope, g=1 / slope, off=off, r2=r2,
                resid_mA=1e3 * np.abs(resid).max() / slope, ts_dev=ts / slope - 1)


def per_channel(text):
    """'2.031' -> the same ohms for every channel; 'A=2532,B=2530' -> per channel."""
    if "=" not in text:
        return dict.fromkeys(CH, float(text))
    return {k.strip(): float(v) for k, v in (kv.split("=") for kv in text.split(","))}


def analyse(dirs, png, rs_by_ch, rload_by_ch):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    print(f"{'ch':>2} {'pol':>4} {'n':>2} {'K':>8} {'g A/V':>7} {'off mV':>7} {'zero mV':>8}"
          f" {'R2':>8} {'res mA':>7} {'TS dev':>7} {'SENS fix':>9}  flags")
    for d in dirs:
        meta, rows = read_points(Path(d) / "points.csv")
        x = meta["ch"]
        if x not in rs_by_ch or x not in rload_by_ch:
            sys.exit(f"{d}: channel {x} needs --rs and --rload")
        rs, rload = rs_by_ch[x], rload_by_ch[x]
        for r in rows:
            r["i_load"] = float(r["v_load"]) / rload
        sens = [float(s) for s in meta["sens"].split(",")]
        ax = axs.flat[CH.index(x)]
        zero = [float(r["v_cs"]) for r in rows if r["pol"] == "zero"]
        for pol, colour in (("0", "C0"), ("180", "C3")):
            pts = [r for r in rows if r["pol"] == pol]
            good = [r for r in pts if r["fault"] == "0"]
            I = np.array([float(r["i_load"]) for r in good])
            V = np.array([float(r["v_cs"]) for r in good])
            bad = [r for r in pts if r["fault"] != "0"]
            ax.plot([float(r["i_load"]) for r in bad], [float(r["v_cs"]) for r in bad],
                    "x", color=colour, alpha=0.5)
            if len(good) < 3:
                print(f"{x:>2} {pol:>4} {len(good):>2}  too few unfaulted points")
                continue
            r = fit(I, V, rs)
            # SENS converts ESP32 ADC volts to amps; scale it by what the load says
            esp = [(float(p["i_load"]), float(p[f"esp_{x}"])) for p in good
                   if p[f"esp_{x}"] not in ("", None) and float(p[f"esp_{x}"]) > 0.5]
            fix = sens[CH.index(x)] * np.median([a / b for a, b in esp]) if esp else np.nan
            flags = " ".join(f for f, on in (
                ("OUTLIERS", abs(r["ts_dev"]) > 0.01),
                ("ANOMALY", any(p["anomaly"] != "0" for p in pts)),
                (f"{len(bad)}FAULT", bad)) if on)
            print(f"{x:>2} {pol:>4} {len(good):>2} {r['K']:8.0f} {r['g']:7.3f} "
                  f"{r['off'] * 1e3:7.1f} {np.mean(zero) * 1e3:8.1f} {r['r2']:8.5f} "
                  f"{r['resid_mA']:7.1f} {r['ts_dev']:+7.2%} {fix:9.3f}  {flags}")
            ax.plot(I, V * 1e3, "o", color=colour, label=f"P{pol}: K={r['K']:.0f}")
            span = np.linspace(0, I.max(), 2)
            ax.plot(span, (r["off"] + span / r["g"]) * 1e3, "-", color=colour, lw=1)
        ax.axhline(np.mean(zero) * 1e3, color="0.6", ls=":", label="drive off")
        ax.set(title=f"{x}  (R_s={rs:.0f} ohm)", xlabel="I_load [A]", ylabel="V_CS [mV]")
        ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=130)
    print(f"-> {png}")


def _self_check():
    rng = np.random.default_rng(0)
    rs, K, off = 2532.0, 7110.0, -0.040
    I = np.linspace(3, 8, 11)
    V = off + I * rs / K + rng.normal(0, 2e-4, I.size)
    r = fit(I, V, rs)
    assert abs(r["K"] / K - 1) < 0.005, r
    assert abs(r["off"] - off) < 1e-3, r
    assert abs(r["g"] - K / rs) < 0.02 and r["r2"] > 0.999, r
    V[-1] += 0.1  # one glitch at the lever end: least squares moves, Theil-Sen does not
    assert abs(fit(I, V, rs)["ts_dev"]) > 0.01

    assert is_fault(4.9, [], [])                       # CS pinned high = VNH5019 fault
    assert is_fault(0.5, ["A"], [])                    # clipped scope channel
    assert is_fault(0.5, [], ["t=1 freq=0.0 | spread=0 bal=0 trip=1"])
    assert not is_fault(0.5, [], ["t=1 | trip=0"])

    line = ("t=5 freq=0.0 | I[A]: A=6.10 B=0.01 C=0.00 D=0.02 | duty[%]: A=100.0 B=0.0 "
            "C=0.0 D=0.0 | spread=6.1 bal=0 trip=0")
    stale = line.replace("A=100.0", "A=0.0").replace("A=6.10", "A=0.00")
    assert esp_amps([line, stale], 100.0, 0) == (6.10, 0.01, 0.00, 0.02)
    assert esp_amps([stale], 100.0, 0) is None

    assert per_channel("2.0") == dict.fromkeys(CH, 2.0)
    assert per_channel("A=2532, C=2542") == {"A": 2532.0, "C": 2542.0}

    class _Fake:
        def __init__(self, lines): self.lines = list(lines)
        def handle_serial_comm(self, _=""): return self.lines.pop(0) if self.lines else None
    wait_idle(_Fake(["boot junk", stale]))             # idle line ends the wait
    try:
        wait_idle(_Fake([line]), timeout_s=0.05)       # a driven channel is not idle
        raise AssertionError("wait_idle accepted a driven board")
    except RuntimeError:
        pass

    assert len(firmware_sens()) == 4
    print("cs_cal self-check OK")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        sys.exit()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--ch", required=True, choices=list(CH))
    r.add_argument("--volts", default="6:10:0.5", help="PSU sweep lo:hi:step, V")
    r.add_argument("--vp", choices=list(CH), help="scope channel on OUTA_X")
    r.add_argument("--vn", choices=list(CH), help="scope channel on OUTB_X")
    r.add_argument("--port")
    a = sub.add_parser("analyse")
    a.add_argument("dirs", nargs="+")
    a.add_argument("--rs", type=per_channel, required=True,
                   help="CS pulldown ohms: one value, or A=..,B=.. (DMM, unpowered)")
    a.add_argument("--rload", type=per_channel, required=True,
                   help="load ohms, leads subtracted: one value, or A=..,B=..")
    a.add_argument("--png", default=str(OUT_DIR / "cs_cal.png"))
    args = p.parse_args()
    record(args) if args.cmd == "record" else analyse(args.dirs, args.png, args.rs, args.rload)
