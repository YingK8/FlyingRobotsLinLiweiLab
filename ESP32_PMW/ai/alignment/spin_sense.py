#!/usr/bin/env python3
"""Spin sense of every solved design-B take, from the once-per-rev coning before the cut.

Before the cut the axis wobbles at 1x the drive frequency -- a body-fixed asymmetry carried
round by the rotor (alignment_rate.py, "CALLED CONING AND NOT PRECESSION") -- so the wobble's
phase advances in the rotor's sense of rotation. Demodulate the axis deviation z = u + iv at +f
and at -f: a matched filter tolerates the irregular, frame-dropping timestamps, and at fs ~230 Hz
a +f line aliases to f - fs, never onto -f. The sign convention is arbitrary but the same for
every take; the low-frequency takes say which sign is normal.
"""

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/meli/Desktop/Kevin/UCB/FlyingRobotsLinLiweiLab/ESP32_PMW")
ROOT = REPO / "results/alignment_rate/20260910_droneB"
PRE = (-2.0, -0.1)                    # s from the cut: coils on, rotor at drive
SLIP = np.linspace(-2.0, 2.0, 81)     # Hz: let the rotor lead or lag the drive a little


def sense(take, f, t_kill):
    a = np.genfromtxt(take / "axis.csv", delimiter=",", names=True)
    t = a["t"]
    n = np.c_[a["nx"], a["ny"], a["nz"]]
    w = (t >= t_kill + PRE[0]) & (t <= t_kill + PRE[1]) & np.isfinite(n).all(axis=1)
    if w.sum() < 100:
        return None
    t, n = t[w], n[w] / np.linalg.norm(n[w], axis=1)[:, None]
    n[(n @ n[0]) < 0] *= -1.0                         # a disc normal has no sign of its own
    n0 = n.mean(axis=0)
    n0 /= np.linalg.norm(n0)
    e1 = np.cross(n0, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)
    z = (n @ e1) + 1j * (n @ e2)
    z -= z.mean()
    tt = t - t[0]
    power = lambda fr: max(abs(np.mean(z * np.exp(-2j * np.pi * (fr + s) * tt))) for s in SLIP)
    return power(f), power(-f), int(w.sum())


TEL_RE = __import__("re").compile(r"^\[(\d+\.\d+)\] <- t=\d+ freq=([\d.]+)")
RAMP_HZ = (15.0, 35.0)                # where the hold test was clean at 10-40 Hz
RAMP_SLIP = np.linspace(-1.0, 1.0, 41)
#: Solved frames needed in the ramp window. Heavy frame drops leave some takes below it
#: (2026-09-13_104421: 121 frames and a 3.2 s gap); lower it only to retry such a take.
RAMP_MIN_N = 200


def _plane(n):
    n = n / np.linalg.norm(n, axis=1)[:, None]
    n[(n @ n[0]) < 0] *= -1.0
    n0 = n.mean(axis=0)
    n0 /= np.linalg.norm(n0)
    e1 = np.cross(n0, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    z = (n @ e1) + 1j * (n @ np.cross(n0, e1))
    return z - z.mean()


def ramp_sense(take, log_path):
    """The same test on the spin-up, while the drive passes RAMP_HZ.

    A locked rotor cannot reverse, so the sense it was captured in at the start of the ramp is
    the sense it holds at the cut -- and at 15-35 Hz the wobble is strong enough to read. The
    sweep.log telemetry is on the video's host clock; its commanded freq is integrated into
    the chirp phase the wobble should follow.
    """

    tel = [(float(m.group(1)), float(m.group(2)))
           for m in map(TEL_RE.match, Path(log_path).read_text(errors="replace").splitlines())
           if m]
    if len(tel) < 10:
        return None
    tt_, ff_ = np.array(tel).T
    a = np.genfromtxt(take / "axis.csv", delimiter=",", names=True)
    t = a["t"]
    n = np.c_[a["nx"], a["ny"], a["nz"]]
    f_t = np.interp(t, tt_, ff_, left=np.nan, right=np.nan)
    rising = np.r_[True, np.diff(np.interp(t, tt_, np.maximum.accumulate(ff_))) >= 0]
    w = (f_t >= RAMP_HZ[0]) & (f_t <= RAMP_HZ[1]) & np.isfinite(n).all(axis=1) & rising
    w &= t < tt_[np.argmax(ff_)]                     # the up-ramp only, not the spin-down
    if w.sum() < RAMP_MIN_N:
        return None
    t, n, f_t = t[w], n[w], f_t[w]
    z = _plane(n)
    phi = 2.0 * np.pi * np.r_[0.0, np.cumsum(0.5 * (f_t[1:] + f_t[:-1]) * np.diff(t))]
    tt = t - t[0]
    power = lambda sgn: max(abs(np.mean(z * np.exp(-1j * sgn * (phi + 2 * np.pi * s * tt))))
                            for s in RAMP_SLIP)
    return power(+1.0), power(-1.0), int(w.sum())


def main():
    settle = {r["take"]: r for r in csv.DictReader(open(ROOT / "report/settling.csv"))}
    trials = {r["take"]: r for r in csv.DictReader(open(ROOT / "report/campaign_trials.csv"))}
    out = []
    for idx in sorted(ROOT.glob("*hz/index.csv")):
        for r in csv.DictReader(open(idx)):
            take = Path(r["flight"])
            s = settle.get(take.name, {})
            if r["outcome"] != "ok" or not (take / "axis.csv").exists():
                continue
            f = float(r["freq_hz"])
            # The hold test needs the settle report's t_kill; the ramp test needs only the log,
            # so a take not yet re-analysed still gets its ramp sense.
            got = sense(take, f, float(s["t_kill"])) if s else None
            pp, pn, n = got if got else (np.nan, np.nan, 0)
            rg = ramp_sense(take, r["log"]) if Path(r["log"]).exists() else None
            rp, rn, rcount = rg if rg else (np.nan, np.nan, 0)
            c = trials.get(take.name, {})
            out.append({"freq_hz": f, "repeat": r["repeat"], "take": take.name,
                        "p_pos": round(pp, 5), "p_neg": round(pn, 5),
                        "log_ratio": round(float(np.log10(pp / pn)), 2),
                        "sense": ("+" if pp > pn else "-") if got else "", "n": n,
                        "ramp_log_ratio": round(float(np.log10(rp / rn)), 2) if rg else "",
                        "ramp_sense": ("+" if rp > rn else "-") if rg else "", "ramp_n": rcount,
                        "swing_deg": s.get("swing_deg", ""), "modal": c.get("modal", ""),
                        "amp_deg": c.get("amp_deg", "")})
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("spin_sense.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(f"{o['freq_hz']:5.0f} Hz r{o['repeat']:>2} {o['take']}  sense {o['sense']} "
              f"log10(P+/P-) {o['log_ratio']:+5.2f}  swing {str(o['swing_deg'])[:5]:>5}  "
              f"modal {o['modal']:5}  amp {str(o['amp_deg'])[:6]}")


if __name__ == "__main__":
    main()
