#!/usr/bin/env python3
"""
Where does the robot leave the pad? Reads one takeoff-ramp CSV written by
`hover_controller_runner` (results/takeoff/<stamp>.csv) and finds the turning point
in z(f): the drive frequency where height stops being pad noise and starts climbing.

The headline number is `f_liftoff`. Everything else is there to say whether to
believe it -- lost-frame fraction, the pad scatter the threshold was set from, and
the max f actually reached (which proves whether the firmware ramp clamp is gone).

Read `captured` first. `f_hz` is the commanded FIELD rate; the `spin` column is the blade
witness's latch (pose/spin.py) -- `turning` / `stopped` / blank, never a rate -- and a
blank is UNKNOWN, not stopped. A rotor that was never captured
at the bottom of the ramp never turns at all, so every z number below is about a robot
lying on the pad while the field sails past it. Capture is the tight margin on this
rig: pull_in_hz(tau_max(3 Hz)) = 4.94 Hz against a 3 Hz start, 1.65x. The mid-ramp slope
is not -- the EASE ramp keeps a wide margin on TorqueLimits.f_dot_max throughout, so
mid-ramp slew is not what breaks sync. (Both figures are against F_STEPOUT_HZ = 225,
measured; the 4.17 / 1.4x this used to quote were computed at the retired 190.)

Also refits the series-RLC coil channel from the logged currents, because this ramp
is the first data this rig has above 100 Hz -- the band where L is visible at all.
See the CAPACITANCE_F comment in z_track.py for why the current constants disagree.

    uv run python controller/control/takeoff_report.py [file.csv]
    uv run python controller/control/takeoff_report.py --compare [n]
Self-check: uv run python controller/control/takeoff_report.py --self-check
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CSV_DIR = ROOT / "results" / "takeoff"

from controller.control import constants as C
from controller.control.z_track import CAPACITANCE_F, INDUCTANCE_H, RESISTANCE_OHM

BIN_HZ = 2.0        # ramp.DEFAULT averages 6.9 Hz/s and peaks at 13.9 (EASE k=2 peaks at
                    # k*df/T, theory.md 18.2), so a bin is 0.15-0.3 s of samples
PAD_F_FRAC = 0.25   # bins under 25% of peak f are "on the pad": at a 160 Hz target that
                    # is below 40 Hz, where the series cap starves the coil (z_track.demo)
PAD_MIN_BINS = 3
RISE_SIGMA = 1.0    # per-bin dz that counts as a rise, in pad-scatter sigmas
HOLD_BINS = 3       # ...and it must hold this many bins, so one noisy bin cannot trigger
LIFT_SIGMA = 5.0    # total rise required before claiming it flew at all
LOST_BAD = 0.25     # above this the pose is too gappy to trust the numbers
FIT_F_MIN = 100.0   # only above here does the inductive term participate
TILT_RMAX_DEG = 6.0  # polar cap: past this the robot is tumbling, not tilted
LOW_F_HZ = 10.0     # "the bottom of the ramp", where capture either happens or does not.
                    # `stopped` can only be emitted under the witness's alias limit
                    # (fps/8, ~3.1 Hz at 25 fps) anyway, so this bound only widens which
                    # `turning` rows count as evidence that the rotor was caught at all.


def load(path) -> dict:
    """
    CSV -> {column: array}, plus `"ramp"`: the profile label the runner stamped above
    the header, or "" for an attempt logged before the stamp existed. Numeric columns are
    float with nan for blanks (no fix, no telemetry yet); `spin` stays a string column,
    since its three states do not have a numeric ordering and a blank there means unknown
    rather than zero.
    """

    import csv

    with open(path, newline="") as fh:
        lines = fh.read().splitlines(keepends=True)
    # The runner stamps `# ramp: <label>` above the header, so an attempt records the
    # profile it actually flew. CSVs written before that stamp existed have no comment
    # line and must still load -- "" is the right answer for them, not a failure.
    stamp = next((l.split(":", 1)[1].strip() for l in lines
                  if l.startswith("# ramp:")), "")
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    out = {"ramp": stamp}
    for k in rows[0]:
        v = [r[k].strip() if r[k] else "" for r in rows]
        try:
            out[k] = np.array([float(x) if x else np.nan for x in v])
        except ValueError:
            out[k] = np.array(v)
    return out


def _bins(f, z):
    """Median z per BIN_HZ bin of f. Returns (left edges, median z, count)."""

    edges = np.arange(math.floor(np.nanmin(f) / BIN_HZ) * BIN_HZ, np.nanmax(f) + BIN_HZ, BIN_HZ)
    idx = np.digitize(f, edges) - 1
    med = np.array([np.median(z[idx == i]) if (idx == i).any() else np.nan
                    for i in range(len(edges))])
    n = np.array([int((idx == i).sum()) for i in range(len(edges))])
    return edges, med, n


def _fit_rlc(f, i_mean):
    """Fit |I|(f) = V/|R + j(2 pi f L - 1/(2 pi f C))|. Returns (L_H, C_F) or (nan, nan)."""

    from scipy.optimize import curve_fit

    def model(f, v, l, c):
        w = 2.0 * np.pi * f
        return v / np.hypot(RESISTANCE_OHM, w * l - 1.0 / (w * c))

    try:
        p, _ = curve_fit(model, f, i_mean, p0=[np.nanmax(i_mean) * RESISTANCE_OHM,
                                               INDUCTANCE_H, CAPACITANCE_F],
                         bounds=([0, 1e-5, 1e-6], [1e3, 1e-1, 1e-2]), maxfev=20000)
        return float(p[1]), float(p[2])
    except Exception:
        return float("nan"), float("nan")


def metrics(d: dict) -> dict:
    """The numbers. `f_liftoff` is nan (with `reason`) when the robot never left the pad."""

    # nanmax/nanmean warn on an all-nan slice, which is the normal dry-run case (stub
    # source, no firmware, so no telemetry ever arrives). A rehearsal must not print
    # what looks like an error, so the empty case is answered rather than warned about.
    finite_f = d["f_hz"][np.isfinite(d["f_hz"])]
    m = {"n_ticks": len(d["t"]),
         # Carried through so every metric block, plot and comparison says which ramp
         # produced it. "" for attempts logged before the stamp existed.
         "ramp": d.get("ramp", ""),
         "f_max_reached": float(finite_f.max()) if finite_f.size else float("nan")}
    # `lost` is live_viz's CUMULATIVE counter (`lost += pose is None`, never reset), not a
    # per-tick flag -- so thresholding it marks every tick after the first miss as lost. A
    # 35 s rehearsal with ONE early miss and a measured 100% fix rate reported 98.6% lost
    # that way, which would have condemned every real flight as untrustworthy.
    # The per-tick truth is whether a fix reached the row at all: a blank z_mm.
    m["lost_frac"] = float(np.mean(~np.isfinite(d["z_mm"]))) if m["n_ticks"] else 0.0
    lost = d.get("lost", np.full(m["n_ticks"], np.nan))
    finite_lost = lost[np.isfinite(lost)]
    m["n_lost_total"] = int(finite_lost.max()) if finite_lost.size else 0

    # Did the rotor ever turn? Three answers, not two, and the two decisive ones are not
    # symmetric: `turning` is provable at any speed, `stopped` is only ever claimed below
    # the alias limit, and a blank is a refusal to answer -- above the limit a strobe can
    # make a spinning rotor look still. Reading a blank as stopped, or "no turning seen"
    # as "not turning", inverts the conclusion on every healthy fast run.
    spin = d.get("spin", np.full(m["n_ticks"], "", dtype="<U8"))
    low = d["f_hz"] <= LOW_F_HZ
    turning, stopped = spin == "turning", spin == "stopped"
    m["n_turning"], m["n_stopped"] = int(turning.sum()), int(stopped.sum())
    m["captured"] = ("turned" if (turning & low).any()
                     else "never turned" if (stopped & low).any() and not turning.any()
                     else "unknown")
    # The strongest lower bound the sensor can give on where the rotor was still moving.
    # There is no upper bound to be had: blanks say nothing, so no step-out follows.
    m["f_turning_max"] = float(np.max(d["f_hz"][turning])) if m["n_turning"] else float("nan")

    # An all-nan column is the normal dry-run case (no firmware, so no telemetry). nanmean
    # raises RuntimeWarning there, and np.errstate does not cover it -- it is a warning, not
    # an FP error. Sum/count by hand instead: no warning, and the empty case is answered.
    coils = np.vstack([d[k] for k in ("i_a", "i_b", "i_c", "i_d")])
    ok = np.isfinite(coils)
    n = ok.sum(axis=0)
    cur = np.where(n > 0, np.where(ok, coils, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)
    fit = np.isfinite(d["f_hz"]) & np.isfinite(cur) & (d["f_hz"] > FIT_F_MIN) & (cur > 0)
    m["n_fit"] = int(fit.sum())
    if m["n_fit"] >= 8:
        m["i_f"], m["i_meas"] = d["f_hz"][fit], cur[fit]
        m["L_H"], m["C_F"] = _fit_rlc(m["i_f"], m["i_meas"])
        if np.isfinite(m["L_H"]):
            m["f_res_hz"] = 1.0 / (2 * math.pi * math.sqrt(m["L_H"] * m["C_F"]))
            m["Q"] = math.sqrt(m["L_H"] / m["C_F"]) / RESISTANCE_OHM

    ramp = (d["state"] <= 1) & np.isfinite(d["f_hz"]) & np.isfinite(d["z_mm"])
    m["n_ramp"] = int(ramp.sum())
    if m["n_ramp"] < 10 * PAD_MIN_BINS:
        return {**m, "f_liftoff": float("nan"), "reason": "too few ramp samples with a fix"}

    f, z = d["f_hz"][ramp], d["z_mm"][ramp]
    edges, med, n = _bins(f, z)
    m["edges"], m["med_z"] = edges, med

    # The pad bins are the noise floor: the robot is sitting there, so its scatter is
    # measurement noise, not motion. Threshold the rise off that rather than off a
    # hard-coded millimetre number, which would be wrong at a different camera range.
    pad = edges < edges[0] + PAD_F_FRAC * (edges[-1] - edges[0])
    pad[:PAD_MIN_BINS] = True
    pad_z = z[f < edges[pad][-1] + BIN_HZ]
    m["z_pad"] = float(np.median(pad_z))
    m["sigma_pad"] = float(1.4826 * np.median(np.abs(pad_z - m["z_pad"])))
    m["n_pad"] = int(len(pad_z))
    sigma = max(m["sigma_pad"], 1e-3)   # a perfectly flat pad would divide by zero

    ok = np.isfinite(med)
    m["z_max"] = float(np.nanmax(med))
    m["f_at_z_max"] = float(edges[np.nanargmax(med)])
    if m["z_max"] - m["z_pad"] < LIFT_SIGMA * sigma:
        return {**m, "f_liftoff": float("nan"), "f_break": float("nan"),
                "reason": f"never rose {LIFT_SIGMA:.0f} sigma ({LIFT_SIGMA * sigma:.1f} mm) "
                          "above the pad"}

    # Turning point: first bin whose dz/bin clears the noise AND stays clear. Gaps
    # (empty bins) are interpolated so a dropout in the middle of the ramp does not
    # split the run of rising bins and hide the real edge.
    zi = np.interp(edges, edges[ok], med[ok])
    rising = np.diff(zi, append=zi[-1]) > RISE_SIGMA * sigma
    run = np.convolve(rising.astype(int), np.ones(HOLD_BINS, int), mode="full")[:len(edges)]
    hit = np.flatnonzero(run[: -HOLD_BINS + 1 or None] == HOLD_BINS)
    first = hit[0] - HOLD_BINS + 2 if len(hit) else 0   # dz[i] is the step INTO bin i+1
    m["f_liftoff"] = float(edges[min(first, len(edges) - 1)]) if len(hit) else float("nan")
    if not len(hit):
        m["reason"] = f"rose, but never {HOLD_BINS} bins in a row above {RISE_SIGMA:.0f} sigma"

    # Above f_break z is back in the pad band: the rotor has lost sync with the field.
    above = np.flatnonzero(ok & (med > m["z_pad"] + LIFT_SIGMA * sigma))
    m["f_break"] = float(edges[above[-1]])

    return m


def _plots(path: Path, m: dict, d: dict) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 6.5),
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(m["edges"], m["med_z"], ".-", lw=1)
    ax.axhline(m["z_pad"], color="0.6", ls="--", label=f"pad {m['z_pad']:.1f} mm")
    if np.isfinite(m.get("f_liftoff", np.nan)):
        ax.axvline(m["f_liftoff"], color="crimson",
                   label=f"f_liftoff {m['f_liftoff']:.0f} Hz")
    if np.isfinite(m.get("f_break", np.nan)):
        ax.axvline(m["f_break"], color="0.3", ls=":", label=f"f_break {m['f_break']:.0f} Hz")
    ax.set(xlabel="f_hz [Hz]", ylabel=f"median z per {BIN_HZ:.0f} Hz bin [mm]", title=path.stem)
    ax.legend(fontsize=8)

    # Commanded field with the witness latch stamped on it: where the rotor was provably
    # turning, where it was provably stopped, and (everywhere else) where nobody knows.
    spin = d.get("spin", np.full(len(d["t"]), ""))
    ax2.plot(d["t"], d["f_hz"], lw=1, color="0.6", label="f_hz commanded (field)")
    for state, colour in (("turning", "tab:green"), ("stopped", "crimson")):
        k = spin == state
        ax2.plot(d["t"][k], d["f_hz"][k], ".", ms=4, color=colour,
                 label=f"{state} ({int(k.sum())} rows)")
    ax2.set(xlabel="t [s]", ylabel="f_hz [Hz]", title=f"rotor: {m['captured']}")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_name(path.stem + "_z_vs_f.png"), dpi=130)
    plt.close(fig)

    if "i_meas" not in m:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m["i_f"], m["i_meas"], ".", ms=3, label="measured (mean of 4 coils)")
    if np.isfinite(m.get("L_H", np.nan)):
        w = 2 * np.pi * np.sort(m["i_f"])
        v = np.median(m["i_meas"] * np.hypot(RESISTANCE_OHM,
                                             2 * np.pi * m["i_f"] * m["L_H"]
                                             - 1.0 / (2 * np.pi * m["i_f"] * m["C_F"])))
        ax.plot(np.sort(m["i_f"]),
                v / np.hypot(RESISTANCE_OHM, w * m["L_H"] - 1.0 / (w * m["C_F"])),
                label=f"fit L={m['L_H'] * 1e3:.2f} mH C={m['C_F'] * 1e6:.0f} uF")
    ax.set(xlabel="f_hz [Hz]", ylabel="coil current [A]", title=path.stem)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_name(path.stem + "_i_vs_f.png"), dpi=130)
    plt.close(fig)


def report(path, plot: bool = True) -> dict:
    path = Path(path)
    d = load(path)
    m = metrics(d)
    lifted = np.isfinite(m.get("f_liftoff", np.nan))
    gappy = m["lost_frac"] > LOST_BAD
    p = lambda k, v: print(f"  {k:<14} {v}")

    print(f"takeoff  {path}")
    # First line of the block: every number below is only interpretable against the ramp
    # that produced it, and comparing two attempts starts here.
    p("ramp", m["ramp"] or "unrecorded (logged before the profile stamp existed)")
    p("ticks", f"{m['n_ticks']}  ({m['n_ramp'] if 'n_ramp' in m else 0} on the ramp with a fix)")
    p("lost", f"{m['lost_frac'] * 100:.1f} % of ticks had no fix "
              f"({m['n_lost_total']} frames dropped in total)"
              + ("   <-- TOO GAPPY TO TRUST" if gappy else ""))
    p("f_max reached", f"{m['f_max_reached']:.1f} Hz")
    if "z_pad" in m:
        p("z_pad", f"{m['z_pad']:.1f} mm   scatter {m['sigma_pad']:.2f} mm "
                   f"(1 sigma, {m['n_pad']} samples)")
        p("z_max", f"{m['z_max']:.1f} mm at {m['f_at_z_max']:.1f} Hz")
    p("rotor", f"{m['captured']}  ({m['n_turning']} turning / {m['n_stopped']} stopped rows, "
               f"rest unknown)"
               + ("   <-- z BELOW IS MEANINGLESS" if m["captured"] == "never turned" else ""))
    p("turning up to", (f"{m['f_turning_max']:.1f} Hz  (lower bound only; blanks above it "
                        "say nothing)") if m["n_turning"] else "never observed turning")
    p("f_liftoff", f"{m['f_liftoff']:.1f} Hz" if lifted else f"nan -- {m.get('reason', '')}")
    if lifted:
        p("f_break", f"{m['f_break']:.1f} Hz  (z back in the pad band above this)")
    if "L_H" in m and np.isfinite(m["L_H"]):
        print(f"  coil refit from {m['n_fit']} points above {FIT_F_MIN:.0f} Hz"
              f"        measured   z_track.py")
        p("  L", f"{m['L_H'] * 1e3:>10.2f} mH {INDUCTANCE_H * 1e3:>9.2f} mH")
        p("  C", f"{m['C_F'] * 1e6:>10.0f} uF {CAPACITANCE_F * 1e6:>9.0f} uF")
        p("  f_res", f"{m['f_res_hz']:>10.0f} Hz "
                     f"{1 / (2 * math.pi * math.sqrt(INDUCTANCE_H * CAPACITANCE_F)):>9.0f} Hz")
        p("  Q", f"{m['Q']:>10.2f}    {math.sqrt(INDUCTANCE_H / CAPACITANCE_F) / RESISTANCE_OHM:>9.2f}")
    else:
        p("coil refit", f"skipped, {m.get('n_fit', 0)} usable current points above "
                        f"{FIT_F_MIN:.0f} Hz")
    verdict = ("ROTOR NEVER TURNED, so nothing above is about a flying robot"
               if m["captured"] == "never turned"
               else f"lifted at {m['f_liftoff']:.0f} Hz, peak {m['z_max']:.0f} mm" if lifted
               else f"NO LIFTOFF -- {m.get('reason', 'unknown')}")
    print(f"  VERDICT        {verdict}; "
          + ("data NOT trustworthy" if gappy else "data ok")
          + f" ({m['lost_frac'] * 100:.0f} % lost)")
    if plot and "edges" in m:
        _plots(path, m, d)
    return m


def compare(n: int = 5, csv_dir=CSV_DIR) -> None:
    """Overlay the z-vs-f curves of the newest n attempts on one axis."""

    files = sorted(Path(csv_dir).glob("*.csv"))[-n:]
    fig, ax = plt.subplots(figsize=(7, 4))
    for f in files:
        m = metrics(load(f))
        if "edges" not in m:
            continue
        # Label by SHAPE, not by clock time: two attempts differ because their ramps
        # differed, and a timestamp cannot say which was which.
        line, = ax.plot(m["edges"], m["med_z"] - m["z_pad"], lw=1,
                        label=f"{f.stem}  {m['ramp'] or 'unrecorded'}")
        if np.isfinite(m.get("f_liftoff", np.nan)):
            ax.axvline(m["f_liftoff"], color=line.get_color(), ls=":", lw=1)
    ax.set(xlabel="f_hz [Hz]", ylabel="median z above pad [mm]",
           title=f"newest {len(files)} takeoff attempts")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(csv_dir) / "compare.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}  ({len(files)} attempts)")


def _synth(path, f_lift=None, f0=3.0, f1=160.0, n=1600, seed=0, spin="lock"):
    """
    A ramp with a known liftoff: flat pad noise, then a rise starting at f_lift.

    `spin` is what the blade witness latches: "lock" (turning, provable out to 50 Hz then
    blank), "zero" (a confident standstill at the bottom of the ramp, blank above the
    alias limit), "blank" (it never answered, so nothing is known either way).
    """

    rng = np.random.default_rng(seed)
    f = np.linspace(f0, f1, n)
    z = rng.normal(0.0, 0.6, n)                       # 0.6 mm pad scatter
    if f_lift is not None:
        z += np.clip(f - f_lift, 0, None) * 1.2       # 1.2 mm per Hz once it lifts
    l, c = 1.4e-3, 1.0 / ((2 * np.pi * 200.0) ** 2 * 1.4e-3)   # f_res = 200 Hz exactly
    w = 2 * np.pi * f
    i = 1.0 / np.hypot(RESISTANCE_OHM, w * l - 1.0 / (w * c))
    rows = [",".join(C.CSV_COLUMNS)]
    n_lost = 0
    for k in range(n):
        lost = k % 40 == 0                             # a few dropouts, blank x/y/z
        n_lost += lost                                 # CUMULATIVE, as live_viz writes it
        zs = "" if lost else f"{z[k]:.3f}"
        # `stopped` only below the alias limit; `turning` is provable higher up, and the
        # witness blanks rather than guessing everywhere else.
        sp = {"lock": "turning" if f[k] <= 50.0 else "",
              "zero": "stopped" if f[k] <= 3.1 else "",
              "blank": ""}[spin]
        rows.append(f"{k * 0.02:.3f},1,{f[k]:.2f},,,{zs},0,0,0,{sp},{n_lost},"
                    + ",".join([f"{i[k]:.4f}"] * 4))
    Path(path).write_text("\n".join(rows) + "\n")


def _self_check() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    for truth in (110.0, 130.0):
        _synth(tmp / f"lift{truth:.0f}.csv", f_lift=truth)
        m = report(tmp / f"lift{truth:.0f}.csv", plot=False)
        assert abs(m["f_liftoff"] - truth) <= 2 * BIN_HZ, (m["f_liftoff"], truth)
        # It flew to the end of the ramp, so f_break must be up at the top of it.
        assert m["f_break"] > 150.0, m["f_break"]
        print()

    _synth(tmp / "flat.csv", f_lift=None)
    m = report(tmp / "flat.csv", plot=False)
    assert math.isnan(m["f_liftoff"]), m["f_liftoff"]
    assert math.isnan(m["f_break"]), m["f_break"]
    assert 0.02 < m["lost_frac"] < 0.05, m["lost_frac"]   # 1 tick in 40 blanked

    # The three capture verdicts must not collapse: a stationary rotor and a witness
    # that never answered are different failures with different fixes.
    _synth(tmp / "nocap.csv", f_lift=None, spin="zero")
    m = report(tmp / "nocap.csv", plot=False)
    assert m["captured"] == "never turned", m["captured"]
    assert m["n_stopped"] > 0 and m["n_turning"] == 0, m
    assert math.isnan(m["f_turning_max"]), m["f_turning_max"]
    print()

    _synth(tmp / "noscan.csv", f_lift=120.0, spin="blank")
    m = report(tmp / "noscan.csv", plot=False)
    assert m["captured"] == "unknown", m["captured"]      # blank is NOT stopped
    assert m["n_stopped"] == 0 and m["n_turning"] == 0, m
    print()

    # Figures and the RLC refit only need to run once; the synth current is a true
    # RLC curve with f_res = 200 Hz, so the fit must find it.
    _synth(tmp / "plots.csv", f_lift=120.0)
    m = report(tmp / "plots.csv", plot=True)
    assert m["captured"] == "turned", m["captured"]
    assert abs(m["f_turning_max"] - 50.0) < 1.0, m["f_turning_max"]   # the 50 Hz bound
    assert (tmp / "plots_z_vs_f.png").exists() and (tmp / "plots_i_vs_f.png").exists()
    assert abs(m["f_res_hz"] - 200.0) < 20.0, m["f_res_hz"]
    print("\nself-check PASS")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--self-check":
        _self_check()
    elif args and args[0] == "--compare":
        compare(int(args[1]) if len(args) > 1 else 5)
    else:
        report(args[0] if args else sorted(CSV_DIR.glob("*.csv"))[-1])


def plot_tilt(paths=None, out=None, labels=None):
    """Rotor-normal trajectory, ONE POLAR PANEL PER RUN, plus tilt vs frequency.

    What is plotted is the normal on **S^2**, not a full SO(3) attitude: the estimator
    recovers the rotor's axis (`theta_deg` from the datum axis, `phi_deg` the azimuth it
    leans along) but NOT the spin angle about that axis, which aliases above ~fps/8 and is
    deliberately not tracked (`theory.md` 18.10). So this is the observable part of the
    attitude -- the geodesic angle from the datum axis, and the direction of the lean.

    One panel per run, NOT overlaid: an earlier version drew both runs into a single polar
    with a shared frequency colormap, which made trimmed and untrimmed indistinguishable
    -- the colour meant frequency, and nothing meant which run.

    Dot colour is drive frequency (dark = slow, bright = fast), so the tilt growing with
    thrust reads as the bright points sitting further out. The rightmost panel compares
    the runs directly.
    """

    import numpy as np

    paths = [Path(p) for p in (paths or sorted(Path(CSV_DIR).glob("*.csv"))[-2:])]
    n = len(paths)
    # Taller than it looks like it needs: the polar panels carry two-line titles
    # and tight_layout clips them against the axes.
    fig = plt.figure(figsize=(4.4 * n + 5.0, 5.4))
    lin = fig.add_subplot(1, n + 1, n + 1)
    sc = None
    for i, p in enumerate(paths):
        d = load(p)
        lab = (labels[i] if labels and i < len(labels)
               else f"{p.stem}  {d.get('ramp', '') or 'unrecorded'}")
        pol = fig.add_subplot(1, n + 1, i + 1, projection="polar")
        pol.set_title(f"{lab}\nrotor normal on $S^2$", fontsize=9, pad=16)
        pol.set_rmax(TILT_RMAX_DEG)
        pol.set_rlabel_position(135)
        if "tilt_deg" not in d:
            pol.text(0, 0, "no tilt columns\n(logged before 2026-09-01)",
                     ha="center", va="center", fontsize=8)
            continue
        f, th, ph = d["f_hz"], d["tilt_deg"], d["tilt_az_deg"]
        m = np.isfinite(f) & np.isfinite(th) & np.isfinite(ph)
        # Clip the radius: a departing robot tumbles to tens of degrees and one such blob
        # squashes the 1-3 deg structure the trim actually moves into the origin. Points
        # beyond the cap are DROPPED, not clamped -- a ring at the rim reads as real
        # attitude and is not.
        keep = m & (th < TILT_RMAX_DEG)
        sc = pol.scatter(np.radians(ph[keep]), th[keep], c=f[keep], s=3, cmap="viridis",
                         alpha=0.55, vmin=0, vmax=140)
        if m.sum() > 20:
            a = np.radians(ph[m & (f > 60)])
            if a.size:
                mean_az = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
                pol.annotate("", xy=(np.radians(mean_az), TILT_RMAX_DEG * 0.9), xytext=(0, 0),
                             arrowprops=dict(color="crimson", width=1.4, headwidth=7))
        edges = np.arange(0, np.nanmax(f[m]) + 10, 10.0)
        idx = np.digitize(f[m], edges) - 1
        med = np.array([np.median(th[m][idx == k]) if (idx == k).sum() > 10 else np.nan
                        for k in range(len(edges))])
        lin.plot(edges, med, marker="o", ms=3, lw=1.5, label=lab)

    if sc is not None:
        fig.colorbar(sc, ax=lin, label="drive frequency [Hz]", pad=0.02)
    lin.set(xlabel="drive frequency [Hz]", ylabel="tilt from datum axis [deg]",
            title="tilt vs frequency")
    lin.grid(alpha=0.3)
    lin.legend(fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(top=0.80, wspace=0.35)
    out = Path(out or Path(CSV_DIR) / "tilt_so2.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}   (red arrow = mean lean direction above 60 Hz)")
    return out
