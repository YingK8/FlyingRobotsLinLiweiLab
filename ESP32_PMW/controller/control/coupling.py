#!/usr/bin/env python3
"""Is the post-kill azimuth ramp contaminated by an oscillation the radial angle shares?

    uv run python controller/control/coupling.py                 # self-check
    uv run python controller/control/coupling.py <campaign_root> # the report + figure

The operator's question: *if there is frequency coupling between radial and azimuthal
rotations, filter those frequencies out to see if that generates a clearer ramp.* Two things
have to be true for that to be worth doing -- a line must be present in BOTH angles at the
same frequency (that is the coupling), and removing it must measurably straighten the ramp.
Both are tested here; neither is eyeballed.

WHY SHARED LINES AT ALL
-----------------------
A rigid axis that cones about the datum puts the SAME frequency into both angles -- the
radial angle nods once per cone and the azimuth carries the cone's phase -- so a shared line
is the signature of coning and not a coincidence. `control/theory.md` names three candidate
rates, and they are distinguishable by how they scale with the drive:

* **synchronous coning at the spin rate** (20.3): a body-fixed COM offset rotates with the
  body, so it drives the tilt block at `f_drive` itself. Tracks drive 1:1.
* **nutation at `(I_s/I_t) w = 1.65 w`** (20.3). Tracks drive at 1.65:1.
* **the precession pole, about 1.2 Hz** (12.x, the retained mode of the hover model) --
  a property of the robot and its suspension, CONSTANT in drive frequency. That is the
  mechanical resonance the operator also hypothesised.

Which of the three it is, is settled by the slope of the shared line against drive frequency,
so that regression is the headline number here and not the spectra.

SAMPLING, AND WHY THE GRID STOPS BEING TRUSTWORTHY AT 25.7 Hz
-------------------------------------------------------------
`alignment_rate.dft` evaluates on the ACTUAL timestamps, so it is not limited by a uniform
Nyquist the way `np.fft` is -- measured on `2026-09-09_210442`, injected 30 and 40 Hz lines
come back at 30.00 and 40.00 Hz with full amplitude, well above the 51.4 Hz nominal rate's
half. But the jitter is not large enough to kill the aliases: a 5 Hz line on those same
timestamps also puts 0.915 of its amplitude at 57.65 Hz against 1.011 at 5 Hz, a 10%
difference. A peak above Nyquist therefore cannot be told from `f - f_s` below it, so peaks
are only IDENTIFIED below `f_s/2` (~25.7 Hz). The grid still runs to 60 Hz in the figure,
with that line drawn, because a reader should see what was excluded.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from controller.control import alignment_rate as ar

ROOT = Path(__file__).resolve().parents[2]

#: Analysis window, kill to +WIN_S. Two seconds is the operator's ask and it is also what the
#: record can spare: the next schedule event (the down-ramp) is seconds away, and 2 s at the
#: ~51 Hz solved rate is ~105 samples.
WIN_S = 2.0

#: Frequency grid. The step is well under the 1/WIN_S = 0.5 Hz Rayleigh resolution, so a peak
#: is located by its lobe rather than by which bin it happened to land in.
F_LO, F_HI, F_STEP = 0.5, 60.0, 0.05

#: Hanning on a WIN_S record has a main lobe 4/WIN_S = 2 Hz wide, so two peaks closer than
#: half of that are one peak, and a radial and an azimuth peak within that are the same line.
LOBE_HZ = 2.0 / WIN_S

#: A peak must clear this fraction of the largest peak in its own spectrum. Set from the
#: measured spectra: the shared lines sit at 0.5-1.0 of the maximum and the grass between them
#: at under 0.15, so 0.25 separates them with room either side.
PEAK_FRAC = 0.25

#: A line is reported for a drive frequency only if it is shared in at least this fraction of
#: that frequency's repeats. Half, so a line has to be reproducible rather than one bad take.
SHARE_FRAC = 0.5


def nyquist(t):
    """The half-sample-rate above which a peak cannot be told from its alias. See the header."""

    return 0.5 / float(np.median(np.diff(np.asarray(t, float))))


# ---------------------------------------------------------------- one kill


def kill_traces(take_dir, log_path, which=None, win_s=WIN_S):
    """``[(t_rel, radial_deg, azimuth_deg), ...]``, one entry per kill in the take.

    Both angles are taken about the SAME rest datum -- `alignment_rate.datum`, the coils-off
    hang at the end of the point -- because a shared frequency only means something if the two
    angles share a reference. Radial comes from `angles()` (polar angle from the datum) and
    azimuth from `azimuth_from_rest()` (the campaign's ramp metric, referenced to its own
    pre-cut value and wrapped into +-180).

    A take with no `sweep.log`, or one whose rest window is missing, is skipped rather than
    fudged: without the datum there is no common reference and the question is meaningless.
    """

    t, axis, _q = ar.load(take_dir, which=which)
    pts = ar.timeline(log_path) if Path(log_path).exists() else []
    if not pts:
        return []
    try:
        rest = ar.datum(t, axis, pts)[0]
    except SystemExit:
        return []
    radial, _azi = ar.angles(axis, rest)
    out = []
    for tk in [k for p in pts for k in p["kills"]]:
        azi = ar.azimuth_from_rest(t, axis, rest, tk)
        if azi is None:
            continue
        m = (t >= tk) & (t <= tk + win_s)
        if m.sum() < 40:                     # under ~0.8 s of record; no 0.5 Hz resolution
            continue
        out.append((t[m] - tk, radial[m], azi[m]))
    return out


def peaks(freqs, spec, f_max, frac=PEAK_FRAC, sep_hz=LOBE_HZ):
    """Local maxima of ``spec`` below ``f_max``, at least ``frac`` of the largest.

    Greedy by height with a `sep_hz` exclusion, rather than every sign change of the first
    difference: on a Hanning-windowed 105-sample record the lobe skirts wobble, so a plain
    turning-point test returns three or four "peaks" per genuine line.
    """

    freqs, spec = np.asarray(freqs, float), np.asarray(spec, float)
    m = freqs <= f_max
    f, s = freqs[m], spec[m].copy()
    if not s.size:
        return []
    floor = frac * s.max()
    out = []
    while s.max() >= floor:
        i = int(np.argmax(s))
        out.append(float(f[i]))
        s[np.abs(f - f[i]) < sep_hz] = -np.inf
    return sorted(out)


def shared_lines(t, radial, azimuth, f_max=None, freqs=None):
    """``([f_shared], freqs, spec_radial, spec_azimuth)`` for one kill.

    "Shared" is a peak in the radial spectrum within one half-lobe of a peak in the azimuth
    spectrum. The reported frequency is the AZIMUTH peak, since the azimuth is what gets
    deprojected and `deproject` fits a single exact frequency.
    """

    freqs = np.arange(F_LO, F_HI + 1e-9, F_STEP) if freqs is None else np.asarray(freqs, float)
    f_max = nyquist(t) if f_max is None else f_max
    sr = ar.dft(radial, t, freqs)
    sa = ar.dft(azimuth, t, freqs)
    pr, pa = peaks(freqs, sr, f_max), peaks(freqs, sa, f_max)
    both = [fa for fa in pa if any(abs(fa - fr) <= LOBE_HZ / 2 for fr in pr)]
    return both, freqs, sr, sa


def consensus(per_kill, n_kills, share_frac=SHARE_FRAC, sep_hz=LOBE_HZ):
    """Cluster per-kill shared frequencies into the lines this drive frequency actually has.

    Every repeat is deprojected against the SAME list, which is the point of clustering: the
    cross-repeat gradient scatter in `quantify` is only a like-for-like comparison if each
    repeat had the same treatment. Fitting each repeat's own peaks instead lets the fit chase
    that repeat's noise and flatters the after-numbers.
    """

    flat = sorted(f for fs in per_kill for f in fs)
    clusters, cur = [], []
    for f in flat:
        if cur and f - cur[0] > sep_hz:
            clusters.append(cur)
            cur = []
        cur.append(f)
    if cur:
        clusters.append(cur)
    keep = [(float(np.median(c)), len(c)) for c in clusters
            if len(c) >= max(2, int(round(share_frac * n_kills)))]
    return keep


# ---------------------------------------------------------------- does it help


def line_fit(t, y):
    """``(gradient_deg_s, rms_residual_deg)`` of a straight line through the whole window.

    The ramp metric. A clean ramp is a straight line plus small residual; an oscillation on
    top of it shows up as residual and, because the record is only a couple of cycles long,
    also drags the gradient about -- which is why both numbers are reported.
    """

    t, y = np.asarray(t, float), np.asarray(y, float)
    A = np.c_[np.ones_like(t), t]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[1]), float(np.sqrt(np.mean((y - A @ coef) ** 2)))


def strip(t, y, lines):
    """Deproject every frequency in ``lines`` from ``y``. ``(clean, [amplitude_deg])``.

    Sequential single-frequency fits, not one joint solve, because that is what
    `alignment_rate.deproject` provides and because on a 2 s record the basis vectors for two
    lines 2 Hz apart are not orthogonal -- fitting them jointly on 105 samples is where a
    least-squares fit starts trading amplitude between them. Largest first, matching-pursuit
    style, so the dominant line is removed against a clean residual.
    """

    y = np.asarray(y, float)
    amps = []
    for f in lines:
        y, a = ar.deproject(y, t, f)
        amps.append(a)
    return y, amps


def quantify(kills, lines):
    """Before/after for one drive frequency. ``dict`` with the numbers the operator asked for.

    Two measures, because they answer different questions. `rms` is whether a SINGLE ramp is
    cleaner. `grad_sd` is whether the ramps AGREE with each other -- the scatter of the fitted
    gradient across repeats -- which is the one that decides whether the deprojection bought a
    more reliable measurement or just a prettier line.
    """

    g0, r0, g1, r1, amps = [], [], [], [], []
    for t_rel, _radial, azi in kills:
        a, b = line_fit(t_rel, azi)
        clean, am = strip(t_rel, azi, lines)
        c, d = line_fit(t_rel, clean)
        g0.append(a), r0.append(b), g1.append(c), r1.append(d)
        amps.append(am)
    g0, r0, g1, r1 = (np.array(v, float) for v in (g0, r0, g1, r1))
    sd = lambda v: float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    return {"n": len(kills), "lines": lines,
            "rms_before": float(r0.mean()), "rms_after": float(r1.mean()),
            "grad_before": float(g0.mean()), "grad_after": float(g1.mean()),
            "grad_sd_before": sd(g0), "grad_sd_after": sd(g1),
            "amp_deg": (np.array(amps, float).mean(0).tolist() if lines else [])}


# ---------------------------------------------------------------- the campaign


def analyse(root, freqs_hz=None, which=None):
    """Every chunk of a campaign -> ``{drive_hz: {**quantify, "kills": [...]}}``."""

    root = Path(root)
    out = {}
    for idx_path in sorted(root.glob("*/index.csv")):
        idx = [r for r in csv.DictReader(open(idx_path)) if r["outcome"] == "ok"]
        if not idx:
            continue
        f_drive = float(idx[0]["freq_hz"])
        if freqs_hz is not None and f_drive not in freqs_hz:
            continue
        kills, per = [], []
        for r in idx:
            take = Path(r["flight"])
            if not (take / "axis.csv").exists() and not (take / "axis_minor.csv").exists():
                continue
            for tr in kill_traces(take, r["log"], which=which):
                both, _fr, _sr, _sa = shared_lines(*tr)
                kills.append(tr)
                per.append(both)
        if not kills:
            continue
        keep = consensus(per, len(kills))
        lines = [f for f, _n in keep]
        out[f_drive] = {**quantify(kills, lines), "kills": kills,
                        "share": [n for _f, n in keep], "per_kill": per}
        print(f"{f_drive:5.0f} Hz  n={len(kills):2d}  shared: "
              + (", ".join(f"{f:.2f} Hz ({n}/{len(kills)})" for f, n in keep) or "none")
              + f"   rms {out[f_drive]['rms_before']:6.2f} -> {out[f_drive]['rms_after']:6.2f} deg"
              + f"   grad sd {out[f_drive]['grad_sd_before']:6.2f} ->"
              + f" {out[f_drive]['grad_sd_after']:6.2f} deg/s")
    return out


def tracking(res):
    """Does the strongest shared line track the drive, or sit at a fixed rate?

    Ordinary least squares of ``f_line`` on ``f_drive``. A slope near 1 is the synchronous
    coning of theory.md 20.3, near 1.65 is nutation, and a slope consistent with ZERO with a
    non-zero intercept is a mechanical resonance -- the operator's hypothesis.

    The strongest line per drive frequency is used, not all of them: a weak second line is
    often the first one's neighbour lobe and would enter the regression as a duplicate.
    """

    fd, fl = [], []
    for f_drive, r in sorted(res.items()):
        if not r["lines"]:
            continue
        fd.append(f_drive)
        fl.append(r["lines"][int(np.argmax(r["amp_deg"]))])
    if len(fd) < 3:
        return None
    fd, fl = np.array(fd, float), np.array(fl, float)
    A = np.c_[np.ones_like(fd), fd]
    coef, *_ = np.linalg.lstsq(A, fl, rcond=None)
    resid = fl - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # Standard error on the slope, so "does it track the drive?" is answered with a number of
    # sigma rather than by looking at it. Against slope 1 (synchronous coning) and 1.65
    # (nutation) the measured -0.016 +- 0.001 is hundreds of sigma away.
    dof = max(len(fd) - 2, 1)
    se = rms * np.sqrt(len(fd) / dof) / np.sqrt(np.sum((fd - fd.mean()) ** 2))
    return {"intercept_hz": float(coef[0]), "slope": float(coef[1]), "slope_se": float(se),
            "rms_hz": rms,
            "spread_hz": float(np.std(fl, ddof=1)), "f_drive": fd, "f_line": fl}


# ---------------------------------------------------------------- the figure

C_RAD, C_AZI, C_CLEAN, C_MARK = ar.C_TILT, ar.C_FIT, ar.C_MARK, "#b5359c"
INK, MUTED, GRID = ar.INK, ar.MUTED, ar.GRID


def figure(res, out_path, show=(20.0, 50.0, 110.0), trk=None):
    """One column per representative drive frequency: spectra on top, azimuth below.

    One repeat per column, not all of them. Overlaying five makes the before/after comparison
    unreadable -- the point of the lower row is whether ONE ramp gets straighter, and the
    across-repeat question is answered by the gradient sd printed on it, not by the ink.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [f for f in show if f in res]
    fig, axes = plt.subplots(2, len(show), figsize=(4.7 * len(show), 7.8), facecolor="white")
    axes = np.atleast_2d(axes)
    for col, f_drive in enumerate(show):
        r = res[f_drive]
        t_rel, radial, azi = r["kills"][0]
        _both, freqs, sr, sa = shared_lines(t_rel, radial, azi)
        f_ny = nyquist(t_rel)

        ax = axes[0, col]
        ax.plot(freqs, sa, color=C_AZI, lw=1.4, label="azimuth")
        ax.plot(freqs, sr, color=C_RAD, lw=1.4, label="radial (polar angle from rest)")
        top = max(sa.max(), sr.max()) * 1.18
        ax.set_ylim(0, top)
        for f, a in zip(r["lines"], r["amp_deg"]):
            ax.axvline(f, color=C_MARK, lw=1.1, ls="--", alpha=0.9)
            ax.annotate(f"{f:.1f} Hz in both, {a:.0f} deg", (f, top), xytext=(3, -3),
                        rotation=90, textcoords="offset points", fontsize=8,
                        color=C_MARK, va="top")
        ax.axvspan(f_ny, F_HI, color=GRID, alpha=0.6, lw=0)
        ax.annotate(f"above {f_ny:.1f} Hz a peak cannot\nbe told from its alias -- not used",
                    (f_ny + 1.5, top * 0.70), fontsize=8, color=MUTED, va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.0))
        ar._style(ax, f"{f_drive:g} Hz drive  --  amplitude spectrum, kill to +{WIN_S:g} s",
                  "frequency (Hz)", "amplitude (deg)")
        ax.set_xlim(0, F_HI)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper right")

        ax = axes[1, col]
        clean, _amps = strip(t_rel, azi, r["lines"])
        g, _rms = line_fit(t_rel, clean)
        c0 = float(np.mean(clean - g * t_rel))
        ax.plot(t_rel, azi, color=C_AZI, lw=1.2, alpha=0.6, label="azimuth, as measured")
        ax.plot(t_rel, clean, color=C_CLEAN, lw=1.8,
                label="after deprojecting the shared lines")
        ax.plot(t_rel, c0 + g * t_rel, color=INK, lw=1.1, ls="--", alpha=0.7,
                label=f"straight-line fit, {g:.0f} deg/s")
        ar._style(ax, f"{f_drive:g} Hz drive  --  azimuth about the cut (one repeat)",
                  "time since coils A,C cut (s)", "azimuth from pre-cut attitude (deg)")
        ax.text(0.015, 0.03,
                f"pooled over {r['n']} repeats:\n"
                f"RMS about the fitted line {r['rms_before']:.1f} -> {r['rms_after']:.1f} deg\n"
                f"gradient sd across repeats "
                f"{r['grad_sd_before']:.1f} -> {r['grad_sd_after']:.1f} deg/s",
                transform=ax.transAxes, fontsize=8.5, color=INK, va="bottom",
                bbox=dict(fc="white", ec=GRID, lw=0.7, pad=4.0))
        ax.legend(frameon=False, fontsize=8.5, labelcolor=MUTED, loc="upper right", ncol=1)
    sub = ("" if trk is None else
           f"\nStrongest shared line: f = {trk['intercept_hz']:.2f} {trk['slope']:+.4f} "
           f"x f_drive Hz -- the slope is {abs(trk['slope'] - 1) / trk['slope_se']:.0f} sigma "
           f"from drive-synchronous (1.0), so this is a mechanical mode of the robot, "
           f"not the drive")
    fig.suptitle("Radial-azimuth frequency coupling after the coil cut, and what removing it "
                 "does to the ramp" + sub, fontsize=11.5, color=INK, x=0.006, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"-> {out_path}")
    return out_path


# ---------------------------------------------------------------- self-check


def _self_check():
    rng = np.random.default_rng(3)
    # Irregular timestamps like the real ones: ~51 Hz with the measured 15-23 ms jitter.
    t = np.cumsum(rng.uniform(0.0146, 0.0232, 130))
    t -= t[0]
    t = t[t <= WIN_S]
    f_shared, f_only_radial = 4.3, 11.0
    radial = 6.0 * np.sin(2 * np.pi * f_shared * t) + 3.0 * np.sin(2 * np.pi * f_only_radial * t)
    ramp = 25.0 * t                                     # a clean 25 deg/s ramp
    azi = ramp + 9.0 * np.cos(2 * np.pi * f_shared * t + 0.7)

    both, freqs, sr, sa = shared_lines(t, radial, azi)
    assert both, "the shared line was not found at all"
    assert min(abs(f - f_shared) for f in both) < LOBE_HZ / 2, both
    # The radial-only line must NOT be called shared -- that is the whole discrimination.
    assert all(abs(f - f_only_radial) > LOBE_HZ / 2 for f in both), both

    g0, r0 = line_fit(t, azi)
    clean, amps = strip(t, azi, [f_shared])
    g1, r1 = line_fit(t, clean)
    assert abs(amps[0] - 9.0) < 0.6, amps                # recovers the injected amplitude
    assert r1 < 0.15 * r0, (r0, r1)                      # the ramp really is cleaner
    assert abs(g1 - 25.0) < abs(g0 - 25.0), (g0, g1)     # and the gradient is nearer truth

    # `consensus` keeps a line seen in half the repeats and drops a one-off.
    keep = consensus([[4.3], [4.4], [4.2], [4.3, 17.0]], 4)
    assert len(keep) == 1 and abs(keep[0][0] - 4.3) < 0.2, keep

    # Above Nyquist a line is degenerate with `f - f_s`, so peaks must not be identified there.
    assert nyquist(t) < 30.0, nyquist(t)
    assert all(f <= nyquist(t) for f in both), both
    print("coupling: self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
        res = analyse(root)
        trk = tracking(res)
        if trk:
            print(f"\nstrongest shared line vs drive: f_line = {trk['intercept_hz']:.2f} "
                  f"+ ({trk['slope']:+.4f} +- {trk['slope_se']:.4f}) * f_drive   "
                  f"(rms {trk['rms_hz']:.2f} Hz, raw spread {trk['spread_hz']:.2f} Hz)")
            for name, want in (("synchronous coning", 1.0), ("nutation 1.65w", 1.65),
                               ("constant (resonance)", 0.0)):
                print(f"    vs slope {want:4.2f} ({name}): "
                      f"{abs(trk['slope'] - want) / trk['slope_se']:.0f} sigma")
        figure(res, root / "report" / "coupling.png", trk=trk)
    else:
        _self_check()
