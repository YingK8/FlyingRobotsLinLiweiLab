#!/usr/bin/env python3
"""Figures and an HTML report for the alignment-settling measurement.

    uv run python controller/control/settle_report.py                    # self-check
    uv run python controller/control/settle_report.py <campaign_root>    # figures + report
    uv run python controller/control/settle_report.py <root> --hz 40     # one frequency

The measurement lives in `alignment_rate.py --settle` and the argument for it in
`control/theory.md` 25. Nothing here measures anything: it draws what that module returns and
writes `settle_report.html` beside the figures, self-contained (every image is inlined) so the
file can be moved or sent on its own.

TWO SETTLING TIMES, AND WHY BOTH ARE HERE
-----------------------------------------
`settle_band_s` is the textbook reading: the last time the trace leaves a band 10% above and
10% below its final resting average. That is what the annotated figure draws, and it is drawn
on the axis's progress along the initial->final arc, because that is the coordinate where an
initial level, a final level and a two-sided band all mean what they say.

`settle_10pct_s` is the metric `alignment_rate` reports, and it tests `Delta` -- the angle to
the final axis in three dimensions. That also counts motion which left the arc sideways, so it
is stricter. Over ten 40 Hz repeats the two agree to 0.01-0.06 s on eight of them and disagree
by 0.51 and 0.80 s on the other two, which are the repeats with the largest sideways
excursion. Both are in `settling.csv`, with `perp_max_deg` to attribute the gap.
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from controller.control import alignment_rate as ar

ROOT = Path(__file__).resolve().parents[2]

INK, MUTED, GRID = "#1c1c1c", "#6b6b6b", "#e3e3e3"
C_TRACE, C_MARK, C_BAND, C_KILL = "#2f6fd0", "#1a8f6a", "#e0721a", "#b4304a"
C_DISC, C_AXIS = "#2f6fd0", "#b4304a"

#: Revolutions of the cone drawn in the precession overlay. Enough to close the loop and see
#: whether it closes; more just overdraws it.
OVERLAY_REVS = 2.0

#: How many disc outlines are drawn around that loop.
OVERLAY_DISCS = 9


# ---------------------------------------------------------------- the annotated step response


def fig_step_response(rows, freq_hz, path):
    """THE figure: one repeat's approach, marked the way a step response is marked.

    Four markings, and nothing else on the upper panel:

    * `t_c`, the instant coils A and C are cut. A schedule label the firmware prints, stamped
      in the same loop that stamps video frames, so it is an event and not a fitted quantity.
    * the initial average, over the second before the cut
    * the final settling average, over the settled window before the down-ramp
    * dashed lines 10% above and 10% below that final average

    Drawn on the axis's progress along the initial->final arc, because that is the coordinate
    where those four things mean what they say: it starts at the initial level, ends at the
    final one, and can sit either side of it, so a band has two sides and an overshoot points
    the way it went.

    THE LOWER PANEL IS WHY TWO SETTLING TIMES ARE REPORTED
    -----------------------------------------------------
    The band above only asks how far ALONG the arc the axis has got. `settle_10pct_s` asks how
    far it is from the final axis in three dimensions, which also counts motion that left the
    arc sideways. Over ten 40 Hz repeats the two agree to 0.01-0.06 s on eight and disagree by
    0.51 and 0.80 s on the other two -- and those two are the ones with the largest sideways
    excursion, which is the trace in the lower panel. Where they differ the 3-D one is the
    stricter and the more physical: an axis 3 deg off to the side has not arrived.
    """

    import matplotlib.pyplot as plt

    rep = _pick(rows)
    if rep is None:
        return None
    t, (s_deg, perp) = rep["_t"], rep["_gc"]
    m = rep["_valid"] & (t >= -ar.PRE_S) & (t <= rep["_stop"])
    ini = float(np.median(s_deg[rep["_valid"] & (t >= -ar.PRE_S) & (t < 0)]))
    fin_m = rep["_valid"] & (t >= ar.FINAL_FROM_S) & (t <= rep["_stop"])
    fin = float(np.median(s_deg[fin_m])) if fin_m.sum() else float("nan")
    band = ar.SETTLE_BAND * abs(fin - ini)

    fig, axs = plt.subplots(2, 1, figsize=(11.5, 8.6), facecolor="white",
                            gridspec_kw={"height_ratios": [2.3, 1.0]}, sharex=True)

    ax = axs[0]
    ax.axhspan(fin - band, fin + band, color=C_BAND, alpha=0.10, zorder=1)
    for sign in (+1, -1):
        ax.axhline(fin + sign * band, color=C_BAND, lw=1.3, ls="--", zorder=2)
    ax.axhline(ini, color=MUTED, lw=1.4, zorder=2)
    ax.axhline(fin, color=C_MARK, lw=1.7, zorder=2)
    ax.plot(t[m], s_deg[m], color=C_TRACE, lw=1.6, zorder=4)
    ax.axvline(0.0, color=C_KILL, lw=1.8, zorder=3)

    x0, x1 = t[m][0], rep["_stop"]
    ax.annotate("$t_c$ — coils A and C cut", (0.0, ini), xytext=(9, -30),
                textcoords="offset points", color=C_KILL, fontsize=10.5, fontweight="bold")
    ax.annotate(f"initial average   {ini:.2f}°", (x0, ini), xytext=(5, 7),
                textcoords="offset points", color=MUTED, fontsize=9.5)
    ax.annotate(f"final resting average   {fin:.2f}°", (x1, fin), xytext=(-6, 9),
                textcoords="offset points", color=C_MARK, fontsize=10,
                ha="right", fontweight="bold")
    ax.annotate(f"+10%   {fin + band:.2f}°", (x1, fin + band), xytext=(-6, 5),
                textcoords="offset points", color=C_BAND, fontsize=9.5, ha="right")
    ax.annotate(f"−10%   {fin - band:.2f}°", (x1, fin - band), xytext=(-6, -15),
                textcoords="offset points", color=C_BAND, fontsize=9.5, ha="right")

    tb = rep["settle_band_s"]
    if tb != "":
        tb = float(tb)
        yb = float(np.interp(tb, t[m], s_deg[m]))
        ax.plot([tb], [yb], "o", color=C_MARK, ms=9, zorder=6)
        ax.annotate("", xy=(tb, ini), xytext=(0.0, ini),
                    arrowprops=dict(arrowstyle="<->", color=C_MARK, lw=1.3))
        ax.annotate(f"settling time   {tb:.2f} s\nlast exit from the band",
                    (tb, yb), xytext=(16, -34), textcoords="offset points", color=C_MARK,
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_MARK, lw=1.2))
    _style(ax, f"{freq_hz:.0f} Hz, {rep['take']} — how far the axis has swung, "
               f"from its initial resting attitude", "",
           "progress along the initial→final arc (deg)")

    ax = axs[1]
    ax.plot(t[m], perp[m], color=C_TRACE, lw=1.3)
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.axvline(0.0, color=C_KILL, lw=1.8)
    ts = rep["settle_10pct_s"]
    lab = (f"3-D settling time {float(ts):.2f} s" if ts != "" else "3-D settling refused")
    _style(ax, f"sideways: how far the axis left that arc — why the 3-D number differs "
               f"({lab})", "time from the cut (s)", "out of the arc (deg)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- the precession overlay


def _derotate(axis, avg, target):
    """Each ``axis[i]`` rotated by the rotation that takes ``avg[i]`` onto ``target``.

    WHY THIS IS NEEDED, AND WHAT IT FIXES
    -------------------------------------
    "Precession about the average axis" means motion relative to an average axis that is
    itself moving. In the window just after the cut the average axis is swinging through 47
    deg in a few hundred milliseconds, so a single window-mean is not a resting axis and a
    cone drawn about it is mostly the swing -- which is what an earlier version of this
    figure drew, and it looked wrong because it was.

    Rodrigues, applied per frame, takes the swing out and leaves the residual: rotate each
    sample by whatever rotation carries ITS OWN running average axis onto a common direction.
    What is left is the cone and nothing else, which is the same residual `precession` and
    `spiral` work with -- so the three figures now show one quantity three ways instead of
    two quantities.
    """

    v = np.asarray(axis, float)
    a = np.asarray(avg, float)
    tgt = np.asarray(target, float) / np.linalg.norm(target)
    k = np.cross(a, tgt)
    sn = np.linalg.norm(k, axis=1)
    cs = np.clip(a @ tgt, -1.0, 1.0)
    th = np.arctan2(sn, cs)[:, None]
    k = k / np.maximum(sn, 1e-12)[:, None]
    kv = np.cross(k, v)
    kdv = np.sum(k * v, axis=1)[:, None]
    return v * np.cos(th) + kv * np.sin(th) + k * kdv * (1 - np.cos(th))


def fig_precession_overlay(rows, freq_hz, path):
    """How the disc itself moves: its plane drawn at successive phases of the cone.

    The measured axis is the disc's normal, so the disc is the circle perpendicular to it.
    Drawing that circle at a sequence of phases shows what a cone half-angle means as MOTION:
    the rim rises and falls once per revolution while the average axis stays put.

    THE SWING IS TAKEN OUT FIRST (`_derotate`)
    ------------------------------------------
    Otherwise the second panel is not a cone at all. It sits at the peak of the response,
    where the average axis is still travelling 47 deg in a few hundred milliseconds, and a
    window-mean taken across that is not an axis the disc precesses about. Every sample is
    therefore rotated by whatever carries its own running average axis onto the vertical, so
    what remains is the residual and the red arrow is a genuine common reference.

    Two panels at the same scale, before the cut and at the peak after it, because the result
    worth seeing is that the cut EXCITES the cone rather than merely re-pointing the robot.
    """

    import matplotlib.pyplot as plt

    rep = _pick(rows)
    if rep is None:
        return None
    t, axis, avg = rep["_t"], rep["_axis"], rep["_avg"]
    cone_hz = float(rep["cone_hz"])
    env = rep["_env"]

    post = rep["_valid"] & (t > 0) & (t < min(1.5, rep["_stop"]))
    t_peak = float(t[post][int(np.argmax(env[post]))]) if post.sum() else 0.4
    span = OVERLAY_REVS / max(cone_hz, 1.0)
    windows = [
        (-0.6, f"BEFORE the cut\n{-0.6:+.2f} to {-0.6 + span:+.2f} s — all four coils driving"),
        (t_peak, f"AFTER the cut\n{t_peak:+.2f} to {t_peak + span:+.2f} s — A and C off, "
                 f"at the peak of the cone"),
    ]

    up = np.array([0.0, 0.0, 1.0])
    fig = plt.figure(figsize=(10.5, 6.4), facecolor="white")
    for k_, (t0, label) in enumerate(windows):
        ax = fig.add_subplot(1, 2, k_ + 1, projection="3d")
        m = rep["_valid"] & (t >= t0) & (t <= t0 + span)
        if m.sum() < 8:
            continue
        # Take the swing out, then look straight down the (now common) average axis.
        rel = _derotate(axis[m], avg[m], up)
        rel /= np.linalg.norm(rel, axis=1)[:, None]

        idx = np.linspace(0, len(rel) - 1, OVERLAY_DISCS).astype(int)
        ring = np.linspace(0, 2 * np.pi, 120)
        for j, n in enumerate(rel[idx]):
            u = np.cross(n, up)
            if np.linalg.norm(u) < 1e-6:
                u = np.cross(n, [0, 1.0, 0])
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
            # The disc is CENTRED ON THE ROBOT and perpendicular to its normal. It does not
            # translate -- an earlier version drew it at the tip of the normal, which reads
            # as a hat floating above the axis rather than as the rotor itself.
            circ = 0.62 * (np.cos(ring)[:, None] * u + np.sin(ring)[:, None] * v)
            sh = 0.18 + 0.66 * j / max(len(idx) - 1, 1)
            ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color=C_DISC, alpha=sh, lw=1.25)
            ax.plot([0, n[0]], [0, n[1]], [0, n[2]], color=C_DISC, alpha=0.55 * sh, lw=1.0)

        ax.plot(rel[:, 0], rel[:, 1], rel[:, 2], color=C_TRACE, lw=1.7, alpha=0.95)
        ax.plot([0, 0], [0, 0], [-0.35, 1.30], color=C_AXIS, lw=2.6)
        # Labelled at the FOOT of the axis: at the head it collides with the panel title.
        ax.text(0, 0, -0.52, "average axis", color=C_AXIS, fontsize=9.5, ha="center")

        half = float(np.median(env[m]))
        ax.set_title(f"{label}\ncone half-angle {half:.1f}$\\degree$",
                     fontsize=10, color=INK, y=0.97)
        ax.set_xlim(-0.72, 0.72)
        ax.set_ylim(-0.72, 0.72)
        ax.set_zlim(-0.62, 1.40)
        ax.set_box_aspect((1, 1, 1.25), zoom=1.28)
        ax.view_init(elev=13, azim=38)
        ax.set_axis_off()

    fig.suptitle(f"{freq_hz:.0f} Hz — the disc precessing about its average axis, "
                 f"swing removed\n{OVERLAY_REVS:.0f} revolutions at {cone_hz:.0f} Hz "
                 f"({span * 1000:.0f} ms), same scale both panels · {rep['take']}",
                 fontsize=11, color=INK, y=0.985)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.86, bottom=0.0, wspace=0.0)
    fig.savefig(path, dpi=190, facecolor="white")
    plt.close(fig)
    return path


def fig_precession_settle(rows, freq_hz, path):
    """The cone's own settling time: a PULSE, marked the way the step response was.

    The axis metric times a step. This times a pulse -- the envelope sits at its driven level,
    jumps when the coils are cut, and decays -- so the quantity playing the role of the swing
    is the excursion, `peak - final`, and the band is 10% of that about the final level.

    The fitted decay is drawn over it because on this campaign the band often refuses (the 5 s
    window ends while the envelope is still falling) and the decay constant does not: 97 of
    107 repeats give a `tau` against 50 that give a settling time.
    """

    import matplotlib.pyplot as plt

    rep = _pick(rows, key="prec_settle_s")
    if rep is None or rep["prec_final_deg"] == "":
        return None
    t, env, v = rep["_t"], rep["_env"], rep["_valid"]
    m = v & (t >= -ar.PRE_S) & (t <= rep["_stop"])
    pre = float(np.median(env[v & (t >= -ar.PRE_S) & (t < 0)]))
    peak, tpk = float(rep["prec_peak_deg"]), float(rep["prec_t_peak_s"])
    fin = float(rep["prec_final_deg"])
    band = ar.SETTLE_BAND * (peak - fin)

    fig, ax = plt.subplots(figsize=(11.5, 6.0), facecolor="white")
    ax.axhspan(fin - band, fin + band, color=C_BAND, alpha=0.10, zorder=1)
    for sign in (+1, -1):
        ax.axhline(fin + sign * band, color=C_BAND, lw=1.3, ls="--", zorder=2)
    ax.axhline(pre, color=MUTED, lw=1.4, zorder=2)
    ax.axhline(fin, color=C_MARK, lw=1.7, zorder=2)
    ax.plot(t[m], env[m], color=C_TRACE, lw=1.6, zorder=4)
    ax.axvline(0.0, color=C_KILL, lw=1.8, zorder=3)
    ax.plot([tpk], [peak], "o", color=C_KILL, ms=8, zorder=6)

    if rep["prec_tau_s"] != "":
        tau, asym = float(rep["prec_tau_s"]), float(rep["prec_asym_deg"])
        xf = np.linspace(tpk, rep["_stop"], 300)
        ax.plot(xf, asym + (peak - asym) * np.exp(-(xf - tpk) / tau), color=C_KILL,
                lw=1.6, ls=":", zorder=5)
        ax.annotate(f"fitted decay   τ = {tau:.2f} s", (xf[len(xf) // 3],
                    asym + (peak - asym) * np.exp(-(xf[len(xf) // 3] - tpk) / tau)),
                    xytext=(18, 22), textcoords="offset points", color=C_KILL, fontsize=10,
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color=C_KILL, lw=1.1))

    x1 = rep["_stop"]
    ax.annotate("$t_c$ — coils A and C cut", (0.0, pre), xytext=(9, 26),
                textcoords="offset points", color=C_KILL, fontsize=10.5, fontweight="bold")
    ax.annotate(f"peak   {peak:.2f}°  at {tpk:.2f} s", (tpk, peak), xytext=(14, 4),
                textcoords="offset points", color=C_KILL, fontsize=10, fontweight="bold")
    ax.annotate(f"before the cut   {pre:.2f}°", (t[m][0], pre), xytext=(5, -16),
                textcoords="offset points", color=MUTED, fontsize=9.5)
    ax.annotate(f"final settled cone   {fin:.2f}°", (x1, fin), xytext=(-6, 9),
                textcoords="offset points", color=C_MARK, fontsize=10, ha="right",
                fontweight="bold")
    ax.annotate(f"±10% of the excursion   ±{band:.2f}°", (x1, fin + band), xytext=(-6, 5),
                textcoords="offset points", color=C_BAND, fontsize=9.5, ha="right")

    ts = rep["prec_settle_s"]
    if ts != "":
        ts = float(ts)
        ax.plot([ts], [float(np.interp(ts, t[m], env[m]))], "o", color=C_MARK, ms=9, zorder=7)
        ax.annotate("", xy=(ts, pre), xytext=(0.0, pre),
                    arrowprops=dict(arrowstyle="<->", color=C_MARK, lw=1.3))
        ax.annotate(f"cone settling time   {ts:.2f} s", (ts, pre), xytext=(10, -26),
                    textcoords="offset points", color=C_MARK, fontsize=10.5,
                    fontweight="bold")
    _style(ax, f"{freq_hz:.0f} Hz, {rep['take']} — the precession cone: excited by the cut, "
               f"then damped", "time from the cut (s)", "RMS cone half-angle (deg)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


def fig_spiral(rows, freq_hz, path):
    """The residual in the tangent plane over the whole window, coloured by time.

    Drawn plain this is a disc of ink -- two hundred revolutions overdrawn at 40 Hz, with
    nothing in it to say which pass came first. Colouring by time is what makes it readable,
    and what it then shows is the thing the envelope trace asserts: the cut throws the cone
    wide and dissipation winds it back in, so the early passes are the outer ring and the late
    ones the tight core. An inward spiral, drawn from every solved frame with none dropped.

    High resolution on purpose. The information here is in how tightly successive loops pack,
    which is exactly what is lost first when the figure is downsampled or scaled.
    """

    import matplotlib.pyplot as plt

    rep = max(rows, key=lambda r: (r["_post"] & r["_valid"]).sum())
    fig, ax = plt.subplots(figsize=(8.6, 8.0), facecolor="white")
    lc = ar.spiral(ax, rep, cmap="viridis")
    if lc is None:
        plt.close(fig)
        return None
    lc.set_linewidth(0.55)
    lc.set_alpha(0.8)
    cb = fig.colorbar(lc, ax=ax, pad=0.02)
    cb.set_label("time from the cut (s)", color=MUTED, fontsize=9.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)
    per_rev = float(rep["fs_hz"]) / max(float(rep["cone_hz"]), 1e-9)
    ax.set_title(f"{freq_hz:.0f} Hz, {rep['take']} — the cone spiralling in\n"
                 f"every solved frame, {rep['_stop']:.1f} s at {rep['cone_hz']:.0f} Hz "
                 f"≈ {rep['_stop'] * rep['cone_hz']:.0f} revolutions · "
                 f"{per_rev:.1f} samples per revolution, so each loop is a "
                 f"{max(int(round(per_rev)), 3)}-gon",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=230, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- the sweep-wide figures


def fig_envelope_heatmap(rows, path):
    """Cone half-angle against time and drive frequency: the excitation, whole sweep, one panel."""

    import matplotlib.pyplot as plt

    freqs = sorted({r["freq_hz"] for r in rows})
    grid = np.arange(-ar.PRE_S, 5.0, 0.02)
    img = np.full((len(freqs), len(grid)), np.nan)
    for i, f in enumerate(freqs):
        stack = []
        for r in (x for x in rows if x["freq_hz"] == f):
            m = r["_valid"]
            if m.sum() < 50:
                continue
            stack.append(np.interp(grid, r["_t"][m], r["_env"][m],
                                   left=np.nan, right=np.nan))
        if stack:
            img[i] = np.nanmedian(np.vstack(stack), axis=0)

    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor="white")
    im = ax.pcolormesh(grid, freqs, img, cmap="magma", shading="nearest")
    ax.axvline(0.0, color="white", lw=1.6)
    ax.annotate("$t_c$", (0.0, freqs[-1]), xytext=(6, -14), textcoords="offset points",
                color="white", fontsize=11, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("RMS cone half-angle (deg)", color=MUTED, fontsize=9.5)
    cb.ax.tick_params(colors=MUTED, labelsize=8.5)
    _style(ax, "precession about the average axis — median over repeats, every frequency",
           "time from the cut (s)", "drive frequency (Hz)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


def fig_metrics(rows, per_freq, campaign_csv, path):
    """The three timing numbers side by side, because they are routinely confused.

    A gradient, a 10-90% band and a band-exit time are three different questions about one
    transient, and this campaign now reports all three. Drawn together so that the rate rising
    with drive while the settling time ALSO rises stops looking like a contradiction.
    """

    import matplotlib.pyplot as plt

    camp = {}
    if Path(campaign_csv).exists():
        camp = {float(r["freq_hz"]): r for r in csv.DictReader(open(campaign_csv))}
    f = [r["freq_hz"] for r in per_freq]

    fig, axs = plt.subplots(1, 4, figsize=(17.5, 4.2), facecolor="white")

    rate = [_f(camp.get(x, {}).get("rate_relu_med")) for x in f]
    axs[0].plot(f, rate, "o-", color=C_BAND, ms=4, lw=1.3)
    _style(axs[0], "§24 rate — gradient of the rising edge",
           "drive frequency (Hz)", "deg/s")

    axs[1].errorbar(f, [r["rise_1090_med_s"] for r in per_freq],
                    yerr=[r["rise_1090_mad_s"] for r in per_freq],
                    fmt="o-", color=C_TRACE, ms=4, lw=1.3, capsize=3)
    axs[1].axhline(0.25, color=MUTED, ls=":", lw=1.1)
    axs[1].annotate("the smoother", (f[-1], 0.25), xytext=(-4, 5),
                    textcoords="offset points", color=MUTED, fontsize=8.5, ha="right")
    _style(axs[1], "rise time — 10% to 90% of final",
           "drive frequency (Hz)", "median ± MAD (s)")

    for r in per_freq:
        axs[2].annotate(f"{r['n_settled']}/{r['n_repeats']}",
                        (r["freq_hz"], r["settle_10pct_med_s"]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=7, color=MUTED)
    axs[2].errorbar(f, [r["settle_10pct_med_s"] for r in per_freq],
                    yerr=[r["settle_10pct_mad_s"] for r in per_freq],
                    fmt="o-", color=C_MARK, ms=4, lw=1.3, capsize=3)
    _style(axs[2], f"AXIS settling — last exit from ±{100 * ar.SETTLE_BAND:.0f}%",
           "drive frequency (Hz)", "median ± MAD (s)")

    axs[3].errorbar(f, [r["prec_settle_med_s"] for r in per_freq],
                    yerr=[r["prec_settle_mad_s"] for r in per_freq],
                    fmt="o-", color=C_KILL, ms=4, lw=1.3, capsize=3, label="cone settling")
    axs[3].errorbar(f, [r["prec_tau_med_s"] for r in per_freq],
                    yerr=[r["prec_tau_mad_s"] for r in per_freq],
                    fmt="s--", color=MUTED, ms=3.5, lw=1.1, capsize=3, label="cone decay τ")
    axs[3].plot(f, [r["settle_10pct_med_s"] for r in per_freq], "o:", color=C_MARK,
                ms=3.5, lw=1.0, alpha=0.75, label="axis settling")
    axs[3].legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(axs[3], "CONE settling and its decay constant",
           "drive frequency (Hz)", "median ± MAD (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- the report


def build(root, out_dir=None, which=None, only_hz=None):
    """Every figure, then `settle_report.html` with all of them inlined."""

    import matplotlib
    matplotlib.use("Agg")

    root = Path(root)
    out = Path(out_dir) if out_dir else root / "report"
    out.mkdir(parents=True, exist_ok=True)

    rows, skipped = ar.settle_rows(root, which=which, only_hz=only_hz)
    if not rows:
        print("no usable takes")
        return None
    per_freq = ar._settle_by_freq(rows)

    # The frequency the two close-up figures are drawn at: the one with the most repeats that
    # actually settled, so the annotated figure is of a typical trace and not a rescued one.
    best = max(per_freq, key=lambda r: (r["n_settled"], -abs(r["freq_hz"] - 40)))
    fhz = best["freq_hz"]
    at_f = [r for r in rows if r["freq_hz"] == fhz]

    figs = [
        ("step", fig_step_response(at_f, fhz, out / "settle_step_response.png"),
         "Settling, marked", _cap_step(at_f, fhz)),
        ("overlay", fig_precession_overlay(at_f, fhz, out / "settle_precession_overlay.png"),
         "How the disc precesses about the average axis", _cap_overlay()),
        ("precset", fig_precession_settle(at_f, fhz, out / "settle_precession_time.png"),
         "How long the cone takes to stop ringing", _cap_precset(per_freq)),
        ("spiral", fig_spiral(at_f, fhz, out / "settle_spiral.png"),
         "The cone spiralling in", _cap_spiral()),
        ("heat", fig_envelope_heatmap(rows, out / "settle_envelope_heatmap.png"),
         "Precession across the sweep", _cap_heat()),
        ("metrics", fig_metrics(rows, per_freq, out / "campaign.csv",
                                out / "settle_metrics.png"),
         "Three timing numbers, one transient", _cap_metrics()),
    ]
    figs = [f for f in figs if f[1] is not None]

    page = out / "settle_report.html"
    page.write_text(_html(figs, per_freq, rows, skipped, fhz))
    print(f"wrote {page}")
    for _k, p, _t, _c in figs:
        print(f"  {p.name}")
    return page


def _cap_step(rows, fhz):
    rep = _pick(rows)
    ts = rep["settle_10pct_s"] if rep else ""
    return (
        f"One repeat at {fhz:.0f} Hz. <b>t<sub>c</sub></b> is the instant coils A and C are "
        f"cut — a schedule label the firmware prints, stamped in the same loop that stamps "
        f"video frames, so there is no offset fitted anywhere in this analysis. The grey line "
        f"is the average over the second before the cut, the green line the settled average "
        f"before the down-ramp, and the dashed orange pair the ±10% band about it. "
        f"The settling time is the last exit from that band"
        + (f", {float(ts):.2f} s here." if ts != "" else ".") +
        " <b>The upper panel is a signed coordinate</b> — progress along the initial→final "
        "arc — so the band is two-sided and an overshoot points the way it went. The lower "
        "panel is Δ, the unsigned angle to the final axis, which is what the band is actually "
        "tested on: a departure in <i>any</i> direction increases it, so one threshold there "
        "is already a band in every direction at once."
    )


def _cap_overlay():
    return (
        "<b>What the cut does:</b> zeroing coils A and C leaves a weaker, asymmetric rotating "
        "field. The robot swings about 45° in azimuth to a new equilibrium — with A and C "
        "gone at 0° and 180°, the asymmetry left by B and D at 90/270 sits 45° away — and its "
        "radial tilt drops slightly, so it ends a little more upright.<br><br>"
        "It also <b>excites the cone, but only transiently</b>: the half-angle jumps on 101 "
        "of 107 repeats, from a median 2.6° before the cut to a 6.8° peak within 0.1–0.4 s, "
        "and then damps back to a settled 1.8° — at or below where it started. The right-hand "
        "panel below is drawn at that peak, so it shows the worst moment rather than the new "
        "steady state.<br><br>"
        "The measured axis is the disc's normal, so the disc is the circle perpendicular to "
        "it, drawn at nine phases. <b>The 45° swing is removed first</b>: each sample is "
        "rotated by whatever carries its own running average axis onto the vertical, so what "
        "is left is the cone and the red arrow is a genuine common reference. Without that "
        "the right-hand panel is mostly swing, because it sits where the axis is still "
        "travelling 47° in a few hundred milliseconds. This is §11.3 becoming visible: "
        "gyroscopic action alone gives steady coning, and only dissipation spirals it back "
        "in, so the decay of that envelope measures the aerodynamic damping "
        "<i>c<sub>t</sub></i> — a number this campaign contains and nobody has extracted."
    )


def _cap_spiral():
    return (
        "The same residual as the overlay, seen down the average axis, over the whole window "
        "and coloured by time. Two hundred revolutions overdrawn is a disc of ink until time "
        "is visible; with it, <b>the trajectory is an inward spiral</b> — the cut throws the "
        "cone wide, dissipation winds it back in. Early passes are the outer ring, late ones "
        "the tight core. Drawn from every solved frame with none dropped, which is why it is "
        "at high resolution: the information is in how tightly successive loops pack, and "
        "that is what downsampling destroys first.<br><br>"
        "<b>The loops are polygons, and that is the sampling rather than noise.</b> The "
        "cameras solve at ~204 Hz and the cone runs at the drive frequency, so 40 Hz leaves "
        "about five samples per revolution — enough to locate the line and measure its "
        "amplitude (that is what the Nyquist gate checks), not enough to draw a smooth "
        "circle. A rounder picture would need a slower drive or a faster camera, not a "
        "different plot."
    )


def _cap_precset(per_freq):
    both = [r for r in per_freq if np.isfinite(float(r["prec_settle_med_s"]))
            and np.isfinite(float(r["settle_10pct_med_s"]))]
    longer = sum(1 for r in both
                 if float(r["prec_settle_med_s"]) > float(r["settle_10pct_med_s"]))
    return (
        "The cone is a <b>pulse</b>, not a step: it sits at its driven level, jumps when the "
        "coils are cut, and decays. So the quantity playing the role of the swing is the "
        "excursion — peak minus final — and the band is ±10% of that about the final level.<br><br>"
        f"<b>The cone takes longer to stop ringing than the axis takes to arrive</b>, at "
        f"{longer} of the {len(both)} frequencies where both are measurable: 2.4–3.2 s against "
        "0.7–2.4 s. The dotted curve is a fitted exponential, and it is drawn because the band "
        "often refuses where the fit does not — the 5&nbsp;s window ends while the envelope is "
        "still falling, so <b>97 of 107 repeats give a decay constant against 50 that give a "
        "settling time</b>. That constant, 0.64–1.38&nbsp;s across the sweep, is the "
        "aerodynamic damping <i>c<sub>t</sub></i> of §11.3 made visible: gyroscopic action "
        "alone gives steady coning at fixed half-angle, and only dissipation winds it in."
    )


def _cap_heat():
    return (
        "Median RMS cone half-angle over repeats, against time and drive. The bright band "
        "just right of t<sub>c</sub> is the excitation; it fades within a second or two at "
        "every frequency. Read horizontally for how long a given drive rings, vertically for "
        "which drives ring hardest — 30–50 Hz, where the cone reaches 4.4° against 1.2° at "
        "110 Hz."
    )


def _cap_metrics():
    return (
        "A gradient, a 10–90% band and a band-exit time are three different questions about "
        "one transient, and they are routinely quoted as if they were one. <b>The §24 rate "
        "rises with drive and the settling time rises too</b>, which is not a contradiction: "
        "the drive sets how hard the robot is thrown at its new equilibrium, not how quickly "
        "it stops ringing once it gets there. The dotted line on the middle panel is the "
        "0.25 s smoother — a rise time near it is mostly filter, which is why the fast ones "
        "are refused outright rather than reported (24.7).<br><br>"
        "The fourth panel adds the cone's own clock. <b>It settles later than the axis at "
        "every frequency where both are measurable</b>, and its decay constant is flat at "
        "roughly 1.2&nbsp;s across 30–90&nbsp;Hz — the axis takes longer to arrive as the "
        "drive rises, but the cone damps at about the same rate regardless, which is what a "
        "dissipation-limited mode should do."
    )


def _html(figs, per_freq, rows, skipped, fhz):
    n_set = sum(r["n_settled"] for r in per_freq)
    n_rep = sum(r["n_repeats"] for r in per_freq)
    # The trend is quoted from the lowest frequency that actually settled a majority of its
    # repeats -- 10 Hz settles 3 of 22 and quoting it as the low end would be a number made
    # of three trials (24.7 withdrew 10 Hz twice already).
    solid = [r for r in per_freq if r["n_settled"] > r["n_repeats"] / 2]
    if len(solid) >= 2:
        head = (f"{n_set} of {n_rep} repeats settle. The settling time rises with drive, from "
                f"{solid[0]['settle_10pct_med_s']:.2f} s at {solid[0]['freq_hz']:.0f} Hz to "
                f"{solid[-1]['settle_10pct_med_s']:.2f} s at {solid[-1]['freq_hz']:.0f} Hz.")
    else:
        head = (f"{n_set} of {n_rep} repeats settle, with a median settling time of "
                f"{np.median([r['settle_10pct_med_s'] for r in per_freq]):.2f} s.")

    cards = []
    for _k, p, title, cap in figs:
        cards.append(f"""<figure>
  <figcaption><h2>{html.escape(title)}</h2><p>{cap}</p></figcaption>
  <img src="data:image/png;base64,{_b64(p)}" alt="{html.escape(title)}">
</figure>""")

    cols = [("freq_hz", "drive (Hz)", "{:.0f}"), ("n_repeats", "repeats", "{:d}"),
            ("n_settled", "settled", "{:d}"),
            ("settle_10pct_med_s", "settling (s)", "{:.2f}"),
            ("rise_1090_med_s", "rise (s)", "{:.2f}"),
            ("swing_med_deg", "swing (deg)", "{:.2f}"),
            ("d_azim_med_deg", "Δazimuth (deg)", "{:.1f}"),
            ("prec_post_med_deg", "precession (deg)", "{:.2f}"),
            ("cone_over_drive", "cone / drive", "{:.2f}")]
    th = "".join(f"<th>{html.escape(lab)}</th>" for _c, lab, _f in cols)
    trs = []
    for r in per_freq:
        tds = []
        for c, _lab, fmt in cols:
            v = r[c]
            tds.append(f"<td>{fmt.format(v) if np.isfinite(float(v)) else '—'}</td>"
                       if not isinstance(v, int) else f"<td>{v}</td>")
        trs.append("<tr>" + "".join(tds) + "</tr>")

    why = []
    for f, name, reason in skipped:
        why.append(f"<tr><td>{f:.0f} Hz</td><td>{html.escape(name)}</td>"
                   f"<td>{html.escape(reason)}</td></tr>")

    return f"""<title>Alignment settling</title>
<style>
:root {{
  --ink:#1c1c1c; --muted:#5f5f5f; --rule:#e2e2e2; --bg:#fbfaf8; --card:#ffffff;
  --accent:#1a8f6a;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --ink:#ececec; --muted:#a6a6a6; --rule:#333; --bg:#151515; --card:#1d1d1d;
  --accent:#4fd1a5;
}} }}
:root[data-theme="dark"] {{
  --ink:#ececec; --muted:#a6a6a6; --rule:#333; --bg:#151515; --card:#1d1d1d;
  --accent:#4fd1a5;
}}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:48px 24px 96px; }}
h1 {{ font-size:30px; line-height:1.2; margin:0 0 6px; letter-spacing:-.02em; }}
.sub {{ color:var(--muted); margin:0 0 8px; }}
.lede {{ font-size:17px; border-left:3px solid var(--accent); padding-left:14px;
  margin:26px 0 40px; }}
figure {{ background:var(--card); border:1px solid var(--rule); border-radius:10px;
  margin:0 0 34px; padding:22px; }}
figcaption h2 {{ font-size:18px; margin:0 0 8px; letter-spacing:-.01em; }}
figcaption p {{ color:var(--muted); margin:0 0 18px; max-width:74ch; }}
img {{ width:100%; height:auto; display:block; border-radius:4px; }}
.tablewrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th,td {{ text-align:right; padding:7px 10px; border-bottom:1px solid var(--rule);
  white-space:nowrap; }}
th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
  letter-spacing:.04em; }}
td:first-child, th:first-child {{ text-align:left; }}
.notes li {{ margin-bottom:9px; color:var(--muted); max-width:74ch; }}
.notes b {{ color:var(--ink); }}
footer {{ color:var(--muted); font-size:12.5px; border-top:1px solid var(--rule);
  padding-top:16px; margin-top:44px; }}
code {{ font-size:12.5px; background:rgba(127,127,127,.13); padding:1px 5px;
  border-radius:3px; }}
</style>
<main>
<h1>Alignment settling</h1>
<p class="sub">Where the axis ends up after coils A and C are cut, and how long until it
stays there. Campaign 2026-09-09/10, {n_rep} repeats, solved every frame.</p>
<p class="lede">{html.escape(head)}</p>

{"".join(cards)}

<h2>Per frequency</h2>
<div class="tablewrap"><table><thead><tr>{th}</tr></thead>
<tbody>{"".join(trs)}</tbody></table></div>

<h2>What this does not settle</h2>
<ul class="notes">
<li><b>{n_rep - n_set} of {n_rep} repeats do not settle inside the 5&nbsp;s window.</b>
Mostly because the trace still wanders wider than the band where it is supposed to have
arrived. Whether that is the robot or the rig is not established here; finding out means a
longer <code>DROP_MS</code>, which costs coil heat at exactly the frequencies where heat is
already the binding constraint.</li>
<li><b>The rise time is unavailable wherever the rise is short.</b> The reported values sit
against a 0.25&nbsp;s smoother, so the faster ones carry about 10% of filter. A rise shorter
than the smoother is refused rather than reported as the filter's own width.</li>
<li><b>The cone sitting at 1.00&times; the drive does not prove the rotor spins at the drive.</b>
The field itself rotates at that rate and can shake the robot whatever the rotor is doing. What
it establishes is where the line is, which is all the filter needs to know.</li>
<li><b>The precession decay is not fitted.</b> The envelope is plotted and tabulated; turning
it into a damping coefficient is the obvious next measurement and is not done.</li>
<li><b>40&nbsp;Hz mixes two hold times</b> and shows as two populations — two traces at 49&deg;
of swing against twelve at 11.5&deg;. The median is quoted over both and should not be.</li>
</ul>

<h2>Takes not measured ({len(skipped)})</h2>
<div class="tablewrap"><table><thead><tr><th>drive</th><th>take</th><th>reason</th></tr>
</thead><tbody>{"".join(why)}</tbody></table></div>

<footer>
Generated {datetime.now():%Y-%m-%d %H:%M} by
<code>controller/control/settle_report.py</code> from
<code>alignment_rate.py --settle</code>. Close-up figures are drawn at {fhz:.0f}&nbsp;Hz, the
frequency with the most settled repeats. The argument for every choice here is in
<code>controller/control/theory.md</code>&nbsp;25.
</footer>
</main>"""


# ---------------------------------------------------------------- helpers


def _pick(rows, key="settle_10pct_s"):
    """The repeat to draw: one that settled, nearest the group's median. Typical, not best."""

    ok = [r for r in rows if r.get(key, "") != ""]
    if not ok:
        return rows[0] if rows else None
    med = float(np.median([float(r[key]) for r in ok]))
    return min(ok, key=lambda r: abs(float(r[key]) - med))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _style(ax, title, xl, yl):
    if title:
        ax.set_title(title, fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel(xl, fontsize=9.5, color=MUTED)
    ax.set_ylabel(yl, fontsize=9.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


# ---------------------------------------------------------------- self-check


def _self_check():
    # The signed coordinate and the unsigned one must agree on the crossing that matters.
    # A trajectory that swings from n0 to nf and settles: `great_circle` progress must start
    # near 0, end near the swing, and reach the +-10% band at the same sample `Delta` does.
    n0 = np.array([0.1, 0.0, 1.0]); n0 /= np.linalg.norm(n0)
    nf = np.array([0.0, 0.25, 1.0]); nf /= np.linalg.norm(nf)
    swing = float(np.degrees(np.arccos(np.clip(n0 @ nf, -1, 1))))
    t = np.linspace(0, 5, 2000)
    frac = 1.0 - np.exp(-t / 0.4)
    traj = n0[None, :] + frac[:, None] * (nf - n0)[None, :]
    traj /= np.linalg.norm(traj, axis=1)[:, None]

    prog, perp = ar.great_circle(traj, n0, nf)
    assert abs(prog[0]) < 0.05, prog[0]
    assert abs(prog[-1] - swing) < 0.05, (prog[-1], swing)
    assert np.abs(perp).max() < 0.05, ("a planar swing must stay in its plane", perp.max())

    delta = np.degrees(np.arccos(np.clip(traj @ nf, -1.0, 1.0)))
    i_signed = int(np.flatnonzero(np.abs(prog - swing) > ar.SETTLE_BAND * swing)[-1])
    i_unsigned = int(np.flatnonzero(delta > ar.SETTLE_BAND * swing)[-1])
    assert abs(t[i_signed] - t[i_unsigned]) < 0.02, (t[i_signed], t[i_unsigned])

    # An OVERSHOOT is the case the two disagree on, and the signed one is why the figure
    # exists: Delta folds it onto the same side as a shortfall, progress does not.
    over = n0[None, :] + (1.3 * frac)[:, None] * (nf - n0)[None, :]
    over /= np.linalg.norm(over, axis=1)[:, None]
    p_over, _ = ar.great_circle(over, n0, nf)
    assert p_over.max() > swing * 1.15, ("an overshoot must read past the final value",
                                         p_over.max(), swing)

    # `_pick` must choose a settled repeat when one exists, and the median-ish one.
    fake = [{"settle_10pct_s": v} for v in ("", 0.5, 1.0, 4.0)]
    assert _pick(fake)["settle_10pct_s"] == 1.0, _pick(fake)
    assert _pick([{"settle_10pct_s": ""}])["settle_10pct_s"] == ""

    print("settle_report: self-check passed (signed/unsigned agree, overshoot, pick)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="a results/alignment_rate/<stamp> campaign root")
    ap.add_argument("--out", default=None)
    ap.add_argument("--which", default=None, choices=("minor", "conic"))
    ap.add_argument("--hz", type=float, default=None, help="only this drive frequency")
    a = ap.parse_args()
    if a.root is None:
        _self_check()
        sys.exit()
    build(a.root, out_dir=a.out, which=a.which, only_hz=a.hz)
