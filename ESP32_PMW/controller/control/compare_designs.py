#!/usr/bin/env python3
"""Two alignment-rate campaigns, one figure: design 1 against design B.

    uv run python controller/control/compare_designs.py                       # self-check
    uv run python controller/control/compare_designs.py rates  TRIALS_1.csv TRIALS_B.csv
    uv run python controller/control/compare_designs.py panels REPORT_1 REPORT_B
    uv run python controller/control/compare_designs.py traces ROOT_1 ROOT_B
    uv run python controller/control/compare_designs.py summary TRIALS_1 TRIALS_B SETTLING_1 SETTLING_B
    uv run python controller/control/compare_designs.py cone SETTLING_1 SETTLING_B

`rates` answers "do the two robots align at different rates?" from every solved repeat:
`campaign_trials.csv`, which `alignment_rate.py --campaign` writes beside `campaign.csv`.
`traces` overlays the two robots' AVERAGE AXIS and CONE HALF-ANGLE against time from the cut,
median and interquartile band over repeats, one column per drive frequency -- the per-take
traces `alignment_rate.settle_rows` computes for the settle report, pooled per design.
`panels` puts six per-frequency medians side by side from each `report/` (`campaign.csv` and
`settling_by_freq.csv`). Output goes to `--out`, default `results/alignment_rate/compare_1_vs_B`.

No measurement happens here. Every number is a median (and MAD) that the two campaign
reports already computed; this only puts them on shared axes.

WHAT THE PANELS CANNOT BE COMPARED ON -- printed under the figure, because each is a real
difference between the campaigns and not a difference between the robots
(`control/theory.md` 24.12, 25.14):

* the settled window: design 1 was clamped to 4-5 s after the kill, design B reads 4-6 s;
* NOT segmentation, though it was first assumed to be: view A against view B agrees to a
  15.9-17.9 deg median on design 1's 10-30 Hz takes and 17.5 on design B's, and design-B takes
  that gave no rate agree exactly as well as those that did (17.5 vs 17.5). Design B's wide
  swing scatter is the robot, not the half ring confusing the ellipse fit;
* coning is unmeasurable above ~104 Hz drive at this capture mode (Nyquist).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from pathlib import Path

#: Categorical slots 1 and 2 of the reference palette (dataviz `references/palette.md`),
#: checked with its validator on this surface. Fixed order: design 1 is always slot 1.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#1a1a19", "#5f5e57", "#e6e5e0"

#: (panel title, csv, median column, MAD column or None, y label). One y-scale per panel.
PANELS = (
    ("Alignment rate", "campaign.csv", "rate_relu_med", "rate_relu_mad", "deg/s"),
    ("Total rotation after the cut", "campaign.csv", "rot_median_deg", "rot_mad_deg", "deg"),
    ("Axis swing", "settling_by_freq.csv", "swing_med_deg", "swing_mad_deg", "deg"),
    ("Settling time (10% band)", "settling_by_freq.csv", "settle_10pct_med_s",
     "settle_10pct_mad_s", "s"),
    ("Cone before the cut", "settling_by_freq.csv", "cone_pre_med_deg", None, "deg"),
    ("Cone decay time constant", "settling_by_freq.csv", "cone_tau_med_s", "cone_tau_mad_s",
     "s"),
)

CAVEATS = (
    "Not like for like: design 1's settled window was clamped to 4-5 s after the cut, "
    "design B's is 4-6 s (theory.md 24.12). Segmentation quality is the same for both "
    "(view A vs B ~16-18 deg median).",
    "Cone panels stop near 104 Hz: above that the once-per-rev line is aliased at this capture "
    "mode (25.14). Error bars are MAD across repeats. n per point is in compare_designs.csv.",
)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def load(root):
    """``{csv name: {freq: row}}`` for one campaign's report."""

    rep = Path(root)
    if (rep / "report").is_dir():
        rep = rep / "report"
    out = {}
    for name in ("campaign.csv", "settling_by_freq.csv"):
        path = rep / name
        if not path.exists():
            raise SystemExit(f"{path} missing -- run alignment_rate.py "
                             f"{'--campaign' if name == 'campaign.csv' else '--settle'} "
                             f"{root} first")
        out[name] = {_num(r["freq_hz"]): r for r in csv.DictReader(open(path))}
    return out


def series(data, name, med, mad):
    """Sorted ``(freqs, medians, mads)`` for one column, NaN rows dropped."""

    pts = []
    for f, r in sorted(data[name].items()):
        m = _num(r.get(med))
        if math.isfinite(f) and math.isfinite(m):
            pts.append((f, m, _num(r.get(mad)) if mad else float("nan")))
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def merged(a, b, labels):
    """One row per (design, frequency): every plotted column plus n_repeats."""

    rows = []
    for label, data in zip(labels, (a, b)):
        freqs = sorted(set(data["campaign.csv"]) | set(data["settling_by_freq.csv"]))
        for f in freqs:
            c = data["campaign.csv"].get(f, {})
            s = data["settling_by_freq.csv"].get(f, {})
            row = {"design": label, "freq_hz": f, "n_repeats": c.get("n_repeats", "")}
            for _, name, med, mad, _ in PANELS:
                src = c if name == "campaign.csv" else s
                row[med] = src.get(med, "")
                if mad:
                    row[mad] = src.get(mad, "")
            rows.append(row)
    return rows


def figure(a, b, labels, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, facecolor=SURFACE)
    for ax, (title, name, med, mad, unit) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for data, color, label in zip((a, b), COLORS, labels):
            f, m, d = series(data, name, med, mad)
            if not f:
                continue
            if mad:
                ax.errorbar(f, m, yerr=[x if math.isfinite(x) else 0.0 for x in d],
                            fmt="none", ecolor=color, elinewidth=1.0, capsize=0, alpha=0.45)
            # 2 px line, >= 8 px marker with a surface ring (marks-and-anatomy)
            ax.plot(f, m, color=color, linewidth=1.6, solid_capstyle="round",
                    marker="o", markersize=6.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=label)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("drive frequency (Hz)")
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", ncol=2, frameon=False,
               bbox_to_anchor=(0.985, 0.995), fontsize=10)
    fig.suptitle("Alignment-rate campaigns: design 1 vs design B", x=0.012, ha="left",
                 y=0.985, fontsize=13, color=INK)
    fig.text(0.012, 0.012, "\n".join(CAVEATS), fontsize=8.5, color=INK_2, ha="left",
             va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


LABELS = ("design 1", "design B (half outer ring)")
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "alignment_rate" / "compare_1_vs_B"


def build(root_a, root_b, labels=LABELS, out_dir=None):
    a, b = load(root_a), load(root_b)
    rep = Path(out_dir) if out_dir else Path(root_b) / "report"
    rep.mkdir(parents=True, exist_ok=True)
    rows = merged(a, b, labels)
    with open(rep / "compare_designs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    figure(a, b, labels, rep / "compare_designs.png")
    print(f"-> {rep / 'compare_designs.png'}\n-> {rep / 'compare_designs.csv'}")
    return rows


def load_trials(path):
    """``{freq: [rate, ...]}`` from a `campaign_trials.csv`: MODAL repeats, finite rates.

    Modal is `campaign`'s own rule (within MODAL_BAND_DEG of the median swing), so a repeat
    that settled into a different equilibrium is excluded here exactly as it is from the
    campaign's median -- both designs are filtered by the same code.
    """

    out = {}
    for r in csv.DictReader(open(path)):
        if str(r.get("modal")).strip().lower() not in ("true", "1"):
            continue
        v = _num(r.get("rate_relu_deg_s"))
        if math.isfinite(v):
            out.setdefault(_num(r["freq_hz"]), []).append(v)
    return out


def load_all(path):
    """``{freq: [(amp_deg, modal, rate_measured), ...]}`` -- EVERY repeat in a trials file.

    The rate panel can only draw repeats with a measurable rate, and on design B that was
    2-4 of 10 per frequency at 10-30 Hz: drawing only those hides the main difference. This
    feeds the swing panel and the k/n counts that show it.
    """

    out = {}
    for r in csv.DictReader(open(path)):
        modal = str(r.get("modal")).strip().lower() in ("true", "1")
        ok = modal and math.isfinite(_num(r.get("rate_relu_deg_s")))
        out.setdefault(_num(r["freq_hz"]), []).append((_num(r.get("amp_deg")), modal, ok))
    return out


def rate_stats(a, b, n_boot=4000, seed=0, min_n=3):
    """Per frequency both designs measured: medians, B/1 ratio with a bootstrap 95% CI, and a
    two-sided Mann-Whitney p, Bonferroni-corrected over the frequencies tested.

    Rank-based and median-based on purpose: above ~50 Hz the repeats are not one population
    (`control/theory.md` 24.5), so a t-test on means would be testing the outliers.

    ``differs`` is decided by the rank test ALONE. The bootstrap CI is descriptive: a
    percentile bootstrap of a median at n ~ 5-10 runs narrow, and in the self-check two draws
    of the SAME distribution gave a ratio CI of [0.94, 1.00] -- clear of 1 -- while
    Mann-Whitney correctly returned p = 0.12. Reading "CI excludes 1" as a difference would
    have called that one.
    """

    import numpy as np
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(seed)
    rows = []
    for f in sorted(set(a) & set(b)):
        x, y = np.asarray(a[f], float), np.asarray(b[f], float)
        if len(x) < min_n or len(y) < min_n:
            continue
        mx = np.median(rng.choice(x, (n_boot, len(x))), axis=1)
        my = np.median(rng.choice(y, (n_boot, len(y))), axis=1)
        lo, hi = np.percentile(my / mx, [2.5, 97.5])
        rows.append({"freq_hz": f, "n_1": len(x), "n_B": len(y),
                     # 3 decimals, not 1: the same table serves cone decays of ~0.1 s
                     "median_1_deg_s": round(float(np.median(x)), 3),
                     "median_B_deg_s": round(float(np.median(y)), 3),
                     "ratio_B_over_1": round(float(np.median(y) / np.median(x)), 3),
                     "ratio_ci_lo": round(float(lo), 3), "ratio_ci_hi": round(float(hi), 3),
                     "p_mannwhitney": float(mannwhitneyu(x, y, alternative="two-sided").pvalue)})
    for r in rows:
        r["p_bonferroni"] = min(1.0, r["p_mannwhitney"] * len(rows))
        r["differs"] = bool(r["p_bonferroni"] < 0.05)
    return rows


def _gapped(xs, ys, step=10.0):
    """Insert NaN where consecutive frequencies are more than one step apart, so a median line
    breaks across frequencies with no data instead of implying values there. Design B has no
    measurable rate at 50 or 60 Hz, and a straight 40 -> 70 Hz segment drew one."""

    ox, oy = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if i and x - xs[i - 1] > 1.5 * step:
            ox.append(float("nan"))
            oy.append(float("nan"))
        ox.append(x)
        oy.append(y)
    return ox, oy


def rate_figure(a, b, stats, out_png, labels=LABELS, all_a=None, all_b=None):
    """Every solved repeat on top; the B/1 ratio with its CI underneath. One y-axis each."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    alls = (all_a, all_b) if all_a is not None and all_b is not None else None
    if alls:
        fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(11, 11.5), sharex=True,
                                            facecolor=SURFACE,
                                            gridspec_kw={"height_ratios": [1.5, 1.25, 1]})
        panels = (top, mid, bot)
    else:
        fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 8.2), sharex=True,
                                       facecolor=SURFACE,
                                       gridspec_kw={"height_ratios": [1.6, 1]})
        panels, mid = (top, bot), None
    for ax in panels:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Repeats, offset left/right of the drive frequency so the two designs never overlap,
    # with a small deterministic jitter so equal values stay visible.
    for data, color, label, dx in zip((a, b), COLORS, labels, (-1.4, 1.4)):
        fs = sorted(data)
        for f in fs:
            v = np.asarray(data[f], float)
            jit = (np.arange(len(v)) - (len(v) - 1) / 2) * (0.9 / max(len(v), 1))
            top.scatter(f + dx + jit, v, s=30, color=color, alpha=0.55, linewidths=0,
                        zorder=2)
        med = [float(np.median(data[f])) for f in fs]
        gx, gy = _gapped([f + dx for f in fs], med)
        top.plot(gx, gy, color=color, linewidth=1.6, marker="o",
                 markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
                 label=f"{label}: median, dots = repeats")
    top.set_ylabel("alignment rate (deg/s)")
    top.set_ylim(bottom=0)
    top.set_title("Alignment rate after the cut: repeats with a measurable rate "
                  "(k/n under each cluster = measurable / analysed)", loc="left",
                  fontsize=10.5, color=INK, pad=8)
    top.legend(loc="upper left", frameon=False, fontsize=9)

    if mid is not None:
        # k/n labels sit in two rows -- design 1 lowest, design B above it -- because the two
        # clusters are only 2.8 Hz apart and side-by-side labels collided ("17/222/10").
        for (data, color, label, dx), row in zip(zip(alls, COLORS, labels, (-1.4, 1.4)),
                                                 (3, 13)):
            fs = sorted(data)
            meds = []
            for f in fs:
                rows = [x for x in data[f] if math.isfinite(x[0])]
                am = np.array([x[0] for x in rows], float)
                mod = np.array([x[1] for x in rows], bool)
                jit = (np.arange(len(am)) - (len(am) - 1) / 2) * (0.9 / max(len(am), 1))
                xs = f + dx + jit
                # filled = modal (eligible for a rate); hollow = outside the modal band.
                # Two encodings, not colour alone, so the distinction survives print/CVD.
                mid.scatter(xs[mod], am[mod], s=30, color=color, alpha=0.6, linewidths=0,
                            zorder=2)
                mid.scatter(xs[~mod], am[~mod], s=30, facecolors="none", edgecolors=color,
                            linewidths=1.2, zorder=2)
                k = sum(1 for x in data[f] if x[2])
                top.annotate(f"{k}/{len(data[f])}", (f + dx, 0.0),
                             xycoords=("data", "axes fraction"), xytext=(0, row),
                             textcoords="offset points", ha="center", va="bottom",
                             fontsize=7.5, color=INK_2)
                meds.append(float(np.median(am)) if len(am) else float("nan"))
            gx, gy = _gapped([f + dx for f in fs], meds)
            mid.plot(gx, gy, color=color, linewidth=1.6, marker="o",
                     markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        mid.axhline(0.0, color=INK_2, linewidth=0.8, zorder=1)
        mid.set_ylabel("swing after the cut (deg)")
        mid.set_title("Swing after the cut, every repeat: filled = modal (rate eligible), "
                      "hollow = outside the 15 deg modal band; line = median", loc="left",
                      fontsize=10.5, color=INK, pad=8)

    bot.axhline(1.0, color=INK_2, linewidth=1.0, zorder=1)
    for r in stats:
        f = r["freq_hz"]
        col = COLORS[1] if r["differs"] else INK_2
        bot.plot([f, f], [r["ratio_ci_lo"], r["ratio_ci_hi"]], color=col, linewidth=2.0,
                 solid_capstyle="round", zorder=2)
        bot.plot(f, r["ratio_B_over_1"], "o", color=col, markersize=7,
                 markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        p = r["p_bonferroni"]
        bot.annotate(f"n {r['n_1']}/{r['n_B']}\np {p:.2g}" if p >= 0.001 else
                     f"n {r['n_1']}/{r['n_B']}\np <0.001",
                     (f, r["ratio_ci_hi"]), textcoords="offset points", xytext=(0, 6),
                     ha="center", va="bottom", fontsize=8, color=INK_2)
    bot.set_ylabel("design B / design 1")
    bot.set_xlabel("drive frequency (Hz)")
    bot.set_title("Ratio of medians with bootstrap 95% CI (descriptive; narrow at n < 10). "
                  "Orange = differs: Mann-Whitney, Bonferroni p < 0.05",
                  loc="left", fontsize=10.5, color=INK, pad=8)
    if stats:
        hi = max(r["ratio_ci_hi"] for r in stats)
        lo = min(r["ratio_ci_lo"] for r in stats)
        bot.set_ylim(min(0.0, lo - 0.1), max(2.0, hi * 1.25))

    fig.suptitle("Do the two robots align at different rates?", x=0.012, ha="left", y=0.99,
                 fontsize=13, color=INK)
    fig.text(0.012, 0.01,
             "Solved takes only: n per frequency is design 1 / design B (modal repeats, "
             "alignment_rate's own rule). Mann-Whitney two-sided, Bonferroni over the "
             "frequencies shown.\nSegmentation is not the cause of design B's scatter: view "
             "A vs B agrees ~16-18 deg median on both designs, and equally on B's takes with and "
             "without a rate.", fontsize=8.5, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_rates(trials_a, trials_b, out_dir=None, labels=LABELS):
    a, b = load_trials(trials_a), load_trials(trials_b)
    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    stats = rate_stats(a, b)
    if stats:
        with open(out / "alignment_rate_stats.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stats[0]))
            w.writeheader()
            w.writerows(stats)
    rate_figure(a, b, stats, out / "alignment_rate_1_vs_B.png", labels,
                all_a=load_all(trials_a), all_b=load_all(trials_b))
    print(f"-> {out / 'alignment_rate_1_vs_B.png'}")
    for r in stats:
        print(f"  {r['freq_hz']:5.0f} Hz  n {r['n_1']:2d}/{r['n_B']:2d}  "
              f"median {r['median_1_deg_s']:7.1f} / {r['median_B_deg_s']:7.1f} deg/s  "
              f"B/1 {r['ratio_B_over_1']:.2f} [{r['ratio_ci_lo']:.2f}, {r['ratio_ci_hi']:.2f}]"
              f"  p_bonf {r['p_bonferroni']:.3g}{'  DIFFERS' if r['differs'] else ''}")
    return stats


#: Common time grid for pooling repeats, seconds from the cut. -1 s is `PRE_S`; 16 s covers
#: design B's whole 15 s drop hold. Each repeat is drawn to its own DOWN label (`_end`), not to
#: the 6 s settle window -- the recordings run the full hold, and stopping at 6 s hid 9 s of it.
T_GRID = (-1.0, 16.0, 0.02)
#: A grid point is drawn only where at least this many repeats cover it, so a band is never
#: one repeat pretending to be a spread.
MIN_COVER = 2
#: Trimmed off the end of every repeat, before its DOWN label. The one-revolution smoother in
#: `average_axis` / `cone_envelope` is ~0.25 s wide, so its last half-window reaches into the
#: 100 ms spin-down: every median jumped in its final samples, at 15 s on design B and 5 s on
#: design 1. 0.3 s clears the half-window plus the spin-down.
END_TRIM_S = 0.3
#: Cut this far BEFORE a detected rotor stop. The axis degrades while the rotor is still
#: slowing, up to ~1 s before the fill confirms the stop: take `002905` reads 7.8 deg at +13 s
#: with camera-A fill still 0.88, and the detector calls the stop at +14.1 s.
SPIN_MARGIN_S = 1.0


def trace_stats(rows):
    """Per frequency: n repeats and ``(median, q25, q75)`` of ``_delta`` and ``_env`` on T_GRID.

    ``_delta`` is the angle of the AVERAGE axis to the final resting axis (deg) -- the
    trajectory with the once-per-rev cone and the 2-4 Hz rod mode removed (`average_axis`).
    ``_env`` is the RMS cone half-angle about that average axis over one revolution
    (`cone_envelope`). Each repeat counts only where ``_valid`` and before its own DOWN label
    (``_end``; ``_stop`` for rows from before that key existed).
    """

    import numpy as np

    grid = np.arange(*T_GRID)
    out = {}
    for f in sorted({float(r["freq_hz"]) for r in rows}):
        stacks = {"delta": [], "env": []}
        for r in (r for r in rows if float(r["freq_hz"]) == f):
            t = np.asarray(r["_t"], float)
            # Stop at the DOWN label or when the rotor stopped, whichever is first: past a stop
            # the "disc" is four still blades and the axis is not measured (`spin_stop_from`).
            end = min(r.get("_end", r["_stop"]) - END_TRIM_S,
                      r.get("_spin_end", float("inf")) - SPIN_MARGIN_S)
            m = np.asarray(r["_valid"], bool) & (t >= grid[0] - 0.05) & (t <= end)
            if m.sum() < 10:
                continue
            inside = (grid >= t[m].min()) & (grid <= t[m].max())
            for key, col in (("delta", "_delta"), ("env", "_env")):
                y = np.full(grid.shape, np.nan)
                v = np.asarray(r[col], float)[m]
                if col == "_delta":
                    # The axis is a LINE (theory.md 25.11): 178 deg from a line is 2 deg from
                    # it. One design-B 10 Hz repeat whose average axis sat on the opposite
                    # branch read ~178 deg for the whole take and set the y-limit for every
                    # panel. `settle_take`'s own metrics are untouched; only the pooling folds.
                    v = np.minimum(v, 180.0 - v)
                y[inside] = np.interp(grid[inside], t[m], v)
                stacks[key].append(y)
        if not stacks["delta"]:
            continue
        stat = {"n": len(stacks["delta"]),
                "n_stopped": sum(1 for r in rows if float(r["freq_hz"]) == f
                                 and r.get("_spin_end", float("inf")) < r.get("_end", r["_stop"]))}
        for key, ys in stacks.items():
            a = np.vstack(ys)
            stat["reps_" + key] = a                  # every repeat, for the thin lines
            cover = np.sum(np.isfinite(a), axis=0)
            with warnings.catch_warnings():        # uncovered grid points are all-NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                q = np.nanpercentile(a, [50, 25, 75], axis=0)
            # The median of whichever repeats are still running is not the median of the set:
            # near each window's end repeats drop out at slightly different times, and design
            # B's 30 Hz median spiked on the last two of nine. Draw it only where at least half
            # contribute; the thin per-repeat lines still show everything.
            q[:, cover < max(MIN_COVER, (len(ys) + 1) // 2)] = np.nan
            stat[key] = q
        out[f] = stat
    return grid, out


def traces_figure(grid, s1, sB, out_png, labels=LABELS, sC=None):
    """Two rows of small multiples: average axis (top), cone half-angle (bottom).

    With ``sC`` a third design is overlaid and the columns are every drive ANY design has
    (operator, 2026-09-14: "up to 120 Hz"); a design without that drive is simply absent there.
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    designs = (s1, sB) if sC is None else (s1, sB, sC)
    pick = set.intersection if sC is None else set.union
    freqs = sorted(pick(*(set(s) for s in designs)))
    if not freqs:
        raise SystemExit("no drive frequency has traces for every design")
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(2, len(freqs), figsize=(3.6 * len(freqs) + 1.2, 7.4),
                             sharex=True, sharey="row", facecolor=SURFACE, squeeze=False)
    rows_spec = (("delta", "average axis: angle to final axis (deg)"),
                 ("env", "cone ('precession') half-angle (deg)"))
    for j, f in enumerate(freqs):
        for i, (key, ylab) in enumerate(rows_spec):
            ax = axes[i][j]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.axvline(0.0, color=INK_2, linewidth=1.0, zorder=1)
            for stats, color, label in zip(designs, COLORS, labels):
                if f not in stats:
                    continue
                med = stats[f][key][0]
                # EVERY repeat, thin, under the median. A band hid that design B's 50 Hz
                # repeats split into two groups -- 3 of 8 swing 8-10 deg at the cut, 5 are already
                # within ~2 deg of their final axis -- so the median was a representative of
                # neither. Its fall BEFORE the cut is real: two of the five were still moving
                # toward their final axis in the last second of the hold (3.8 -> 0.2 and
                # 6.0 -> 2.1 deg). Only the individual lines can show that.
                for y in stats[f]["reps_" + key]:
                    ax.plot(grid, y, color=color, linewidth=0.6, alpha=0.3, zorder=2)
                ax.plot(grid, med, color=color, linewidth=1.8, zorder=3,
                        label=f"{label}: bold = median, thin = each repeat")
            if i == 0:
                n = " / ".join(str(s[f]["n"]) if f in s else "-" for s in designs)
                stop = " / ".join(str(s[f]["n_stopped"]) if f in s else "-" for s in designs)
                ax.set_title(f"{f:g} Hz   n {n}   stopped {stop}",
                             loc="left", fontsize=10.5, color=INK, pad=6)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("time from the cut (s)")
    # ONE limit per row, set after every panel is drawn. Setting it panel by panel on shared
    # axes froze the whole row at the first panel's range and cut the pre-cut level off the top
    # of every other frequency.
    for i, (key, _) in enumerate(rows_spec):
        # 99th percentile of every repeat, so one wild repeat cannot flatten the rest
        allv = np.concatenate([st[f]["reps_" + key].ravel() for st in designs for f in freqs
                               if f in st])
        top = float(np.nanpercentile(allv, 99))
        axes[i][0].set_ylim(0.0, 1.08 * top)
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", ncol=len(designs), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.995, 0.995))
    abc = sC is not None
    fig.suptitle("Average axis and cone half-angle after the cut: "
                 + ("design A vs design B vs design C" if abc else "design 1 vs design B"),
                 x=0.008, ha="left", y=0.985, fontsize=13, color=INK)
    fig.text(0.008, 0.008,
             f"n = repeats, {'A / B / C' if abc else 'design 1 / design B'}; stopped = repeats "
             "whose rotor stopped before the "
             "hold ended -- each is cut where it stopped, since past that the disc is still "
             "blades and no axis is measured. Vertical line = the cut (coils A and C off). "
             "y clipped at the 99th percentile of all repeats. "
             "Average axis = the trajectory with the once-per-rev cone and the 2-4 Hz rod mode "
             "removed (alignment_rate.average_axis).\nThe cone is what the campaign calls "
             "coning: it turns at exactly the drive frequency, so it is synchronous coning, not "
             "free precession (theory.md 25.10). Each repeat runs to its own spin-down: 5 s after the "
             "cut on design 1 (5 s hold), 15 s on design B (15 s hold)."
             + (" Design A = design 1; design C also holds 15 s. " + ABC_RAMP if abc else ""),
             fontsize=8.5, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def build_traces(root_1, root_b, root_c=None, out_dir=None, labels=LABELS):
    """Solve nothing: pool the settle traces the campaigns already compute, and draw them.

    With ``root_c`` it is the A/B/C figure + csv (`traces_A_vs_B_vs_C`) in `OUT_DIR_ABC`.
    """

    from controller.control import alignment_rate as ar

    abc = root_c is not None
    out = Path(out_dir or (OUT_DIR_ABC if abc else OUT_DIR))
    out.mkdir(parents=True, exist_ok=True)
    stem, labels = ("traces_A_vs_B_vs_C", LABELS_ABC) if abc else ("traces_1_vs_B", labels)
    grid, s1 = trace_stats(ar.settle_rows(root_1)[0])
    stats = [s1] + [trace_stats(ar.settle_rows(r)[0])[1] for r in (root_b, root_c) if r]
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "freq_hz", "n", "t_s", "delta_med", "delta_q25", "delta_q75",
                    "env_med", "env_q25", "env_q75"])
        for label, st in zip(labels, stats):
            for f, d in sorted(st.items()):
                for k, t in enumerate(grid):
                    w.writerow([label, f, d["n"], round(float(t), 3)]
                               + [round(float(v), 4) for v in d["delta"][:, k]]
                               + [round(float(v), 4) for v in d["env"][:, k]])
    traces_figure(grid, stats[0], stats[1], out / f"{stem}.png", labels,
                  sC=stats[2] if abc else None)
    print(f"-> {out / stem}.png\n-> {out / stem}.csv")
    for f in sorted(set.union(*(set(s) for s in stats))):
        counts = " / ".join(f"{s[f]['n']:2d}" if f in s else " -" for s in stats)
        print(f"  {f:4.0f} Hz  n {counts}")
    return stats


def load_settle(path, max_stop_s=6.0, column="settle_10pct_s", max_hz=math.inf):
    """``({freq: [value, ...]}, {freq: (measured, analysed)})`` for one `settling.csv` column.

    ``column`` defaults to the 10% settling time. A repeat counts as measured when the column is
    finite (blank when `settling` refused -- never re-entering the 10% band inside the window,
    or no cone decay fitted). A design-B repeat whose rotor stopped before ``max_stop_s``
    (`spin_stop_s`, `alignment_rate.POST_TO_S`) is dropped outright: past the stop the axis is
    not measured, so neither is anything read from it. Drives above ``max_hz`` are left out.
    """

    vals, counts = {}, {}
    for r in csv.DictReader(open(path)):
        f = _num(r["freq_hz"])
        if f > max_hz:
            continue
        stop = _num(r.get("spin_stop_s"))
        if math.isfinite(stop) and stop < max_stop_s:
            continue
        m, n = counts.get(f, (0, 0))
        v = _num(r.get(column))
        if math.isfinite(v):
            vals.setdefault(f, []).append(v)
            m += 1
        counts[f] = (m, n + 1)
    return vals, counts


def _counts_from_trials(path):
    """``{freq: (measurable rate, analysed)}`` from a `campaign_trials.csv`."""

    out = {}
    for f, rows in load_all(path).items():
        out[f] = (sum(1 for x in rows if x[2]), len(rows))
    return out


#: A median bar and trend line are drawn only from at least this many measured repeats. One
#: design-B 10 Hz repeat settled "instantly" (1 of 10 measured) and its lone value drew a median
#: at 0 s and dragged the trend line down; the dots are always shown.
MIN_FOR_MEDIAN = 3


SUMMARY_SPEC = (("Alignment rate after the cut", "deg/s"),
                ("Settling time: last entry into a 10% band about the final axis", "s"))


def summary_figure(rows, stats, counts, out_png, labels=LABELS, spec=SUMMARY_SPEC,
                   title="Design 1 vs design B: how fast the axis realigns, and when it settles",
                   note=None):
    """Per-repeat metrics against drive frequency, both designs: one row per ``spec`` entry,
    ``rows[i]`` = (design 1 {freq: [values]}, design B). Default: alignment rate and settling.

    Each frequency shows the two designs side by side: dots = repeats, a bar = the median, a
    whisker = the middle 50%. A thin line joins every median, unbroken across missing frequencies.
    Under each cluster, measured / analysed. An asterisk marks a frequency where the designs
    differ (two-sided Mann-Whitney, Bonferroni over the frequencies tested on that row).
    """

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    n_rows = len(spec)
    # Three designs share each 10 Hz slot: at 12 in their m/n counts run into each other.
    width = 12 if len(rows[0]) == 2 else 17
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, 4.6 * n_rows if n_rows > 1 else 6.2),
                             sharex=True, facecolor=SURFACE, squeeze=False)
    axes = axes[:, 0]
    dx = (-1.9, 1.9) if len(rows[0]) == 2 else (-2.8, 0.0, 2.8)
    all_f = sorted({f for row in rows for d in row for f in d})
    for ax, (row_title, unit), data, st, cnt in zip(axes, spec, rows, stats, counts):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for k, (d, color, label) in enumerate(zip(data, COLORS, labels)):
            fs = sorted(d)
            meds = []
            med_f = []
            for f in fs:
                v = np.asarray(d[f], float)
                x0 = f + dx[k]
                jit = (np.arange(len(v)) - (len(v) - 1) / 2) * (1.1 / max(len(v), 1))
                ax.scatter(x0 + jit, v, s=22, color=color, alpha=0.45, linewidths=0, zorder=2)
                if len(v) < MIN_FOR_MEDIAN:
                    continue
                q25, med, q75 = np.percentile(v, [25, 50, 75])
                ax.plot([x0, x0], [q25, q75], color=color, linewidth=2.2,
                        solid_capstyle="round", zorder=3)
                ax.plot([x0 - 0.9, x0 + 0.9], [med, med], color=color, linewidth=3.0,
                        solid_capstyle="round", zorder=4)
                meds.append(med)
                med_f.append(f)
            # Joined through every median, gaps included (operator's request, 2026-09-13): the
            # dots and the m/n counts already show where a design has too few repeats.
            ax.plot([f + dx[k] for f in med_f], meds, color=color, linewidth=1.0, alpha=0.7,
                    zorder=1,
                    label=f"{label}: dots = repeats, bar = median, whisker = middle 50%")
            for f in all_f:
                if f in cnt[k]:
                    m, n = cnt[k][f]
                    ax.annotate(f"{m}/{n}", (f + dx[k], 0.0), xycoords=("data", "axes fraction"),
                                xytext=(0, 3), textcoords="offset points", ha="center",
                                va="bottom", fontsize=7.5, color=INK_2)
        top = max(float(np.max(v)) for d in data for v in d.values())
        for r in st:
            if r["differs"]:
                ax.annotate(f"*\np={r['p_bonferroni']:.2g}", (r["freq_hz"], top * 1.02),
                            ha="center", va="bottom", fontsize=8.5, color=INK)
        # A strip below zero for the m/n counts: at y = 0 they sat on the dots of any metric
        # that reaches zero (cone decays of ~0.1 s, the instant design-B 10 Hz settle).
        ax.set_ylim(-0.07 * top, top * 1.16)
        ax.set_title(row_title, loc="left", fontsize=11.5, color=INK, pad=8)
        ax.set_ylabel(unit)
    axes[-1].set_xlabel("drive frequency (Hz)")
    axes[-1].set_xticks(all_f)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper left", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.01, 0.955))
    fig.suptitle(title, x=0.012, ha="left", y=0.99, fontsize=13.5, color=INK)
    import textwrap
    note = note or ("Numbers under each cluster: measured / analysed repeats (rate: modal repeats with a "
            "detectable jump; settling: repeats that re-entered the band). Median bar and line "
            f"need >= {MIN_FOR_MEDIAN} measured repeats. * = designs differ: two-sided "
            "Mann-Whitney, Bonferroni per row, >= 3 repeats a side. Settling windows differ: "
            "design 1's was clamped at 5 s after the cut, design B's runs to 6 s, so a repeat "
            "slower than 5 s is refused on design 1 but measured on B (theory.md 24.12). "
            "Design-B repeats whose rotor stopped inside the window are excluded (24.13).")
    fig.text(0.012, 0.008, "\n".join(textwrap.wrap(note, 190)), fontsize=8.3, color=INK_2,
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.065, 1, 0.925))
    fig.savefig(out_png, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def build_summary(trials_1, trials_b, settling_1, settling_b, out_dir=None, labels=LABELS):
    """The rate + settling-time comparison figure and its statistics table."""

    out = Path(out_dir or OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    rates = (load_trials(trials_1), load_trials(trials_b))
    (s1, c1), (sb, cb) = load_settle(settling_1), load_settle(settling_b)
    settles = (s1, sb)
    st_rate, st_settle = rate_stats(*rates), rate_stats(*settles)
    counts = ((_counts_from_trials(trials_1), _counts_from_trials(trials_b)), (c1, cb))
    with open(out / "rate_and_settling_stats.csv", "w", newline="") as fh:
        rows = [{"metric": "alignment_rate", **r} for r in st_rate] + \
               [{"metric": "settle_10pct_s", **r} for r in st_settle]
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summary_figure((rates, settles), (st_rate, st_settle), counts,
                   out / "rate_and_settling_1_vs_B.png", labels)
    print(f"-> {out / 'rate_and_settling_1_vs_B.png'}")
    for name, st in (("rate", st_rate), ("settling", st_settle)):
        for r in st:
            print(f"  {name:8s} {r['freq_hz']:5.0f} Hz  n {r['n_1']:2d}/{r['n_B']:2d}  "
                  f"median {r['median_1_deg_s']:8.2f} / {r['median_B_deg_s']:8.2f}  "
                  f"B/1 {r['ratio_B_over_1']:.2f}  p_bonf {r['p_bonferroni']:.3g}"
                  f"{'  DIFFERS' if r['differs'] else ''}")
    return st_rate, st_settle


#: Above this drive the once-per-rev coning line is aliased at the 640x400 / ~208 fps capture
#: mode (Nyquist, theory.md 25.14), so a fitted cone decay there measures nothing. The cone
#: panels in `panels` stop at the same place.
CONE_MAX_HZ = 104.0


def build_cone(settling_1, settling_b, settling_c=None, out_dir=None, labels=LABELS):
    """Decay time constant of the cone ('precession') half-angle vs drive, two or three designs.

    Per repeat `cone_tau_s` from each `settling.csv` -- the values behind `settling_by_freq`'s
    `cone_tau_med_s` -- with the settling row's rotor-stop exclusion, below `CONE_MAX_HZ`.
    With `settling_c` it is the A/B/C figure + csv in `OUT_DIR_ABC`; the statistics stay A vs B.
    """

    out = Path(out_dir or (OUT_DIR if settling_c is None else OUT_DIR_ABC))
    out.mkdir(parents=True, exist_ok=True)
    (t1, c1), (tb, cb) = (load_settle(p, column="cone_tau_s", max_hz=CONE_MAX_HZ)
                          for p in (settling_1, settling_b))
    if settling_c is not None:
        return _build_cone_abc(t1, c1, tb, cb, settling_c, out)
    st = rate_stats(t1, tb)
    with open(out / "cone_tau_stats.csv", "w", newline="") as fh:
        if st:
            w = csv.DictWriter(fh, fieldnames=["metric"] + list(st[0]))
            w.writeheader()
            w.writerows({"metric": "cone_tau_s", **r} for r in st)
    note = ("Dots: one repeat's fitted decay time constant of the RMS cone half-angle after the "
            "cut (cone_tau_s in settling.csv, the values behind cone_tau_med_s). Numbers under "
            "each cluster: repeats with a fitted decay / analysed. Median bar and line need >= "
            f"{MIN_FOR_MEDIAN} repeats. Drives above {CONE_MAX_HZ:.0f} Hz are left out: the "
            "coning line is aliased at this capture mode (theory.md 25.14). Design-B repeats whose "
            "rotor stopped inside the window are excluded (24.13). * = designs differ: two-sided "
            "Mann-Whitney, Bonferroni, >= 3 repeats a side.")
    png = out / "cone_tau_1_vs_B.png"
    summary_figure(((t1, tb),), (st,), ((c1, cb),), png, labels,
                   spec=(("Cone ('precession') half-angle: decay time constant after the cut",
                          "s"),),
                   title="Design 1 vs design B: how fast the coning dies away", note=note)
    print(f"-> {png}")
    for r in st:
        print(f"  cone tau {r['freq_hz']:5.0f} Hz  n {r['n_1']:2d}/{r['n_B']:2d}  "
              f"median {r['median_1_deg_s']:6.3f} / {r['median_B_deg_s']:6.3f} s  "
              f"B/1 {r['ratio_B_over_1']:.2f}  p_bonf {r['p_bonferroni']:.3g}"
              f"{'  DIFFERS' if r['differs'] else ''}")
    return st


#: Design A is the campaign the two-design figures call "design 1" (`20260909_205843`).
LABELS_ABC = ("design A", "design B (half outer ring)", "design C")
OUT_DIR_ABC = OUT_DIR.parent / "compare_A_B_C"
#: Every A/B/C figure carries this. The stars come from `rate_stats`, which is pairwise, so only
#: A vs B is tested; C is left untested because its ramp differs (tilt_run --seg2-rate 10).
ABC_RAMP = ("NOT like-for-like: design C's ramp climbed to the drive at 10 Hz/s, A and B at "
            "3.5 Hz/s (2.8 from 60 Hz).")
ABC_CAVEAT = "* = A and B differ (two-sided Mann-Whitney, Bonferroni); C is not tested. " + ABC_RAMP


def _abc_outputs(values, counts, st, out, stem, spec, title, note):
    """`<stem>.png` and, beside it, `<stem>.csv`: one row per design x drive holding what the
    figure draws -- measured / analysed, median, middle 50%, and every repeat's value."""

    import numpy as np

    summary_figure((values,), (st,), (counts,), out / f"{stem}.png", LABELS_ABC,
                   spec=(spec,), title=title, note=note)
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "freq_hz", "n_measured", "n_analysed", "median", "q25", "q75",
                    "values"])
        for design, d, cnt in zip("ABC", values, counts):
            for f in sorted(set(d) | set(cnt)):
                v = np.asarray(d.get(f, ()), float)
                q = (np.percentile(v, [50, 25, 75]) if len(v) >= MIN_FOR_MEDIAN
                     else (math.nan,) * 3)
                m, n = cnt.get(f, (len(v), len(v)))
                w.writerow([design, f"{f:g}", m, n, *(f"{x:.4g}" for x in q),
                            ";".join(f"{x:.4g}" for x in v)])
    print(f"-> {out / stem}.png + .csv")


def _build_cone_abc(t1, c1, tb, cb, settling_c, out):
    """`build_cone` with a third design (see `ABC_CAVEAT` for what is and is not tested)."""

    tc, cc = load_settle(settling_c, column="cone_tau_s", max_hz=CONE_MAX_HZ)
    st = rate_stats(t1, tb)
    note = ("Dots: one repeat's fitted decay time constant of the RMS cone half-angle after the "
            "cut (cone_tau_s in settling.csv). Numbers under each cluster: repeats with a fitted "
            f"decay / analysed. Median bar and line need >= {MIN_FOR_MEDIAN} repeats. Drives "
            f"above {CONE_MAX_HZ:.0f} Hz are left out (aliased, theory.md 25.14). Repeats whose "
            "rotor stopped inside the window are excluded (24.13). " + ABC_CAVEAT)
    # Stem is the operator's rename of the first output (was cone_tau_A_vs_B_vs_C).
    _abc_outputs((t1, tb, tc), (c1, cb, cc), st, out, "cone_time_constant_A_vs_B_vs_C",
                 ("Cone ('precession') half-angle: decay time constant after the cut", "s"),
                 "Design A vs B vs C: how fast the coning dies away", note)
    return st


def build_rate_abc(trials_a, trials_b, trials_c, out_dir=None):
    """Alignment rate after the cut vs drive for designs A, B and C, from each
    `campaign_trials.csv` (the per-repeat rows `alignment_rate.py --campaign` writes)."""

    out = Path(out_dir or OUT_DIR_ABC)
    out.mkdir(parents=True, exist_ok=True)
    paths = (trials_a, trials_b, trials_c)
    rates = tuple(load_trials(p) for p in paths)
    st = rate_stats(rates[0], rates[1])
    note = ("Dots: one modal repeat's alignment rate on the rising edge (campaign_trials.csv). "
            "Numbers under each cluster: modal repeats with a detectable jump / analysed. Median "
            f"bar and line need >= {MIN_FOR_MEDIAN} repeats. " + ABC_CAVEAT)
    _abc_outputs(rates, tuple(_counts_from_trials(p) for p in paths), st, out,
                 "alignment_rate_A_vs_B_vs_C", ("Alignment rate after the cut", "deg/s"),
                 "Design A vs B vs C: how fast the axis realigns", note)
    return st


def _self_check():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        roots = []
        for k, scale in enumerate((1.0, 0.8)):
            rep = Path(d) / f"c{k}" / "report"
            rep.mkdir(parents=True)
            with open(rep / "campaign.csv", "w") as fh:
                fh.write("freq_hz,n_repeats,rate_relu_med,rate_relu_mad,rot_median_deg,"
                         "rot_mad_deg\n")
                for f in (10, 20, 30):
                    fh.write(f"{f},5,{900 * scale},{50},{40 * scale},{3}\n")
            with open(rep / "settling_by_freq.csv", "w") as fh:
                fh.write("freq_hz,swing_med_deg,swing_mad_deg,settle_10pct_med_s,"
                         "settle_10pct_mad_s,cone_pre_med_deg,cone_tau_med_s,"
                         "cone_tau_mad_s\n")
                for f in (10, 20, 30):
                    fh.write(f"{f},{10 * scale},1,1.5,0.2,{3 * scale},1.1,0.1\n")
                fh.write("40,nan,nan,nan,nan,nan,nan,nan\n")      # a refused frequency
            roots.append(Path(d) / f"c{k}")
        rows = build(*roots)
        # 3 campaign freqs + the settle-only 40 Hz, per design
        assert len(rows) == 8, len(rows)
        assert (roots[1] / "report" / "compare_designs.png").stat().st_size > 10_000
        # NaN rows are dropped from a series, not plotted as zero
        f, m, _ = series(load(roots[0]), "settling_by_freq.csv", "swing_med_deg",
                         "swing_mad_deg")
        assert f == [10.0, 20.0, 30.0] and m == [10.0, 10.0, 10.0], (f, m)
    # rate statistics: the same distribution twice is not a difference; a 2x shift is
    import numpy as np
    rng = np.random.default_rng(1)
    same = {20.0: list(rng.normal(900, 60, 10))}
    st_same = rate_stats(same, {20.0: list(rng.normal(900, 60, 10))})
    # The rank test decides; the CI is descriptive and may clear 1 at this n (docstring).
    assert not st_same[0]["differs"] and abs(st_same[0]["ratio_B_over_1"] - 1.0) < 0.1, st_same
    st_diff = rate_stats(same, {20.0: list(rng.normal(1800, 60, 10))})
    assert st_diff[0]["ratio_ci_lo"] > 1.0 and st_diff[0]["differs"], st_diff
    gx, _ = _gapped([40.0, 70.0, 80.0], [1.0, 2.0, 3.0])
    assert math.isnan(gx[1]) and gx[2:] == [70.0, 80.0], gx
    # a frequency with fewer than 3 repeats on either side is not tested
    assert rate_stats({30.0: [1.0, 2.0]}, {30.0: [1.0, 2.0, 3.0]}) == []
    # load_trials keeps modal, finite rows only, and the figure renders
    with tempfile.TemporaryDirectory() as d:
        pa, pb = Path(d) / "a.csv", Path(d) / "b.csv"
        for path, mu in ((pa, 900), (pb, 1300)):
            with open(path, "w") as fh:
                fh.write("freq_hz,repeat,take,rate_relu_deg_s,amp_deg,modal\n")
                for f in (20, 40):
                    for k in range(6):
                        fh.write(f"{f},{k + 1},t{k},{mu + 10 * k},40,True\n")
                fh.write("20,9,tx,5000,90,False\n20,10,ty,nan,40,True\n")
        got = load_trials(pa)
        assert sorted(got) == [20.0, 40.0] and len(got[20.0]) == 6, got
        stats = build_rates(pa, pb, out_dir=d)
        assert (Path(d) / "alignment_rate_1_vs_B.png").stat().st_size > 10_000
        assert all(r["differs"] for r in stats), stats
    # traces: two synthetic repeats per design pool onto the grid; a repeat stops at _stop,
    # and a point covered by fewer than MIN_COVER repeats is left blank
    def fake(f, swing, n, stop):
        t = np.arange(-1.2, stop + 0.2, 0.004)
        out_ = []
        for k in range(n):
            d = np.where(t < 0, swing, swing * np.exp(-t / 0.3)) + 0.1 * k
            out_.append({"freq_hz": f, "_t": t, "_delta": d, "_env": np.full(t.shape, 3.0 + k),
                         "_valid": np.ones(t.shape, bool), "_stop": stop})
        return out_
    grid, s1 = trace_stats(fake(40.0, 30.0, 3, 5.0))
    # the pooled axis angle is folded onto a line, and the end of each repeat is trimmed
    flip = fake(20.0, 5.0, 2, 5.0)
    for r_ in flip:
        r_["_delta"] = 180.0 - r_["_delta"]
    _, sF = trace_stats(flip)
    assert np.nanmax(sF[20.0]["reps_delta"]) <= 90.0
    assert np.isnan(sF[20.0]["reps_delta"][0][np.argmin(abs(grid - 4.9))])
    _, sB = trace_stats(fake(40.0, 10.0, 2, 6.0) + fake(50.0, 5.0, 1, 6.0))
    assert s1[40.0]["n"] == 3 and sB[40.0]["n"] == 2 and sB[50.0]["n"] == 1
    assert s1[40.0]["reps_delta"].shape == (3, grid.size), s1[40.0]["reps_delta"].shape
    # a repeat whose rotor stopped is cut there, and counted
    stop = fake(80.0, 5.0, 2, 6.0)
    for r_ in stop:
        r_["_end"], r_["_spin_end"] = 6.0, 3.0
    _, sS = trace_stats(stop)
    assert sS[80.0]["n_stopped"] == 2
    assert np.isnan(sS[80.0]["reps_delta"][0][np.argmin(abs(grid - 4.0))])
    med = s1[40.0]["delta"][0]
    assert abs(med[np.argmin(abs(grid + 0.5))] - 30.1) < 1e-6        # pre-cut = swing
    assert np.isnan(med[grid > 5.05]).all()                           # design 1 stops at 5 s
    assert np.isnan(sB[50.0]["env"][0]).all()                         # one repeat < MIN_COVER
    # 4 repeats, 3 ending at 5 s and 1 at 6 s: past 5 s only 1 of 4 remain -> no median there
    mix = fake(60.0, 5.0, 3, 5.0) + fake(60.0, 5.0, 1, 6.0)
    _, sM = trace_stats(mix)
    assert np.isnan(sM[60.0]["delta"][0][np.argmin(abs(grid - 5.5))])
    assert np.isfinite(sM[60.0]["reps_delta"][3][np.argmin(abs(grid - 5.5))])
    # a repeat carrying `_end` is drawn past its 6 s settle window, to the DOWN label
    long_ = fake(30.0, 8.0, 2, 6.0)
    for r_ in long_:
        r_["_t"] = np.arange(-1.2, 15.2, 0.004)
        r_["_delta"] = np.full(r_["_t"].shape, 1.0)
        r_["_env"] = np.full(r_["_t"].shape, 2.0)
        r_["_valid"] = np.ones(r_["_t"].shape, bool)
        r_["_end"] = 15.0
    _, sL = trace_stats(long_)
    assert np.isfinite(sL[30.0]["delta"][0][np.argmin(abs(grid - 14.0))])
    with tempfile.TemporaryDirectory() as d:
        traces_figure(grid, s1, sB, Path(d) / "t.png")
        assert (Path(d) / "t.png").stat().st_size > 10_000
        # a drive only one design has still gets its column in the A/B/C figure
        sC = {f: v for f, v in sB.items() if f != min(sB)}
        traces_figure(grid, s1, sB, Path(d) / "t3.png", LABELS_ABC, sC=sC)
        assert (Path(d) / "t3.png").stat().st_size > 10_000

    # settling: refused repeats are counted but not plotted; a rotor stop inside the window
    # drops the repeat altogether; and the summary figure renders from both files
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "s.csv"
        sp.write_text("freq_hz,settle_10pct_s,spin_stop_s,cone_tau_s\n"
                      "20,1.0,,0.5\n20,1.2,,0.6\n20,,,0.7\n20,0.9,4.0,0.4\n20,1.1,9.0,\n"
                      "110,1.0,,0.9\n")
        v_, c_ = load_settle(sp)
        assert sorted(v_[20.0]) == [1.0, 1.1, 1.2] and c_[20.0] == (3, 4), (v_, c_)
        # another column, same stop rule; an aliased drive is dropped, not plotted
        tau_, ctau_ = load_settle(sp, column="cone_tau_s", max_hz=CONE_MAX_HZ)
        assert sorted(tau_) == [20.0] and sorted(tau_[20.0]) == [0.5, 0.6, 0.7], tau_
        assert ctau_ == {20.0: (3, 4)}, ctau_
        build_cone(sp, sp, out_dir=d)
        assert (Path(d) / "cone_tau_1_vs_B.png").stat().st_size > 10_000
        build_cone(sp, sp, sp, out_dir=d)
        assert (Path(d) / "cone_time_constant_A_vs_B_vs_C.png").stat().st_size > 10_000
        abc = list(csv.DictReader(open(Path(d) / "cone_time_constant_A_vs_B_vs_C.csv")))
        assert [r["design"] for r in abc] == ["A", "B", "C"], abc
        assert abc[2]["values"] == "0.5;0.6;0.7" and abc[2]["median"] == "0.6", abc[2]
        assert (abc[2]["n_measured"], abc[2]["n_analysed"]) == ("3", "4"), abc[2]
        tp = Path(d) / "t.csv"
        tp.write_text("freq_hz,repeat,take,rate_relu_deg_s,amp_deg,modal\n"
                      + "".join(f"20,{k},t{k},{900 + 10 * k},40,True\n" for k in range(5)))
        build_summary(tp, tp, sp, sp, out_dir=d)
        assert (Path(d) / "rate_and_settling_1_vs_B.png").stat().st_size > 10_000
        build_rate_abc(tp, tp, tp, out_dir=d)
        assert (Path(d) / "alignment_rate_A_vs_B_vs_C.png").stat().st_size > 10_000
        abc = list(csv.DictReader(open(Path(d) / "alignment_rate_A_vs_B_vs_C.csv")))
        assert [r["design"] for r in abc] == ["A", "B", "C"], abc
        assert len(abc[0]["values"].split(";")) == 5 and abc[0]["median"] == "920", abc[0]

    print("compare_designs: self-check passed (load, merge, NaN handling, figures, "
          "rate statistics, traces, settling summary, cone decay)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        sys.exit()
    if sys.argv[1] == "summary":
        if len(sys.argv) != 6:
            raise SystemExit("summary TRIALS_1 TRIALS_B SETTLING_1 SETTLING_B")
        build_summary(*sys.argv[2:6])
        sys.exit()
    if sys.argv[1] == "abc":
        # Files, not campaign roots: design A's campaign_trials.csv lives in
        # compare_1_vs_B/design1_campaign/, not in its own report/.
        if len(sys.argv) not in (8, 10) or (len(sys.argv) == 10 and sys.argv[8] != "--out"):
            raise SystemExit("abc TRIALS_A TRIALS_B TRIALS_C SETTLING_A SETTLING_B SETTLING_C "
                             "[--out DIR]")
        out_abc = sys.argv[9] if len(sys.argv) == 10 else None
        build_rate_abc(*sys.argv[2:5], out_dir=out_abc)
        build_cone(*sys.argv[5:8], out_dir=out_abc)
        sys.exit()
    if sys.argv[1] == "cone":
        if len(sys.argv) not in (4, 5, 7) or (len(sys.argv) == 7 and sys.argv[5] != "--out"):
            raise SystemExit("cone SETTLING_A SETTLING_B [SETTLING_C [--out DIR]]")
        build_cone(*sys.argv[2:5], out_dir=sys.argv[6] if len(sys.argv) == 7 else None)
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("rates", "panels", "traces"))
    ap.add_argument("design_1", help="rates: campaign_trials.csv; panels: a report dir")
    ap.add_argument("design_b")
    ap.add_argument("design_c", nargs="?", help="traces only: a third campaign root (A/B/C)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.design_c and a.mode != "traces":
        raise SystemExit("a third design is only supported by `traces` (and `abc` / `cone`)")
    if a.mode == "rates":
        build_rates(a.design_1, a.design_b, out_dir=a.out)
    elif a.mode == "traces":
        build_traces(a.design_1, a.design_b, a.design_c, out_dir=a.out)
    else:
        build(a.design_1, a.design_b, out_dir=a.out or OUT_DIR)
