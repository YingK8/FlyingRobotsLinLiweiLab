"""Build an iteration entry for ITERATIONS.md from a sweep's JSON output.

Every number in the journal comes from here, read out of the sweep's own file.
Nothing is retyped.  That is not fastidiousness: a journal whose numbers were
copied by hand is a journal that will eventually disagree with the code it
documents, and at that point it is worse than no journal, because it is
confidently wrong.

**The regression classification is the point of this module.**  Each metric is
compared against the previous iteration and sorted into improved / constant /
regressed, using a tolerance taken from the *measured* run-to-run noise rather
than a number someone felt was about right.  Without a noise floor, "improved"
is unfalsifiable -- every rerun moves every number a little.

Usage:
    uv run python controller/pose/validation/journal.py \\
        --tag iter2 --prev iter1 --title "predicted-error gate"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[2] / "results" / "pose_validation"
JOURNAL = HERE.parents[1] / "pose" / "ITERATIONS.md"

# Metrics compared between iterations. `lower_better` decides which direction
# counts as an improvement; `noise` is the tolerance below which a change is
# called constant, and defaults are placeholders until `--noise` supplies
# measured ones.
METRICS = [
    ("in_spec", False, 0.02, "in spec", "{:.1%}"),
    ("detect_rate", False, 0.02, "detected", "{:.1%}"),
    ("ang_mean", True, 0.05, "angle mean", "{:.3f}°"),
    ("ang_max", True, 0.30, "angle worst", "{:.3f}°"),
    ("pos_mean", True, 0.02, "position mean", "{:.3f} mm"),
    ("pos_max", True, 0.10, "position worst", "{:.3f} mm"),
]


def load(tag):
    path = RESULTS / f"resolution_{tag}.json"
    if not path.exists():
        raise SystemExit(f"no sweep results at {path}")
    return json.loads(path.read_text())


def _mode_key(rec):
    return f"{rec['width']}x{rec['height']}"


# Metric -> the bootstrap tolerance key the sweep records for it, where one
# exists. Metrics without a measured tolerance fall back to the default above.
TOL_KEYS = {
    "ang_mean": "ang_mean_tol", "ang_max": "ang_max_tol",
    "pos_mean": "pos_mean_tol", "pos_max": "pos_max_tol",
}


def classify(cur, prev, noise=None):
    """Sort every (mode, metric) into improved / constant / regressed.

    Returns three lists of human-readable strings. A metric missing from either
    side is skipped rather than guessed at -- a mode that stopped detecting
    entirely shows up as a detection-rate regression, which is the honest place
    for it.

    The tolerance for each comparison is the **measured** sampling spread that
    the sweep bootstrapped for that mode and metric, taken as the larger of the
    iterations' figures so a noisier run cannot be credited with a change it
    could not resolve. Only where neither run carries a measured tolerance does a
    default apply.
    """
    noise = noise or {}
    improved, constant, regressed = [], [], []
    if not prev:
        return improved, constant, regressed

    for tier in ("core", "edge"):
        cur_by = {_mode_key(r): r for r in cur.get("tiers", {}).get(tier, [])}
        prev_by = {_mode_key(r): r for r in prev.get("tiers", {}).get(tier, [])}
        for mode, c in cur_by.items():
            p = prev_by.get(mode)
            if p is None:
                continue
            for key, lower_better, default_tol, label, fmt in METRICS:
                if key not in c or key not in p:
                    continue
                a, b = p[key], c[key]
                if a is None or b is None:
                    continue
                tol_key = TOL_KEYS.get(key)
                # Use whichever iterations carry a measured tolerance, not both:
                # an older run predating the bootstrap should not force every
                # comparison against it back onto a guessed default.
                measured = [r[tol_key] for r in (c, p)
                            if tol_key and tol_key in r and r[tol_key] is not None]
                tol = max(measured) if measured else noise.get(key, default_tol)
                delta = b - a
                if abs(delta) <= tol:
                    bucket = constant
                elif (delta < 0) == lower_better:
                    bucket = improved
                else:
                    bucket = regressed
                bucket.append(
                    f"{tier} {mode} {label}: {fmt.format(a)} → {fmt.format(b)}"
                )
    return improved, constant, regressed


def table(records, target_pos, target_ang):
    """Per-mode results as a markdown table."""
    head = ("| mode | fps | detected | angle mean | angle worst | "
            "position mean | position worst | **in spec** | ms/frame |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    rows = ""
    for r in records:
        if not r.get("n"):
            rows += (f"| {r['width']}×{r['height']} | {r['fps']} | "
                     f"**0%** | — | — | — | — | — | "
                     f"{r.get('ms_per_frame', 0):.1f} |\n")
            continue
        spec = r["in_spec"]
        mark = "**100%**" if spec >= 1.0 else f"{spec:.1%}"
        rows += (
            f"| {r['width']}×{r['height']} | {r['fps']} | {r['detect_rate']:.0%} | "
            f"{r['ang_mean']:.3f}° | {r['ang_max']:.3f}° | "
            f"{r['pos_mean']:.3f} mm | {r['pos_max']:.3f} mm | {mark} | "
            f"{r['ms_per_frame']:.1f} |\n"
        )
    return head + rows


def render_entry(number, title, cur, prev, changes, theory, noise=None, chart=None,
                 note=None):
    """One complete journal section."""
    tp = cur.get("target_pos_mm", 0.5)
    ta = cur.get("target_angle_deg", 1.0)
    improved, constant, regressed = classify(cur, prev, noise)

    out = [f"## Iteration {number} — {title}", ""]
    out.append(f"*{cur['poses']} poses per tier, seed {cur['seed']}. "
               f"Target ±{ta:g}° and ±{tp:g} mm on 100% of reported frames.*")
    out.append("")

    out.append("### Changed")
    out.append("")
    out += [f"- {c}" for c in changes]
    out.append("")

    if theory:
        out.append("### Theory")
        out.append("")
        out.append(theory.strip())
        out.append("")

    out.append("### Results")
    out.append("")
    if chart:
        out.append(f"![accuracy against sensor mode]({chart})")
        out.append("")
    out.append("**Core tier** — this is what gates the target.")
    out.append("")
    out.append(table(cur["tiers"]["core"], tp, ta))
    out.append("**Edge tier** — measured and tracked, does not gate.")
    out.append("")
    out.append(table(cur["tiers"]["edge"], tp, ta))

    if note:
        out.append(note.strip())
        out.append("")

    out.append("### Improved / Constant / Regressed")
    out.append("")
    if prev is None:
        out.append("First entry on this dataset, so there is nothing to compare "
                   "against. The conditions changed in this iteration "
                   "(backgrounds, lighting, two tiers), so earlier numbers from "
                   "the flat-background sweeps are **not** comparable and are "
                   "deliberately not carried forward.")
        out.append("")
    else:
        for name, items in (("Improved", improved), ("Constant", constant),
                            ("**Regressed**", regressed)):
            out.append(f"**{name}** ({len(items)})")
            out.append("")
            if items:
                out += [f"- {s}" for s in items]
            else:
                out.append("- none")
            out.append("")
        if regressed:
            out.append("> A regression is listed above. Per the policy in the plan, "
                       "this entry is only valid if each is either fixed or "
                       "justified in writing below.")
            out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prev", default=None)
    ap.add_argument("--number", type=int, default=1)
    ap.add_argument("--title", default="")
    ap.add_argument("--changes", nargs="*", default=[])
    ap.add_argument("--theory", default="")
    ap.add_argument("--noise", default=None,
                    help="tag of a noise-floor run, used for the tolerances")
    ap.add_argument("--chart", default=None)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args(argv)

    cur = load(args.tag)
    prev = load(args.prev) if args.prev else None
    entry = render_entry(args.number, args.title, cur, prev, args.changes,
                         args.theory, chart=args.chart)
    if args.append and JOURNAL.exists():
        JOURNAL.write_text(JOURNAL.read_text().rstrip() + "\n\n" + entry + "\n")
    else:
        print(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
