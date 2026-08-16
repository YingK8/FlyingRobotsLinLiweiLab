"""Turn a sweep CSV into a text summary and figures.

    uv run python controller/pose/validation/report.py controller/pose/validation_results.csv

Reports medians rather than means throughout.  The error distributions have long
tails -- a handful of cells where segmentation lost the ring entirely, or where
the wrong ambiguity branch was chosen, sit orders of magnitude above the rest --
and a mean would report those outliers as if they were the typical case.  The
tail is shown separately, as a failure count and a p95, which is the honest way
to present it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Targets from the plan: what "working" means before this goes near a controller.
TARGET_NORMAL_DEG = 2.0
TARGET_POS_MM = 1.0
FPS_BUDGETS = (240, 420)


def load(path):
    df = pd.read_csv(path, comment="#")
    return df


def _fmt(v, unit="", nd=2):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}{unit}"


def summarise(path, tilt_limit=40.0):
    """Print the summary. ``tilt_limit`` marks where the flat-circle model holds."""
    df = load(path)
    n = len(df)
    det = df[df["detected"] == 1].copy()

    print()
    print("=" * 78)
    print(f"pose validation summary  --  {Path(path).name}")
    print("=" * 78)
    print(f"cells {n} | detected {len(det)} ({len(det)/max(1,n):.1%}) | "
          f"missed {n - len(det)}")

    if det.empty:
        print("nothing detected; no metrics to report")
        return df

    inside = det[det["tilt_deg"] <= tilt_limit]
    outside = det[det["tilt_deg"] > tilt_limit]

    print()
    print(f"-- accuracy within the flat-circle envelope (tilt <= {tilt_limit:g} deg, "
          f"{len(inside)} cells)")
    _accuracy_block(inside)

    if not outside.empty:
        print()
        print(f"-- outside the envelope (tilt > {tilt_limit:g} deg, {len(outside)} cells)")
        print("   the mast and magnet dominate the silhouette here; expected to degrade")
        _accuracy_block(outside)

    print()
    print("-- ambiguity (two back-projection branches; one frame cannot choose)")
    wrong = det["branch_wrong"].sum()
    print(f"   wrong branch chosen in {wrong}/{len(det)} cells ({wrong/len(det):.1%})")
    print(f"   median normal error, chosen branch : {_fmt(det['normal_err_deg'].median(), ' deg')}")
    print(f"   median normal error, best branch   : {_fmt(det['normal_err_best_deg'].median(), ' deg')}")
    print(f"   median ambiguity margin            : {_fmt(det['ambiguity_margin_deg'].median(), ' deg')}")

    print()
    print("-- compute time")
    tot = det["t_total_ms"]
    print(f"   median {_fmt(tot.median(), ' ms', 3)} | p95 {_fmt(tot.quantile(0.95), ' ms', 3)} "
          f"| max {_fmt(tot.max(), ' ms', 3)}")
    print(f"   segmentation {_fmt(det['t_seg_ms'].median(), ' ms', 3)} | "
          f"back-projection {_fmt(det['t_est_ms'].median(), ' ms', 3)}")
    print(f"   sustainable {1e3/tot.median():.0f} Hz at the median, "
          f"{1e3/tot.quantile(0.95):.0f} Hz at p95")
    for fps in FPS_BUDGETS:
        budget = 1e3 / fps
        over = (tot > budget).mean()
        print(f"   vs {fps} fps ({budget:.2f} ms): {over:6.2%} of cells over budget"
              f"   [{'OK' if over < 0.01 else 'OVER'}]")

    print()
    print("-- detection rate by condition (share of cells with any detection)")
    for axis in ("alpha", "bg_level", "ambient"):
        rates = df.groupby(axis)["detected"].mean()
        cells = "  ".join(f"{k:g}:{v:.0%}" for k, v in rates.items())
        print(f"   {axis:10s} {cells}")

    print()
    print("-- median position error by condition (mm, detected cells only)")
    for axis in ("alpha", "bg_level", "ambient"):
        med = det.groupby(axis)["pos_err_mm"].median()
        cells = "  ".join(f"{k:g}:{v:.2f}" for k, v in med.items())
        print(f"   {axis:10s} {cells}")

    print()
    print("-- worst lighting rigs by median position error")
    by_light = det.groupby("light").agg(
        n=("pos_err_mm", "size"),
        pos=("pos_err_mm", "median"),
        normal=("normal_err_best_deg", "median"),
        detect=("detected", "mean"),
    ).sort_values("pos", ascending=False)
    for name, row in by_light.head(4).iterrows():
        print(f"   {name[:44]:44s} pos {row['pos']:6.2f} mm  normal {row['normal']:5.2f} deg")

    print()
    _verdict(inside)
    return df


def _accuracy_block(d):
    if d.empty:
        print("   (no cells)")
        return
    for label, col, unit in (
        ("position       ", "pos_err_mm", " mm"),
        ("  depth (z)    ", "dz_mm", " mm"),
        ("normal (best)  ", "normal_err_best_deg", " deg"),
        ("normal (chosen)", "normal_err_deg", " deg"),
        ("tilt bias      ", "tilt_err_deg", " deg"),
        ("segmentation IoU", "seg_iou", ""),
    ):
        v = d[col].abs() if col == "dz_mm" else d[col]
        print(f"   {label} median {_fmt(v.median(), unit)} | p95 {_fmt(v.quantile(0.95), unit)}"
              f" | max {_fmt(v.max(), unit)}")


def _verdict(inside):
    if inside.empty:
        return
    pos = inside["pos_err_mm"].median()
    nrm = inside["normal_err_best_deg"].median()
    ok_pos = pos < TARGET_POS_MM
    ok_nrm = nrm < TARGET_NORMAL_DEG
    print(f"-- verdict against plan targets (median, within envelope)")
    print(f"   position  {_fmt(pos, ' mm')} vs target {TARGET_POS_MM} mm   "
          f"[{'PASS' if ok_pos else 'MISS'}]")
    print(f"   normal    {_fmt(nrm, ' deg')} vs target {TARGET_NORMAL_DEG} deg  "
          f"[{'PASS' if ok_nrm else 'MISS'}]")


def figures(path, outdir=None):
    """Heatmaps, a timing histogram and an error-vs-tilt curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = load(path)
    det = df[df["detected"] == 1]
    outdir = Path(outdir or Path(path).with_suffix("")).with_name(
        (Path(path).stem + "_figures")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    _heatmap(axes[0, 0], det, "alpha", "bg_level", "pos_err_mm",
             "median position error (mm)", plt)
    _heatmap(axes[0, 1], det, "tilt_deg", "ambient", "normal_err_best_deg",
             "median normal error, best branch (deg)", plt)

    ax = axes[1, 0]
    curve = det.groupby("tilt_deg").agg(
        best=("normal_err_best_deg", "median"),
        chosen=("normal_err_deg", "median"),
        pos=("pos_err_mm", "median"),
    )
    ax.plot(curve.index, curve["best"], "o-", label="normal err (best branch)")
    ax.plot(curve.index, curve["chosen"], "s--", label="normal err (chosen)", alpha=0.7)
    ax.plot(curve.index, curve["pos"], "^-", label="position err (mm)")
    ax.axvline(40, color="k", ls=":", lw=1)
    ax.text(40.5, ax.get_ylim()[1] * 0.9, "flat-circle\nenvelope", fontsize=8, va="top")
    ax.set_xlabel("ground-truth tilt (deg)")
    ax.set_ylabel("median error")
    ax.set_title("error vs tilt")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(det["t_total_ms"], bins=50, color="tab:blue", alpha=0.8)
    for fps, colour in zip(FPS_BUDGETS, ("tab:orange", "tab:red")):
        ax.axvline(1e3 / fps, color=colour, ls="--", lw=1.5, label=f"{fps} fps budget")
    ax.set_xlabel("compute time per frame (ms)")
    ax.set_ylabel("cells")
    ax.set_title("compute time vs frame-rate budget")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"pose validation - {Path(path).name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = outdir / "summary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"figures -> {out}")
    return out


def _heatmap(ax, det, xcol, ycol, vcol, title, plt):
    if det.empty:
        ax.set_title(title + " (no data)")
        return
    piv = det.pivot_table(index=ycol, columns=xcol, values=vcol, aggfunc="median")
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns], fontsize=8)
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index], fontsize=8)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title, fontsize=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if v < np.nanmedian(piv.values) else "k")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", help="sweep results CSV")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--tilt-limit", type=float, default=40.0)
    args = ap.parse_args(argv)

    summarise(args.csv, tilt_limit=args.tilt_limit)
    if not args.no_figures:
        figures(args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
