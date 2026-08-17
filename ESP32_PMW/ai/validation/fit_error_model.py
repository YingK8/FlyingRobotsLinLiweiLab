"""
Fit the predicted-error model, and check it on data it never saw.

The gate in `uncertainty.py` decides which frames the estimator is allowed to
answer on.  If it is optimistic the target is missed; if it is pessimistic the
detection rate collapses and the estimator becomes useless while appearing
perfect.  So it is fitted on one split and judged on another, and the judgement
reported here is the one that matters:

* **coverage** -- what fraction of accepted frames actually met the spec. This is
  the number the whole exercise is about, and it has to be 100%.
* **acceptance** -- what fraction of frames it let through. Coverage is trivial
  to achieve by rejecting everything, so it is meaningless without this beside it.

Training data comes from a resolution sweep run with ``keep_features``, so the
model sees the same observables at fit time and at run time, computed by the same
function.

Run: uv run python controller/pose/validation/fit_error_model.py --poses 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [
    str(HERE),
    str(HERE.parent / "validation"),
    str(_C / "pose"),
    str(_C / "calib"),
    str(_C / "camera"),
]

import resolution_sweep as rsweep  # noqa: E402
import uncertainty  # noqa: E402

RESULTS = HERE.parents[2] / "results" / "pose_validation"


def collect(n_poses, seed, tiers=("core", "edge")):
    """
    Render, score with features kept, and return the training samples.

        **Both tiers are used to fit.** The gate has to behave sensibly on frames it
        should reject, and a model fitted only on good conditions has never seen a
        bad one -- it would have no basis for predicting a large error and would let
        the worst frames through.
    """

    frames, render_rig = rsweep.render_frames(n_poses, seed)
    samples = []
    for tier in tiers:
        print(f"\n  scoring {tier} for features:", flush=True)
        _, s = rsweep.score(frames[tier], render_rig, keep_features=True)
        for row in s:
            row["tier"] = tier
        samples += s
    return samples


def split3(samples, seed=0):
    """
    Split the **modes** three ways: fit / select / report.

        Two splits are not enough here, and the reason is specific.  ``k`` is a
        single global multiplier, so it does not reorder frames -- it only slides the
        accept/reject threshold along a fixed ranking.  Choosing it is therefore
        choosing an operating point, and an operating point picked on the same data
        it is reported against is guaranteed to look good.

        So: the **scale** is fitted on one set of modes, the **threshold** is chosen
        on a second, and the numbers quoted come from a third that neither step saw.
    """

    modes = sorted({s["mode"] for s in samples})
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(modes))
    n_fit = max(1, len(order) // 2)
    n_sel = max(1, (len(order) - n_fit) // 2)
    fit_m = set(order[:n_fit])
    sel_m = set(order[n_fit : n_fit + n_sel])
    rep_m = set(order[n_fit + n_sel :])
    pick = lambda ms: [s for s in samples if s["mode"] in ms]
    return (
        pick(fit_m),
        pick(sel_m),
        pick(rep_m),
        sorted(fit_m),
        sorted(sel_m),
        sorted(rep_m),
    )


def choose_threshold(
    samples,
    target_pos,
    target_ang,
    blunder_pos=5.0,
    blunder_ang=10.0,
    quantiles=None,
    verbose=True,
):
    """
    Pick the operating point by **leave-one-mode-out**, mid-plateau.

        Two corrections over the obvious approach, both of which cost real accuracy
        when they were got wrong:

        *Selection set.*  A single held-out split makes the choice hostage to which
        modes happen to land in it -- one draw put 160x120 there, a mode with almost
        no valid frames, and the threshold it chose then missed 100% on fresh data.
        Every mode is held out in turn instead and the results pooled, so the
        threshold answers "does this generalise across resolution" rather than
        "does this work on the two modes I happened to draw".

        *Where on the plateau.*  Coverage as a function of the quantile is a step:
        100% above some value, decaying below it. The most permissive point holding
        100% is the plateau's **edge**, and an edge is exactly where resampling
        flips the answer -- measured, q=0.80 sat one step past it and delivered
        94-97% instead of 100%. So the midpoint of the plateau is taken, trading
        acceptance for a margin that survives new data.
    """

    quantiles = quantiles if quantiles is not None else np.arange(0.95, 0.40, -0.01)
    modes = sorted({s["mode"] for s in samples})
    core_n = max(1, sum(1 for s in samples if s["tier"] == "core"))
    holding = []
    for q in quantiles:
        pooled = []
        for held in modes:
            fit = [
                s
                for s in samples
                if s["mode"] != held
                and s["pos_err"] <= blunder_pos
                and s["ang_err"] <= blunder_ang
            ]
            if len(fit) < 40:
                continue
            m = uncertainty.ErrorModel.fit(
                [r["features"] for r in fit],
                [r["pos_err"] for r in fit],
                [r["ang_err"] for r in fit],
                quantile=float(q),
            )
            pooled += [
                s
                for s in samples
                if s["mode"] == held
                and s["tier"] == "core"
                and m.accepts(s["features"], target_pos, target_ang)
            ]
        if not pooled:
            continue
        cov = np.mean(
            [s["pos_err"] <= target_pos and s["ang_err"] <= target_ang for s in pooled]
        )
        # **Per-mode**, not pooled. A pooled average lets a weak resolution hide
        # inside a strong one: measured, a threshold at 100% pooled coverage
        # still left 640x480 at 85.7% in spec, because the frames it got wrong
        # were a small share of a set dominated by 1280x800. The specification is
        # per test case, so every mode that answers at all must be perfect.
        by_mode = {}
        for s in pooled:
            by_mode.setdefault(s["mode"], []).append(
                s["pos_err"] <= target_pos and s["ang_err"] <= target_ang
            )
        worst_mode = min(np.mean(v) for v in by_mode.values())
        if verbose:
            print(
                f"    q={q:.2f}  accept {len(pooled)/core_n:6.1%}  "
                f"pooled {cov:7.2%}  worst mode {worst_mode:7.2%}"
            )
        if worst_mode >= 1.0:
            holding.append((float(q), len(pooled) / core_n))
    if not holding:
        return None
    qs = [q for q, _ in holding]
    chosen = float(np.median(qs))
    # snap to an actually-evaluated quantile
    chosen = min(qs, key=lambda q: abs(q - chosen))
    acc = dict(holding)[chosen]
    if verbose:
        print(
            f"\n  100% plateau spans q = {min(qs):.2f} .. {max(qs):.2f}; "
            f"taking the midpoint q = {chosen:.2f} (pooled acceptance {acc:.1%})"
        )
    return chosen, acc, (min(qs), max(qs))


def split(samples, frac=0.5, seed=0):
    """
    Split by **mode**, not at random.

        A random split would put frames of the same pose at 1280x800 and at 640x480
        on both sides, and those are far from independent -- the model would be
        graded partly on data it had effectively seen. Splitting by resolution asks
        the harder and more useful question: does a model fitted at some resolutions
        generalise to others?
    """

    modes = sorted({s["mode"] for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(modes)
    cut = max(1, int(len(modes) * frac))
    train_modes = set(modes[:cut])
    train = [s for s in samples if s["mode"] in train_modes]
    test = [s for s in samples if s["mode"] not in train_modes]
    return train, test, sorted(train_modes), sorted(set(modes) - train_modes)


def evaluate(model, samples, target_pos, target_ang, label=""):
    """
    Coverage and acceptance on a set of samples.
    """

    accepted, met = 0, 0
    worst_pos = worst_ang = 0.0
    for s in samples:
        if not model.accepts(s["features"], target_pos, target_ang):
            continue
        accepted += 1
        ok = s["pos_err"] <= target_pos and s["ang_err"] <= target_ang
        met += int(ok)
        worst_pos = max(worst_pos, s["pos_err"])
        worst_ang = max(worst_ang, s["ang_err"])
    cov = met / accepted if accepted else float("nan")
    acc = accepted / len(samples) if samples else 0.0
    print(
        f"  {label:<12} accepted {acc:6.1%} of {len(samples):5d}   "
        f"coverage {cov:7.2%}   worst accepted: {worst_pos:.3f} mm / "
        f"{worst_ang:.3f}°"
    )
    return {
        "accepted": acc,
        "coverage": cov,
        "n": len(samples),
        "worst_pos": worst_pos,
        "worst_ang": worst_ang,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poses", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--quantile", type=float, default=uncertainty.RATIO_QUANTILE)
    ap.add_argument("--write", action="store_true", help="save the fitted model")
    ap.add_argument("--blunder-pos-mm", type=float, default=5.0)
    ap.add_argument("--blunder-ang-deg", type=float, default=10.0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--samples",
        default=None,
        help="reuse a saved sample file instead of re-rendering",
    )
    args = ap.parse_args(argv)

    cache = (
        Path(args.samples)
        if args.samples
        else (RESULTS / f"error_samples_{args.poses}_{args.seed}.json")
    )
    if cache.exists():
        print(f"reusing samples from {cache}")
        samples = json.loads(cache.read_text())
    else:
        samples = collect(args.poses, args.seed)
        cache.write_text(json.dumps(samples))
        print(f"cached samples to {cache}")
    print(f"\n{len(samples)} feature/error samples")

    # Single held-out report split for the quoted numbers; the *threshold* is
    # chosen by leave-one-mode-out inside the remaining data, never on this.
    fit_s, sel_s, rep_s, fm, sm, rm = split3(samples, seed=args.seed)
    select_pool = fit_s + sel_s
    print(
        f"  select pool:  {', '.join(sorted(set(fm) | set(sm)))}  "
        f"({len(select_pool)} samples)"
    )
    print(f"  report modes: {', '.join(rm)}  ({len(rep_s)} samples)")

    print("\n  choosing the operating point by leave-one-mode-out:")
    chosen = choose_threshold(
        select_pool,
        uncertainty.TARGET_POS_MM,
        uncertainty.TARGET_ANGLE_DEG,
        args.blunder_pos_mm,
        args.blunder_ang_deg,
        verbose=args.verbose,
    )
    if chosen is None:
        print("\nNo operating point holds 100% coverage. Not writing a model.")
        return 1
    q, sel_acc, plateau = chosen

    # Fit the shipped scale on the whole selection pool, precision regime only.
    # A blunder has ordinary-looking features and an error two orders out, so no
    # monotone function of these features predicts it; leaving them in lets them
    # set the quantile (measured: k reached 674, and the gate then rejected every
    # frame). They stay in the *evaluation*, so a blunder the hard gates miss
    # surfaces as a coverage failure rather than being excused.
    fit_rows = [
        s
        for s in select_pool
        if s["pos_err"] <= args.blunder_pos_mm and s["ang_err"] <= args.blunder_ang_deg
    ]
    print(
        f"  fitting scale on {len(fit_rows)}/{len(select_pool)} "
        f"(excluded {len(select_pool) - len(fit_rows)} blunders)"
    )
    model = uncertainty.ErrorModel.fit(
        [r["features"] for r in fit_rows],
        [r["pos_err"] for r in fit_rows],
        [r["ang_err"] for r in fit_rows],
        quantile=q,
        meta={
            "source": "resolution_sweep",
            "poses": args.poses,
            "seed": args.seed,
            "chosen_quantile": q,
            "plateau": list(plateau),
            "select_modes": sorted(set(fm) | set(sm)),
            "report_modes": rm,
        },
    )

    print("\nfitted scale coefficients")
    for name, coef in (
        ("position", zip(uncertainty.FEATURES_POS, model.coef_pos)),
        ("angle", zip(uncertainty.FEATURES_ANG, model.coef_ang)),
    ):
        print(f"  {name:9s} " + "  ".join(f"{n}={c:+.4g}" for n, c in coef))
    print(f"  quantile inflation  k_pos={model.k_pos:.3f}  k_ang={model.k_ang:.3f}")

    print(
        f"\nwith the gate at ±{uncertainty.TARGET_ANGLE_DEG:g}° / "
        f"±{uncertainty.TARGET_POS_MM:g} mm "
        f"(report split -- neither fitted nor selected on):"
    )
    ev = {"chosen_quantile": q, "plateau": list(plateau), "select_acceptance": sel_acc}
    ev["report"] = evaluate(
        model,
        rep_s,
        uncertainty.TARGET_POS_MM,
        uncertainty.TARGET_ANGLE_DEG,
        "report all",
    )
    for tier in ("core", "edge"):
        sub = [s for s in rep_s if s["tier"] == tier]
        if sub:
            ev[f"report_{tier}"] = evaluate(
                model,
                sub,
                uncertainty.TARGET_POS_MM,
                uncertainty.TARGET_ANGLE_DEG,
                f"report {tier}",
            )

    out = RESULTS / "error_model_fit.json"
    out.write_text(
        json.dumps(
            {
                "evaluation": ev,
                "quantile": args.quantile,
                "poses": args.poses,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    if args.write:
        print(f"wrote {model.save()}")
    else:
        print("(not saved; pass --write to install it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
