#!/usr/bin/env python3
"""
Coil array optimisation: sample-based global search with gradient polish. theory.md sect. 15.4.

Section 14 swept a two-parameter coplanar family exhaustively and found nothing laterally
stable. This searches the family that section 15 opens up instead: two rotating rings at
independent heights with the upper one reverse-wound, plus a steady ring that couples to the
rotor's axial moment and nothing else.

The objective is the real one. Not section 14's R, which is a scalar summary of a reduced
third-order model, but the stability margin of the **full eight-state plant** linearised at its
own trim with the currents held fixed:

    maximise  sigma = -max Re lambda(A),   A = stability_cert.linearize(...)

That matters because R and the static stiffness C_net are two separate necessary conditions and
neither implies the other: the two-ring array reaches R = 0.55 while still being laterally
expelling. The eigenvalues fold both in and cannot be gamed by satisfying one of them.

Optimiser: `differential_evolution`, which is population-based sampling, iterated, with an
L-BFGS-B gradient polish on the winner. Global optimality is not provable for a problem this
non-convex, so the run reports best-of-N restarts with a spread diagnostic instead of claiming it.

    uv run python controller/control/optimise_array.py --self-check
    uv run python controller/control/optimise_array.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

import coil_geometry as cg
import drive_model as dm
import spatial_model as sm
import stability_cert as sc

OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "array_optimisation"

# name, low, high. Lengths in m, angles in deg, current in A, frequency in Hz.
BOUNDS = [
    ("r_lo",   0.025, 0.070),
    ("z_lo",  -0.040, 0.010),
    ("r_hi",   0.025, 0.070),
    ("z_hi",   0.025, 0.100),
    ("tilt_lo", -30.0, 30.0),
    ("tilt_hi", -30.0, 30.0),
    ("pitch",  0.019, 0.032),
    ("r_dc",   0.030, 0.110),
    ("z_dc",  -0.010, 0.045),
    ("i_dc",   0.0,   5.0),
    ("f",      70.0,  160.0),
]
NAMES = [b[0] for b in BOUNDS]
LIMITS = [(b[1], b[2]) for b in BOUNDS]

Z_WORK = 0.020          # nominal working height, used only for the camera cone
LOCK_MAX = 0.8
CLEAR_MIN = -1e-6       # m. The rig as built has pitch = 2R exactly, so its coils touch;
                        # touching is fine, interpenetrating is not.
WORK_CLEAR_MIN = 0.008  # m, coil body to the robot
BAD = 1e3               # cost floor for designs that do not even have a trim


def unpack(x):
    return dict(zip(NAMES, np.asarray(x, float)))


def build(params, m_axial_frac=0.03, amp=None, v_rail=dm.V_RAIL, model="loop"):
    """``(coils, plant, amp)`` for a parameter vector, with the drive current set by the rail."""

    q = params if isinstance(params, dict) else unpack(params)
    if amp is None:
        # The supply, not the coil rating, is what caps the current (drive_model).
        amp = dm.max_current(q["f"], v_rail=v_rail, square=True)
    coils = cg.c4_array(
        r_lo=q["r_lo"], z_lo=q["z_lo"], r_hi=q["r_hi"], z_hi=q["z_hi"],
        tilt_lo=q["tilt_lo"], tilt_hi=q["tilt_hi"], pitch=q["pitch"], amp=amp,
        n_per_ring=2, r_dc=q["r_dc"], z_dc=q["z_dc"], i_dc=q["i_dc"], model=model,
    )
    robot = sm.replace(sm.robot_params(), m_axial=m_axial_frac * sm.robot_params().mdip)
    plant = sm.Plant(coils, robot, f_hover=110.0, use_grad=True)
    return coils, plant, amp


def evaluate(params, m_axial_frac=0.03, v_rail=dm.V_RAIL, model="loop"):
    """Full metric set for one design, or a dict with ``feasible=False`` and a graded penalty."""

    q = params if isinstance(params, dict) else unpack(params)
    coils, plant, amp = build(q, m_axial_frac, v_rail=v_rail, model=model)

    gap = cg.pair_clearance(coils)
    if gap < CLEAR_MIN:
        return dict(**q, feasible=False, why="coils interfere", penalty=1e3 * (CLEAR_MIN - gap),
                    clearance=gap, margin=np.nan, z_eq=np.nan)

    t = sc.stable_trim(plant, q["f"])
    if t is None:
        return dict(**q, feasible=False, why="no stable trim", penalty=10.0,
                    clearance=gap, margin=np.nan, z_eq=np.nan)
    z_eq, k_az = t

    work = np.array([0.0, 0.0, z_eq])
    w_clear = cg.workspace_clearance(coils, work)
    cam = cg.camera_clearance(coils, work)
    if w_clear < WORK_CLEAR_MIN:
        return dict(**q, feasible=False, why="coil in the workspace",
                    penalty=100.0 * (WORK_CLEAR_MIN - w_clear), clearance=gap,
                    margin=np.nan, z_eq=z_eq)

    u, v, _, _ = sm.field_at(work, coils, None, False)
    lock = sm.lock_margin(u, v, plant.robot.mdip, plant.k_drag * q["f"] ** 2)
    if lock > LOCK_MAX:
        return dict(**q, feasible=False, why="step-out", penalty=10.0 * (lock - LOCK_MAX),
                    clearance=gap, lock=lock, margin=np.nan, z_eq=z_eq)

    if cam < 0.0:
        return dict(**q, feasible=False, why="coil blocks a camera",
                    penalty=1.0 * (-cam), clearance=gap, lock=lock, camera=cam,
                    margin=np.nan, z_eq=z_eq)

    a_mat = sc.linearize(plant, q["f"], z_eq)
    sigma = sc.margin(a_mat)
    b0, _, b2 = cg.field_shape(coils, z_eq)
    w_z = math.sqrt(max(-k_az, 0.0))
    om = 0.5 * plant.robot.mdip * b0 / (plant.robot.I_spin * 2 * math.pi * q["f"])

    return dict(
        **q, feasible=True, why="", penalty=0.0, amp=amp, z_eq=z_eq, margin=sigma,
        w_axial=w_z, Omega_align=om, R=math.sqrt(2) * om / w_z if w_z > 0 else np.inf,
        B_mean=b0, d2B=b2, lock=lock, clearance=gap, work_clearance=w_clear,
        camera=cam, power=dm.power(amp), fom=cg.figure_of_merit(coils, z_eq),
    )


def cost(x, m_axial_frac=0.03, v_rail=dm.V_RAIL, w_min=0.0, model="loop"):
    """Scalar to minimise: minus the stability margin, plus graded infeasibility penalties."""

    try:
        m = evaluate(x, m_axial_frac, v_rail, model)
    except (sm.StepOut, ValueError, np.linalg.LinAlgError):
        return BAD
    if not m["feasible"]:
        return BAD + m["penalty"]
    if w_min > 0.0 and m["w_axial"] < w_min:
        # A floor on the axial trap, not a term traded against the margin. Every constraint
        # here gates rather than trades: a soft camera penalty on the same scale as the
        # margin bought clearance by giving up real stability, which is not the deal wanted.
        return BAD + 10.0 * (w_min - m["w_axial"])
    return -m["margin"]


# --------------------------------------------------------------------------


# The two-ring array of section 15.1, as a warm start. Not an answer, a place to begin.
X_SEED = np.array([0.037, -0.010, 0.037, 0.050, 0.0, 0.0, 0.021, 0.060, 0.020, 2.0, 110.0])


def _safe_workers(workers):
    """Fall back to serial when multiprocessing cannot re-import the entry point.

    `workers=-1` needs a real, importable __main__. Run the module from stdin (a heredoc, or
    `exec` in a notebook) and every worker tries to import `<stdin>`, fails, and the pool
    retries forever: the run does not error, it hangs. Cheaper to detect than to debug.
    """

    if workers == 1:
        return 1
    main = sys.modules.get("__main__")
    return workers if getattr(main, "__file__", None) else 1


def optimise(seed=0, maxiter=60, popsize=18, w_min=0.0, m_axial_frac=0.03,
             v_rail=dm.V_RAIL, workers=-1, model="loop", x0=None):
    """One global run: differential evolution, then an L-BFGS-B polish on the winner.

    ``x0`` seeds one member of the initial population, which makes the run monotone against
    that point: differential evolution never returns worse than its best initial member.
    """

    args = (m_axial_frac, v_rail, w_min, model)
    workers = _safe_workers(workers)
    res = differential_evolution(
        cost, LIMITS, args=args, seed=seed, maxiter=maxiter, popsize=popsize,
        tol=1e-6, mutation=(0.4, 1.0), recombination=0.8, polish=True,
        workers=workers, updating="deferred", init="sobol", x0=x0,
    )
    # Refine locally around the global answer AND around the seed, keeping whichever wins.
    #
    # The second is not redundant. The feasible region is a thin sliver bounded by the camera
    # cone and the best designs sit on its edge; differential evolution's mutation walks out
    # of the sliver and never returns to refine the seed. Measured: a 3960-evaluation global
    # run returned -1.17 /s where a local refinement of the same seed returned +0.067 /s.
    #
    # The refinement is another differential evolution in a small box, not L-BFGS-B. Two
    # reasons, both measured. It parallelises, where `minimize` is serial and spends 12
    # finite-difference evaluations per iteration on 11 parameters; and truncating that to a
    # budget which does parallelise left it unconverged, at -0.57 /s.
    cands = [(float(res.fun), res.x)]
    for st in [res.x] + ([np.asarray(x0, float)] if x0 is not None else []):
        cands.append(_refine(st, args, seed=seed, workers=workers))
    fun, x = min(cands, key=lambda c: c[0])
    return x, fun, res


def _refine(x, args, frac=0.10, maxiter=25, seed=0, workers=-1):
    """Local differential evolution in a box of +/- ``frac`` of the range around ``x``."""

    x = np.asarray(x, float)
    span = np.ptp(LIMITS, axis=1)
    box = [(max(lo, xi - frac * sp), min(hi, xi + frac * sp))
           for xi, sp, (lo, hi) in zip(x, span, LIMITS)]
    r = differential_evolution(cost, box, args=args, seed=seed, maxiter=maxiter, popsize=10,
                               tol=1e-8, polish=False, workers=_safe_workers(workers),
                               updating="deferred", x0=x)
    return float(r.fun), r.x


def multistart(n=6, **kw):
    """Best of ``n`` independent global runs, with the spread across them as the diagnostic.

    Global optimality is not provable here. What is reportable is whether independent runs
    agree; if they scatter, the answer is a good design and not a proven optimum, and the
    printed spread is what says so.
    """

    rows = []
    for k in range(n):
        # Every restart is seeded from the two-ring array, jittered, rather than only the
        # first: an unseeded run in this landscape rarely finds the feasible sliver at all.
        rng = np.random.default_rng(k)
        span = np.ptp(LIMITS, axis=1)
        x0 = X_SEED if k == 0 else np.clip(
            X_SEED + 0.12 * span * rng.normal(size=len(X_SEED)),
            [lo for lo, _ in LIMITS], [hi for _, hi in LIMITS])
        x, f, _ = optimise(seed=k, x0=x0, **kw)
        m = evaluate(x, kw.get("m_axial_frac", 0.03), kw.get("v_rail", dm.V_RAIL),
                     kw.get("model", "loop"))
        rows.append({**m, "seed": k, "cost": f})
        print(f"  seed {k}: cost {f:+.4f}  margin {m.get('margin', float('nan')):+.4f}/s  "
              f"{'' if m['feasible'] else m['why']}")
    df = pd.DataFrame(rows)
    ok = df[df["feasible"]]
    if len(ok) > 1:
        print(f"  spread of the margin across {len(ok)} feasible runs: "
              f"{ok['margin'].std():.4f} /s on a mean of {ok['margin'].mean():+.4f}")
    return df


def pareto(w_mins=(0.0, 0.5, 1.5, 3.0), n=2, **kw):
    """Stability margin against the axial trap it is bought with. This is the real answer."""

    rows = []
    for w in w_mins:
        print(f"axial floor {w:.1f} rad/s ({w/2/math.pi:.2f} Hz):")
        df = multistart(n=n, w_min=w, **kw)
        ok = df[df["feasible"]]
        if len(ok):
            best = ok.loc[ok["margin"].idxmax()].to_dict()
            rows.append({**best, "w_min": w})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------


def _self_check():
    # 1. The parameter vector round-trips and the builder honours it.
    x0 = np.array([0.037, -0.010, 0.037, 0.050, 0.0, 0.0, 0.021, 0.060, 0.020, 2.0, 110.0])
    q = unpack(x0)
    coils, plant, amp = build(q)
    assert len(coils.pos) == 20, len(coils.pos)
    assert abs(amp - dm.max_current(110.0)) < 1e-12
    print(f"build       : 16 rotating + 4 steady coils, drive {amp:.3f} A set by the "
          f"{dm.V_RAIL:.0f} V rail")

    # 2. The metric set is complete and agrees with stability_cert on the same design.
    m = evaluate(q)
    assert m["feasible"], m
    a = sc.linearize(plant, 110.0, m["z_eq"])
    assert abs(sc.margin(a) - m["margin"]) < 1e-9
    print(f"evaluate    : z_eq {m['z_eq']*1e3:.2f} mm, margin {m['margin']:+.4f}/s, "
          f"R {m['R']:.3f}, lock {m['lock']:.3f}, {m['power']:.1f} W")

    # 3. Infeasibility is graded, not a cliff: a global sampler needs a slope to follow.
    tight = dict(q, pitch=0.0195, r_lo=0.026)
    c_bad = cost(np.array([tight[k] for k in NAMES]))
    worse = dict(tight, pitch=0.019)
    c_worse = cost(np.array([worse[k] for k in NAMES]))
    assert c_bad >= BAD and c_worse >= c_bad, (c_bad, c_worse)
    assert cost(x0) < BAD
    print(f"penalties   : feasible {cost(x0):+.4f}, interfering {c_bad:.1f} -> {c_worse:.1f} "
          f"as it gets worse")

    # 4. The optimiser finds a planted optimum on a synthetic objective, which is the only way
    # to test the search itself rather than the physics.
    target = np.array([0.5 * (a + b) for a, b in LIMITS])

    def quad(z, *_):
        return float(np.sum(((np.asarray(z) - target) / np.ptp(LIMITS, axis=1)) ** 2))

    res = differential_evolution(quad, LIMITS, seed=0, maxiter=60, popsize=12,
                                 tol=1e-12, polish=True, workers=1, init="sobol")
    err = np.max(np.abs(res.x - target) / np.ptp(LIMITS, axis=1))
    assert err < 1e-3, (err, res.x, target)
    print(f"optimiser   : recovers a planted optimum to {err:.1e} of the box width")

    # 5. A short real run must improve on its own starting point. Deliberately tiny: this
    # checks the wiring, not the search, which item 4 already covers.
    x_best, f_best, _ = optimise(seed=0, maxiter=3, popsize=3, workers=1, x0=x0)
    assert f_best <= cost(x0) + 1e-9, (f_best, cost(x0))
    mb = evaluate(x_best)
    print(f"short run   : cost {cost(x0):+.4f} -> {f_best:+.4f}"
          + (f", margin {mb['margin']:+.4f}/s at "
             f"r_lo {mb['r_lo']*1e3:.1f} z_hi {mb['z_hi']*1e3:.1f} f {mb['f']:.0f} Hz"
             if mb["feasible"] else f" ({mb['why']})"))

    print("self-check PASS")


def run(out_dir=OUT_DIR, n=2, maxiter=25, popsize=12, v_rail=dm.V_RAIL, m_axial_frac=0.03):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Pareto: stability margin against the axial trap it costs\n")
    df = pareto(n=n, maxiter=maxiter, popsize=popsize, v_rail=v_rail,
                m_axial_frac=m_axial_frac)
    df.to_csv(out_dir / "pareto.csv", index=False)
    if len(df):
        best = df.loc[df["margin"].idxmax()]
        print(f"\nbest margin {best['margin']:+.4f} /s at axial floor {best['w_min']:.1f} rad/s")
        print(f"  r_lo {best['r_lo']*1e3:.1f} mm, z_lo {best['z_lo']*1e3:.1f} mm, "
              f"r_hi {best['r_hi']*1e3:.1f} mm, z_hi {best['z_hi']*1e3:.1f} mm")
        print(f"  tilt {best['tilt_lo']:.1f} / {best['tilt_hi']:.1f} deg, "
              f"pitch {best['pitch']*1e3:.1f} mm, f {best['f']:.1f} Hz")
        print(f"  DC ring r {best['r_dc']*1e3:.1f} mm, z {best['z_dc']*1e3:.1f} mm, "
              f"{best['i_dc']:.2f} A")
        print(f"  z_eq {best['z_eq']*1e3:.2f} mm, B {best['B_mean']*1e3:.3f} mT, "
              f"R {best['R']:.3f}, lock {best['lock']:.3f}, {best['power']:.1f} W")
    print(f"\nwrote {out_dir}/pareto.csv")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--rail", type=float, default=dm.V_RAIL)
    ap.add_argument("--m-axial", type=float, default=0.03, help="axial moment, fraction of mdip")
    args = ap.parse_args()
    if args.self_check:
        _self_check()
    else:
        run(args.out, n=args.restarts, maxiter=args.maxiter, v_rail=args.rail,
            m_axial_frac=args.m_axial)
