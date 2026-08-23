#!/usr/bin/env python3
"""
Closed-loop simulation and viewer for the multi-coil spatial plant.

Headless (default) runs a scripted target sequence, writes spatial_mpc_sim.png, and asserts the
tracking and constraint bounds. --live animates in real time: click the top-down panel to move
the target, the slider sets its height. The live sim advances by wall-clock elapsed time, so a
slow solve shows as the clock slowing rather than as a trajectory pretending to be real time.

The controller runs against its own model, so the position traces measure the control law and
nothing else. The panels carrying real information are the two constraint traces: the lock
margin, which is how close the robot is to losing synchronisation, and the channel currents,
which are what the amplifier has to deliver. A run that tracks beautifully while riding either
limit has not solved the problem.

Usage:
    uv run python controller/control/simulate_spatial.py [--live] [--grad]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from spatial_model import StepOut, make_state
from spatial_mpc import closed_loop, default_mpc

HOVER = np.array([0.0, 0.0, 0.0130])  # default target, m
START = np.array([0.006, -0.004, 0.0180])


def script(t):
    """Scripted target: hold, step across, climb, then a diagonal."""

    if t < 2.0:
        return HOVER
    if t < 5.0:
        return HOVER + np.array([0.006, 0.0, 0.0])
    if t < 8.0:
        return HOVER + np.array([0.006, 0.0, 0.004])
    return HOVER + np.array([-0.004, 0.005, 0.002])


def plot(run, mpc, path):
    """Five panels: 3-D path, per-axis tracking, lock margin, current, input."""

    import matplotlib.pyplot as plt

    t = run["t"]
    fig = plt.figure(figsize=(13, 8))
    ax3 = fig.add_subplot(2, 3, 1, projection="3d")
    ax3.plot(*(run["r"].T * 1e3), lw=1.2, label="flown")
    ax3.plot(*(run["ref"].T * 1e3), "--", lw=1.0, alpha=0.7, label="target")
    ax3.scatter(*(run["r"][0] * 1e3), c="k", s=25, label="start")
    ax3.scatter(*(run["ref"][-1] * 1e3), marker="*", c="r", s=90, label="final target")
    ax3.set_xlabel("x [mm]")
    ax3.set_ylabel("y [mm]")
    ax3.set_zlabel("z [mm]")
    ax3.legend(fontsize=7)
    ax3.set_title("trajectory")

    ax = fig.add_subplot(2, 3, 2)
    for i, name in enumerate("xyz"):
        line = ax.plot(t, run["r"][:, i] * 1e3, label=name)[0]
        ax.plot(t, run["ref"][:, i] * 1e3, "--", lw=0.9, color=line.get_color())
    ax.set_ylabel("position [mm]")
    ax.set_xlabel("t [s]")
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("solid = flown, dashed = target")

    ax = fig.add_subplot(2, 3, 3)
    err = np.linalg.norm(run["r"] - run["ref"], axis=1) * 1e3
    ax.semilogy(t, np.maximum(err, 1e-4))
    ax.axhline(mpc.w.pos_tol * 1e3, color="g", ls=":", label="tolerance")
    ax.set_ylabel("|error| [mm]")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=8)
    ax.set_title("tracking error")

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, run["ratio"])
    ax.axhline(mpc.lim.lock_max, color="orange", ls=":", label="soft limit")
    ax.axhline(1.0, color="r", ls="--", label="step-out")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"required $\sin\varphi$")
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=8)
    ax.set_title("lock margin")

    ax = fig.add_subplot(2, 3, 5)
    for g in range(run["amps"].shape[1]):
        ax.plot(t, run["amps"][:, g], lw=0.9, label=f"ch{g}")
    ax.axhline(mpc.lim.i_max, color="r", ls="--", label="limit")
    ax.set_ylabel("channel current [A]")
    ax.set_xlabel("t [s]")
    ax.legend(ncol=3, fontsize=7)
    ax.set_title("coil current")

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, np.degrees(np.arctan(run["u"][:, 0])), label="field tilt x")
    ax.plot(t, np.degrees(np.arctan(run["u"][:, 1])), label="field tilt y")
    ax.set_ylabel("field tilt [deg]")
    ax.set_xlabel("t [s]")
    axf = ax.twinx()
    axf.plot(t, run["u"][:, 2], color="k", lw=0.9, label="f")
    axf.set_ylabel("f drive [Hz]")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("input")

    for a in fig.axes:
        a.grid(True, alpha=0.35)
    fig.suptitle(
        f"spatial MPC, {1/mpc.ts:.0f} Hz, horizon {mpc.horizon*mpc.ts:.1f} s, "
        f"gradient force {'on' if mpc.plant.use_grad else 'off'}"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def live(mpc, speed=1.0):
    """Real-time animation with a mouse-driven target."""

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    target = HOVER.copy()
    x = make_state(START, s=(0.0, 0.0, 1.0))
    mpc.design_lqr(target)
    u = mpc.u_trim.copy()

    fig = plt.figure(figsize=(12, 6.5))
    ax_xy = fig.add_axes([0.05, 0.28, 0.42, 0.64])
    ax_t = fig.add_axes([0.56, 0.58, 0.40, 0.34])
    ax_c = fig.add_axes([0.56, 0.16, 0.40, 0.30])
    ax_s = fig.add_axes([0.05, 0.10, 0.42, 0.03])

    lim = mpc.lim.box_xy * 1e3
    ax_xy.set_xlim(-lim, lim)
    ax_xy.set_ylim(-lim, lim)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_xy.grid(True, alpha=0.35)
    for g in range(mpc.plant.coils.n_channels):
        sel = mpc.plant.coils.channel == g
        ax_xy.scatter(
            *(mpc.plant.coils.pos[sel][:, :2].T * 1e3), s=6, c="0.75", marker="o"
        )
    (trail,) = ax_xy.plot([], [], "-", lw=1.0, alpha=0.8)
    (robot,) = ax_xy.plot([], [], "o", ms=9, c="tab:blue")
    (goal,) = ax_xy.plot([], [], "*", ms=16, c="tab:red")

    hist = {k: [] for k in ("t", "z", "zr", "e", "ratio", "amp")}
    (lz,) = ax_t.plot([], [], label="z")
    (lzr,) = ax_t.plot([], [], "--", label="z target")
    ax_t.set_ylabel("z [mm]")
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.35)
    (lr,) = ax_c.plot([], [], label=r"lock $\sin\varphi$")
    (la,) = ax_c.plot([], [], label="max |I| / limit")
    ax_c.axhline(1.0, color="r", ls="--", lw=0.8)
    ax_c.set_ylim(0, 1.2)
    ax_c.set_xlabel("t [s]")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.35)

    slider = Slider(
        ax_s,
        "target z [mm]",
        mpc.lim.box_z[0] * 1e3,
        mpc.lim.box_z[1] * 1e3,
        valinit=HOVER[2] * 1e3,
    )

    def on_click(event):
        if event.inaxes is ax_xy and event.xdata is not None:
            target[0] = np.clip(event.xdata * 1e-3, -mpc.lim.box_xy, mpc.lim.box_xy)
            target[1] = np.clip(event.ydata * 1e-3, -mpc.lim.box_xy, mpc.lim.box_xy)

    fig.canvas.mpl_connect("button_press_event", on_click)

    state = {"t": 0.0, "wall": time.perf_counter(), "rt": 1.0, "lost": False}
    trailxy = []

    def frame(_):
        nonlocal x, u
        if state["lost"]:
            return trail, robot, goal, lz, lzr, lr, la

        now = time.perf_counter()
        elapsed = min(now - state["wall"], 0.25) * speed
        state["wall"] = now
        target[2] = slider.val * 1e-3

        steps = max(1, int(round(elapsed / mpc.ts)))
        t_solve = 0.0
        for _ in range(steps):
            u, diag = mpc.solve(x, target, u_prev=u)
            t_solve += diag["solve_s"]
            x, info, ok = mpc.apply(x, u, mpc.ts)
            state["t"] += mpc.ts
            if not ok:
                state["lost"] = True
                break
            hist["t"].append(state["t"])
            hist["z"].append(x[2] * 1e3)
            hist["zr"].append(target[2] * 1e3)
            hist["ratio"].append(info["ratio"])
            hist["amp"].append(np.abs(mpc.allocate(u[:2])).max() / mpc.lim.i_max)
            trailxy.append((x[0] * 1e3, x[1] * 1e3))
        state["rt"] = steps * mpc.ts / max(t_solve, 1e-6)

        tr = np.array(trailxy[-400:])
        trail.set_data(tr[:, 0], tr[:, 1])
        robot.set_data([x[0] * 1e3], [x[1] * 1e3])
        goal.set_data([target[0] * 1e3], [target[1] * 1e3])

        w = slice(-600, None)
        lz.set_data(hist["t"][w], hist["z"][w])
        lzr.set_data(hist["t"][w], hist["zr"][w])
        lr.set_data(hist["t"][w], hist["ratio"][w])
        la.set_data(hist["t"][w], hist["amp"][w])
        for a in (ax_t, ax_c):
            a.relim()
            a.autoscale_view(scalex=True, scaley=(a is ax_t))
        ax_c.set_ylim(0, 1.2)

        err = np.linalg.norm(x[0:3] - target) * 1e3
        ax_xy.set_title(
            f"t={state['t']:5.1f} s   err={err:6.2f} mm   "
            f"real-time x{state['rt']:.2f}   "
            + ("LOST LOCK" if state["lost"] else "click to move target")
        )
        return trail, robot, goal, lz, lzr, lr, la

    from matplotlib.animation import FuncAnimation

    anim = FuncAnimation(fig, frame, interval=33, blit=False, cache_frame_data=False)
    fig._anim = anim  # keep a reference alive
    plt.show()


def _self_check(mpc, run):
    err = np.linalg.norm(run["r"] - run["ref"], axis=1)
    # Sample just before each target change, once the loop has had time to settle.
    settled = [err[int(w / mpc.ts) - 1] for w in (2.0, 5.0, 8.0, 11.0)]
    print(
        "settled err : " + ", ".join(f"{e*1e3:.3f} mm" for e in settled)
    )
    print(
        f"constraints : lock max {run['ratio'].max():.3f} (limit "
        f"{mpc.lim.lock_max}), current max {run['amps'].max():.2f} A "
        f"(limit {mpc.lim.i_max})"
    )
    acc = np.gradient(run["v"], mpc.ts, axis=0)
    jerk = np.gradient(acc, mpc.ts, axis=0)
    print(
        f"smoothness  : peak |a| {np.linalg.norm(acc,axis=1).max():.2f} m/s^2, "
        f"peak |j| {np.linalg.norm(jerk,axis=1).max():.1f} m/s^3"
    )
    print(
        f"solve time  : median {np.median(run['solve_s'])*1e3:.1f} ms, "
        f"p95 {np.percentile(run['solve_s'],95)*1e3:.1f} ms "
        f"(budget {mpc.ts*1e3:.0f} ms)"
    )

    assert run["lost_at"] is None, run["lost_at"]
    assert max(settled) < 1.5e-3, settled
    assert run["ratio"].max() < 1.0, run["ratio"].max()
    assert run["amps"].max() < mpc.lim.i_max * 1.01, run["amps"].max()
    assert np.median(run["solve_s"]) < mpc.ts, np.median(run["solve_s"])
    print("self-check PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="interactive real-time viewer")
    ap.add_argument("--grad", action="store_true", help="include the gradient force")
    ap.add_argument("--rate", type=float, default=20.0, help="control rate, Hz")
    ap.add_argument("--speed", type=float, default=1.0, help="--live time scaling")
    ap.add_argument("--out", default="spatial_mpc_sim.png")
    args = ap.parse_args()

    ts = 1.0 / args.rate
    # Tuned to run in real time: at 20 Hz this solves in ~44 ms against a 50 ms
    # budget. The horizon is 0.4 s, one SQP iteration per step, and the inputs
    # are free for two steps then revert to trim. One iteration is enough only
    # because the warm start is the LQR solution rather than trim.
    mpc = default_mpc(
        use_grad=args.grad,
        ts=ts,
        horizon=max(6, int(round(0.4 / ts))),
        blocks=2,
        iters=1,
        substeps=3,
    )

    if args.live:
        live(mpc, speed=args.speed)
        return

    mpc.design_lqr(HOVER)
    print(
        "open poles  : "
        + ", ".join(f"{v.real:+.2f}{v.imag:+.2f}j" for v in np.sort_complex(mpc.open_poles))
    )
    print(
        "lqr poles   : "
        + ", ".join(f"{v.real:+.2f}{v.imag:+.2f}j" for v in np.sort_complex(mpc.lqr_poles))
    )
    t0 = time.perf_counter()
    run = closed_loop(mpc, make_state(START, s=(0.0, 0.0, 1.0)), script, 13.0)
    print(f"ran 13.0 s of flight in {time.perf_counter()-t0:.1f} s wall")
    plot(run, mpc, args.out)
    print(f"wrote {args.out}")
    _self_check(mpc, run)


if __name__ == "__main__":
    main()
