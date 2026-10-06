#!/usr/bin/env python3
"""Rotor tilt through the 30% duty drop, per frequency point, from two independent sensors.

Inputs, all in one `results/tilt_sweep/<take>/` folder:

    sweep.log         the firmware's labels and telemetry; gives each point's start, its
                      duty-drop instant (`duty[%]: A` -> 30) and its end
    stereo_pose.csv   `live_viz.from_recording` with `disc_pose.DiscStereoEstimator`:
                      the rotor normal, world frame, from the disc ellipse in both views
    mast_pass.csv     `disc_pose.mast_pass`: the mast's world direction, triangulated
                      from the hub-to-bead line in both views

TILT IS MEASURED FROM THE REST ATTITUDE, AND THAT IS THE ONLY DEFENSIBLE DATUM
------------------------------------------------------------------------------
"Tilt change from what?" was the right objection to the first version of this analysis.
The rig's world frame is camera A -- `stereo_rig.json` says so, and `prime_zero`
measured 38 deg between world +z and a hovering rotor -- so nothing in that frame is
"up", and a raw `theta_deg` is an angle from wherever camera A points.

What the recording does contain is the robot at rest: after every point the carrier is
cut and it sits for 10 s with nothing driving it. The mast's direction over those windows
is the datum, and every tilt here is the angle from it. It is a physical zero (the
attitude the robot returns to with no field), it is measured by the same sensor in the
same frame as the data, and it is exactly what `zeroing.Zero` does for a flight -- just
taken from the mast, which does not strobe, rather than from a disc of frozen blades.

TWO SENSORS, ONE AXIS
---------------------
The rotor normal and the mast direction are the same physical axis measured two ways:
the disc from its foreshortening, the mast from its projection. They share no failure
mode -- the disc is worst when the blades are frozen, the mast when a blade outscores it
-- so their agreement per frame is the validity measure, and a spectral peak that both
carry is a real wobble while one that only one carries is not. The disc normal's sign is
resolved against the mast (that is the "rod as prior"): the ellipse alone cannot tell
rotor-up from rotor-down, and the pipeline's `orient` reference is camera A's +z.

    uv run python controller/control/tilt_report.py results/tilt_sweep/<take>
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

from controller.pose import body_angle, disc_pose

#: Coils-off rest window relative to a point's DOWN label: 4 s of down-ramp, then the
#: carrier is cut for 10 s. Trimmed at both ends so the spin-down and the next ramp's
#: first frames stay out.
REST_FROM_S, REST_TO_S = 5.0, 13.5
#: `perp` is diagnostic only since `find_mast` moved to a reach-based score (the frames
#: the datum comes from have no disc to be perpendicular to). Off.
MIN_PERP = -1.0
SMOOTH_S = 0.25
FPS = 208.0
# The fusion constants and `disc_quality` live with the estimator now (`disc_pose`), so
# the live loop (`tilt_servo`) and this report blend the two sensors with one function.
SIGMA_DISC_DEG, SIGMA_MAST_DEG = disc_pose.SIGMA_DISC_DEG, disc_pose.SIGMA_MAST_DEG
AGREE_MAX_DEG = disc_pose.AGREE_MAX_DEG
RATIO_EDGE_MIN, RATIO_EDGE_OK = disc_pose.RATIO_EDGE_MIN, disc_pose.RATIO_EDGE_OK
disc_quality = disc_pose.disc_quality


# ---- loading --------------------------------------------------------------------------
def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def load(take_dir):
    """``(pts, frames)``: the timeline, and one merged row per frame that both passes solved."""

    take_dir = Path(take_dir)
    pts = body_angle.timeline(take_dir / "sweep.log")
    st = {int(r["frame"]): r for r in _read(take_dir / "stereo_pose.csv")}
    ms = {int(r["frame"]): r for r in _read(take_dir / "mast_pass.csv")}
    out = []
    for f in sorted(set(st) & set(ms)):
        a, b = st[f], ms[f]
        n = np.array([float(a["nx"]), float(a["ny"]), float(a["nz"])])
        m = np.array([float(b["mast_x"]), float(b["mast_y"]), float(b["mast_z"])])
        perp = min(float(b.get("A_perp", "nan")), float(b.get("B_perp", "nan")))
        # the thinner view's minor/major; nan when a view had no disc
        ratios = [float(b.get(f"{tg}_disc_minor", "nan")) / max(float(b.get(f"{tg}_disc_major", "nan")), 1e-9)
                  for tg in ("A", "B")]
        ratio_min = min(ratios) if all(np.isfinite(ratios)) else float("nan")
        # each view's mast plane (`disc_pose.mast_plane`), for the frames with one view only
        planes = [np.array([float(b.get(f"{tg}_p{k}", "nan")) for k in "xyz"]) for tg in ("A", "B")]
        planes = [p for p in planes if np.isfinite(p).all()]
        out.append({"frame": f, "t": float(a["t_capture"]), "n": n, "mast": m,
                    "perp": perp, "disc_mm": float(a["discrepancy_mm"]),
                    "rms_px": float(a["fit_rms_px"]), "ratio_min": ratio_min,
                    "planes": planes})
    return pts, out



# ---- the datum --------------------------------------------------------------------------
def rest_windows(pts):
    return [(p["t_end"] + REST_FROM_S, p["t_end"] + REST_TO_S) for p in pts]


def datum(frames, pts):
    """Unit "up" from the mast over every rest window. ``(up, n_used, spread_deg)``.

    Spread is the RMS angle of the individual directions about the mean: a number that
    says how still the robot actually was. Under ~2 deg is a datum; over ~5 says the
    robot was swinging on its wire and this needs a longer window or a different one.
    """

    wins = rest_windows(pts)
    vs = [r["mast"] for r in frames
          if np.all(np.isfinite(r["mast"])) and r["perp"] >= MIN_PERP
          and any(a <= r["t"] <= b for a, b in wins)]
    if len(vs) < 50:
        raise SystemExit(f"only {len(vs)} rest-window mast directions; no datum")
    V = np.array(vs, dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)   # unit rows, or arccos reads noise as tilt
    up = V.mean(0)
    up /= np.linalg.norm(up)
    spread = float(np.sqrt(np.mean(np.degrees(np.arccos(np.clip(V @ up, -1, 1))) ** 2)))
    return up, len(vs), spread


def _angle(v, up):
    return float(np.degrees(np.arccos(np.clip(float(v @ up) / np.linalg.norm(v), -1, 1))))


def rest_basis(up):
    """``(e1, e2)`` spanning the plane perpendicular to the datum; see theory.md 23.4.

    ``e1`` is world x (camera A's x) projected onto that plane, ``e2 = up x e1``. Any pair
    would do -- the choice only names the two lean axes -- and camera A's x is the one
    direction in this rig that is fixed and physically identifiable.
    """

    e1 = np.array([1.0, 0.0, 0.0]) - up[0] * up
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def lean_table(frames, pts, up, min_tilt_deg=0.5):
    """Disc normal as a tilt vector about the datum, one row per solved frame.

    ``(lean1, lean2)`` are the tilt magnitude times the unit lean direction's components
    along `rest_basis` -- so ``hypot(lean1, lean2) == tilt`` exactly, and each is the lean
    about one rest axis. Azimuth is ``atan2(lean2, lean1)`` and undefined below
    ``min_tilt_deg`` (as `estimator._angles_from_normal`), where it is nan rather than noise.

    A frame outside every frequency point -- a ramp, or a coils-off rest window -- keeps
    its row with ``freq_hz`` -1 and ``t_rel_drop_s`` nan. Its tilt is then the datum's own
    scatter, which is the check that the datum is a datum, and the per-point plots select
    on ``freq_hz`` so nothing downstream sees them.

    The normal is the disc's blended with the mast's at `SIGMA_DISC_DEG` / `SIGMA_MAST_DEG`
    when the frame has a mast (``fused`` = 1), the disc's alone when it does not, and
    nothing when the two are more than `AGREE_MAX_DEG` apart. `pose/theory.md` 20.4.
    """

    e1, e2 = rest_basis(up)
    rows = []
    for r in frames:
        p = next((p for p in pts if p["t_start"] <= r["t"] <= p["t_end"]), None)
        ok_m = np.all(np.isfinite(r["mast"])) and r["perp"] >= MIN_PERP
        got = disc_pose.fused_axis(r["n"], mast=r["mast"] if ok_m else None,
                                   planes=r.get("planes", ()),
                                   ratio_min=r.get("ratio_min"), ref=up)
        if got is None:
            continue                            # disc and rod disagree; not a data point
        n, fused = got
        if float(n @ up) < 0:
            n = -n
        tilt = _angle(n, up)
        lat = n - float(n @ up) * up
        norm = np.linalg.norm(lat)
        u = lat / norm if norm > 1e-9 else np.zeros(3)
        l1, l2 = tilt * float(u @ e1), tilt * float(u @ e2)
        az = float(np.degrees(np.arctan2(l2, l1))) if tilt >= min_tilt_deg else float("nan")
        rows.append({"frame": r["frame"], "t": r["t"],
                     "freq_hz": p["freq"] if p else -1,
                     "t_rel_drop_s": r["t"] - p["t_drop"] if p else float("nan"), "lean1_deg": l1, "lean2_deg": l2,
                     "tilt_deg": tilt, "azimuth_deg": az, "fused": int(fused)})
    return rows


# ---- per-frame tilt -----------------------------------------------------------------------
def tilt_table(frames, pts, up):
    """One row per frame with both sensors' tilt from the datum and their agreement."""

    rows = []
    for r in frames:
        n = r["n"] / np.linalg.norm(r["n"])
        ok_m = np.all(np.isfinite(r["mast"])) and r["perp"] >= MIN_PERP
        m = r["mast"] / np.linalg.norm(r["mast"]) if ok_m else None
        # Rod as prior: the disc's sign is whichever agrees with the mast, else with up.
        ref = m if m is not None else up
        if float(n @ ref) < 0:
            n = -n
        hz, rel = -1, float("nan")
        for p in pts:
            if p["t_start"] <= r["t"] <= p["t_end"]:
                hz, rel = p["freq"], r["t"] - p["t_drop"]
                break
        rows.append({
            "frame": r["frame"], "t": r["t"], "freq_hz": hz, "t_rel_drop_s": rel,
            "tilt_disc_deg": _angle(n, up),
            "tilt_mast_deg": _angle(m, up) if m is not None else float("nan"),
            "agree_deg": _angle(n, m) if m is not None else float("nan"),
            "disc_mm": r["disc_mm"], "rms_px": r["rms_px"], "perp": r["perp"],
        })
    return rows


# ---- per-point summary ----------------------------------------------------------------------
def summary(rows, pts, pre_s=1.0, settle_s=2.0):
    n_sm = int(round(SMOOTH_S * FPS))
    out = []
    for p in pts:
        r = sorted((x for x in rows if x["freq_hz"] == p["freq"]), key=lambda x: x["t"])
        if len(r) < 4 * n_sm:
            continue
        t = np.array([x["t_rel_drop_s"] for x in r])
        rec = {"freq": p["freq"], "n": len(r)}
        for key, name in (("tilt_disc_deg", "disc"), ("tilt_mast_deg", "mast")):
            y = np.array([x[key] for x in r])
            ok = np.isfinite(y)
            pre = y[ok & (t >= -pre_s) & (t < 0)]
            post = y[ok & (t > settle_s) & (t <= settle_s + 2.0)]
            rec[f"{name}_pre"] = float(np.median(pre)) if len(pre) else float("nan")
            rec[f"{name}_post"] = float(np.median(post)) if len(post) else float("nan")
            rec[f"{name}_scatter"] = float(np.std(pre)) if len(pre) > 5 else float("nan")
            # wobble after the drop, same estimator as body_angle.oscillation
            w = ok & (t >= 0.2) & (t <= 5.0)
            if w.sum() > 200:
                yy = body_angle._median_filter(y[w], 11)
                yy = yy - yy.mean()
                fs = 1.0 / float(np.median(np.diff(t[w])))
                Y = np.abs(np.fft.rfft(yy * np.hanning(len(yy))))
                fq = np.fft.rfftfreq(len(yy), 1.0 / fs)
                sel = (fq > 0.3) & (fq < 20)
                k = int(np.argmax(Y[sel]))
                rec[f"{name}_wobble_hz"] = float(fq[sel][k])
                rec[f"{name}_wobble_amp"] = float(2 * Y[sel][k] / len(yy))
            else:
                rec[f"{name}_wobble_hz"] = rec[f"{name}_wobble_amp"] = float("nan")
        ag = np.array([x["agree_deg"] for x in r])
        rec["agree_med"] = float(np.nanmedian(ag)) if np.isfinite(ag).any() else float("nan")
        rec["mast_frac"] = float(np.isfinite(ag).mean())
        out.append(rec)
    return out


# ---- output ----------------------------------------------------------------------------------
def report(take_dir, plot=True, out_dir=None, name=None):
    """``out_dir`` defaults to the take dir; `controller/report.py` points it at
    `<take>/report/` so one folder holds everything a run produced."""

    take_dir = Path(take_dir)
    out_dir = Path(out_dir or take_dir)
    name = name or take_dir.name
    pts, frames = load(take_dir)
    up, n_up, spread = datum(frames, pts)
    print(f"datum: mast over {n_up} rest frames, up = [{up[0]:+.3f} {up[1]:+.3f} {up[2]:+.3f}] "
          f"(world = camera A), spread {spread:.2f} deg")
    rows = tilt_table(frames, pts, up)
    cols = ["frame", "t", "freq_hz", "t_rel_drop_s", "tilt_disc_deg", "tilt_mast_deg",
            "agree_deg", "disc_mm", "rms_px", "perp"]
    with open(out_dir / "tilt_frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    sm = summary(rows, pts)
    with open(out_dir / "tilt_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(sm[0].keys()) if sm else ["freq"])
        w.writeheader()
        w.writerows(sm)
    print(f"\n{'Hz':>4} {'n':>5} | {'disc pre':>8} {'post':>6} {'sd':>5} | {'mast pre':>8} "
          f"{'post':>6} {'sd':>5} | {'agree':>5} {'mast%':>5} | {'wobble disc/mast Hz':>19}")
    for s in sm:
        print(f"{s['freq']:4d} {s['n']:5d} | {s['disc_pre']:8.1f} {s['disc_post']:6.1f} "
              f"{s['disc_scatter']:5.1f} | {s['mast_pre']:8.1f} {s['mast_post']:6.1f} "
              f"{s['mast_scatter']:5.1f} | {s['agree_med']:5.1f} {100*s['mast_frac']:5.0f} | "
              f"{s['disc_wobble_hz']:6.2f} / {s['mast_wobble_hz']:5.2f}")
    lean = lean_table(frames, pts, up)
    lcols = ["frame", "t", "freq_hz", "t_rel_drop_s", "lean1_deg", "lean2_deg", "tilt_deg",
             "azimuth_deg", "fused"]
    with open(out_dir / "normal_angle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, lcols)
        w.writeheader()
        w.writerows(lean)
    if plot:
        _plot(rows, pts, sm, out_dir, up, spread, name)
        plot_normal(lean, pts, out_dir, spread, name)
    return rows, sm


def _plot(rows, pts, sm, out_dir, up, spread, name=None):
    name = name or Path(out_dir).name
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sm = int(round(SMOOTH_S * FPS))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), pts):
        r = sorted((x for x in rows if x["freq_hz"] == p["freq"]), key=lambda x: x["t"])
        if not r:
            ax.set_title(f"{p['freq']} Hz (no data)")
            continue
        t = np.array([x["t_rel_drop_s"] for x in r])
        m = (t >= -2) & (t <= 5)
        for key, col, lab in (("tilt_disc_deg", "#1f77b4", "rotor disc (stereo)"),
                              ("tilt_mast_deg", "#d62728", "mast (stereo)")):
            y = np.array([x[key] for x in r])
            ok = np.isfinite(y)
            ax.scatter(t[m & ok], y[m & ok], s=1.5, color=col, alpha=0.35, linewidths=0)
            if (m & ok).sum() > n_sm:
                ys = body_angle._median_filter(y[ok], n_sm)
                tt = t[ok]
                mm = (tt >= -2) & (tt <= 5)
                ax.plot(tt[mm], ys[mm], color=col, lw=1.8, label=lab)
        ax.axvline(0, color="k", ls="--", lw=1.3)
        ax.axvspan(-1, 0, color="0.88", zorder=0)
        ax.set_title(f"{p['freq']} Hz", fontsize=11)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[1]:
        ax.set_xlabel("time from 30% duty drop on ch0 (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("tilt from rest attitude (deg)")
    fig.suptitle(f"{name} - rotor tilt through the channel-0 30% duty drop\n"
                 f"absolute, from the rest attitude (mast over coils-off windows, spread "
                 f"{spread:.1f} deg); dashed = drop; faint = per frame, bold = 0.25 s median",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "tilt_panels.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    f = [s["freq"] for s in sm]
    for key, col, lab in (("disc", "#1f77b4", "rotor disc"), ("mast", "#d62728", "mast")):
        axes[0].plot(f, [s[f"{key}_pre"] for s in sm], "o--", color=col, alpha=0.5,
                     label=f"{lab}, before drop")
        axes[0].plot(f, [s[f"{key}_post"] for s in sm], "o-", color=col, label=f"{lab}, after")
        axes[1].plot(f, [s[f"{key}_post"] - s[f"{key}_pre"] for s in sm], "o-", color=col,
                     label=lab)
        axes[2].plot(f, [s[f"{key}_wobble_hz"] for s in sm], "o-", color=col, label=lab)
        axes[3].plot(f, [s[f"{key}_scatter"] for s in sm], "o-", color=col, label=lab)
    axes[3].plot(f, [s["agree_med"] for s in sm], "s-", color="k", label="disc vs mast")
    axes[0].set_ylabel("tilt from rest (deg)")
    axes[1].set_ylabel("tilt change at the drop (deg)")
    axes[2].set_ylabel("post-drop wobble (Hz)")
    axes[2].annotate("believe it only where\nboth sensors agree", (0.03, 0.86),
                     xycoords="axes fraction", fontsize=8)
    axes[3].set_ylabel("pre-drop scatter / disagreement (deg)")
    for ax, ttl in zip(axes, ("Attitude", "Effect of the drop", "Wobble", "Quality")):
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("drive frequency (Hz)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"{name} - channel-0 30% duty drop against drive frequency, two sensors",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_dir / "tilt_summary.png", dpi=130)
    plt.close(fig)


def plot_normal(lean, pts, out_dir, spread, name=None):
    """One figure per frequency: lean about each rest axis, tilt, azimuth vs time from the
    drop, every solved frame as a point (no lines -- a line through a strobing rotor
    invents structure), whole point from ramp start to spin-down end."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = name or Path(out_dir).name
    panels = (("lean1_deg", "lean about rest e1 (deg)"), ("lean2_deg", "lean about rest e2 (deg)"),
              ("tilt_deg", "tilt from rest (deg)"), ("azimuth_deg", "lean azimuth (deg)"))
    for p in pts:
        r = [x for x in lean if x["freq_hz"] == p["freq"]]
        if not r:
            continue
        t = np.array([x["t_rel_drop_s"] for x in r])
        fused = np.array([x["fused"] for x in r], int)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for ax, (key, lab) in zip(axes, panels):
            y = np.array([x[key] for x in r], float)
            for code, col, lab_f in ((0, "0.6", "disc only (no mast this frame)"),
                                     (2, "#7fb8e0", "disc in one view's mast plane"),
                                     (1, "#1f77b4", "disc + mast, blended")):
                s = fused == code
                ax.scatter(t[s], y[s], s=2, color=col, linewidths=0, label=lab_f)
            ax.axvline(0, color="k", ls="--", lw=1.2)
            ax.set_ylabel(lab)
            ax.grid(alpha=0.25)
        axes[3].set_ylim(-180, 180)
        axes[3].set_yticks(range(-180, 181, 90))
        axes[0].annotate("ch0 -> 30% duty", (0, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(4, -4), textcoords="offset points", va="top", fontsize=9)
        axes[3].set_xlabel("time from 30% duty drop on ch0 (s)")
        fig.suptitle(f"{name} - rotor normal about the rest attitude, {p['freq']} Hz point\n"
                     f"e1 = camera-A x projected onto the rest plane, e2 = up x e1; datum "
                     f"spread {spread:.1f} deg\nevery solved frame, no smoothing; frames with "
                     f"disc and mast > {AGREE_MAX_DEG:.0f} deg apart left out", fontsize=11)
        axes[0].legend(fontsize=8, loc="upper right", markerscale=4)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out_dir / f"normal_angle_f{p['freq']:03d}hz.png", dpi=130)
        plt.close(fig)


def _self_check():
    """Synthetic frames: the datum must come out as the rest direction and the tilts as
    the angles that were put in, with the disc sign flipped back by the mast."""

    up_true = np.array([0.1, -0.98, 0.15])
    up_true /= np.linalg.norm(up_true)
    pts = [{"freq": 100, "t_start": 0.0, "t_drop": 10.0, "t_end": 15.0}]
    frames = []
    # rest: mast exactly up with tiny noise
    rng = np.random.default_rng(0)
    for i in range(300):
        t = 15.0 + REST_FROM_S + i * 0.02
        frames.append({"frame": i, "t": t, "n": -up_true, "mast": up_true + rng.normal(0, 0.005, 3),
                       "perp": 0.9, "disc_mm": 1.0, "rms_px": 1.0})
    # hold: tilted 12 deg, disc reported with the WRONG sign
    ax = np.cross(up_true, [1, 0, 0]); ax /= np.linalg.norm(ax)
    c, s = np.cos(np.radians(12)), np.sin(np.radians(12))
    v = up_true * c + np.cross(ax, up_true) * s
    for i in range(300):
        frames.append({"frame": 1000 + i, "t": 11.0 + i * 0.01, "n": -v, "mast": v,
                       "perp": 0.9, "disc_mm": 1.0, "rms_px": 1.0})
    up, n_used, spread = datum(frames, pts)
    assert n_used == 300 and spread < 1.0, (n_used, spread)
    assert np.degrees(np.arccos(up @ up_true)) < 0.5
    rows = tilt_table(frames, pts, up)
    hold = [r for r in rows if r["freq_hz"] == 100]
    assert all(abs(r["tilt_disc_deg"] - 12) < 0.6 for r in hold), "disc sign not resolved"
    assert all(abs(r["tilt_mast_deg"] - 12) < 0.6 for r in hold)
    assert all(r["agree_deg"] < 0.1 for r in hold)
    lean = lean_table(frames, pts, up)
    hold = [r for r in lean if r["freq_hz"] == 100]
    assert len(lean) == 600 and len(hold) == 300, (len(lean), len(hold))
    assert all(r["freq_hz"] == -1 and math.isnan(r["t_rel_drop_s"])
               for r in lean if r not in hold), "rest frames carry no point"
    assert all(r["fused"] for r in hold), "mast present on every hold frame; all fused"
    # a frame whose disc is 30 deg off the mast is dropped, not blended
    bad = dict(frames[-1], frame=9999, n=np.cross(v, [0, 0, 1]))
    assert lean_table([bad], pts, up) == [], "disagreeing frame must be left out"
    # edge-on: a disc 10 deg off the mast is neither dropped nor believed; the mast carries it
    c10, s10 = np.cos(np.radians(10)), np.sin(np.radians(10))
    off = v * c10 + np.cross(ax, v) * s10
    edge = dict(frames[-1], frame=9998, n=off, ratio_min=0.10)
    (row,) = lean_table([edge], pts, up)
    assert abs(row["tilt_deg"] - 12) < 1.0, row["tilt_deg"]
    assert disc_quality(0.4) == 1.0 and disc_quality(float("nan")) == 1.0
    # one view only: no mast direction, but a plane the axis lies in. A disc 10 deg out
    # of that plane is projected back into it and lands on the true axis.
    p = np.cross(v, [1.0, 0.0, 0.0]); p /= np.linalg.norm(p)          # a plane containing v
    n_out = v * c10 + p * s10                                          # 10 deg out of it
    one = dict(frames[-1], frame=9997, n=n_out, mast=np.full(3, np.nan), planes=[p])
    (row,) = lean_table([one], pts, up)
    assert row["fused"] == 2 and abs(row["tilt_deg"] - 12) < 0.6, row
    mag = [math.hypot(r["lean1_deg"], r["lean2_deg"]) for r in hold]
    assert all(abs(m - 12) < 0.6 and abs(m - r["tilt_deg"]) < 1e-9 for m, r in zip(mag, hold))
    az = np.array([r["azimuth_deg"] for r in hold])
    assert np.isfinite(az).all() and az.std() < 0.5, az.std()
    e1, e2 = rest_basis(up)
    assert abs(e1 @ up) < 1e-9 and abs(e2 @ up) < 1e-9 and abs(e1 @ e2) < 1e-9
    print("tilt_report: self-check passed (datum recovered, 12 deg tilt, disc sign from mast, "
          "lean vector)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report(sys.argv[1])
    else:
        _self_check()
