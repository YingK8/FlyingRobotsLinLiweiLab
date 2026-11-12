#!/usr/bin/env python3
"""One command from a tilt-sweep take to a folder someone can read.

    uv run python controller/report.py results/tilt_sweep/<take>
    uv run python controller/report.py --all
    uv run python controller/report.py <take> --force     # redo the cached passes too

The passes this chains all existed already; what did not exist was an order to run them
in, a record of what the firmware was commanding at each instant, and one plot of the
whole take rather than one per frequency point. Everything lands in `<take>/report/`:

    overlay.mp4      both views, the disc mask tinted, the fitted ellipse, the mast pick
                     and the fused rotor axis drawn on every kept frame
    timeline.csv     one row per solved frame: position, tilt, azimuth, the lean pair,
                     and the commanded frequency and four duties in force at that frame
    timeline.png     those columns against time, every frame its own point, every command
                     change marked
    events.csv       the command changes themselves: each `label=` the firmware emitted
                     and every duty step over DUTY_STEP, with the time it happened
    telemetry.csv    every telemetry line the firmware sent -- frequency, four currents,
                     four duties. `events.csv` is derived from this and this is the record
    tilt_frames.csv / tilt_summary.csv / normal_angle.csv / tilt_*.png / normal_angle_f*.png
                     `control/tilt_report.py`, unchanged: the two-sensor per-point analysis

The two heavy passes stay in the take dir, not in `report/`, because they are inputs and
are ~10 MB each: `stereo_pose.csv` (`live_viz.from_recording` with
`pose/disc_pose.DiscStereoEstimator`) and `mast_pass.csv` (`disc_pose.mast_pass`). Both
are reused when present, so a second run only redraws. `--force` deletes them first.

READ THE CHANGES, NOT THE ABSOLUTE NUMBERS -- twice over. The angles are from the mast's
coils-off rest attitude, which is the only datum this rig has (`control/theory.md` 23.4),
and the POSITION is scaled by a rim radius this rotor does not have, so `x/y/z_mm` is a
direction and a trend and not a distance (`pose/disc_pose.py` header).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
from pathlib import Path

import numpy as np

from controller.control import tilt_report
from controller.pose import disc_pose, normal_video

ROOT = Path(__file__).resolve().parents[1]
SWEEPS = ROOT / "results" / "tilt_sweep"

#: One telemetry line: the commanded frequency and the four carrier duties.
TEL_RE = re.compile(r"freq=([\d.]+).*?duty\[%\]:\s*A=([\d.]+)\s+B=([\d.]+)\s+C=([\d.]+)\s+D=([\d.]+)")
CUR_RE = re.compile(r"I\[A\]:\s*A=(-?[\d.]+)\s+B=(-?[\d.]+)\s+C=(-?[\d.]+)\s+D=(-?[\d.]+)")
LABEL_RE = re.compile(r"label=([A-Z0-9_]+)")
#: Smallest duty move worth calling an event, in percent. The telemetry prints one
#: decimal and the schedule's steps are 20 % apart, so this only has to clear rounding.
DUTY_STEP = 0.5

TEL_COLS = ["t", "freq_hz"] + [f"duty_{c}" for c in "ABCD"] + [f"i_{c}" for c in "ABCD"]
TIMELINE_COLS = ["frame", "t", "t_s", "freq_hz", "t_rel_drop_s", "x_mm", "y_mm", "z_mm",
                 "tilt_deg", "azimuth_deg", "lean1_deg", "lean2_deg", "fused",
                 "fit_rms_px", "discrepancy_mm", "cmd_freq_hz"] + [f"duty_{c}" for c in "ABCD"]


def _read(path):
    with open(path) as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def _write(path, cols, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    return path


def _f(row, key):
    """``row[key]`` as a float, nan for missing or unparseable -- a gap, not a zero."""

    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


# ---- what the firmware was doing ------------------------------------------------------
def commands(log_path):
    """``(telemetry, events)`` from a run log.

    `telemetry` is one row per line the firmware sent; `events` is the subset that is a
    CHANGE -- every `label=` (the schedule's own point markers, `FREQ_020HZ`,
    `DROP_020HZ`, `DOWN_020HZ`) and every duty step over `DUTY_STEP` on any channel.
    Frequency is not evented: it ramps continuously, so every sample would qualify. It is
    a column in `telemetry.csv` and a trace on the plot instead.
    """

    from controller.control import sync

    entries, _ = sync.read_log(log_path)
    tel, events, prev = [], [], None
    for t, direction, text, _ in entries:
        if direction != "<-":
            continue
        lb = LABEL_RE.search(text)
        if lb:
            events.append({"t": t, "kind": "label", "detail": lb.group(1)})
        m = TEL_RE.search(text)
        if not m:
            continue
        cur = CUR_RE.search(text)
        row = {"t": t, "freq_hz": float(m.group(1))}
        for i, c in enumerate("ABCD"):
            row[f"duty_{c}"] = float(m.group(2 + i))
            row[f"i_{c}"] = float(cur.group(1 + i)) if cur else float("nan")
        for c in "ABCD":
            if prev is not None and abs(row[f"duty_{c}"] - prev[f"duty_{c}"]) >= DUTY_STEP:
                events.append({"t": t, "kind": "duty",
                               "detail": f"{c}: {prev[f'duty_{c}']:.0f} -> {row[f'duty_{c}']:.0f} %"})
        tel.append(row)
        prev = row
    events.sort(key=lambda e: e["t"])
    return tel, events


def _command_at(tel, times, t):
    """The telemetry row in force at time `t` -- sample and hold, since a command stands
    until the next one. ``{}`` before the first line."""

    i = bisect.bisect_right(times, t) - 1
    return tel[i] if i >= 0 else {}


# ---- the whole take, one row per frame ------------------------------------------------
def timeline(take_dir, out_dir, tel):
    """Join the stereo pose, the angles about the rest datum, and the command in force.

    Every frame the stereo pass solved gets a row, whether or not it sits inside a
    frequency point: outside one, `freq_hz` is -1 and the angles are the datum's own
    scatter, which is the check that the datum is a datum.
    """

    pose = {int(r["frame"]): r for r in _read(Path(take_dir) / "stereo_pose.csv")}
    # Missing when the take has no frequency points and so no rest datum (`build`); the
    # angle columns are then nan and the position and command columns still stand.
    angles = Path(out_dir) / "normal_angle.csv"
    ang = {int(r["frame"]): r for r in _read(angles)} if angles.exists() else {}
    times = [r["t"] for r in tel]
    t0 = min((float(r["t_capture"]) for r in pose.values()), default=0.0)
    rows = []
    for f in sorted(pose):
        p, a = pose[f], ang.get(f, {})
        t = float(p["t_capture"])
        c = _command_at(tel, times, t)
        row = {"frame": f, "t": t, "t_s": t - t0,
               "freq_hz": int(_f(a, "freq_hz")) if a else -1,
               "t_rel_drop_s": _f(a, "t_rel_drop_s"),
               "cmd_freq_hz": c.get("freq_hz", float("nan")),
               "fused": int(_f(a, "fused")) if a else -1}
        for k in ("x_mm", "y_mm", "z_mm", "fit_rms_px", "discrepancy_mm"):
            row[k] = _f(p, k)
        for k in ("tilt_deg", "azimuth_deg", "lean1_deg", "lean2_deg"):
            row[k] = _f(a, k)
        for ch in "ABCD":
            row[f"duty_{ch}"] = c.get(f"duty_{ch}", float("nan"))
        rows.append(row)
    return rows, t0


def _mark(ax, events, t0, label=False):
    """Every command change as a rule: the duty drop red, the spin-down blue, a point's
    start grey. Labelled on the top panel only, or the frequency labels stack."""

    for e in events:
        d = e["detail"]
        if e["kind"] == "duty" or d.startswith("DROP"):
            col, ls = "#d62728", "--"
        elif d.startswith("DOWN"):
            col, ls = "#1f77b4", ":"
        else:
            col, ls = "0.6", ":"
        ax.axvline(e["t"] - t0, color=col, ls=ls, lw=0.8, alpha=0.7, zorder=0)
        if label and e["kind"] == "label" and d.startswith("FREQ_"):
            ax.annotate(d[len("FREQ_"):].lower(), (e["t"] - t0, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(2, -2),
                        textcoords="offset points", va="top", fontsize=7, color="0.35")


def plot_timeline(rows, events, t0, out_dir, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t_s"] for r in rows])
    col = lambda k: np.array([r[k] for r in rows], float)
    fig, axes = plt.subplots(4, 1, figsize=(17, 11), sharex=True)
    for k, c in (("x_mm", "#1f77b4"), ("y_mm", "#2ca02c"), ("z_mm", "#d62728")):
        axes[0].scatter(t, col(k), s=1.5, color=c, linewidths=0, label=k)
    axes[0].set_ylabel("position (mm, nominal scale)")
    axes[0].legend(fontsize=8, markerscale=5, loc="upper right", ncol=3)

    fused = col("fused")
    for code, c, lab in ((0, "0.6", "disc only"), (2, "#7fb8e0", "disc in one mast plane"),
                         (1, "#1f77b4", "disc + mast, blended")):
        s = fused == code
        axes[1].scatter(t[s], col("tilt_deg")[s], s=1.5, color=c, linewidths=0, label=lab)
        axes[2].scatter(t[s], col("azimuth_deg")[s], s=1.5, color=c, linewidths=0)
    axes[1].set_ylabel("radial angle: tilt from rest (deg)")
    axes[1].legend(fontsize=8, markerscale=5, loc="upper right")
    axes[2].set_ylabel("azimuthal angle of the lean (deg)")
    axes[2].set_ylim(-180, 180)
    axes[2].set_yticks(range(-180, 181, 90))

    axes[3].scatter(t, col("cmd_freq_hz"), s=1.5, color="#7f7f7f", linewidths=0,
                    label="commanded freq")
    axes[3].set_ylabel("drive frequency (Hz)")
    twin = axes[3].twinx()
    for ch, c in zip("ABCD", ("#d62728", "#ff7f0e", "#2ca02c", "#9467bd")):
        twin.scatter(t, col(f"duty_{ch}"), s=1.5, color=c, linewidths=0, label=f"duty {ch}")
    twin.set_ylabel("carrier duty (%)")
    twin.set_ylim(-5, 105)
    twin.legend(fontsize=8, markerscale=5, loc="lower right", ncol=4)
    axes[3].legend(fontsize=8, markerscale=5, loc="upper right")
    axes[3].set_xlabel("time from the first solved frame (s)")

    for i, ax in enumerate(axes):
        _mark(ax, events, t0, label=(i == 0))
        ax.grid(alpha=0.25)
    fig.suptitle(
        f"{name} - whole take, every solved frame a point\n"
        "angles are from the mast's coils-off rest attitude (control/theory.md 23.4); "
        "position is scaled by a rim radius this rotor lacks, so read the change and not "
        "the value (pose/disc_pose.py)\n"
        "rules: grey = a point's label, red = duty step or DROP, blue = spin-down. "
        "Every one is in events.csv", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(Path(out_dir) / "timeline.png", dpi=130)
    plt.close(fig)


# ---- the pipeline ---------------------------------------------------------------------
README = """# {name}

Generated by `uv run python controller/report.py results/tilt_sweep/{name}`.

| file | what | made by |
|---|---|---|
| `overlay.mp4` | both views: disc mask tinted, fitted ellipse, mast pick, fused rotor axis | `pose/normal_video.py --seg` |
| `timeline.csv` | one row per solved frame: position, tilt, azimuth, lean, and the command in force | `controller/report.py` |
| `timeline.png` | those against time, every frame a point, every command change marked | `controller/report.py` |
| `events.csv` | the command changes: every `label=` and every duty step | `controller/report.py` |
| `telemetry.csv` | every telemetry line the firmware sent (freq, currents, duties) | `controller/report.py` |
| `tilt_frames.csv`, `tilt_summary.csv`, `normal_angle.csv` | the two-sensor per-point analysis | `control/tilt_report.py` |
| `tilt_panels.png`, `tilt_summary.png`, `normal_angle_f*.png` | its plots | `control/tilt_report.py` |

Inputs, left in the take dir because they are ~10 MB each and are reused:
`../stereo_pose.csv` (`live_viz.from_recording` + `pose/disc_pose.DiscStereoEstimator`),
`../mast_pass.csv` (`disc_pose.mast_pass`), `../sweep.log`, and the video under
`results/flights/`.

Angles are from the mast's coils-off rest attitude, the only datum this rig has
(`control/theory.md` 23.4). Position is scaled by the rim radius this rotor does not
have: it is a trend, not a distance (`pose/disc_pose.py` header).
"""


def build(take_dir, stride=7, force=False, video=True):
    take = Path(take_dir)
    flight = normal_video.flight_of(take)
    if not (flight / "meta.json").exists():
        print(f"{take.name}: no meta.json in {flight.name}; recording never finalised, skipped")
        return None
    out = take / "report"
    out.mkdir(exist_ok=True)

    pose_csv, mast_csv = take / "stereo_pose.csv", take / "mast_pass.csv"
    if force:
        for p in (pose_csv, mast_csv):
            p.unlink(missing_ok=True)
    normal_video.ensure_pose(take, flight)
    if not mast_csv.exists():
        print(f"{take.name}: mast pass -> {mast_csv.name}")
        tmp = mast_csv.with_name(mast_csv.name + ".part")   # see `normal_video.ensure_pose`
        disc_pose.mast_pass(flight, tmp)
        tmp.replace(mast_csv)

    tel, events = commands(take / "sweep.log")
    _write(out / "telemetry.csv", TEL_COLS, tel)
    _write(out / "events.csv", ["t", "kind", "detail"], events)
    print(f"{take.name}: {len(tel)} telemetry lines, {len(events)} command changes")

    try:
        tilt_report.report(take, out_dir=out, name=take.name)
    except SystemExit as e:
        # A duty-percent take has no `FREQ_*` points, so no coils-off rest windows and no
        # datum. Its command record and position are still worth having; its angles are
        # not measurable and are left nan rather than referred to a datum that is not one.
        print(f"{take.name}: no angle analysis -- {e}. Position and commands only")

    rows, t0 = timeline(take, out, tel)
    _write(out / "timeline.csv", TIMELINE_COLS, rows)
    plot_timeline(rows, events, t0, out, take.name)
    (out / "README.md").write_text(README.format(name=take.name))

    if video:
        normal_video.render(take, out_path=out / "overlay.mp4", stride=stride, seg=True)
    print(f"{take.name}: {out}")
    return out


def build_all(**kw):
    for take in sorted(SWEEPS.iterdir()):
        if take.is_dir() and re.match(r"\d{8}_\d{6}", take.name) and (take / "sweep.log").exists():
            build(take, **kw)


def _self_check():
    """The log parser on a synthetic log, and the join on synthetic rows."""

    import tempfile

    log = ("[100.000] <> t0\n"
           "[100.000] <- t=1 freq=1.0 | I[A]: A=0.10 B=-0.20 C=0.30 D=0.40 | "
           "duty[%]: A=100.0 B=100.0 C=100.0 D=100.0 | spread=1.0 bal=0 trip=0\n"
           "[100.500] <- label=FREQ_020HZ\n"
           "[101.000] <- t=2 freq=20.0 | I[A]: A=0.10 B=0.20 C=0.30 D=0.40 | "
           "duty[%]: A=100.0 B=100.0 C=100.0 D=100.0 | spread=1.0 bal=0 trip=0\n"
           "[102.000] <- t=3 freq=20.0 | I[A]: A=0.10 B=0.20 C=0.30 D=0.40 | "
           "duty[%]: A=30.0 B=100.0 C=100.0 D=100.0 | spread=1.0 bal=0 trip=0\n"
           "[103.000] <- label=DOWN_020HZ\n"
           "   this line has no clock and must be dropped\n")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sweep.log"
        p.write_text(log)
        tel, ev = commands(p)
    assert len(tel) == 3, tel
    assert tel[0]["i_B"] == -0.2 and tel[2]["duty_A"] == 30.0, tel
    kinds = [(e["kind"], e["detail"]) for e in ev]
    assert kinds == [("label", "FREQ_020HZ"), ("duty", "A: 100 -> 30 %"),
                     ("label", "DOWN_020HZ")], kinds
    # the ramp from 1 to 20 Hz is not an event: frequency is a trace, not a step
    assert not any(e["kind"] == "freq" for e in ev)
    times = [r["t"] for r in tel]
    assert _command_at(tel, times, 99.0) == {}, "no command before the first line"
    assert _command_at(tel, times, 101.9)["duty_A"] == 100.0    # holds until the step
    assert _command_at(tel, times, 102.5)["duty_A"] == 30.0     # and after it
    assert np.isnan(_f({"x": ""}, "x")) and np.isnan(_f({}, "x"))
    print("report: self-check passed (log parsed, ramp not evented, command held per frame)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", nargs="?", help="results/tilt_sweep/<take>")
    ap.add_argument("--all", action="store_true", help="every take that has a sweep.log")
    ap.add_argument("--force", action="store_true", help="redo the stereo and mast passes")
    ap.add_argument("--stride", type=int, default=7, help="video: keep every Nth frame (7 = real time)")
    ap.add_argument("--no-video", action="store_true", help="plots and CSVs only")
    a = ap.parse_args()
    kw = dict(stride=a.stride, force=a.force, video=not a.no_video)
    if a.all:
        build_all(**kw)
    elif a.take:
        build(a.take, **kw)
    else:
        _self_check()
