#!/usr/bin/env python3
"""Every processed take as ONE labelled CSV, plus the campaign tables, in one zip.

    uv run python controller/control/package_takes.py                          # self-check
    uv run python controller/control/package_takes.py --out results/alignment_rate/tilt_flight_processed_csv.zip
    uv run python controller/control/package_takes.py --designs 1 --out ...    # one campaign

One file per good take, named ``tilt_flight_design_<1|b>_<freq>hz_<repeat>.csv``. The repeat is
the campaign index's OWN repeat number, gaps and all (an aborted repeat is re-recorded under
the same number, so e.g. design 1 at 10 Hz has no repeat 19), which means every file traces
back to exactly one row of its `index.csv`.

Each file merges the four outputs `disc_axis.solve` writes for a take, on the frame number:

    axis.csv       -> nx ny nz agree_deg              the fused 3-D axis
    axis_minor.csv -> minor_* x_mm y_mm z_mm radius_mm the minor-axis estimate, disc centre, radius
    tilt_A.csv     -> a_*                             camera A's ellipse fit
    tilt_B.csv     -> b_*                             camera B's ellipse fit

and adds what no single output carries: time from the cut (`t_from_cut_s`, from the take's
`KILL_` label, `alignment_rate.timeline`), the schedule phase at every frame (from the
`sweep.log` labels), and `rotor_stopped` (`alignment_rate.spin_stop`) -- past a stop the "disc"
is still blades and the axis columns do not measure anything (`theory.md` 24.13).

Nothing on disk is moved or deleted. The zip is written directly; a take folder that cannot
be reached stops the run (`alignment_rate._take_or_die`) rather than leaving a quiet gap.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AR = ROOT / "results" / "alignment_rate"

#: (design tag, campaign root, per-repeat rate table). Design 1's trials were re-analysed into
#: the comparison folder so its tracked report stayed untouched (`compare_1_vs_B/README.md`).
CAMPAIGNS = (
    ("1", AR / "20260909_205843", AR / "compare_1_vs_B" / "design1_campaign" / "campaign_trials.csv"),
    ("b", AR / "20260910_droneB", AR / "20260910_droneB" / "report" / "campaign_trials.csv"),
)

#: sweep.log label prefix -> the phase name written on every frame from that label on.
PHASES = {"FREQ": "ramp", "SETTLE": "settle", "HOLD": "hold", "KILL": "drop",
          "DOWN": "spindown", "TILT": "off"}

SOURCES = (("axis.csv", ""), ("axis_minor.csv", "minor_"), ("tilt_A.csv", "a_"),
           ("tilt_B.csv", "b_"))
#: Columns of `axis_minor.csv` that are not an axis estimate keep their own names.
UNPREFIXED = {"x_mm", "y_mm", "z_mm", "radius_mm"}


def csv_name(design, freq_hz, repeat):
    return f"tilt_flight_design_{design}_{int(round(float(freq_hz)))}hz_{int(repeat)}.csv"


def label_times(log_path):
    """Sorted ``[(host_time, phase)]`` from a sweep.log's ``label=`` lines."""

    out = []
    for m in re.finditer(r"^\[(\d+\.\d+)\].*?label=([A-Z]+)",
                         Path(log_path).read_text(errors="replace"), re.M):
        phase = PHASES.get(m.group(2))
        if phase:
            out.append((float(m.group(1)), phase))
    return sorted(out)


def _phase_at(t, labels):
    phase = "boot"
    for tl, ph in labels:
        if tl > t:
            break
        phase = ph
    return phase


def merge_take(take, log_path, t_kill, spin_stop_s):
    """``(header, rows)`` for one take: the four outputs joined on ``frame``."""

    tables = {}
    header = ["frame", "t_s", "t_from_cut_s", "phase", "rotor_stopped"]
    for fname, prefix in SOURCES:
        path = Path(take) / fname
        if not path.exists():
            continue
        rows = list(csv.DictReader(open(path)))
        cols = [c for c in (rows[0].keys() if rows else []) if c not in ("frame", "t")]
        named = [(c, c if (prefix == "minor_" and c in UNPREFIXED) else prefix + c)
                 for c in cols]
        header += [n for _, n in named]
        tables[fname] = ({int(r["frame"]): r for r in rows}, named)
    labels = label_times(log_path) if log_path and Path(log_path).exists() else []
    frames = sorted(set().union(*(set(tb[0]) for tb in tables.values()))) if tables else []
    out = []
    for fr in frames:
        row = {"frame": fr}
        t = None
        for fname, _ in SOURCES:          # fused time first, then each fallback in order
            if fname in tables and fr in tables[fname][0] and t is None:
                t = float(tables[fname][0][fr]["t"])
        row["t_s"] = f"{t:.6f}" if t is not None else ""
        rel = (t - t_kill) if (t is not None and t_kill is not None) else None
        row["t_from_cut_s"] = f"{rel:.4f}" if rel is not None else ""
        row["phase"] = _phase_at(t, labels) if t is not None else ""
        row["rotor_stopped"] = (int(rel >= spin_stop_s) if rel is not None
                                and spin_stop_s is not None and math.isfinite(spin_stop_s) else 0)
        for fname, (by_frame, named) in tables.items():
            src = by_frame.get(fr, {})
            for c, n in named:
                row[n] = src.get(c, "")
        out.append(row)
    return header, out


def _by_take(path, key="take"):
    if not Path(path).exists():
        return {}
    return {r[key]: r for r in csv.DictReader(open(path))}


def build(out_zip, designs=("1", "b")):
    """Write the zip. Returns the manifest rows."""

    from controller.control import alignment_rate as ar

    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for design, root, trials_path in CAMPAIGNS:
            if design not in designs:
                continue
            trials = _by_take(trials_path)
            settle = _by_take(root / "report" / "settling.csv")
            for idx in sorted(root.glob("*hz/index.csv")):
                for r in csv.DictReader(open(idx)):
                    if r["outcome"] != "ok":
                        continue
                    take = ar._take_or_die(r["flight"])
                    if not (take / "axis.csv").exists():
                        print(f"  {take.name}: not solved -- skipped (no axis.csv)")
                        continue
                    pts = ar.timeline(r["log"]) if Path(r["log"]).exists() else []
                    kills = [k for p in pts for k in p["kills"]]
                    t_kill = kills[0] if kills else None
                    stop = ar.spin_stop(take, t_kill) if t_kill is not None else float("inf")
                    header, rows = merge_take(take, r["log"], t_kill, stop)
                    name = csv_name(design, r["freq_hz"], r["repeat"])
                    buf = io.StringIO()
                    w = csv.DictWriter(buf, fieldnames=header, lineterminator="\n")
                    w.writeheader()
                    w.writerows(rows)
                    zf.writestr(f"takes/design_{design}/{name}", buf.getvalue())
                    tr, se = trials.get(take.name, {}), settle.get(take.name, {})
                    # What a blank cell means depends on the take: design 1's per-camera fits
                    # and frames.csv are not on disk at all (its video was archived into
                    # results/flights/Archive.zip and the take folders stripped), while a blank
                    # in a design-B file means that output had no solution for that frame.
                    per_view = (take / "tilt_A.csv").exists()
                    video = (take / "A" / "A.mp4").exists()
                    manifest.append({
                        "file": f"takes/design_{design}/{name}", "design": design,
                        "freq_hz": float(r["freq_hz"]), "repeat": int(r["repeat"]),
                        "take": take.name, "video_dir": str(take), "log": r["log"],
                        "t_kill_host_s": f"{t_kill:.4f}" if t_kill is not None else "",
                        "per_view_columns": int(per_view), "video_present": int(video),
                        "rotor_stop_s": f"{stop:.2f}" if math.isfinite(stop) else "",
                        "n_frames": len(rows),
                        "rate_relu_deg_s": tr.get("rate_relu_deg_s", ""),
                        "swing_amp_deg": tr.get("amp_deg", ""),
                        "modal_repeat": tr.get("modal", ""),
                        "settle_swing_deg": se.get("swing_deg", ""),
                        "settle_10pct_s": se.get("settle_10pct_s", ""),
                        "settle_note": se.get("note", ""),
                    })
                    print(f"  {name}  {len(rows)} frames")
            # the campaign's own result tables, under their design
            for src, dst in ((root / "report" / "campaign.csv", "campaign.csv"),
                             (trials_path, "campaign_trials.csv"),
                             (root / "report" / "settling.csv", "settling.csv"),
                             (root / "report" / "settling_by_freq.csv", "settling_by_freq.csv")):
                if Path(src).exists():
                    zf.write(src, f"summary/design_{design}_{dst}")
        for name in ("alignment_rate_stats.csv", "rate_and_settling_stats.csv",
                     "traces_1_vs_B.csv"):
            src = AR / "compare_1_vs_B" / name
            if src.exists():
                zf.write(src, f"summary/compare_{name}")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(manifest[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(sorted(manifest, key=lambda m: (m["design"], m["freq_hz"], m["repeat"])))
        zf.writestr("manifest.csv", buf.getvalue())
        zf.writestr("README.md", _readme(manifest))
    print(f"-> {out_zip}: {len(manifest)} takes")
    return manifest


def _readme(manifest):
    n = {d: sum(1 for m in manifest if m["design"] == d) for d in ("1", "b")}
    pv = {d: sum(1 for m in manifest if m["design"] == d and m["per_view_columns"])
          for d in ("1", "b")}
    vid = {d: sum(1 for m in manifest if m["design"] == d and m["video_present"])
           for d in ("1", "b")}
    return f"""# Alignment-rate campaigns: processed CSVs

Design 1 (`results/alignment_rate/20260909_205843`, {n['1']} takes) and design B, the half outer
ring (`results/alignment_rate/20260910_droneB`, {n['b']} takes). Made by
`controller/control/package_takes.py`.

## takes/design_<d>/tilt_flight_design_<d>_<freq>hz_<repeat>.csv

One file per good take; `<repeat>` is the campaign index's own repeat number. One row per frame.

| column | meaning |
|---|---|
| frame | video frame index, shared by all columns in the row |
| t_s | host capture time, s (monotonic clock, same as the sweep.log stamps) |
| t_from_cut_s | seconds from the cut (coils A and C off; the take's `KILL_` label) |
| phase | ramp, settle, hold, drop (after the cut), spindown, off -- from the sweep.log labels |
| rotor_stopped | 1 once the rotor has stopped (`alignment_rate.spin_stop`); axis columns are meaningless after that |
| nx ny nz | fused 3-D rotor axis, unit vector, world frame (camera A's axis is +z, NOT up) |
| agree_deg | angle between the axes from camera A and camera B |
| minor_nx minor_ny minor_nz | the minor-axis (two-plane intersection) axis estimate |
| minor_vs_conic_deg | its angle to the conic-backprojection estimate |
| x_mm y_mm z_mm | triangulated disc centre, mm, world frame |
| radius_mm | disc radius from the unforeshortened major axis, mm |
| a_* / b_* | camera A / B ellipse fit: theta_deg (axis-ratio tilt), area_px, cx cy (centre, px), d1 d2 (axis lengths, px), ang_deg |

A blank cell means that output had no solution for that frame -- EXCEPT where the source file
does not exist for that take at all; `manifest.csv` says which.

## What exists per design

| | design 1 | design B |
|---|---|---|
| takes here | {n['1']} | {n['b']} |
| fused axis + minor axis (`nx..`, `minor_*`, `x_mm..`) | all | all |
| per-camera fits (`a_*`, `b_*`) | {pv['1']} of {n['1']} | {pv['b']} of {n['b']} |
| video beside the take | {vid['1']} of {n['1']} | {vid['b']} of {n['b']} |

Design 1's takes were stripped after solving: their `A/A.mp4`, `B/B.mp4` and `frames.csv` live
in `results/flights/Archive.zip`, and `tilt_A.csv` / `tilt_B.csv` were never kept, so its files
carry no per-camera columns. Re-solving those takes from the archive would produce them.
Design B's video is on the USB stick (`/Volumes/UBUNTU 24_0/ESP32_PMW_flights/droneB/`), which
`manifest.csv` records per take in `video_dir`.

## manifest.csv

One row per take file: design, freq_hz, repeat, take stamp, where its video and log live,
the cut time on the host clock, when the rotor stopped (blank = never within the take), and
the take's own results -- rate_relu_deg_s, swing_amp_deg and modal_repeat from the rate
analysis; settle_swing_deg, settle_10pct_s and settle_note from the settling analysis.
`per_view_columns` and `video_present` say whether that take's per-camera fits and video are
on disk, so a blank column can be told from a missing source.

## summary/

Each campaign's result tables (campaign, campaign_trials, settling, settling_by_freq) and the
design comparison's statistics and pooled traces. Their provenance and caveats are in
`results/alignment_rate/compare_1_vs_B/README.md` and `controller/control/theory.md` 24.12-24.13.
"""


def _self_check():
    import tempfile

    assert csv_name("b", 10.0, 2) == "tilt_flight_design_b_10hz_2.csv"
    assert csv_name("1", "120.0", "6") == "tilt_flight_design_1_120hz_6.csv"
    with tempfile.TemporaryDirectory() as d:
        take = Path(d)
        (take / "axis.csv").write_text("frame,t,nx,ny,nz,agree_deg\n"
                                       "0,100.0,0,0,1,5\n2,100.2,0,0.1,0.99,6\n")
        (take / "axis_minor.csv").write_text(
            "frame,t,nx,ny,nz,vs_conic_deg,x_mm,y_mm,z_mm,radius_mm\n"
            "0,100.0,0,0,1,1,1,2,3,6\n1,100.1,0,0,1,1,1,2,3,6\n")
        view = "frame,t,theta_deg,area_px,cx,cy,d1,d2,ang_deg\n0,100.0,45,100,1,1,10,10,0\n"
        (take / "tilt_A.csv").write_text(view)
        (take / "tilt_B.csv").write_text(view)
        log = take / "sweep.log"
        log.write_text("[99.0] label=FREQ_010HZ\n[99.5] label=HOLD_010HZ\n"
                       "[100.05] label=KILL_010HZ\n[101.0] label=DOWN_010HZ\n")
        header, rows = merge_take(take, log, t_kill=100.05, spin_stop_s=0.1)
        # every source column lands, prefixed; the minor file's centre keeps its own name
        for c in ("nx", "agree_deg", "minor_nx", "minor_vs_conic_deg", "x_mm", "radius_mm",
                  "a_theta_deg", "b_d2"):
            assert c in header, (c, header)
        # the union of frames, in order, with blanks where an output did not solve
        assert [r["frame"] for r in rows] == [0, 1, 2]
        assert rows[1]["nx"] == "" and rows[1]["minor_nx"] == "0"
        # time from the cut, phase from the labels, and the rotor-stop flag
        assert rows[0]["t_from_cut_s"] == "-0.0500" and rows[0]["phase"] == "hold"
        assert rows[2]["phase"] == "drop" and rows[2]["rotor_stopped"] == 1
        assert rows[1]["rotor_stopped"] == 0          # 0.05 s after the cut, stop at 0.1
    print("package_takes: self-check passed (names, merge, blanks, phases, rotor flag)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _self_check()
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="1,b", help="comma list of design tags")
    a = ap.parse_args()
    build(a.out, tuple(x.strip() for x in a.designs.split(",")))
