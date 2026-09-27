#!/usr/bin/env python3
"""Solve one take to CSV, check the CSV is sane, and only then copy it and the video to USB.

    uv run python ai/alignment/process_block.py results/flights/<take> [--no-upload]

WHY THIS EXISTS
---------------
The sweep records a block, then waits. That wait is dead time, and it is also the only safe
window to do anything expensive: the cameras and the USB drive share a bus, so a solve or a
copy running DURING a recording competes with the capture and shows up as dropped frames.

So the order is: record -> stop -> solve -> validate -> upload -> pause -> next block. Nothing
here runs while a recorder is live, and `--no-upload` exists for checking a take without
touching the drive.

WHY `disc_axis` AND NOT `ai/rim_track.py`
-----------------------------------------
`ai/rim_track.py` imports `stereo_axis`, `_load_rig`, `VIRTUAL_F`, `_min_plate` and `_trim_fit`
from `ai/alignment_rate.py`, which is gitignored and was never committed -- it exists in no
branch and on no machine here. `controller/pose/disc_axis.py` is the rebuilt producer of the
SAME layout: its docstring records that `theta_deg` matches the missing script's output to
0.0008 deg over 7065 frames, and it writes `axis.csv` with the same `AXIS_COLS` that
`segments.py` and `theta_lp.py` read. The two differ in the ellipse fit (RANSAC on the rim
edge against `disc_pose.segment_disc`), not in the geometry.

Output goes to `results/rim/<take>/`, which is where `segments.py` looks for it.

WHAT "REASONABLE" MEANS
-----------------------
A solve that produces a file is not a solve that produced data. The checks below are the ones
that would have caught the failures this project has actually had: a take whose tracker lost
the disc, one where the two views disagree (the segmenter fitting different shapes), and one
where the axis sign flipped end-for-end so every differenced rate is garbage.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RIM = ROOT / "results" / "rim"
USB = Path("/Volumes/UBUNTU 24_0/ESP32_PMW_flights/whole_ring")

#: Fraction of the take's frames that must carry a solved axis. The tracker legitimately loses
#: the disc while the rotor is stopped and the silhouette is edge-on, so this is not 1.0.
MIN_COVERAGE = 0.50
#: Median view-A-against-view-B disagreement, degrees, and the gate is deliberately loose.
#:
#: `agree_deg` is PRODUCER-SPECIFIC and must only be compared between takes solved by the
#: same code. `results/rim/` holds takes written by the missing `ai/rim_track.py`, which used a
#: VIRTUAL focal length (`VIRTUAL_F`) and a RANSAC fit on the rim edge; they read 3.6-4.3 deg.
#: Takes this repo can still solve are written by `controller/pose/disc_axis.py`, which uses the
#: real rig intrinsics and `disc_pose.segment_disc`, and over 29 alignment-rate takes those read
#: **15.1-19.6 deg** median with p90 up to 36. Calibrating this gate on the 3.6 figure -- which
#: is what a first pass did on 2026-09-27 -- rejects perfectly good takes for a difference in
#: camera model, not in data quality.
#:
#: 28 sits above every good disc_axis take measured (19.6) and above the 2026-09-27 whole-ring
#: take (21.6, p90 22.9 -- tight, unlike the noisiest good take whose p90 is 36), while still
#: catching a real segmentation failure, which drives this to 40+ or to non-finite garbage.
MAX_AGREE_MED_DEG = 28.0
#: The axis must actually move. A take where the rotor never tilted has no step to measure.
MIN_TILT_SPAN_DEG = 5.0
#: Sign flips: adjacent frames whose axis reverses. The sign is carried forward frame to frame,
#: so a handful is noise; a run of them means the carry failed and rates are meaningless.
MAX_FLIP_FRAC = 0.02
#: The capture rate the driver is asked for, and how far below it a take may fall before it is
#: worth saying so. The request is `record.CAP_FPS_DRIVER_CLAMP`; what is DELIVERED is lower,
#: because the encoder and the Python read loop cost something the sensor does not. Measured
#: 2026-09-27: 209.2 pairs/s for the bare source, 193.7-206.0 in a real take. So this is a
#: report, not a gate -- a take at 200 fps is perfectly usable, and refusing it would throw
#: away good data over a number that does not change the physics.
TARGET_FPS = 210.0
FPS_WARN_FRAC = 0.05
#: Frames the encoder queue may lose before the take is called damaged. `frames.csv` keeps a
#: row for a dropped frame with `written=0`, and `record.read_index` filters those out, so the
#: TIMING survives and the analysis stays correct -- what is lost is coverage. 2026-09-27
#: exposed the gap: `2026-09-27_115535_whole120` dropped 234 of 90085 (0.26%) and nothing in
#: this file mentioned it, so the only record was the column in `frame_rates.txt`. A tenth of a
#: percent is harmless; a few percent means the take has holes in it wherever the encoder fell
#: behind, which is exactly where a step might have been missed.
MAX_DROP_FRAC = 0.02


def solve(take):
    """Run `disc_axis.solve` into `results/rim/<take>/`. Returns the output directory."""

    out = RIM / Path(take).name
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "controller/pose/disc_axis.py", str(take), "--out", str(out)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    tail = (r.stdout or "").strip().splitlines()[-3:]
    for line in tail:
        print(f"    {line}")
    if r.returncode:
        print(f"    !! disc_axis exited {r.returncode}")
        print((r.stderr or "")[-600:])
    return out


def validate(take, out):
    """Check the solved CSV. Returns ``(ok, report_lines)``."""

    lines, ok = [], True
    axis_p, frames_p = out / "axis.csv", Path(take) / "frames.csv"
    if not axis_p.exists():
        return False, ["    !! no axis.csv was written"]
    rows = list(csv.DictReader(open(axis_p)))
    if not rows:
        return False, ["    !! axis.csv is empty"]

    n_frames = sum(1 for _ in open(frames_p)) - 1 if frames_p.exists() else 0
    cov = len(rows) / n_frames if n_frames else 0.0
    lines.append(f"    solved {len(rows)}/{n_frames} frames ({cov:.0%})")
    if cov < MIN_COVERAGE:
        ok = False
        lines.append(f"    !! coverage under {MIN_COVERAGE:.0%} -- the tracker lost the disc")

    # The capture rate, from `meta.json`, because it is the one number the take's name and
    # directory cannot tell you and the analysis cares about (it sets the practical limit for
    # resolving the spin line). Reported, not gated: see TARGET_FPS.
    meta_p = Path(take) / "meta.json"
    if meta_p.exists():
        m = json.loads(meta_p.read_text())
        got = float(m.get("fps_measured") or 0.0)
        asked = m.get("cap_fps_requested")
        declared = m.get("fps")
        lines.append(
            f"    capture {got:.1f} fps delivered (asked {asked or '-'}, "
            f"declared {declared:g})" if got else "    capture: no fps_measured recorded"
        )
        if got and (TARGET_FPS - got) / TARGET_FPS > FPS_WARN_FRAC:
            lines.append(
                f"    note: {got:.1f} fps against the {TARGET_FPS:.0f} requested "
                f"({got / TARGET_FPS:.0%}). Expected -- the pipeline is encoder-bound, not "
                f"sensor-bound, so the request is a ceiling and not a setting."
            )
        # A declared rate far from the delivered one means the mp4 plays at the wrong speed.
        if got and declared and abs(float(declared) - got) / got > 0.10:
            lines.append(
                f"    note: mp4 declares {float(declared):g} fps but {got:.1f} was captured, "
                f"so it plays {float(declared) / got:.2f}x. Timing is unaffected: every "
                f"timestamp comes from frames.csv, not from frame indices."
            )
        # Frames the encoder queue lost. `read_index` drops their rows, so timing survives and
        # only coverage is affected -- see MAX_DROP_FRAC.
        dropped = int(m.get("dropped") or 0)
        if dropped:
            frac = dropped / max(dropped + int(m.get("n_frames") or 0), 1)
            lines.append(f"    dropped {dropped} frames ({frac:.2%}) -- rows kept, timing OK")
            if frac > MAX_DROP_FRAC:
                ok = False
                lines.append(
                    f"    !! {frac:.1%} of frames were dropped, over the {MAX_DROP_FRAC:.0%} "
                    f"limit -- the take has holes wherever the encoder fell behind, which is "
                    f"where a step could have been missed"
                )

    n = np.array([[float(r["nx"]), float(r["ny"]), float(r["nz"])] for r in rows])
    if not np.isfinite(n).all():
        ok = False
        lines.append("    !! non-finite axis components")
        n = n[np.isfinite(n).all(axis=1)]
    if len(n) < 2:
        return False, lines + ["    !! too few usable rows"]

    agree = np.array([float(r["agree_deg"]) for r in rows])
    agree = agree[np.isfinite(agree)]
    med = float(np.median(agree)) if len(agree) else float("nan")
    lines.append(f"    view A vs B: median {med:.1f} deg, p90 {np.percentile(agree, 90):.1f}")
    if not (med <= MAX_AGREE_MED_DEG):
        ok = False
        lines.append(f"    !! views disagree by more than {MAX_AGREE_MED_DEG} deg -- "
                     f"the segmenter is fitting different shapes")

    # Tilt span, from the axis against its own median direction: a datum-free way to ask
    # "did the rotor actually lean", which is what the experiment measures.
    #
    # MEASURED TO THE AXIS LINE, NOT THE VECTOR. An ellipse fixes a disc normal only up to
    # sign -- (x,y,z) and (-x,-y,-z) are the same physical axis -- so one failed sign carry
    # makes every later frame antiparallel and `acos(u @ ref)` parks at 180 deg. On the
    # 2026-09-27 take exactly that happened at t=115 s and the first version reported a
    # 179.2 deg "span" while the axis was stable to 0.5 deg either side of the flip and had
    # moved ~20 deg in total. Folding with |u @ ref| removes the ambiguity the carry papers
    # over, so this measures motion rather than bookkeeping.
    u = n / np.linalg.norm(n, axis=1, keepdims=True)
    ref = np.median(u, axis=0)
    ref /= np.linalg.norm(ref)
    tilt = np.degrees(np.arccos(np.clip(np.abs(u @ ref), 0.0, 1.0)))
    span = float(np.percentile(tilt, 99) - np.percentile(tilt, 1))
    lines.append(f"    tilt span {span:.1f} deg (1-99 pct, sign-invariant)")
    if span < MIN_TILT_SPAN_DEG:
        ok = False
        lines.append(f"    !! tilt span under {MIN_TILT_SPAN_DEG} deg -- nothing moved")

    # Sign flips. REPORTED, NOT GATED: a flip is the bookkeeping artefact folded out above,
    # so rejecting on it would discard a take whose geometry is sound. What the count is good
    # for is telling a reader that a flip happened and where to look.
    flips = int((np.sum(u[:-1] * u[1:], axis=1) < 0).sum())
    frac = flips / max(len(u) - 1, 1)
    lines.append(f"    sign flips {flips} ({frac:.3%}) -- bookkeeping, folded out above")

    for name in ("axis_minor.csv", "tilt_A.csv", "tilt_B.csv"):
        if not (out / name).exists():
            lines.append(f"    note: {name} missing")
    return ok, lines


def upload(take, out):
    """Copy the video and the solved CSV to the USB. Returns the destination."""

    dest = USB / "flights" / Path(take).name
    for sub in ("A", "B"):
        src = Path(take) / sub / f"{sub}.mp4"
        if src.exists():
            (dest / sub).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / sub / f"{sub}.mp4")
    for name in ("frames.csv", "meta.json"):
        if (Path(take) / name).exists():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(take) / name, dest / name)
    solved = USB / "analysis" / Path(take).name
    solved.mkdir(parents=True, exist_ok=True)
    for name in ("axis.csv", "axis_minor.csv", "tilt_A.csv", "tilt_B.csv"):
        if (out / name).exists():
            shutil.copy2(out / name, solved / name)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("take")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args(argv)

    take = Path(a.take)
    if not (take / "frames.csv").exists():
        raise SystemExit(f"{take}: no frames.csv -- not a take directory")
    print(f"  processing {take.name}")

    out = solve(take)
    ok, lines = validate(take, out)
    for line in lines:
        print(line)

    if not ok:
        print("  VERDICT: NOT USABLE -- not uploading. The take stays on disk.")
        return 1
    print("  VERDICT: usable")
    if a.no_upload:
        print("  --no-upload: skipping the copy")
        return 0
    if not USB.parent.exists():
        print(f"  !! {USB.parent} is not mounted -- skipping the copy")
        return 1
    dest = upload(take, out)
    print(f"  uploaded -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())