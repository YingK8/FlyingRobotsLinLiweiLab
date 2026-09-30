#!/usr/bin/env python3
"""Line takes recorded before the `written` flag up at the cut, from their dropped-frame pattern.

Before 2026-09-13, `FlightWriter` kept a `frames.csv` row for every frame the encoder queue
dropped, and `disc_axis` paired mp4 frame i with row i ("forward"). With drops, every frame
after the first drop got an earlier row's time. Which rows were dropped was never recorded,
but the video is dropped-free at one end in many takes:

  back     all drops came BEFORE the cut: pair mp4 frame i with row (rows - n_mp4 + i), counting
           from the end. Exact from the last drop onward, which covers the cut and after.
  forward  all drops came AFTER the cut: the existing pairing is exact up to the cut; left as is.
  none     no drops: nothing to do.

The category per take comes from the alignment test (swing onset within 0.04 s of that
frequency's drop-free lag under one pairing). Takes where neither pairing lands the swing on
the cut were re-recorded, not retimed.

Rewrites the `t` column of every per-frame CSV in the take that has `frame` and `t` columns,
keeping the original beside it as `<name>.forward_t.csv` (never deleted), and writes
`retime.json` with the method, so an analysis can tell exact post-cut timing from approximate.

    uv run python ai/alignment/retime_takes.py CATEGORIES.csv [--dry-run]

CATEGORIES.csv columns: take_dir, category (back | forward | none), onset_forward_s, onset_back_s.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PER_FRAME = ("axis.csv", "axis_minor.csv", "tilt_A.csv", "tilt_B.csv")


def backward_times(take):
    """Row time (camera A) for each mp4 frame index, pairing from the end of the recording."""

    rows = np.genfromtxt(take / "frames.csv", delimiter=",", names=True)
    dropped = int(json.load(open(take / "meta.json")).get("dropped", 0))
    n_mp4 = len(rows) - dropped
    return rows["t_a"][len(rows) - n_mp4:], dropped


def retime(take, category, onset_fwd, onset_back, dry_run=False):
    take = Path(take)
    done = []
    if category == "back":
        t_frame, dropped = backward_times(take)
        for name in PER_FRAME:
            p = take / name
            if not p.exists():
                continue
            with open(p) as fh:
                rows = list(csv.DictReader(fh))
            if not rows or "frame" not in rows[0] or "t" not in rows[0]:
                continue
            backup = take / (p.stem + ".forward_t.csv")
            if backup.exists():
                raise SystemExit(f"{backup} exists: {take.name} was already retimed")
            fields = list(rows[0])
            for r in rows:
                r["t"] = f"{t_frame[int(r['frame'])]:.6f}"
            if not dry_run:
                shutil.copy2(p, backup)
                with open(p, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields)
                    w.writeheader()
                    w.writerows(rows)
            done.append(name)
    else:
        dropped = int(json.load(open(take / "meta.json")).get("dropped", 0))
    info = {"method": category, "dropped": dropped, "files": done,
            "onset_forward_s": onset_fwd, "onset_back_s": onset_back,
            "post_cut_exact": category in ("back", "none"),
            "why": "frames.csv kept rows for dropped frames (pre-2026-09-13 FlightWriter)"}
    if not dry_run:
        (take / "retime.json").write_text(json.dumps(info, indent=2))
    return info


def _self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        take = Path(d)
        # 10 captured rows at 0..9 s, 3 dropped before the cut -> the mp4 holds 7 frames,
        # which are really rows 3..9
        (take / "frames.csv").write_text("index,t_capture,skew_s,t_a,t_b\n"
                                         + "".join(f"{i},{i}.0,0,{i}.0,{i}.0\n" for i in range(10)))
        (take / "meta.json").write_text(json.dumps({"dropped": 3}))
        (take / "axis.csv").write_text("frame,t,nx\n" + "".join(f"{i},{i}.0,0.1\n" for i in range(7)))
        info = retime(take, "back", 3.0, 0.0)
        got = [float(r["t"]) for r in csv.DictReader(open(take / "axis.csv"))]
        assert got == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], got
        assert (take / "axis.forward_t.csv").exists() and info["post_cut_exact"]
        try:
            retime(take, "back", 3.0, 0.0)
            raise AssertionError("a second retime must refuse")
        except SystemExit:
            pass
        assert retime(take, "forward", 0.0, 3.0)["post_cut_exact"] is False
    print("retime_takes: self-check passed (backward pairing, backup kept, no double retime)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _self_check()
        sys.exit()
    dry = "--dry-run" in sys.argv
    for r in csv.DictReader(open(sys.argv[1])):
        info = retime(r["take_dir"], r["category"], float(r["onset_forward_s"]),
                      float(r["onset_back_s"]), dry_run=dry)
        print(Path(r["take_dir"]).name, info["method"], info["dropped"], info["files"])
