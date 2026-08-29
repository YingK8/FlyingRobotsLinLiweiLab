#!/usr/bin/env python3
"""
The refusing shutter: board pair images onto disk, and nothing else.

`theory.md` section 16. A capture writes one **bag** -- a directory holding `A/`, `B/`,
`pairs.csv` and `meta.json` -- and `capture` reuses a bag that already exists rather than
re-shooting it. Overwriting an hour of board-waving must be asked for: ``override=True``.

    capture(PAIR_DIR)                        # shoot, or reuse what is there
    capture(PAIR_DIR, override=True)         # shoot again, replacing it
    capture(PAIR_DIR, append=True)           # add to it
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path[:0] = [str(Path(__file__).resolve().parent.parent / "camera")]
import sources  # noqa: E402

from calibrate import (MIN_COMMON_CORNERS, MIN_CORNERS, MIN_ORIENT_SPREAD_DEG,
                       MAX_INCIDENCE_DEG, NATIVE_H, NATIVE_W, PAIR_DIR,
                       SPEC, board_incidence_deg, board_normal, detect, fit_corners,
                       orientation_spread_deg, solve_board_pose)

# A moving board costs a *product* -- (board speed) x (sync window) -- not a pixel count,
# so that is what the gate is on (section 16.2). At 11 px/mm with 4.5 ms of skew a 40 mm/s
# hand sweep injects 1.98 px, and the run that produced it failed at 1.08 px joint RMS.
MAX_PAIR_ERROR_PX = 0.10   # a fifth of the 0.5 px the whole calibration is allowed
MAX_BLUR_PX = 0.50         # smear within one exposure, which no amount of sync removes
EXPOSURE_S = 0.008         # the frame period at the mode's top rate; macOS refuses to
                           # report exposure through OpenCV

MIN_KEEP_ROT_DEG = 8.0   # a saved shot must differ from every other by this much in board
MIN_KEEP_SHIFT_PX = 80.0 # orientation, or this much in where it sits in frame
INCIDENCE_BANDS = (0.0, 15.0, 30.0, 45.0)   # how the usable tilt range is covered
BAND_FULL = 6            # shots in one band before the shutter asks for a different tilt.
                         # A rotation-blind gate lets the operator slide the board around
                         # without tilting it: 97 views that way gave 16.0 deg of spread
                         # against the 20 deg the intrinsics want, and more frames cannot
                         # fix a spread problem.


def annotate(image, spec, corners, ids, rvec=None, tvec=None, K=None, dist=None):
    """Copy of ``image`` with detected corners and, if posed, the board axes."""
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    if ids is not None and len(ids):
        # OpenCV 5 hands back flat arrays; the drawing helpers still want (N,1,*).
        cv2.aruco.drawDetectedCornersCharuco(
            out, np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2),
            np.asarray(ids, dtype=np.int32).reshape(-1, 1))
    if rvec is not None and K is not None:
        cv2.drawFrameAxes(out, K, dist, rvec, tvec, 2.0 * spec.square_mm)
    return out

NATIVE_W, NATIVE_H = 1280, 800     # ELP OV9281 native: 119 fps, full field of view
FAST_W, FAST_H = 640, 400          # exact 0.5x rescale: 217 fps, same field of view
# Which index is which camera: `camera/elp.py :: probe_indices`. USB cameras enumerate
# BEFORE the built-in FaceTime, so the ELP is normally index 0.


# ---- the bag ------------------------------------------------------------------------
def write_meta(bag, spec, indices, mode, rotate180, n_pairs, n_solo, skew):
    """What this bag is, beside the images. Its presence is what marks a bag finished."""

    meta = {"board": spec.summary(), "camera_indices": list(indices), "mode": list(mode),
            "rotate180": bool(rotate180), "n_pairs": n_pairs, "n_solo": list(n_solo),
            "skew": skew, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (Path(bag) / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def read_meta(bag):
    """This bag's `meta.json`, or ``{}`` if it was never finished."""

    path = Path(bag) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


MIN_KEEP_ROT_DEG = 8.0   # a saved pair must differ from every other saved pair by this much
MIN_KEEP_SHIFT_PX = 80.0 # ...in board orientation, or this much in where it sits in frame
INCIDENCE_BANDS = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)   # how the tilt range is covered
BAND_FULL = 6            # shots in one band before the shutter asks for a different tilt. A
                         # rotation-blind gate lets the operator slide the board around the
                         # frame without tilting it: 97 views that way gave 16.0 deg of
                         # spread against the 20 deg `calibrate_intrinsics` wants, and more
                         # frames cannot fix a spread problem.


def _rough_K(shape):
    """A made-up focal length, enough for an angle. Nothing metric may be read off it."""

    f = max(shape[:2]) * 1.2
    return np.array([[f, 0, shape[1] / 2.0], [0, f, shape[0] / 2.0], [0, 0, 1.0]])


def _look(spec, frame, prev=None, dt=None):
    """What one frame shows: corners, ids, count, incidence, and how fast it is moving.

    ``prev`` is the previous frame's ``{id: corner}``; the motion figure is the median
    travel over the corners both frames saw. Median, so one mis-detected corner cannot
    veto a shot. With ``dt`` that becomes ``speed`` in px/s, which is the number the gates
    are written in -- px per frame would mean something different at every loop rate.
    """

    corners, ids = detect(spec, frame)
    n_det = 0 if ids is None else len(ids)
    inc, rvec = float("nan"), None
    if n_det >= MIN_CORNERS:
        got = solve_board_pose(spec, corners, ids, _rough_K(frame.shape), np.zeros(5))
        if got is not None:
            rvec = got[0]
            inc = board_incidence_deg(rvec)

    here = (None if ids is None else
            {int(i): c for i, c in zip(ids.ravel(), corners.reshape(-1, 2))})
    motion = float("nan")
    if here and prev:
        shared = [np.linalg.norm(prev[i] - c) for i, c in here.items() if i in prev]
        if shared:
            motion = float(np.median(shared))
    speed = motion / dt if (dt and not np.isnan(motion)) else float("nan")
    return {"corners": corners, "ids": ids, "n": n_det, "incidence": inc, "rvec": rvec,
            "motion": motion, "speed": speed, "by_id": here}


def _pose_key(look):
    """``(rotation, image centre, incidence)`` -- what pose novelty is judged on."""

    return (Rotation.from_rotvec(look["rvec"].ravel()),
            look["corners"].reshape(-1, 2).mean(axis=0), look["incidence"])


def _novel(look, saved):
    """Whether this pose differs from every pose already saved for this camera."""

    rot = Rotation.from_rotvec(look["rvec"].ravel())
    centre = look["corners"].reshape(-1, 2).mean(axis=0)
    return not any(np.degrees((rot * r.inv()).magnitude()) < MIN_KEEP_ROT_DEG
                   and np.linalg.norm(centre - c) < MIN_KEEP_SHIFT_PX for r, c, _ in saved)


def solo_ok(look, saved, exposure_s=EXPOSURE_S, max_blur_px=MAX_BLUR_PX, spec=SPEC):
    """Whether one camera's frame is worth keeping for its own intrinsics alone.

    The pair gates in `gates` are the extrinsic's, and two of them -- shared corners and
    skew -- are about the *other* camera. A frame the pair gates refuse for those reasons
    still constrains this camera's $K$ perfectly well, and refusing it is what starved the
    intrinsics: pairs only exist where both cameras see the board, which on a 90 deg rig
    is a narrow band of tilts (`theory.md` section 14.3).
    """

    return (look["n"] >= fit_corners(spec) and look["rvec"] is not None
            and look["incidence"] <= MAX_INCIDENCE_DEG
            and not (look["speed"] * exposure_s > max_blur_px)
            and _novel(look, saved))


def gates(spec, looks, skew_s, saved, exposure_s=EXPOSURE_S, max_blur_px=MAX_BLUR_PX):
    """Whether this pair is worth saving, and the name of the first thing wrong with it.

    Every gate is a physical quantity rather than a preference -- `theory.md` section 16.5.
    Returns ``(ok, [(name, passed, detail), ...])``; the list drives the overlay, so the
    operator sees which gate is holding the shutter rather than guessing.
    """

    speed = max([lk["speed"] for lk in looks], default=float("nan"))
    shared = (len(np.intersect1d(looks[0]["ids"], looks[1]["ids"]))
              if len(looks) > 1 and all(lk["ids"] is not None for lk in looks) else
              (looks[0]["n"] if looks else 0))
    novel, band_ok, band_detail = True, True, ""
    if looks and looks[0]["rvec"] is not None:
        novel = _novel(looks[0], saved)

        inc = looks[0]["incidence"]
        band = int(np.searchsorted(INCIDENCE_BANDS, inc, side="right") - 1)
        counts = np.zeros(len(INCIDENCE_BANDS), dtype=int)
        for _, _, saved_inc in saved:
            counts[int(np.searchsorted(INCIDENCE_BANDS, saved_inc, side="right") - 1)] += 1
        spread = (orientation_spread_deg([board_normal(r.as_rotvec()) for r, _, _ in saved])
                  if len(saved) > 1 else 0.0)
        # Once the spread is there, stop steering -- coverage of position matters too.
        band_ok = spread >= MIN_ORIENT_SPREAD_DEG or counts[band] < BAND_FULL
        band_detail = (f"{int(INCIDENCE_BANDS[band])}-"
                       f"{int(INCIDENCE_BANDS[band + 1]) if band + 1 < len(INCIDENCE_BANDS) else int(MAX_INCIDENCE_DEG)}"
                       f" deg has {counts[band]}, spread {spread:.1f}/{MIN_ORIENT_SPREAD_DEG:.0f}")

    checks = [
        ("board in view", all(lk["n"] >= fit_corners(spec) for lk in looks),
         "/".join(str(lk["n"]) for lk in looks) + f" of {spec.n_corners}, "
         f"need {fit_corners(spec)}"),
        ("board posed", all(lk["rvec"] is not None for lk in looks),
         "/".join("y" if lk["rvec"] is not None else "n" for lk in looks)),
        ("shared corners", shared >= MIN_COMMON_CORNERS, f"{shared}"),
        ("tilt usable", all(lk["incidence"] <= MAX_INCIDENCE_DEG for lk in looks),
         "/".join(f"{lk['incidence']:.0f}" for lk in looks) + f" of {MAX_INCIDENCE_DEG:.0f} deg"),
        ("still (pair)", not (speed * skew_s > MAX_PAIR_ERROR_PX),
         f"{speed * skew_s:.2f} px" if np.isfinite(speed) else "unknown"),
        ("still (blur)", not (speed * exposure_s > max_blur_px),
         f"{speed * exposure_s:.2f} px" if np.isfinite(speed) else "unknown"),
        ("new pose", novel, "yes" if novel else "duplicate"),
        ("tilt spread", band_ok, band_detail),
    ]
    return all(ok for _, ok, _ in checks), checks


def _overlay(spec, frames, looks, checks, footer):
    """The live view: one annotated panel per camera, side by side, and the gate strip."""

    panels = []
    for frame, look, tag in zip(frames, looks, "AB"):
        panel = annotate(frame, spec, look["corners"], look["ids"])
        cv2.putText(panel, f"{tag}  {look['n']}/{spec.n_corners} corners  "
                    f"incidence {look['incidence']:5.1f} deg  "
                    f"{look['speed']:6.1f} px/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0)
                    if look["n"] >= MIN_CORNERS else (0, 0, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    view = np.hstack(panels)

    for i, (name, ok, detail) in enumerate(checks):
        cv2.putText(view, f"{'OK ' if ok else '.. '}{name:16s}{detail}",
                    (10, 66 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 200, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
    ready = all(ok for _, ok, _ in checks)
    cv2.putText(view, footer, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 200, 0) if ready else (255, 255, 255), 2, cv2.LINE_AA)
    return view


def capture(out_dir=PAIR_DIR, indices=(0, 1), spec=None, width=NATIVE_W, height=NATIVE_H,
            rotate180=True, max_skew_s=0.002, append=False, auto=True, rate_hz=10.0,
            exposure_s=EXPOSURE_S, max_blur_px=MAX_BLUR_PX, override=False):
    """Live preview that shoots by itself. SPACE pauses and resumes, q quits.

    The shutter tries a shot ``rate_hz`` times a second and saves the pair only if it clears
    every gate in `gates`. There is nothing to press and nothing to time: place the board,
    and it is photographed as soon as it settles.

    **The refusal is the whole design** (`theory.md` section 16). A hand-held board
    photographed mid-sweep is what failed the first real run at 1.08 px joint RMS: at
    11 px/mm a 40 mm/s sweep across 2 ms of skew is 1.98 px of disagreement against a
    0.5 px budget. The gates also do the rationing, so nothing floods the disk -- a board
    left alone yields exactly one pair, and 10 Hz is faster than a hand can move it.

    One session serves both calibrations (`theory.md` section 16.5a). A shot that clears
    every gate is saved as a pair; a shot only one camera can use is saved for that camera
    alone, under its own stem, and feeds the intrinsics that the pairs cannot condition.

    Skew is handled where it is cheapest: `sources.StereoCamera` re-reads until the two
    frames land inside ``max_skew_s``. On this bench 2 ms yields 16 pairs/s at 1.02 ms
    median against 7.71 ms unfiltered, still above ``rate_hz``.
    """

    spec = spec or SPEC
    out_dir = Path(out_dir)
    if not (override or append) and (out_dir / "meta.json").exists():
        meta = read_meta(out_dir)
        print(f"reusing {out_dir}: {meta['n_pairs']} pair(s), {meta['n_solo']} solo, "
              f"board {meta['board']['board']}, shot {meta['created']}")
        print("  capture(..., override=True) re-shoots it, append=True adds to it")
        return out_dir
    idx = [indices] if isinstance(indices, int) else list(indices)
    tags = "AB"[:len(idx)]
    for tag in tags:
        (out_dir / tag).mkdir(parents=True, exist_ok=True)

    src = (sources.open_source(f"camera:{idx[0]}", width=width, height=height,
                               grayscale=True, rotate180=rotate180) if len(idx) == 1 else
           sources.open_stereo([f"camera:{i}" for i in idx], max_skew_s=max_skew_s,
                               width=width, height=height, grayscale=True,
                               rotate180=rotate180))

    old = [f for tag in tags for f in (out_dir / tag).glob("*.png")]
    n = len(list((out_dir / tags[0]).glob("pair_*.png"))) if append else 0
    m = [len(list((out_dir / tag).glob("solo_*.png"))) if append else 0 for tag in tags]
    if old:
        print(f"{out_dir} already holds {len(old)} image(s); "
              + ("continuing from there" if append else "override: they will be replaced"))
    saved, solo, rows = [], [[] for _ in idx], []
    prev, t_prev, t_try = [None] * len(idx), None, 0.0
    period = 1.0 / rate_hz if rate_hz else 0.0
    print(f"{'auto' if auto else 'paused'}: SPACE pauses and resumes, q quits")
    try:
        while True:
            item = src.read()
            if item is None:
                print("source ended")
                break
            t, payload = item
            frames = list(payload) if isinstance(payload, (list, tuple)) else [payload]
            skew_s = getattr(src, "last_skew", 0.0)
            if not np.isfinite(skew_s):
                skew_s = max_skew_s

            dt, t_prev = (t - t_prev if t_prev else None), t
            looks = [_look(spec, f, prev[k], dt) for k, f in enumerate(frames)]
            prev = [lk["by_id"] for lk in looks]
            ok, checks = gates(spec, looks, skew_s, saved, exposure_s, max_blur_px)

            due = auto and (t - t_try) >= period
            if due:
                t_try = t
            keep = [k for k, lk in enumerate(looks)
                    if due and not ok and solo_ok(lk, solo[k], exposure_s, max_blur_px, spec)]
            if (due and ok) or keep:
                # Clearing happens on the first saved shot, not when the window opens:
                # opening the preview to look at an expensive set must not destroy it. The
                # hazard is mixing sets -- a stale frame is invisible in the solve.
                if old and not append:
                    for tag in tags:
                        for f in (out_dir / tag).glob("*.png"):
                            f.unlink()
                    print(f"cleared {len(old)} old image(s) from {out_dir}")
                    old = []
            if due and ok:
                stem = f"pair_{n:03d}"
                for tag, frame in zip(tags, frames):
                    cv2.imwrite(str(out_dir / tag / f"{stem}.png"), frame)
                saved.append(_pose_key(looks[0]))
                for k, lk in enumerate(looks):     # a pair covers each camera's K as well
                    solo[k].append(_pose_key(lk))
                rows.append((n, t, skew_s * 1e3, looks[0]["speed"],
                             *[lk["incidence"] for lk in looks]))
                print(f"{stem}: skew {skew_s*1e3:.2f} ms, {looks[0]['speed']:.0f} px/s, "
                      + ", ".join(f"{tag} {lk['n']} corners @ {lk['incidence']:.0f} deg"
                                  for tag, lk in zip(tags, looks)))
                n += 1
            for k in keep:
                # Its own stem per camera, so `pair_views` can never match two solo frames
                # into a pair that the skew gate refused.
                tag = tags[k]
                cv2.imwrite(str(out_dir / tag / f"solo_{tag}_{m[k]:03d}.png"), frames[k])
                solo[k].append(_pose_key(looks[k]))
                m[k] += 1
            blocked = next((c[0] for c in checks if not c[1]), None)
            tally = f"saved {n} pair + " + "/".join(f"{t}{c}" for t, c in zip(tags, m))
            footer = (f"{tally}   AUTO {rate_hz:.0f} Hz"
                      + (f"   waiting on '{blocked}'" if blocked else "   SHOOTING")
                      if auto else f"{tally}   PAUSED")
            footer += "   SPACE = pause, q = quit" if auto else "   SPACE = resume, q = quit"
            cv2.imshow("calibration capture", _overlay(spec, frames, looks, checks, footer))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                auto = not auto
                print("auto capture on" if auto else "paused")
    finally:
        cv2.destroyAllWindows()
        stats = src.skew_stats() if hasattr(src, "skew_stats") else {}
        src.close()

    with open(out_dir / "pairs.csv", "w") as fh:
        for k, v in stats.items():
            fh.write(f"# skew_{k}, {v}\n")
        fh.write("index,t_capture,skew_ms,speed_px_s,"
                 + ",".join(f"incidence_{tag.lower()}_deg" for tag in tags) + "\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    write_meta(out_dir, spec, idx, (width, height), rotate180, n, m, stats)
    print(f"\n{n} pair(s) in {out_dir}, plus "
          + ", ".join(f"{c} solo for {tag}" for tag, c in zip(tags, m)))
    if rows:
        sk = np.array([r[2] for r in rows])
        print(f"  skew of the pairs saved: median {np.median(sk):.2f} ms, "
              f"worst {sk.max():.2f} ms")
    return out_dir
