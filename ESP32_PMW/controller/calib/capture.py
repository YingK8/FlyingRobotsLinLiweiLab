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
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from controller.camera import identify
from controller.camera import sources
from controller.camera import uvc

from controller.calib.calibrate import (MIN_COMMON_CORNERS, MIN_CORNERS, MIN_ORIENT_SPREAD_DEG,
                       MAX_INCIDENCE_DEG, NATIVE_H, NATIVE_W, PAIR_DIR,
                       SPEC, board_incidence_deg, board_normal, detect, fit_corners,
                       orientation_spread_deg, solve_board_pose)

# A moving board costs a *product* -- (board speed) x (sync window) -- not a pixel count,
# so that is what the gate is on (section 16.2). At 11 px/mm with 4.5 ms of skew a 40 mm/s
# hand sweep injects 1.98 px, and the run that produced it failed at 1.08 px joint RMS.
MAX_PAIR_ERROR_PX = 0.10   # a fifth of the 0.5 px the whole calibration is allowed
MAX_BLUR_PX = 0.50         # smear within one exposure, which no amount of sync removes
EXPOSURE_S = 0.002         # set on the cameras over UVC (camera/uvc.py), section 16.4a:
                           # at 1 ms skew the pair gate binds first (100 px/s), where 2 ms
                           # blurs 0.2 px. Costs 4x the light of auto; read the frame means
GAIN = 100                 # the ELP's own default (GET_DEF, 2026-11-04); range 16-255
AUTO_EXPOSURE_S = 0.008    # exposure_s=None leaves the cameras as they are: assume a full
                           # frame period, since macOS will not report what auto chose

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

FAST_W, FAST_H = 640, 400          # exact 0.5x rescale: 217 fps, same field of view
# Which index is an ELP: `camera/identify.py :: elp_indices`, which probes rather
# than reading a device listing -- no macOS listing is in OpenCV's index order.


# ---- the bag ------------------------------------------------------------------------
def write_meta(bag, spec, indices, mode, rotate180, n_pairs, n_solo, skew, exposure_s=None,
               camera_settings=None):
    """What this bag is, beside the images. Its presence is what marks a bag finished."""

    # Which devices shot this, as a set: an index means nothing after the session,
    # and no listing can say which index was which device. Enough to detect that a
    # different camera has since been swapped in, which is what it is for.
    meta = {"board": spec.summary(), "camera_indices": list(indices),
            "elp_ids": elp_ids_now(), "mode": list(mode),
            "rotate180": (bool(rotate180) if isinstance(rotate180, bool)
                          else sources.rotations(rotate180, len(indices))), "n_pairs": n_pairs, "n_solo": list(n_solo),
            "skew": skew, "exposure_s": exposure_s,
            "camera_settings": camera_settings, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (Path(bag) / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def elp_ids_now():
    """
    The connected ELPs' device IDs, sorted, or ``[]`` if they cannot be read.

        Never fatal: a bag is still worth having if identification failed, and an
        empty list reads downstream as "this rig has no tripwire".
    """

    try:
        return sorted(identify.elp_ids())
    except Exception as exc:
        print(f"  camera identity unavailable ({exc}); bag records indices only")
        return []


def read_meta(bag):
    """This bag's `meta.json`, or ``{}`` if it was never finished."""

    path = Path(bag) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


MIN_KEEP_ROT_DEG = 8.0   # a saved pair must differ from every other saved pair by this much
MIN_KEEP_SHIFT_PX = 80.0 # ...in board orientation, or this much in where it sits in frame:
                         # px at 1280 wide, scaled to the frame's width in `_novel`
INCIDENCE_BANDS = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)   # how the tilt range is covered
BAND_FULL = 6            # shots in one band before the shutter asks for a different tilt. A
                         # rotation-blind gate lets the operator slide the board around the
                         # frame without tilting it: 97 views that way gave 16.0 deg of
                         # spread against the 20 deg `calibrate_intrinsics` wants, and more
                         # frames cannot fix a spread problem.

# The pose guide (theory.md 16.5c): face-on plus eight tilts, per camera. The textbook set
# (face-on + 40 deg four ways) scores 35.8 deg of spread against MIN_ORIENT_SPREAD_DEG;
# 35 deg is a guess that keeps that margin while leaving room under MAX_INCIDENCE_DEG for
# the other camera, 90 deg round the rig, to still see the board.
GUIDE_TILT_DEG = 35.0
GUIDE_HIT_DEG = 12.0     # a saved pose this close to a target fills it; a guess, ~half the
                         # 45 deg azimuth step projected at 35 deg of tilt


def _guide_targets(tilt_deg=GUIDE_TILT_DEG):
    """``[(hint, normal)]``: face-on, then the eight compass tilts, in camera coordinates.

    `board_normal` is in the +z hemisphere, so a normal with +x has the screen-right edge
    nearer the camera (+y: the bottom edge) -- `test_guide_targets` holds that sign.
    """
    a = np.radians(tilt_deg)
    out = [("face-on", np.array([0.0, 0.0, 1.0]))]
    for phi in np.radians(np.arange(0, 360, 45)):
        n = np.array([np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi), np.cos(a)])
        edges = [e for e, v in (("right", n[0]), ("left", -n[0]), ("bottom", n[1]), ("top", -n[1]))
                 if v > 1e-6]
        out.append(("+".join(edges) + f" edge nearer, {tilt_deg:.0f} deg", n))
    return out


GUIDE_TARGETS = _guide_targets()


def _angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def open_targets(saved, targets=GUIDE_TARGETS, hit_deg=GUIDE_HIT_DEG):
    """The guide targets no saved pose of this camera is within ``hit_deg`` of."""

    have = [board_normal(r.as_rotvec()) for r, _, _ in saved]
    return [t for t in targets if not any(_angle_deg(t[1], h) < hit_deg for h in have)]


def next_target(look, saved, targets=GUIDE_TARGETS, hit_deg=GUIDE_HIT_DEG):
    """The guide target this camera should fill next, nearest the current pose; None once
    every target has a saved pose. ``saved`` is this camera's `_pose_key` list."""

    open_ = open_targets(saved, targets, hit_deg)
    if not open_ or look.get("rvec") is None:
        return open_[0] if open_ else None
    now = board_normal(look["rvec"])
    return min(open_, key=lambda t: _angle_deg(t[1], now))


def guide_quads(spec, look, normal, K):
    """``(now, target)`` board outlines in pixels: the current pose, and the same board
    turned about its centre so its normal is ``normal``. Rough K, so a direction to tilt,
    not a measurement."""

    R = Rotation.from_rotvec(np.ravel(look["rvec"]))
    cur = board_normal(look["rvec"])
    delta = Rotation.align_vectors([normal], [cur])[0]
    w, h = spec.cols * spec.square_mm, spec.rows * spec.square_mm
    outline = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)
    centre = R.apply([w / 2, h / 2, 0]) + np.ravel(look["tvec"])

    def project(rot, t):
        pts, _ = cv2.projectPoints(outline, rot.as_rotvec(), t, K, np.zeros(5))
        return pts.reshape(-1, 2)

    R_t = delta * R
    return project(R, np.ravel(look["tvec"])), project(R_t, centre - R_t.apply([w / 2, h / 2, 0]))


# ---- pair mode (theory.md 16.5d) ------------------------------------------------------
PAIR_TILTS_DEG = (0.0, 15.0, 30.0)  # rings round the bisector; the outer one survives only
                                    # where it keeps both cameras under the incidence cap
PAIR_MARGIN_DEG = 8.0               # targets sit this far under MAX_INCIDENCE_DEG, so a hand
                                    # a few degrees off still clears `not edge-on`. 3 kept the
                                    # +/-30 ring at 56/56 deg on the 99.6 deg rig: 5 deg of lean
                                    # toward one camera put the other past 60, unholdable by
                                    # hand (operator, 2026-11-11). 8 keeps >= 9.5 (theory 16.5d)


def rel_rotation(looks, last=None):
    """Camera B -> camera A rotation from this frame if both cameras posed it, else ``last``.
    Rough K on both sides, so good to a few degrees: enough to aim a guide, nothing more."""

    if len(looks) < 2 or any(lk.get("rvec") is None for lk in looks[:2]):
        return last
    R_a = Rotation.from_rotvec(np.ravel(looks[0]["rvec"]))
    R_b = Rotation.from_rotvec(np.ravel(looks[1]["rvec"]))
    return R_a * R_b.inv()


def pair_targets(R_ab, tilts=PAIR_TILTS_DEG, max_inc=MAX_INCIDENCE_DEG - PAIR_MARGIN_DEG):
    """``[(hint, normal)]`` in camera A's frame: the bisector of the two optical axes (the
    one normal both cameras see equally steeply), then rings round it, keeping only the
    normals BOTH cameras see under ``max_inc``. On a rig 99 deg apart the bisector is
    ~50 deg from each, so the ring survives across the baseline and not along it."""

    z = np.array([0.0, 0.0, 1.0])
    zb = R_ab.apply(z)                        # B's optical axis, in A's frame
    b = (z + zb) / np.linalg.norm(z + zb)
    u = np.cross(z, zb)
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(b, u)
    out = []
    for t in np.radians(tilts):
        for phi in (np.radians(np.arange(0, 360, 45)) if t else [0.0]):
            n = np.cos(t) * b + np.sin(t) * (np.cos(phi) * u + np.sin(phi) * v)
            inc = [np.degrees(np.arccos(min(1.0, abs(float(n @ ax))))) for ax in (z, zb)]
            if max(inc) <= max_inc:
                out.append((f"pair {np.degrees(t):.0f} deg off the bisector", n))
    return out


def in_camera(normal, k, R_ab):
    """A target normal (A's frame) in camera ``k``'s frame, in its +z hemisphere."""

    n = normal if k == 0 else R_ab.inv().apply(normal)
    return n if n[2] >= 0 else -n


def turn_words(now, want):
    """Which screen edges to bring nearer to go from normal ``now`` to ``want`` (same frame
    convention as `_guide_targets`: +x on the normal = right edge nearer)."""

    d = np.asarray(want) - np.asarray(now)
    edges = [e for e, v in (("right", d[0]), ("left", -d[0]), ("bottom", d[1]), ("top", -d[1]))
             if v > 0.05]
    return ("+".join(edges) + " edge nearer") if edges else "hold"


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
    inc, rvec, tvec = float("nan"), None, None
    if n_det >= MIN_CORNERS:
        got = solve_board_pose(spec, corners, ids, _rough_K(frame.shape), np.zeros(5))
        if got is not None:
            rvec, tvec = got
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
            "tvec": tvec, "motion": motion, "speed": speed, "by_id": here, "width": frame.shape[1]}


def _pose_key(look):
    """``(rotation, image centre, incidence)`` -- what pose novelty is judged on."""

    return (Rotation.from_rotvec(look["rvec"].ravel()),
            look["corners"].reshape(-1, 2).mean(axis=0), look["incidence"])


def _novel(look, saved):
    """Whether this pose differs from every pose already saved for this camera."""

    rot = Rotation.from_rotvec(look["rvec"].ravel())
    centre = look["corners"].reshape(-1, 2).mean(axis=0)
    return not any(np.degrees((rot * r.inv()).magnitude()) < MIN_KEEP_ROT_DEG
                   and np.linalg.norm(centre - c) < MIN_KEEP_SHIFT_PX * look.get("width", NATIVE_W) / NATIVE_W
                   for r, c, _ in saved)


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


def gates(spec, looks, skew_s, saved, exposure_s=EXPOSURE_S, max_blur_px=MAX_BLUR_PX,
          variety=None):
    """Whether this pair is worth saving, and the name of the first thing wrong with it.

    Every gate is a physical quantity rather than a preference -- `theory.md` section 16.5.
    Returns ``(ok, [(name, passed, detail), ...])``; the list drives the overlay, so the
    operator sees which gate is holding the shutter rather than guessing.

    ``variety``: camera A's whole saved set (pairs + solos), which is what its intrinsics
    are fitted from, so what `tilt variety` measures spread over. Defaults to ``saved``.
    Measured over the pairs alone it capped pairs at ~6 a session on a rig whose every
    pair lands in one band (theory.md 16.5d).
    """

    variety = saved if variety is None else variety
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
        spread = (orientation_spread_deg([board_normal(r.as_rotvec()) for r, _, _ in variety])
                  if len(variety) > 1 else 0.0)
        # Once the spread is there, stop steering -- coverage of position matters too.
        band_ok = spread >= MIN_ORIENT_SPREAD_DEG or counts[band] < BAND_FULL
        span = (f"{int(INCIDENCE_BANDS[band])}-"
                f"{int(INCIDENCE_BANDS[band + 1]) if band + 1 < len(INCIDENCE_BANDS) else int(MAX_INCIDENCE_DEG)}")
        band_detail = (f"A has {counts[band]}/{BAND_FULL} pairs at {span} deg tilt, "
                       f"A's variety {spread:.0f}/{MIN_ORIENT_SPREAD_DEG:.0f} deg"
                       + ("" if band_ok else " -> tilt to another angle"))

    def per_cam(fmt):
        return " / ".join(f"{t} {fmt(lk)}" for t, lk in zip("AB", looks))

    steep = [t for t, lk in zip("AB", looks) if not lk["incidence"] <= MAX_INCIDENCE_DEG]
    still_pair = not (speed * skew_s > MAX_PAIR_ERROR_PX)
    still_blur = not (speed * exposure_s > max_blur_px)
    # Names are the gate's identity (tests, theory.md 16.5); details say what to DO.
    checks = [
        ("board in view", all(lk["n"] >= fit_corners(spec) for lk in looks),
         per_cam(lambda lk: lk["n"]) + f" corners, need {fit_corners(spec)} each"),
        ("board posed", all(lk["rvec"] is not None for lk in looks),
         per_cam(lambda lk: "ok" if lk["rvec"] is not None else "no pose")),
        ("shared corners", shared >= MIN_COMMON_CORNERS,
         f"{shared} seen by both, need {MIN_COMMON_CORNERS}"
         + ("" if shared >= MIN_COMMON_CORNERS else " -> face the board between A and B")),
        ("not edge-on", not steep,
         per_cam(lambda lk: f"{lk['incidence']:.0f}") + f" deg from face-on, max {MAX_INCIDENCE_DEG:.0f}"
         + (f" -> turn the board toward {'+'.join(steep)}" if steep else "")),
        ("still (pair)", still_pair,
         (f"{speed * skew_s:.2f} of {MAX_PAIR_ERROR_PX} px" if np.isfinite(speed) else "unknown")
         + ("" if still_pair else " -> hold still")),
        ("still (blur)", still_blur,
         (f"{speed * exposure_s:.2f} of {max_blur_px} px" if np.isfinite(speed) else "unknown")
         + ("" if still_blur else " -> hold still")),
        ("new pose", novel, "yes" if novel else
         f"already saved -> turn {MIN_KEEP_ROT_DEG:.0f}+ deg or move {MIN_KEEP_SHIFT_PX:.0f}+ px"),
        ("tilt variety", band_ok, band_detail),
    ]
    return all(ok for _, ok, _ in checks), checks


def _text(img, s, org, scale, color, thickness=1):
    """Text on a black box, legible over the black-and-white board. A box, not an outline:
    OpenCV 5 spaces glyphs by stroke weight, so a thicker black copy does not line up."""
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(img, (org[0] - 2, org[1] - h - 3), (org[0] + w + 2, org[1] + base), (0, 0, 0), -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _overlay(spec, frames, looks, checks, footer, max_width=None, *, check_heading=None,
             saved=None, pairs=None, pair_R=None):
    """The live view: one annotated panel per camera, side by side, and the gate strip.

    With ``saved`` (per camera `_pose_key` lists) each panel also carries the pose guide:
    the board outline now (yellow), where to turn it (cyan), and the hint in words. With
    ``pairs`` (the saved pairs, camera A's keys) it guides pairs first -- both panels, one
    target, from ``pair_R`` (`rel_rotation`) -- then each camera alone (theory.md 16.5d)."""

    p_targets = pair_targets(pair_R) if pair_R is not None else []
    p_open = open_targets(pairs, p_targets) if pairs is not None else []
    pair_mode = pairs is not None and (pair_R is None or bool(p_open))
    p_next = next_target(looks[0], pairs, p_targets) if pair_mode and pair_R is not None else None

    # Guide ONE camera at a time, the one with more targets open (ties: A). Two guides on
    # a rig 90 deg round ask for opposite tilts and neither is satisfiable (theory.md 16.5c).
    lead = (max(range(len(frames)), key=lambda k: (len(open_targets(saved[k])), -k))
            if saved is not None else None)
    panels = []
    for k, (frame, tag) in enumerate(zip(frames, "AB")):
        K = _rough_K(frame.shape)
        scale = min(1.0, max_width / frame.shape[1]) if max_width else 1.0
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        look = looks[k]
        corners = look["corners"]
        panel = annotate(frame, spec, corners * scale if corners is not None else None, look["ids"])
        label = f"{tag}  {look['n']}/{spec.n_corners} corners"
        if np.isfinite(look["incidence"]):
            label += f"  incidence {look['incidence']:.0f} deg"
        if np.isfinite(look["speed"]):
            label += f"  {look['speed']:.0f} px/s"
        color = (0, 200, 0) if look["n"] >= fit_corners(spec) else (0, 0, 255)
        cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        if pair_mode:
            done = f"pair {len(p_targets) - len(p_open)}/{len(p_targets)}" if p_targets else "pairs first"
            if p_next is None:
                hint = f"{tag} pairs first: show the board to both A and B"
            elif look["rvec"] is None:
                hint = f"{tag} {done}: board not seen"
            else:
                want, cur = in_camera(p_next[1], k, pair_R), board_normal(look["rvec"])
                now, ghost = guide_quads(spec, look, want, K)
                for quad, c in ((now, (0, 220, 255)), (ghost, (255, 255, 0))):
                    cv2.polylines(panel, [np.int32(quad * scale)], True, c, 2, cv2.LINE_AA)
                hint = f"{tag} {done}: {turn_words(cur, want)} ({_angle_deg(want, cur):.0f} deg off)"
            _text(panel, hint, (8, panel.shape[0] - 40), 0.5, (255, 255, 0))
        elif saved is not None:
            target = next_target(look, saved[k])
            done = len(GUIDE_TARGETS) - len(open_targets(saved[k]))
            if target is None:
                hint = f"{tag} angles covered: move the board to empty parts of the frame"
            elif k != lead:
                hint = f"{tag} {done}/{len(GUIDE_TARGETS)} angles: follow {'AB'[lead]} for now"
            else:
                hint = f"{tag} {done}/{len(GUIDE_TARGETS)} angles, next: {target[0]}"
                if look["rvec"] is not None:
                    now, want = guide_quads(spec, look, target[1], K)
                    for quad, c in ((now, (0, 220, 255)), (want, (255, 255, 0))):
                        cv2.polylines(panel, [np.int32(quad * scale)], True, c, 2, cv2.LINE_AA)
                    hint += f"  ({_angle_deg(target[1], board_normal(look['rvec'])):.0f} deg off)"
            _text(panel, hint, (8, panel.shape[0] - 40), 0.5, (255, 255, 0))
        panels.append(panel)
    view = np.hstack(panels)

    if check_heading:
        cv2.putText(view, check_heading, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for i, (name, ok, detail) in enumerate(checks):
        _text(view, f"{'OK ' if ok else '.. '}{name:16s}{detail}", (10, 66 + 22 * i), 0.5,
              (0, 200, 0) if ok else (0, 0, 255))
    ready = not check_heading and bool(checks) and all(ok for _, ok, _ in checks)
    cv2.putText(view, footer, (8, view.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 200, 0) if ready else (255, 255, 255), 2, cv2.LINE_AA)
    return view


def capture(out_dir=PAIR_DIR, indices=None, spec=None, width=NATIVE_W, height=NATIVE_H,
            rotate180=True, max_skew_s=0.002, append=False, auto=True, rate_hz=10.0,
            exposure_s=EXPOSURE_S, max_blur_px=MAX_BLUR_PX, override=False, backend=None,
            fps=None, gain=GAIN, auto_exposure=False, src=None, sink=None):
    """Live preview that shoots by itself. SPACE pauses and resumes, q quits.

    ``src``/``sink``: an already-open stereo source and a display to draw on (the viser
    app's). The caller keeps ownership of ``src``; ``sink.show`` returning ``q`` ends it.

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
    frames land inside the target window (2 ms by default), or returns the closest
    pair after bounded retries. The shutter uses that pair's actual skew.

    ``exposure_s`` and ``gain`` are SET on every ELP over UVC once the streams are open, with
    every automatic function off (`camera/uvc.apply`, section 16.4a), and left that way.
    The read-back goes into `meta.json`. ``exposure_s=None`` leaves the cameras as
    they are unless ``auto_exposure=True`` explicitly restores auto exposure.
    """

    spec = spec or SPEC
    out_dir = Path(out_dir)
    if not (override or append) and (out_dir / "meta.json").exists():
        meta = read_meta(out_dir)
        print(f"reusing {out_dir}: {meta['n_pairs']} pair(s), {meta['n_solo']} solo, "
              f"board {meta['board']['board']}, shot {meta['created']}")
        print("  capture(..., override=True) re-shoots it, append=True adds to it")
        return out_dir
    # None means 'the ELPs, as of now' -- resolved here rather than by the caller,
    # because a caller that resolved it earlier is holding a stale list.
    idx = (identify.elp_indices(backend=backend) if indices is None
           else [indices] if isinstance(indices, int) else list(indices))
    tags = "AB"[:len(idx)]
    for tag in tags:
        (out_dir / tag).mkdir(parents=True, exist_ok=True)

    owns_src = src is None
    if not owns_src:
        caller_skew, src.max_skew_s = src.max_skew_s, max_skew_s
    src = src or (sources.open_source(f"camera:{idx[0]}", width=width, height=height, fps=fps, grayscale=True,
                               rotate180=sources.rotations(rotate180, 1)[0], backend=backend) if len(idx) == 1 else
           sources.open_stereo([f"camera:{i}" for i in idx], max_skew_s=max_skew_s,
                               width=width, height=height, fps=fps, grayscale=True,
                               rotate180=rotate180, backend=backend))

    for cam, i in zip(getattr(src, "sources", [src]), idx):
        got = cam.actual
        if (got["width"], got["height"]) != (width, height):
            if owns_src:
                src.close()
            raise OSError(
                f"camera index {i} delivered {got['width']}x{got['height']}, not "
                f"{width}x{height}. That is not an ELP -- run camera/identify.py to "
                f"see what is connected, or pass indices=None to resolve them here."
            )

    settings = None
    if exposure_s or auto_exposure:
        try:
            settings = uvc.configure(exposure_s, gain, auto_exposure=auto_exposure)
        except Exception:
            if owns_src:
                src.close()
            raise
        print("camera settings (UVC): " + "; ".join(map(str, settings)))
    # Auto exposure is not measured per frame. Use at least a requested frame period
    # for the still-board blur gate; never present this bound as an exposure reading.
    gate_exposure_s = exposure_s or max(AUTO_EXPOSURE_S, 1.0 / fps if fps else 0.0)

    old = [f for tag in tags for f in (out_dir / tag).glob("*.png")]
    n = len(list((out_dir / tags[0]).glob("pair_*.png"))) if append else 0
    m = [len(list((out_dir / tag).glob("solo_*.png"))) if append else 0 for tag in tags]
    if old:
        print(f"{out_dir} already holds {len(old)} image(s); "
              + ("continuing from there" if append else "override: they will be replaced"))
    saved, solo, rows = [], [[] for _ in idx], []
    prev, t_prev, t_try = [None] * len(idx), None, 0.0
    period = 1.0 / rate_hz if rate_hz else 0.0
    # Renders in the output cell under Jupyter, where a HighGUI window never opens.
    # Opened on the main thread; see sources.Sink.open.
    sink = (sink or sources.Sink("calibration capture", hz=20)).open()
    preview_width = 480 if sink.inline else None
    checks = []
    last_detection = None
    pair_R = None   # B -> A, from the last frame both cameras posed; aims the pair guide

    def show_position(frames):
        if sink.due():
            # Detect on the displayed frames. Preview observations never enter motion
            # history, shutter gates, saved images, or the final verification report.
            live_looks = [_look(spec, frame) for frame in frames]
            heading = ("Awaiting first shutter check" if last_detection is None else
                       f"Last shutter check ({time.monotonic() - last_detection:.1f}s ago)")
            footer = f"saved {n} pair   live corners | matching A/B   interrupt = stop"
            sink.show(_overlay(spec, frames, live_looks, checks, footer, preview_width,
                                  check_heading=heading, saved=solo, pairs=saved,
                                  pair_R=rel_rotation(live_looks, pair_R)))

    # Also render while the stereo reader searches for a synchronised pair. Only
    # the reader's returned pair goes to the shutter below.
    live_stereo = sink.inline and len(idx) > 1
    if live_stereo:
        target = f"配对目标 {max_skew_s * 1000:g} ms" if max_skew_s is not None else "无配对时间限制"
        print(f"实时角点预览 · 每路 {min(width, preview_width)} px · {target} · Interrupt 结束", flush=True)
    elif not sink.inline:
        print(f"{'auto' if auto else 'paused'}: SPACE pauses and resumes, q quits")
    try:
      try:
        while True:
            item = src.read(on_skew_retry=show_position) if live_stereo else src.read()
            if item is None:
                print("source ended")
                break
            t, payload = item
            frames = list(payload) if isinstance(payload, (list, tuple)) else [payload]
            skew_s = getattr(src, "last_skew", 0.0)
            if not np.isfinite(skew_s):
                skew_s = max_skew_s

            if t_prev is None:   # the light check CLAUDE.md asks for: healthy ~59/82, max 235
                print("frame mean/max: " + ", ".join(f"{f.mean():.0f}/{f.max()}" for f in frames))
            dt, t_prev = (t - t_prev if t_prev else None), t
            looks = [_look(spec, f, prev[k], dt) for k, f in enumerate(frames)]
            prev = [lk["by_id"] for lk in looks]
            ok, checks = gates(spec, looks, skew_s, saved, gate_exposure_s, max_blur_px,
                               variety=solo[0])
            pair_R = rel_rotation(looks, pair_R)
            last_detection = time.monotonic()

            due = auto and (t - t_try) >= period
            if due:
                t_try = t
            keep = [k for k, lk in enumerate(looks)
                    if due and not ok and solo_ok(lk, solo[k], gate_exposure_s, max_blur_px, spec)]
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
            footer += ("   interrupt = stop" if sink.inline else
                       "   SPACE = pause, q = quit" if auto else
                       "   SPACE = resume, q = quit")
            # After a long retry burst the best pair can be older than the live
            # image already displayed. Use it for the shutter, not for playback.
            fresh_view = not live_stereo or time.monotonic() - t < 0.15
            key = (sink.show(_overlay(spec, frames, looks, checks, footer, preview_width, saved=solo,
                                    pairs=saved, pair_R=pair_R))
                   if fresh_view and sink.due() else -1)
            if key == ord("q"):
                break
            if key == ord(" "):
                auto = not auto
                print("auto capture on" if auto else "paused")
      except KeyboardInterrupt:
        # A cell has no q. Interrupting must still leave a finished bag behind,
        # so it breaks the loop rather than unwinding past the writes below.
        print("\ninterrupted")
    finally:
        sink.close()
        stats = src.skew_stats() if hasattr(src, "skew_stats") else {}
        if owns_src:
            src.close()
        else:
            src.max_skew_s = caller_skew

    with open(out_dir / "pairs.csv", "w") as fh:
        for k, v in stats.items():
            fh.write(f"# skew_{k}, {v}\n")
        fh.write("index,t_capture,skew_ms,speed_px_s,"
                 + ",".join(f"incidence_{tag.lower()}_deg" for tag in tags) + "\n")
        for r in rows:
            fh.write(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + "\n")
    write_meta(out_dir, spec, idx, (width, height), rotate180, n, m, stats, exposure_s, settings)
    print(f"\n{n} pair(s) in {out_dir}, plus "
          + ", ".join(f"{c} solo for {tag}" for tag, c in zip(tags, m)))
    if rows:
        sk = np.array([r[2] for r in rows])
        print(f"  skew of the pairs saved: median {np.median(sk):.2f} ms, "
              f"worst {sk.max():.2f} ms")
    return out_dir
