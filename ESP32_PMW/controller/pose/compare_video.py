#!/usr/bin/env python3
"""Side-by-side flight comparison, synced on the ramp and labelled with live frequency.

Two takes rarely start at the same moment: `record` begins when `fly()` opens the cameras,
but the ramp is not commanded until PRIME_FIXES consecutive fixes exist, which takes a
different time each run. Cutting both from file start would show one run seconds ahead of
the other and read as a difference in behaviour. So each clip is cut from ITS OWN
`seq=go` -- the first tick whose logged `f_hz` is non-zero -- and the drive frequency
drawn on each frame comes from that run's CSV, interpolated onto the frame's timestamp.

Labels are drawn with OpenCV, not ffmpeg `drawtext`: the ffmpeg here is built without
freetype, so `drawtext` does not exist.

    uv run python controller/pose/compare_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _ramp_start(csv_path):
    """(t of the first commanded drive, t of the last logged tick), seconds from loop start."""

    from controller.control import takeoff_report as R

    d = R.load(csv_path)
    t, f = d["t"], d["f_hz"]
    m = np.isfinite(f) & (f > 0.5)
    if not m.any():
        raise SystemExit(f"{csv_path}: never commanded a drive frequency")
    return float(t[m][0]), float(t[np.isfinite(f)][-1]), d


def _band(img, lines, y0=6):
    """Dark band with white text, top-left. Readable over a bright pad or a dark rig."""

    # FIXED line pitch, not per-line text height: the heights differ with scale, so
    # stacking by each line's own height let a small line ride up into the one above.
    pitch = 24
    for i, (txt, scale, thick) in enumerate(lines):
        (w, h), _ = cv2.getTextSize(txt, FONT, scale, thick)
        y = y0 + pitch * (i + 1)
        cv2.rectangle(img, (6, y - h - 5), (6 + w + 12, y + 6), (0, 0, 0), -1)
        cv2.putText(img, txt, (12, y), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def build(pairs, out=None, size=(640, 400), fps=60.0, tail_s=6.0, sync="time"):
    """`pairs` is [(flight_dir, csv_path, title, subtitle), ...] -- one panel each.

    `sync="freq"` steps a COMMON DRIVE FREQUENCY and shows each run at the moment it
    passed that frequency. Use it whenever the runs flew different profiles: 082621
    climbed to 210 Hz over 20 s and 092547 to 110 Hz over 12 s, so syncing on elapsed
    time put 25.9 Hz beside 41.7 Hz -- the same instant, but not the same ramp, which is
    exactly the misleading comparison this is meant to avoid.

    `sync="time"` keeps the old behaviour, correct only when both runs flew one profile.
    """

    caps, meta = [], []
    for flight, csv, title, sub in pairs:
        v = Path(flight) / "A" / "A.mp4"
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {v} -- is the take finalised? (moov atom)")
        t0, t_end, d = _ramp_start(csv)
        caps.append(cap)
        meta.append(dict(t0=t0, dur=t_end - t0 + tail_s, d=d, title=title, sub=sub,
                         src_fps=cap.get(cv2.CAP_PROP_FPS) or 60.0))
    # Common frequency sweep: from the highest of the runs' starts to the lowest of their
    # maxima, so every step exists in every run and nothing is extrapolated.
    f_lo = max(float(np.nanmin(m["d"]["f_hz"][np.isfinite(m["d"]["f_hz"])
                                              & (m["d"]["f_hz"] > 0.5)])) for m in meta)
    f_hi = min(float(np.nanmax(m["d"]["f_hz"][np.isfinite(m["d"]["f_hz"])])) for m in meta)
    dur = min(m["dur"] for m in meta)
    out = Path(out or ROOT / "results/takeoff/trim_vs_untrimmed.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = size
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps,
                         (W * len(pairs), H), True)
    if not wr.isOpened():
        raise SystemExit("no avc1 writer on this build")

    # SEQUENTIAL reads, one source frame per output frame. Seeking per frame with
    # CAP_PROP_POS_MSEC lands on the nearest decodable frame, so a slow stretch of the
    # sync curve resolves to the SAME source frame many times over and the result
    # stutters -- which is what frequency-sync did through the 6 s capture segment: a
    # linear sweep over 102 Hz gave that 2-8 Hz stretch only ~80 frames for 6 s of
    # footage, about 13 effective fps.
    #
    # `sync="time"` therefore plays both runs at natural speed from their own `seq=go`.
    # The cost is that the panels only hold the SAME FREQUENCY if both runs flew the same
    # ramp; 082621 climbed to 210 Hz over 20 s and 092547 to 110 Hz over 12 s, so their
    # frequencies diverge and the on-frame readout shows it rather than hiding it. Two
    # runs of the same profile are the only way to have both, and that needs a re-fly.
    dur_total = dur + tail_s
    n = int(dur_total * fps)
    for cap, m in zip(caps, meta):
        cap.set(cv2.CAP_PROP_POS_MSEC, m["t0"] * 1000.0)
        m["src_i"] = 0
    for k in range(n):
        t = k / fps
        panels = []
        for cap, m in zip(caps, meta):
            want = int(round(t * m["src_fps"]))
            frame = m.get("last")
            while m["src_i"] <= want:
                ok, fr = cap.read()
                m["src_i"] += 1
                if not ok:
                    break
                frame = fr
            m["last"] = frame
            if frame is None:
                frame = np.zeros((H, W, 3), np.uint8)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (W, H))
            d = m["d"]
            t_src = m["t0"] + t
            t_last = float(d["t"][np.isfinite(d["t"])][-1])
            if t_src <= t_last:
                f = float(np.interp(t_src, d["t"], np.nan_to_num(d["f_hz"])))
                z = float(np.interp(t_src, d["t"], np.nan_to_num(d["z_mm"])))
                read = f"drive {f:6.1f} Hz     z {z:+5.2f} mm     t {t:4.1f} s"
                phase = "ramp"
            else:
                read = f"coils stopped                      t {t:4.1f} s"
                phase = "run ended"
            _band(frame, [(m["title"], 0.52, 1), (m["sub"], 0.44, 1),
                          (read, 0.52, 1), (phase, 0.42, 1)])
            panels.append(frame)
        wr.write(np.hstack(panels))
    wr.release()
    for c in caps:
        c.release()
    print(f"wrote {out}  ({dur_total:.1f}s at {fps:.0f} fps, time-synced on each run's "
          f"ramp start, 1 source frame per output frame, {len(pairs)} panels)")
    return out


def demo():
    build([
        (ROOT / "results/flights/2026-09-01_082624",
         ROOT / "results/takeoff/20260901_082621.csv",
         "UNTRIMMED  (no weak direction)",
         "lifts 126 Hz, then departs sideways"),
        (ROOT / "results/flights/2026-09-01_092550",
         ROOT / "results/takeoff/20260901_092547.csv",
         "TRIMMED  weak dir 0 deg (coil A), strength 0.40",
         "tilt ~40% lower through the climb"),
    ])


if __name__ == "__main__":
    demo()
