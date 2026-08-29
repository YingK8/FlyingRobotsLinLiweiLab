#!/usr/bin/env python3
"""
The rim's effective radius, from flight footage and nothing else.

`estimator.RADIUS_BY_APPEARANCE` is where the threshold cuts a shaded edge, not where the
mesh says the rim is, so it is a property of the rig *and* of `segment.DARK_THRESH`
together and has to be re-measured when either moves.

Two cameras 83 deg apart each back-project the rim ellipse to a position of its own, and
those two answers only agree at the right radius: too small and each camera puts the robot
too close along its own axis, too large and too far, and the axes point different ways so
the error does not cancel. Sweep the radius, take the minimum of the median disagreement.
No renderer, no second target, and every flight already on disk is a measurement.

    python fit_radius.py                        # every flight under results/flights
    python fit_radius.py <flight> [<flight>]    # named ones
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import background as bgmod  # noqa: E402
import cv2  # noqa: E402
import rig as rigmod  # noqa: E402
from record import open_recording  # noqa: E402
from shape import CentreCalibration, TiltCalibration  # noqa: E402
from stereo import StereoPoseEstimator  # noqa: E402

FLIGHTS = HERE.parents[1] / "results" / "flights"
STEP = 12          # every twelfth stereo frame: the sweep needs spread, not every frame


def _frames(flight, step=STEP):
    caps, _ = open_recording(flight)
    out = []
    try:
        for i in range(100000):
            fs = []
            for c in caps:
                ok, f = c.read()
                if ok:
                    fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
            if len(fs) < 2:
                break
            if i % step == 0:
                out.append((i, fs))
    finally:
        for c in caps:
            c.release()
    return out


def discrepancy(flight, radii, rig=None, plates=None, frames=None):
    """``{radius: median cross-view disagreement in mm}`` for one flight."""

    rig = rig or rigmod.StereoRig.load(rigmod.DEFAULT_PATH)
    plates = bgmod.load_for_flight(flight) if plates is None else plates
    frames = _frames(flight) if frames is None else frames
    out = {}
    for r in radii:
        est = StereoPoseEstimator(rig, radius_mm=r, backgrounds=plates,
                                  tilt_cal=TiltCalibration.load(),
                                  centre_cal=CentreCalibration.load())
        d = [p.discrepancy_mm for i, fs in frames
             if (p := est.update(fs, t=i / 60.0, frame_index=i)) is not None]
        out[r] = float(np.median(d)) if d else float("nan")
    return out


def fit(flights, radii=np.arange(9.8, 11.01, 0.1)):
    """Sweep every flight, print the curves, return the radius each one prefers."""

    radii = [round(float(r), 3) for r in radii]
    rig = rigmod.StereoRig.load(rigmod.DEFAULT_PATH)
    print("median cross-view discrepancy, mm")
    print(f"{'flight':>18s} " + " ".join(f"{r:>6.1f}" for r in radii))
    best = {}
    for f in flights:
        d = discrepancy(f, radii, rig=rig)
        vals = [d[r] for r in radii]
        if not np.isfinite(vals).any():
            print(f"{Path(f).name:>18s}   no poses at any radius -- wrong appearance for "
                  f"this footage, or no rig")
            continue
        best[Path(f).name] = radii[int(np.nanargmin(vals))]
        print(f"{Path(f).name:>18s} " + " ".join(f"{v:>6.2f}" for v in vals)
              + f"   -> {best[Path(f).name]}", flush=True)
    if best:
        agree = len(set(best.values())) == 1
        print(f"\n{'all flights agree on' if agree else 'flights disagree:'} "
              f"{sorted(set(best.values()))} mm"
              + ("" if agree else "  -- the rig or the threshold moved between them"))
    return best


if __name__ == "__main__":
    import segment
    print(f"appearance {segment.APPEARANCE}, dark threshold {segment.DARK_THRESH}\n")
    args = [Path(a) for a in sys.argv[1:]]
    fit(args or sorted(d for d in FLIGHTS.iterdir()
                       if d.is_dir() and (d / "A" / "A.mp4").exists()))
