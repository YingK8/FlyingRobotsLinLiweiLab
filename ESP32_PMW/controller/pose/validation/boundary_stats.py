"""Compare the simulated silhouette boundary against real photographs.

Every accuracy result in this package is measured in simulation, and every
robustness mechanism in it -- the axial re-weighting, the one-sided loss, the
trimming experiments -- exists to survive a *rough* boundary. So the question of
whether the simulated boundary is as rough as a real one is not a detail: it
decides whether any of those results transfer. It had never been asked, and when
it was, the answer was no.

The metric is **hull vertices per pixel of perimeter**. A convex hull gains a
vertex at every outward excursion of the outline, so the count measures
roughness; dividing by perimeter makes it dimensionless, so a 150 px rendered
rotor and a 440 px photographed one are directly comparable with nothing to tune.

Measured with `render_frames` against `vision/drone_orientation/*.jpeg`:

    simulated (core)   0.1249   [0.0980 - 0.1369]
    simulated (edge)   0.1133   [0.0881 - 0.1367]
    real               0.0221   [0.0139 - 0.0470]

The simulated boundary is **5.7x rougher**, and the ranges do not overlap: the
roughest real capture is smoother than the smoothest simulated frame. A method
whose job is to reject boundary outliers is therefore being graded on a problem
substantially harder than the one it will face -- which flatters it.

**Neither side is yet a valid reference for flight.** The photographs are of a
stationary robot under good light, so they understate the roughness of a frame
blurred by a 310-350 Hz spin; the renders carry noise drawn from an exposure
model that has never been checked against the camera's measured read noise. The
honest statement is that the two disagree by 5.7x and that the disagreement is
unexplained.

This module exists so that comparison is a standing check rather than a one-off.
When real captures under flight conditions exist, run it: if the simulated and
measured distributions overlap, simulation results carry weight for the real
system, and until they do, they do not.

Run: uv run python controller/pose/validation/boundary_stats.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import segment as segmod  # noqa: E402

REAL_DIR = HERE.parents[1] / "vision" / "drone_orientation"


def _perimeter(major, minor):
    """Ramanujan's approximation -- exact enough well past our eccentricities."""
    a, b = major / 2.0, minor / 2.0
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def roughness(gray, thresh=segmod.THRESH):
    """Hull vertices per pixel of perimeter, or None if nothing is detected.

    Dimensionless by construction, so it compares across object sizes and
    resolutions without a scale factor to choose.
    """
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    hull, _ = segmod.silhouette_hull(mask)
    if hull is None or len(hull) < 5:
        return None
    fit = segmod.fit_ellipse(hull, axial=False)
    if fit is None:
        return None
    (_, _), (major, minor), _ = fit[0]
    per = _perimeter(major, minor)
    if not np.isfinite(per) or per <= 0:
        return None
    return len(hull) / per, major


def real_captures(directory=REAL_DIR):
    """Roughness of every real photograph available."""
    out = []
    for f in sorted(Path(directory).glob("*.jpeg")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        r = roughness(img)
        if r is not None:
            out.append((f.name, r[0], r[1]))
    return out


def simulated(n_poses=16, seed=20260812):
    """Roughness of rendered frames, both condition tiers."""
    import resolution_sweep as rsweep  # noqa: E402  (imports a GL context)

    frames, _ = rsweep.render_frames(n_poses, seed)
    out = {}
    for tier, entries in frames.items():
        vals = []
        for entry in entries:
            for im in entry[0]:
                g = im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                r = roughness(g)
                if r is not None:
                    vals.append(r)
        out[tier] = np.array(vals) if vals else np.empty((0, 2))
    return out


def _summary(name, v):
    if not len(v):
        return f"  {name:<18} no detections"
    return (f"  {name:<18} n={len(v):3d}  median {np.median(v):.5f}  "
            f"[{np.percentile(v, 10):.5f} – {np.percentile(v, 90):.5f}]")


def main(argv=None):
    real = real_captures()
    print("Boundary roughness — hull vertices per pixel of perimeter\n")
    print("real captures:")
    for name, r, major in real:
        print(f"  {name:<14} major {major:6.0f} px   {r:.5f}")
    rv = np.array([r for _, r, _ in real])
    print(_summary("REAL", rv))

    print("\nsimulated:")
    sim = simulated()
    for tier, arr in sim.items():
        print(_summary(f"SIM {tier}", arr[:, 0] if len(arr) else arr))

    if len(rv) and any(len(a) for a in sim.values()):
        sv = np.concatenate([a[:, 0] for a in sim.values() if len(a)])
        ratio = np.median(sv) / np.median(rv)
        print(f"\n  simulated / real  =  {ratio:.1f}x")
        overlap = (sv.min() <= rv.max()) and (rv.min() <= sv.max())
        print(f"  distributions overlap: {'yes' if overlap else 'NO'}")
        if not overlap:
            print("  -> the roughest real frame is smoother than the smoothest "
                  "simulated one.\n     Robustness results measured in simulation "
                  "do not transfer unchecked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
