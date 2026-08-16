"""Monochrome backgrounds: gradients and bands, always darker than the robot.

The validation set used flat grey at three levels.  Real backdrops are not flat:
they have falloff from a light, edges, and repeating structure.  This generates
those, in greyscale, because the camera is greyscale.

**Everything here stays below the segmenter's threshold.**  That is a hard
invariant, asserted in `test_backgrounds.py`, and it is a deliberate choice about
what these images are for.  A background that crosses grey 128 stops being a
background: it becomes foreground, the convex hull swallows it, and the pose
error goes to tens of millimetres.  That cliff is real and already characterised
in `README.md` -- position error 232 mm at background grey 0.5 -- but it is a
*segmentation* failure, and mixing it in here would drown the geometric signal
this set exists to measure.  Difficulty comes from structure and contrast
instead.

The ceilings are grounded in the only real dark-backdrop photographs in the
repo.  `vision/drone_orientation/drone1.jpeg` and `drone3.jpeg` are the physical
robot on a matte black cloth; excluding the robot, their background maxima are
79 and 71 grey, with **zero** pixels above 128.  So:

* ``CORE_PEAK = 90`` -- a little above what the real backdrop does.
* ``EDGE_PEAK = 120`` -- harder than anything real, still sub-threshold.

Each generator returns ``uint8`` of the requested shape.  They are pure functions
of the supplied ``rng``, so a dataset is reproducible from its seed.
"""

from __future__ import annotations

import math

import numpy as np

# The segmenter's threshold. Nothing generated here may reach it.
THRESHOLD = 128
CORE_PEAK = 90
EDGE_PEAK = 120

# Below this the field is indistinguishable from black at 8-bit precision and
# the generator has no effect worth measuring.
MIN_SPAN = 4


def _finish(field, peak):
    """Clamp to ``[0, peak]`` and return uint8.

    The clamp is the invariant, applied in one place rather than trusted to each
    generator's arithmetic. ``peak`` is passed rather than assumed so the caller
    chooses the tier, and it is itself clamped below the threshold so no caller
    can opt out of the guarantee.
    """
    peak = int(min(peak, THRESHOLD - 1))
    out = np.clip(field, 0.0, float(peak))
    return out.astype(np.uint8)


def _axes(shape):
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    return x / max(w - 1, 1), y / max(h - 1, 1)


def flat(shape, rng, peak=CORE_PEAK):
    """A uniform field. The original behaviour, kept for comparability."""
    return _finish(np.full(shape, rng.uniform(0.0, peak)), peak)


def gradient(shape, rng, peak=CORE_PEAK):
    """Linear ramp at a random orientation between two random levels."""
    x, y = _axes(shape)
    ang = rng.uniform(0.0, 2.0 * math.pi)
    t = x * math.cos(ang) + y * math.sin(ang)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-9)
    lo, hi = sorted(rng.uniform(0.0, peak, 2))
    if hi - lo < MIN_SPAN:
        hi = min(peak, lo + MIN_SPAN)
    return _finish(lo + (hi - lo) * t, peak)


def vignette(shape, rng, peak=CORE_PEAK):
    """Radial falloff, bright at the centre or bright at the edges.

    The second case is the one worth having: a backdrop lit from behind the
    camera is brightest where the robot is *not*, which puts the strongest
    background gradient right where the silhouette's boundary sits.
    """
    x, y = _axes(shape)
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    r = np.hypot(x - cx, y - cy)
    r = r / max(r.max(), 1e-9)
    lo, hi = sorted(rng.uniform(0.0, peak, 2))
    if hi - lo < MIN_SPAN:
        hi = min(peak, lo + MIN_SPAN)
    profile = r if rng.random() < 0.5 else (1.0 - r)
    return _finish(lo + (hi - lo) * profile, peak)


def bands(shape, rng, peak=CORE_PEAK):
    """Sinusoidal stripes at a random orientation, frequency and phase.

    Frequency is drawn in cycles across the frame.  The interesting range is a
    few cycles: much lower and it is a gradient, much higher and the morphology
    in `segment.py` averages it away before the hull sees anything.
    """
    x, y = _axes(shape)
    ang = rng.uniform(0.0, math.pi)
    freq = rng.uniform(1.5, 9.0)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    t = np.sin(2.0 * math.pi * freq * (x * math.cos(ang) + y * math.sin(ang)) + phase)
    mid = rng.uniform(0.25 * peak, 0.75 * peak)
    amp = min(mid, peak - mid) * rng.uniform(0.4, 1.0)
    return _finish(mid + amp * t, peak)


def square_bands(shape, rng, peak=CORE_PEAK):
    """Hard-edged stripes.

    Worse than sinusoidal ones on purpose.  A step edge survives the 3x3 opening
    and 7x7 closing in `segment.py` with its gradient intact, so it is the
    background structure most likely to be mistaken for a real boundary.
    """
    x, y = _axes(shape)
    ang = rng.uniform(0.0, math.pi)
    freq = rng.uniform(1.0, 6.0)
    phase = rng.uniform(0.0, 1.0)
    t = np.sin(2.0 * math.pi * freq * (x * math.cos(ang) + y * math.sin(ang))
               + 2.0 * math.pi * phase)
    duty = rng.uniform(0.3, 0.7)
    lo, hi = sorted(rng.uniform(0.0, peak, 2))
    if hi - lo < MIN_SPAN:
        hi = min(peak, lo + MIN_SPAN)
    return _finish(np.where(t > (2.0 * duty - 1.0), hi, lo), peak)


def mixed(shape, rng, peak=CORE_PEAK):
    """A gradient with bands laid over it, each at reduced amplitude.

    Generated at half peak each so the sum still respects the ceiling without
    the clamp having to do the work -- a clamped field has flat regions that are
    not representative of anything.
    """
    g = gradient(shape, rng, peak=peak * 0.6).astype(np.float64)
    b = bands(shape, rng, peak=peak * 0.4).astype(np.float64)
    return _finish(g + b - float(b.mean()) * 0.5, peak)


GENERATORS = {
    "flat": flat,
    "gradient": gradient,
    "vignette": vignette,
    "bands": bands,
    "square_bands": square_bands,
    "mixed": mixed,
}

# Flat is kept common because it is the condition every earlier result was
# measured under; the structured ones share the rest between them.
CORE_WEIGHTS = {"flat": 0.30, "gradient": 0.20, "vignette": 0.15,
                "bands": 0.15, "square_bands": 0.10, "mixed": 0.10}
EDGE_WEIGHTS = {"flat": 0.10, "gradient": 0.15, "vignette": 0.15,
                "bands": 0.25, "square_bands": 0.20, "mixed": 0.15}


def sample(shape, rng, peak=CORE_PEAK, weights=None):
    """One background, drawn from the weighted mix. Returns ``(field, name)``."""
    weights = weights or CORE_WEIGHTS
    names = list(weights)
    probs = np.array([weights[n] for n in names], dtype=np.float64)
    name = names[int(rng.choice(len(names), p=probs / probs.sum()))]
    return GENERATORS[name](shape, rng, peak=peak), name


def contact_sheet(shape=(120, 160), seed=0, peak=CORE_PEAK, cols=6, rows=3):
    """A tiled sample of every generator, for looking at.

    Worth having because these are judged by eye as much as by statistic: a
    generator that is technically in range but visually nothing like a backdrop
    is still the wrong generator.
    """
    rng = np.random.default_rng(seed)
    names = list(GENERATORS)
    tiles = []
    for r in range(rows):
        row = []
        for c in range(cols):
            name = names[(r * cols + c) % len(names)]
            row.append(GENERATORS[name](shape, rng, peak=peak))
        tiles.append(np.hstack(row))
    return np.vstack(tiles)
