"""Tests for the synthetic backgrounds.

One of these matters far more than the others: **nothing generated may reach the
segmenter's threshold**.  If a background crosses grey 128 it stops being a
background and becomes foreground, the convex hull swallows it, and every pose
downstream is wrong by tens of millimetres -- while the images still look
perfectly reasonable.  That is exactly the kind of failure a validation set is
supposed to expose rather than contain, so it is asserted exhaustively rather
than spot-checked.

Deliberately dependency-free (no pytest) to match the repo's loose-script
convention.

Run: uv run python controller/pose/test_backgrounds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Scratch may depend on the whole pipeline, so all four stages go on the path.
# (This is the one direction the layering allows to be unrestricted: ai/ is not
# a stage, it is what the stages are exercised by.)
_C = HERE.parents[1] / "controller"
sys.path[:0] = [str(HERE), str(HERE.parent / "validation"),
                str(_C / "pose"), str(_C / "calib"), str(_C / "camera")]

import backgrounds as bg  # noqa: E402

SHAPES = [(120, 160), (480, 640), (800, 1280)]
TRIALS = 200


def test_never_reaches_threshold():
    """The invariant. Every generator, many seeds, several shapes."""
    worst = {}
    for name, fn in bg.GENERATORS.items():
        peak_seen = 0
        for i in range(TRIALS):
            rng = np.random.default_rng(i)
            shape = SHAPES[i % len(SHAPES)]
            peak = bg.EDGE_PEAK if i % 2 else bg.CORE_PEAK
            field = fn(shape, rng, peak=peak)
            assert field.dtype == np.uint8, f"{name} returned {field.dtype}"
            assert field.shape == shape, f"{name} returned {field.shape} for {shape}"
            hi = int(field.max())
            peak_seen = max(peak_seen, hi)
            assert hi < bg.THRESHOLD, (
                f"{name} reached grey {hi} at seed {i}, threshold is {bg.THRESHOLD}"
            )
        worst[name] = peak_seen
    detail = ", ".join(f"{k} {v}" for k, v in sorted(worst.items()))
    print(f"  below threshold     max grey per generator: {detail} (limit {bg.THRESHOLD})")


def test_respects_the_requested_peak():
    """A core-tier request must not quietly produce an edge-tier field."""
    for name, fn in bg.GENERATORS.items():
        for i in range(60):
            field = fn((240, 320), np.random.default_rng(1000 + i), peak=bg.CORE_PEAK)
            assert int(field.max()) <= bg.CORE_PEAK, (
                f"{name} exceeded the core peak {bg.CORE_PEAK} with {field.max()}"
            )
    print(f"  peak honoured       core <= {bg.CORE_PEAK}, edge <= {bg.EDGE_PEAK}")


def test_structured_generators_have_structure():
    """A 'gradient' that comes out flat is a silent failure of the whole set.

    Checks the field actually varies -- otherwise a bug that collapsed every
    generator to a constant would pass the threshold test perfectly and quietly
    undo the entire point of the exercise.
    """
    flat_like = {"flat"}
    for name, fn in bg.GENERATORS.items():
        spans = []
        for i in range(40):
            field = fn((240, 320), np.random.default_rng(2000 + i), peak=bg.EDGE_PEAK)
            spans.append(int(field.max()) - int(field.min()))
        median_span = float(np.median(spans))
        if name in flat_like:
            assert median_span == 0, f"{name} should be uniform, spans {median_span}"
        else:
            assert median_span >= bg.MIN_SPAN, (
                f"{name} is effectively flat: median span {median_span}"
            )
    print("  structure present   every non-flat generator varies across the frame")


def test_deterministic_from_seed():
    """Same seed, same field -- otherwise a dataset cannot be reproduced."""
    for name, fn in bg.GENERATORS.items():
        a = fn((160, 240), np.random.default_rng(7), peak=bg.CORE_PEAK)
        b = fn((160, 240), np.random.default_rng(7), peak=bg.CORE_PEAK)
        assert np.array_equal(a, b), f"{name} is not reproducible from its seed"
    print("  reproducible        identical seeds give identical fields")


def test_sample_mixes_and_labels():
    """`sample` must return the name it used, or a run cannot be diagnosed."""
    rng = np.random.default_rng(3)
    seen = {}
    for _ in range(400):
        field, name = bg.sample((120, 160), rng, peak=bg.CORE_PEAK)
        assert name in bg.GENERATORS, f"unknown generator name {name!r}"
        assert int(field.max()) < bg.THRESHOLD
        seen[name] = seen.get(name, 0) + 1
    missing = set(bg.GENERATORS) - set(seen)
    assert not missing, f"generators never drawn in 400 samples: {missing}"
    order = ", ".join(f"{k} {v / 4:.0f}%" for k, v in sorted(seen.items()))
    print(f"  weighted mix        {order}")


def test_contrast_is_meaningful_at_the_edge_tier():
    """The edge tier has to be harder than the core one, not merely different."""
    rng = np.random.default_rng(11)
    core = [bg.bands((240, 320), rng, peak=bg.CORE_PEAK).std() for _ in range(40)]
    rng = np.random.default_rng(11)
    edge = [bg.bands((240, 320), rng, peak=bg.EDGE_PEAK).std() for _ in range(40)]
    assert np.median(edge) > np.median(core), (
        f"edge bands are not higher contrast: {np.median(edge):.1f} vs "
        f"{np.median(core):.1f}"
    )
    print(f"  edge is harder      band contrast {np.median(core):.1f} -> "
          f"{np.median(edge):.1f} grey sd")


if __name__ == "__main__":
    print("background generators")
    fail = 0
    for fn in (
        test_never_reaches_threshold,
        test_respects_the_requested_peak,
        test_structured_generators_have_structure,
        test_deterministic_from_seed,
        test_sample_mixes_and_labels,
        test_contrast_is_meaningful_at_the_edge_tier,
    ):
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
            fail += 1
    print("all passed" if not fail else f"{fail} FAILED")
    sys.exit(1 if fail else 0)
