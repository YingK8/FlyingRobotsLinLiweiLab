"""
Capture the empty-rig frame that `segment.background_mask` subtracts.

Coils, wires, support box, the room beyond the backdrop and the backdrop itself are
all **fixed to the rig**, so one subtraction removes all of them. It is also the
cheapest option by a wide margin, on a 1280x800 frame:

    background subtraction     0.056 ms
    backdrop finder (1/4 res)  2.43 ms
    backdrop finder (full res) 48.4 ms

The camera period at 1280x800 is 8.3 ms and `segment()` already costs 7.9 ms
single-core, so 0.06 against 2.4 ms decides whether the loop keeps up.

**A median, not a mean**, because the likely failure is a frame with the robot half
in shot or a hand withdrawing. A mean smears that in at low contrast, invisibly and
permanently; a median ignores it while it stays a minority.

**A stale background is silently wrong.** Moving the camera, refocusing or changing
the lighting invalidates it and nothing downstream can tell: the subtraction just
starts reporting shifted edges as robot. Run `check()` after any change to the rig.

Usage:
    background.capture(index=0, frames=30)   # robot out of frame
    background.check()                       # is the stored one still this scene?
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# Pipeline layering: a stage sees only the stages before it, so a forward import
# fails at once instead of quietly creating a cycle. pose is stage 3 of 4.
sys.path[:0] = [str(HERE), str(HERE.parent / "calib"), str(HERE.parent / "camera")]

import elp as cammod  # noqa: E402
import segment as segmod  # noqa: E402

DEFAULT_OUT = HERE.parents[0] / "pose" / "background_dark.png"

# Above this fraction of the frame differing, the stored background no longer
# describes the scene. Deliberately generous: the robot and its rod are a few
# percent of the frame, so anything near 10% is the *scene* having changed, not
# the robot being in it.
STALE_FRACTION = 0.10


def median_background(source, n_frames, timeout=2.0):
    """
    Median of ``n_frames`` grabbed frames, as uint8 grayscale.
    """

    stack = []
    while len(stack) < n_frames:
        item = cammod.as_frames(source.read())
        if item is None:
            break
        f = item[1][0]
        stack.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
    if not stack:
        return None, 0
    return np.median(np.stack(stack), axis=0).astype(np.uint8), len(stack)


def differing_fraction(gray, bg, thresh=None):
    """
    Fraction of pixels that differ from the background by more than the threshold.
    """

    m = segmod.background_mask(gray, bg, thresh)
    return float("nan") if m is None else float(m.mean() / 255.0)


def write(bg, out_path, meta_extra=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bg)
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "controller/elp/background.py",
        "mode": f"{bg.shape[1]}x{bg.shape[0]}",
        "diff_thresh": segmod.BG_DIFF_THRESH,
        "note": "Median of empty-rig frames; consumed by segment.load_background(). "
        "Recapture after moving the camera, refocusing or changing the "
        "lighting -- a stale background is silently wrong. Check with "
        "`background.py --check`.",
    }
    meta.update(meta_extra or {})
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def capture(index=0, mode=None, frames=30, out=DEFAULT_OUT):
    """
    Record the empty scene. Returns the sidecar metadata.

        Take the robot out of frame first: this is the reference `valid_region`
        subtracts against, and a background with the robot in it carves a
        robot-shaped hole out of every later detection.
    """

    out = Path(out)
    source, cams = cammod.open_group(index, mode=mode, grayscale=True)
    try:
        # Read `actual` while the capture is still open. Afterwards every
        # CAP_PROP query returns -1, which serialises into the sidecar as a
        # plausible-looking record of a mode that was never used.
        actual = cams[0].actual
        print(f"camera {index}: {actual}")
        print(f"capturing {frames} frames ...")
        for _ in range(10):  # settle before measuring
            source.read()
        bg, n = median_background(source, frames)
    finally:
        source.close()

    if bg is None:
        raise OSError("no frames -- nothing written")

    meta = write(
        bg, out, {"n_frames": n, "index": index, "actual": actual, "n_samples": n}
    )
    print(f"wrote {out}  (median of {n} frames, {meta['mode']})")
    print(f"      {out.with_suffix('.json')}")
    print(
        "\nsegment.valid_region will now use background subtraction (0.06 ms) "
        "instead of the backdrop finder (2.4 ms)."
    )
    return meta


def check(index=0, mode=None, out=DEFAULT_OUT):
    """
    Is the stored background still this scene? Returns the differing fraction.

        A background goes stale silently: the region it produces stays plausible
        while describing a room that has moved. Above `STALE_FRACTION` the difference
        is the scene having changed rather than the robot being in it.
    """

    out = Path(out)
    bg = segmod.load_background(out)
    if bg is None:
        raise FileNotFoundError(f"no background at {out}")

    source, cams = cammod.open_group(index, mode=mode, grayscale=True)
    try:
        for _ in range(10):  # let exposure settle
            source.read()
        item = cammod.as_frames(source.read())
        if item is None:
            raise OSError("no frames")
        gray = item[1][0]
    finally:
        source.close()

    if gray.shape != bg.shape:
        raise ValueError(
            f"background is {bg.shape[1]}x{bg.shape[0]}, camera is "
            f"{gray.shape[1]}x{gray.shape[0]} -- unusable at this resolution, recapture"
        )

    frac = differing_fraction(gray, bg)
    print(
        f"{frac:.1%} of the frame differs from the stored background "
        f"(threshold {segmod.BG_DIFF_THRESH})"
    )
    if frac > STALE_FRACTION:
        print(
            f"STALE: above {STALE_FRACTION:.0%}, this is the scene having changed, "
            f"not the robot being in it. Recapture."
        )
    else:
        print("looks like the same scene")
    return frac
