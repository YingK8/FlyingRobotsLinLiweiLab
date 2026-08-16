"""Capture the empty-rig frame that `segment.background_mask` subtracts.

The coils, the wires, the support box, the dark room beyond the backdrop and the
backdrop itself are all **fixed to the rig**. One subtraction removes every one of
them, which is why this is preferred over any amount of cleverness about what the
clutter looks like -- and why it is worth a bench step to capture.

It is also, by a wide margin, the cheapest option. Measured on a 1280x800 frame:

    background subtraction     0.056 ms
    backdrop finder (1/4 res)  2.43 ms
    backdrop finder (full res) 48.4 ms

At 1280x800 the camera period is 8.3 ms and `segment()` alone already costs 7.9 ms
single-core, so the difference between 0.06 and 2.4 ms is the difference between a
loop that keeps up with the camera and one that drops every fourth frame.

**A median, not a mean.** The failure this is most likely to suffer is a frame with
the robot half in shot, or a hand still withdrawing. A mean smears that across the
result at low contrast, where it is invisible and permanent; a median is unmoved by
it as long as it is in a minority of frames.

**A stale background is silently wrong**, which is the one real hazard here. Moving
the camera, refocusing, or changing the lighting invalidates it, and nothing
downstream can tell -- the subtraction simply starts reporting the shifted edges as
robot. `--check` measures how much of the frame currently differs from the stored
background and says whether it still looks like the same scene. Re-run after any
change to the rig.

Usage:
    uv run python controller/elp/background.py --index 0 --frames 30
    uv run python controller/elp/background.py --check
"""

from __future__ import annotations

import argparse
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
    """Median of ``n_frames`` grabbed frames, as uint8 grayscale."""
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
    """Fraction of pixels that differ from the background by more than the threshold."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--mode", default=None, help="WxH@FPS, e.g. 1280x800@120")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true",
                    help="compare the live scene against the stored background")
    args = ap.parse_args(argv)

    out = Path(args.out)

    if args.check:
        bg = segmod.load_background(out)
        if bg is None:
            print(f"no background at {out} -- nothing to check")
            return 1
        source, cams = cammod.open_group(args.index, mode=args.mode, grayscale=True)
        try:
            for _ in range(10):                       # let exposure settle
                source.read()
            item = cammod.as_frames(source.read())
            if item is None:
                print("no frames")
                return 1
            gray = item[1][0]
        finally:
            source.close()
        if gray.shape != bg.shape:
            print(f"SIZE MISMATCH: background is {bg.shape[1]}x{bg.shape[0]}, "
                  f"camera is {gray.shape[1]}x{gray.shape[0]}")
            print("the background is unusable at this resolution -- recapture")
            return 1
        frac = differing_fraction(gray, bg)
        print(f"{frac:.1%} of the frame differs from the stored background "
              f"(threshold {segmod.BG_DIFF_THRESH})")
        if frac > STALE_FRACTION:
            print(f"STALE: above {STALE_FRACTION:.0%}, this is the scene having changed, "
                  f"not the robot being in it. Recapture.")
            return 1
        print("looks like the same scene")
        return 0

    source, cams = cammod.open_group(args.index, mode=args.mode, grayscale=True)
    try:
        # Read `actual` while the capture is still open. Afterwards every
        # CAP_PROP query returns -1, which serialises into the sidecar as a
        # plausible-looking record of a mode that was never used.
        actual = cams[0].actual
        print(f"camera {args.index}: {actual}")
        print(f"take the robot out of frame, then capturing {args.frames} frames ...")
        for _ in range(10):                           # settle before measuring
            source.read()
        bg, n = median_background(source, args.frames)
    finally:
        source.close()

    if bg is None:
        print("no frames -- nothing written")
        return 1

    meta = write(bg, out, {"n_frames": n, "index": args.index,
                           "actual": actual,
                           "n_samples": n})
    print(f"wrote {out}  (median of {n} frames, {meta['mode']})")
    print(f"      {out.with_suffix('.json')}")
    print("\nsegment.valid_region will now use background subtraction (0.06 ms) "
          "instead of the backdrop finder (2.4 ms).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
