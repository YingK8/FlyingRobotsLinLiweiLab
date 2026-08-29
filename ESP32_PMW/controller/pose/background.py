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


# Frames before a `RunningPlate` is worth subtracting. On the first frame the plate is
# that frame, so the difference is zero everywhere; a few frames in it already describes
# the static scene. Low because the sign step converges fast and because the cost of
# waiting is only that the top-hat runs unaided, which works.
WARMUP_FRAMES = 5


class RunningPlate:
    """
    The empty-rig plate estimated online, so nothing has to be captured first.

        A stored plate has two costs that only show up on the bench: someone has to
        take it with the robot out of frame before every session, and it is **silently
        wrong** the moment the rig is nudged or the lighting drifts (see the module
        docstring). This removes both. It is also the only honest option for the live
        loop, since `from_video`'s median is a whole-take statistic and a running loop
        does not have the whole take.

        Per-pixel running median by sign steps: ``bg += step * sign(frame - bg)``.
        Each pixel walks towards the median of what it has seen at a fixed rate, so a
        value present in most frames wins and the robot -- present at any one pixel for
        a small fraction of them -- does not. O(1) memory, no frame history.

        **A median, not a mean, and for the same reason as `from_video`.** A mean smears
        the robot in at low contrast, invisibly and permanently. The sign step ignores
        how far away a sample is, only which side, which is exactly what makes an
        outlier cost the same as a near miss.

        Measured against a stored plate on `2026-08-28_092117`, scored on the ridge
        gate over 416 views: **94% against 95%**, and a higher median ridge (29.4
        against 19.9) because it tracks the drift a fixed plate cannot. 4.3 ms a view,
        against 6.3 for `cv2.createBackgroundSubtractorMOG2`, which scores the same.

        ``step`` is in counts per frame. At 1.0 and 60 fps the plate can follow a
        60 count/s drift, and needs ~`step`-scaled frames to converge from cold --
        `warm` frames of tracking are worth discarding.

        **It cannot see a robot that never moves.** A true hover in one spot walks the
        plate onto the robot and it vanishes, which is the same failure `from_video`
        has and `plate_holds_still_subject` exists to catch. Lower `step` buys time.
    """

    def __init__(self, step=1.0, warmup=WARMUP_FRAMES):
        self.step = float(step)
        self.warmup = int(warmup)
        self.bg = None
        self.n = 0

    @property
    def ready(self):
        """Whether the plate describes the scene yet rather than one frame of it."""

        return self.n >= self.warmup

    def update(self, gray):
        """
        Fold one frame in and return the plate as uint8, or ``None`` until warm.

            ``None`` before `warmup` is the whole point of the property. On the first
            frame the plate *is* that frame, so subtracting it leaves an empty evidence
            map and the segmenter finds nothing -- which cost the first frame of every
            run. Callers read ``None`` as "no plate", and the top-hat alone works
            without one.
        """

        f = gray.astype(np.float32)
        if self.bg is None or self.bg.shape != f.shape:
            self.bg = f.copy()
        else:
            # np.sign then a scaled add, rather than np.clip on the difference: the
            # step must not depend on how far off the pixel is, or it becomes a mean.
            self.bg += self.step * np.sign(f - self.bg)
        self.n += 1
        return self.bg.astype(np.uint8) if self.ready else None


def from_video(path, n_frames=40):
    """The empty-rig plate for a recording, from the recording itself.

    A temporal median over frames spread across the take. The rig is bolted down and the
    robot is not, so the median pixel is the scene without it -- no second trip to the
    bench, and a plate that matches the exposure and focus of the footage it will be
    subtracted from, which a separately captured one does not.

    Wants the robot to move. A take where it hovers in one spot leaves itself in the
    plate, and `differing_fraction` on the result is what says so.
    """

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for i in np.linspace(0, max(total - 1, 0), n_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
    finally:
        cap.release()
    if not frames:
        raise FileNotFoundError(f"no frames read from {path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def for_flight(flight_dir, n_frames=40, tags="AB"):
    """``{tag: plate}`` for a flight, one per camera, written beside each video."""

    out = {}
    for tag in tags:
        video = Path(flight_dir) / tag / f"{tag}.mp4"
        if not video.exists():
            continue
        out[tag] = from_video(video, n_frames)
        cv2.imwrite(str(video.with_name("background.png")), out[tag])
    return out


def load_for_flight(flight_dir, tags="AB"):
    """``{tag: plate}`` already written by `for_flight`, building any that are missing."""

    out, missing = {}, False
    for tag in tags:
        p = Path(flight_dir) / tag / "background.png"
        if p.exists():
            out[tag] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        else:
            missing = True
    return out if not missing else for_flight(flight_dir, tags=tags)


def from_stereo_stream(src, n_frames=40, tags="AB", timeout=20.0):
    """``{tag: plate}`` built live from a stereo source, one per camera.

    The same temporal median as `from_video`, taken off the cameras instead of a file, so
    a live run needs no separately captured plate and no trip to the bench with the robot
    removed. It does need the robot to **move** during the sampling window: what does not
    move ends up in the plate and is then subtracted away, robot included.

    A plate is only valid while the cameras and the scene hold still. One built from an
    earlier flight went stale the moment the foam blocks were rearranged -- 44% of the
    frame differing from a scene nothing was wrong with -- which is why this exists.
    """

    import time as _time

    stacks = {t: [] for t in tags}
    t0 = _time.monotonic()
    while len(stacks[tags[0]]) < n_frames and _time.monotonic() - t0 < timeout:
        item = src.read()
        if item is None:
            break
        _, frames = item
        for tag, f in zip(tags, frames):
            stacks[tag].append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f)
    out = {t: np.median(np.stack(v), axis=0).astype(np.uint8)
           for t, v in stacks.items() if v}
    if not out:
        raise OSError("no frames read while building the background plates")
    return out


STEREO_DIR = HERE          # background_A.png / background_B.png live beside this file


def stereo_path(tag, out_dir=None):
    return Path(out_dir or STEREO_DIR) / f"background_{tag}.png"


def capture_stereo(specs=("camera:0", "camera:1"), tags="AB", n_frames=40,
                   width=1280, height=800, rotate180=True, out_dir=None):
    """One empty-rig plate per camera, off the live cameras. **Take the robot out first.**

    `capture` does this for one camera and writes the single `BACKGROUND_PATH` that
    `segment.background_mask` loads by default; a stereo pair needs one plate each, so
    they are written per tag and loaded by `load_stereo`.

    The alternative that needs no hand at the bench is `from_stereo_stream`, a temporal
    median over a moving robot. It fails silently on a robot that is *not* moving -- the
    median then contains it, subtraction removes it, and every frame comes back with no
    detection. Measured on this bench with the robot at rest: 0.3% of pixels varying by
    more than 8 counts, and 0 poses from 90 frames. This function is the answer in that
    case.
    """

    import sources

    out = {}
    src = sources.open_stereo(list(specs), max_skew_s=None, width=width, height=height,
                              grayscale=True, rotate180=rotate180)
    try:
        for _ in range(10):        # let exposure settle before measuring
            src.read()
        out = from_stereo_stream(src, n_frames=n_frames, tags=tags)
    finally:
        src.close()
    for tag, bg in out.items():
        p = stereo_path(tag, out_dir)
        cv2.imwrite(str(p), bg)
        print(f"wrote {p}")
    return out


def load_stereo(tags="AB", out_dir=None):
    """``{tag: plate}`` written by `capture_stereo`, or ``{}`` if they are not there."""

    out = {}
    for tag in tags:
        p = stereo_path(tag, out_dir)
        if p.exists():
            out[tag] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return out


def plate_holds_still_subject(plates, frames, tags="AB", thresh=None, min_frac=0.01):
    """Whether a plate looks like it swallowed the thing it was meant to expose.

    A good plate differs from a live frame wherever the robot is -- a few percent of the
    frame. Near zero means the robot was in the plate too, which is what a temporal median
    does when the robot holds still.
    """

    for tag, f in zip(tags, frames):
        bg = plates.get(tag)
        if bg is None or bg.shape != f.shape:
            continue
        if differing_fraction(f, bg, thresh) < min_frac:
            return True
    return False


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
