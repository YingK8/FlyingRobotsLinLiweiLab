"""
The live tracker: capture and pose in C++, this thin layer in Python.

    `pmw_pose.Tracker` owns both cameras, both grabber queues and the pose worker.
    Python does not see a frame unless it asks for one, and the pose rate no longer
    depends on how fast the interpreter goes round a loop.

    **What this buys is the pairing, not the language.** The cameras already delivered
    205-209 fps at 640x400 through `sources.py` (five flights' `meta.json`, zero drops).
    What capped the pose rate was `StereoCamera.read` *consuming* its slot: it waits for
    a fresh frame from each camera, so a pair costs the slower camera's period and the
    observation rate can never beat one camera's rate. The C++ slot is never consumed,
    which lets a pose fire on every frame from *either* camera -- roughly twice the
    observations at the same resolution, FOV and calibration. `native/src/tracker.h`
    carries the argument and `pose/theory.md` 22 the measurement.

    **The rate ceiling is the sensor, and 300 Hz is above it.** One ELP tops out near
    208 fps at 640x400. Downsampling cannot move that -- binning happens after the USB
    transfer -- and 19.3 measured a 2x2 bin at roughly double the position bias for 6 Hz.
    Interleaving is the only route past one camera's rate on this hardware.

    Everything numeric is the same core `stereo_native` drives, configured from the same
    `native_config`, so there is still exactly one home for every tuning constant.
"""

from __future__ import annotations

import numpy as np

from controller.calib import rig as rigmod
from controller.pose import segment as segmod
from controller.pose import background as bgmod
from controller.pose import stereo_native
from controller.pose.stereo_native import stereo_pose

try:
    import pmw_pose
except ImportError:                                   # pragma: no cover - build-dependent
    pmw_pose = None


def available():
    """Whether the compiled tracker is importable."""

    return pmw_pose is not None and hasattr(pmw_pose, "Tracker")


#: A pair whose two views are further apart than this is refused rather than fused.
#:
#: Two jobs in one number. As an *accuracy* gate it is the skew `refine` cannot correct
#: for (`fuse` does, from the per-view stamps, but the evidence maps it fits are each
#: fixed at their own instant). As a *liveness* gate it is what replaces
#: `MonoCamera.read`'s 2 s timeout: with a slot that is never consumed, a camera that
#: stops delivering would otherwise pair its frozen last frame forever, and here its skew
#: instead grows without bound until every pair is refused and `wait` starts timing out.
#: 1.5 frame periods -- above the one period a healthy free-running pair can reach, well
#: under anything a stopped camera produces.
SKEW_LIMIT_PERIODS = 1.5

#: The frame rate `background.RunningPlate`'s `step` is quoted against.
#:
#: Its docstring says "``step`` is in counts per frame. At 1.0 and 60 fps the plate can
#: follow a 60 count/s drift" -- so the constant is per-frame but the thing it means is
#: per-second, and tripling the pose rate triples how fast the plate adapts. Left alone at
#: 220 Hz it walks at 220 counts/s instead of 60, and "it cannot see a robot that never
#: moves ... lower `step` buys time" then buys 3.7x less of it. Measured: a preview on a
#: stationary robot ran ~5 minutes and then lost 2915 consecutive frames while both
#: cameras stayed fresh.
#:
#: Third instance of one mistake -- see `window_frames` below and `pose/theory.md` 22.8.
#: A constant in frames is a duration in disguise, and this pipeline's frame rate moved.
PLATE_STEP_REF_HZ = 60.0

#: The take every published pose number is measured on (`pose/theory.md` 21.1).
BENCH_TAKE = "results/flights/New Folder With Items/2026-08-29_231418"


def camera_ids(rig):
    """
    The two calibrated cameras' device ids, A first, from the rig itself.

        `camera/identify.py` finds the ELPs by opening each OpenCV index and checking
        the delivered size, because "neither macOS listing enumerates in OpenCV's order"
        and a unique-id "cannot be tied to an OpenCV index". Addressing AVFoundation
        directly, it can be: the rig records the pair it was calibrated against, in
        order, so which camera is A is a fact about the calibration rather than about
        probe order and a moved cable.
    """

    ids = list(rig.meta.get("elp_ids") or ())
    if len(ids) < 2:
        raise ValueError(
            f"rig has no elp_ids; cannot say which camera is A. Devices visible now: "
            f"{pmw_pose.list_cameras() if pmw_pose else '(pmw_pose not built)'}")
    return ids[:2]


class Tracker:
    """
    Two cameras, a pose worker and a latest-pose slot, all in C++.

        Built from a `NativeStereoPoseEstimator` rather than from a pile of arguments,
        so `stereo_native.native_config` stays the single source of every constant and
        the estimator carries the zeroing, the gates and the frame scale that
        `stereo_pose` needs to rebuild a `StereoPose`.

        Not a `StereoPoseEstimator`: it is a *source* of poses, not something you hand
        frames to. `read()` is the whole interface.
    """

    def __init__(self, est, width=640, height=400, fps=210.0, rotate180=True,
                 pair_mode="interleave", max_skew_s=None, ids=None):
        if not available():
            raise ImportError("pmw_pose.Tracker is not built; run `uv sync --extra native`")
        self.est = est
        self.width, self.height, self.fps = int(width), int(height), float(fps)
        self.pair_mode = pair_mode
        self.ids = list(ids) if ids else camera_ids(est.rig)
        self.max_skew_s = (SKEW_LIMIT_PERIODS / self.fps if max_skew_s is None
                           else float(max_skew_s))
        self.frame_index = 0
        self._seq = 0

        # The rig is calibrated at 1280x800 and the loop flies at 640x400, an exact 0.5x
        # decimation. `_match_scale` is what halves fx, fy, cx, cy and the pixel
        # constants; it keys off the frame it is handed, so it is driven here with a
        # blank of the size the cameras are about to deliver rather than being trusted to
        # happen later -- the C++ config is read once, at construction.
        est._match_scale(np.zeros((self.height, self.width), np.uint8))

        cfg = stereo_native.native_config(est)
        # A count, but the thing it means is a duration. `stereo.py:253` calls
        # WINDOW_FRAMES "a quarter second at 60 fps, matching DROPOUT_S", and
        # `window_normal` does gate on DROPOUT_S in seconds -- but the deque is trimmed
        # by count, so the window is only a quarter second at 60 Hz. At 400 Hz the
        # constant would span 36 ms and the branch-flip prior would lose most of its
        # memory. Derived from the rate here; `stereo.py`'s constant is left alone so
        # nothing moves under `native_parity`. See theory.md 22.
        cfg["window_frames"] = max(int(stereo_native.stereo.WINDOW_FRAMES),
                                   int(round(est.dropout_s * self._pose_rate_guess())))

        # Same correction, same reason: hold the plate's adaptation at the counts per
        # SECOND it was tuned for rather than per frame. See PLATE_STEP_REF_HZ.
        self.plate_step = min(1.0, PLATE_STEP_REF_HZ / self._pose_rate_guess())

        self._t = pmw_pose.Tracker(
            cams=[stereo_native.camera_dict(c) for c in est.rig.cameras],
            cfg=cfg,
            centre_cal=stereo_native.centre_cal_dict(est.centre_cal),
            reference=np.ascontiguousarray(est.reference, dtype=np.float64),
            ids=self.ids,
            width=self.width, height=self.height, fps=self.fps, rotate180=bool(rotate180),
            pair_mode=pair_mode, max_skew_s=self.max_skew_s,
            plate_step=self.plate_step, plate_warmup=bgmod.WARMUP_FRAMES,
            plate_refresh=segmod.PLATE_REFRESH_FRAMES)

    def _pose_rate_guess(self):
        """Poses per second this pairing will produce, before anything is measured."""

        return self.fps * (2.0 if self.pair_mode == "interleave" else 1.0)

    # ---- lifecycle -------------------------------------------------------------------
    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._t.stop()

    def reset(self):
        self._t.reset()
        self.est.reset()
        self._seq = 0

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- what the loop reads ---------------------------------------------------------
    def read(self, timeout=2.0):
        """
        The next pose, or ``None`` if none arrived within ``timeout``.

            ``None`` means the tracker produced nothing, which is a camera that stopped
            or a pair whose views are too far apart -- not a lost frame. A frame the
            estimator saw and could not solve comes back as a `StereoPose`-shaped miss
            in `stats()["n_lost"]`, exactly as it does through `est.update`.
        """

        seq, d = self._t.wait(self._seq, float(timeout))
        if d is None:
            return None
        self._seq = seq
        self.frame_index += 1
        return self._to_pose(d)

    def latest(self):
        """The newest pose without waiting, ``None`` if there has not been one yet."""

        seq, d = self._t.latest()
        if d is None:
            return None
        self._seq = seq
        return self._to_pose(d)

    def _to_pose(self, d):
        # The same conversion `stereo_native.update` runs, off the same dict. The
        # estimator carries the zeroing and the jump gate's memory, so it is the object
        # the conversion mutates -- there is no second copy of that state.
        pose = stereo_pose(self.est, d, float(d["t"]), self.frame_index, float(d["t"]))
        return self.est._gate_predicted(pose)

    def frames(self):
        """Newest ``(t, frameA, frameB, stamps)``, or ``None``. For the recorder and viz."""

        return self._t.frames()

    def set_motion(self, motion):
        """
        Push the filter's velocity in, for `fuse`'s per-view time correction.

            The worker fires on its own thread, so the velocity has to be handed to it
            rather than passed per call. It is one pose stale, which is what the live
            path already does: `live_viz.py:1503` passes `filt.pos` and only updates the
            filter at `:1504`.
        """

        if motion is None:
            self._t.set_motion(None, None)
            return
        self._t.set_motion(np.ascontiguousarray(motion.rate, dtype=np.float64),
                           np.ascontiguousarray(motion.rate_cov, dtype=np.float64))

    def set_thresh(self, level):
        """
        The segmentation threshold, live -- what the viser slider drives.

            On the native core this was dead: `thresh` reaches C++ as `Config.level`,
            read once at construction, and `_ensure_native` only rebuilds on a frame
            scale change. So `live_viz`'s `est.thresh = viz.thresh` has done nothing
            since the native core became the default.
        """

        if level is None:
            return
        self.est.thresh = int(level)
        self._t.set_thresh(int(level))

    def stats(self):
        """Per-camera grab rate, pose rate, losses and measured skew."""

        return self._t.stats()

    # ---- offline ---------------------------------------------------------------------
    def push(self, ci, gray, t):
        """Feed a frame as if a camera had delivered it. Pairs with `pump`."""

        self._t.push_frame(int(ci), np.ascontiguousarray(gray, dtype=np.uint8), float(t))

    def pump(self):
        """Run one pairing step synchronously. Returns whether a pose came out."""

        return self._t.pump()


def _frames(rec, n, scale=0.5):
    """`n` grayscale pairs and their per-camera stamps, as the loop would see them."""

    import cv2
    from controller.camera import record as recmod

    caps, stamps = recmod.open_recording(rec)
    out = []
    for i in range(n):
        got = [c.read() for c in caps]
        if not all(ok for ok, _ in got):
            break
        gs = [np.ascontiguousarray(
                  cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f,
                             None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
              for _, f in got]
        out.append((gs, [float(x) for x in stamps[i]]))
    for c in caps:
        c.release()
    return out


def _self_check():
    """
    Pairing, the staleness guard and the view cache, from a recording with no camera.

        Writing our own capture layer cost the one thing `cv::VideoCapture` would have
        given free: opening an mp4 to exercise the threading offline. `push`/`pump` buy
        it back -- everything below the AVFoundation call itself runs here.
    """

    from pathlib import Path
    from controller.viz import live_viz

    # The bench take, named rather than `latest_flight()`. The recent flights are
    # tilt-sweep takes of a rotor with no rim, which the default segmenter solves 0% of
    # by design (`pose/theory.md` 20) -- a self-check pointed at one would fail for a
    # reason that has nothing to do with the tracker.
    rec = Path(BENCH_TAKE)
    assert rec.is_dir(), f"bench recording missing: {rec}"
    rig, est = live_viz._stereo_estimator(backgrounds="running")
    pairs = _frames(rec, 30)
    assert len(pairs) >= 20, f"only {len(pairs)} frames read from {rec}"
    # `ids` is never opened: `push`/`pump` bypass the cameras entirely.
    mk = lambda mode: Tracker(est, pair_mode=mode, ids=["-", "-"])

    # 1. `both` pairs one-for-one, as `StereoCamera.read` does, and solves.
    trk = mk("both")
    n_pose = 0
    for gs, row in pairs:
        for ci, g in enumerate(gs):
            trk.push(ci, g, row[ci])
        n_pose += trk.pump()
    assert n_pose > 0, "pair_mode='both' produced no pose at all"

    # 2. The staleness guard. With a slot that is never consumed this is what replaces
    #    `MonoCamera.read`'s 2 s timeout, so it gets its own assertion: a pair beyond
    #    `max_skew_s` is refused AND counted, never quietly fused.
    before = trk.stats()["n_skew_dropped"]
    gs, row = pairs[0]
    trk.push(0, gs[0], 1000.0)
    trk.push(1, gs[1], 1000.0 + 10 * trk.max_skew_s)
    assert not trk.pump(), "a pair beyond max_skew_s produced a pose"
    assert trk.stats()["n_skew_dropped"] == before + 1, "the stale pair was not counted"

    # 3. **The view cache is exact.** This is the one claim the interleaved rate rests
    #    on: when only camera A has a new frame, view B is reused rather than segmented
    #    again, and that must not change the answer. Both trackers below see identical
    #    pixels; only one of them is allowed to reuse. Pushing B's unchanged frame again
    #    bumps its seq, which forces the recompute the other one skips.
    est.reset()
    cached, recomp = mk("interleave"), mk("interleave")
    got_c = got_r = None
    for k, (gs, row) in enumerate(pairs):
        for t in (cached, recomp):
            t.push(0, gs[0], row[0])
        # Every frame for `recomp`; only when B genuinely changes for `cached`. On the
        # last step B is held back from `cached`, so it must reuse.
        hold = k == len(pairs) - 1
        recomp.push(1, gs[1] if not hold else pairs[k - 1][0][1], row[1] if not hold else pairs[k - 1][1][1])
        if not hold:
            cached.push(1, gs[1], row[1])
        cached.pump()
        recomp.pump()
        if hold:
            got_c, got_r = cached.latest(), recomp.latest()
    assert got_c is not None and got_r is not None, "no pose on the cache-reuse step"
    dx = float(np.max(np.abs(np.asarray(got_c.xyz_mm) - np.asarray(got_r.xyz_mm))))
    dn = float(np.max(np.abs(np.asarray(got_c.normal) - np.asarray(got_r.normal))))
    assert dx == 0.0 and dn == 0.0, (
        f"the reused view changed the answer: {dx:.3e} mm, {dn:.3e} in the normal. "
        f"A cache that is not exact is a bug in it, not a speed-up.")

    # 4. Interleaving really does fire on a single camera. `both` on the same input
    #    would produce nothing here, which is the whole point of the pairing change.
    solo = mk("interleave")
    gs, row = pairs[0]
    solo.push(0, gs[0], row[0])
    solo.push(1, gs[1], row[1])
    solo.pump()
    n_before = solo.stats()["n_pose"] + solo.stats()["n_lost"]
    # A's new frame lands within one frame period of B's, which is what a free-running
    # pair actually produces: when A delivers, B's newest is uniform on [t-T, t]. Using
    # the recording's own next-row stamp instead would be a full period later and the
    # skew guard would (correctly) refuse it -- that case is check 2's job, not this one.
    solo.push(0, pairs[1][0][0], row[0] + 0.5 * solo.max_skew_s)   # camera A only
    solo.pump()
    assert solo.stats()["n_pose"] + solo.stats()["n_lost"] == n_before + 1, (
        "interleave did not fire on a single camera's frame")

    # 5. The window is a duration, not the raw count, once the rate is high enough.
    assert mk("interleave")._pose_rate_guess() > mk("both")._pose_rate_guess()

    print(f"tracker: {n_pose} poses paired from {rec.name}; skew guard fires; "
          f"reused view is bit-identical; interleave fires on one camera")


if __name__ == "__main__":
    _self_check()
