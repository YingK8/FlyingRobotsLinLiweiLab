"""
Frame sources, so the pipeline does not care where pixels come from.

There is no high-speed camera on this rig yet -- the bench C270 is measured at
~28 fps flat at every resolution, and only the built-in FaceTime camera is
attached today.  So the estimator is written against a source interface, and
throughput is proven against the 240 fps footage in `writeup/` while the real
hardware is still a purchase order.  When it arrives, one new class implements
`read()` and nothing else changes.

`MonoCamera` is the piece the repo has never had.  Everything camera-side
here today is a synchronous `cap.read()` in a notebook cell with
`CAP_PROP_BUFFERSIZE = 1` to stop latency piling up.  That works at 28 fps and
falls apart at 400: a slow frame stalls acquisition, and dropped frames vanish
without trace.  Here a dedicated thread does nothing but grab into a
drop-oldest slot, so estimation can never back-pressure the camera, and drops
are counted rather than hidden.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class Capture:
    """
    Interface: ``read()`` returns ``(t_capture, frame)`` or ``None`` at end.

        ``t_capture`` is seconds on the `time.monotonic` clock, as close to shutter
        as the backend allows.
    """

    def read(self):
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        while True:
            item = self.read()
            if item is None:
                return
            yield item


class Image(Capture):
    """
    Replay a list of in-memory frames or a directory of image files.

        Used to feed rendered frames through the identical path a camera would take,
        so a synthetic run exercises the same code as a live one.
    """

    def __init__(
        self, frames=None, directory=None, pattern="*.png", rate_hz=None, loop=False
    ):
        if frames is None and directory is None:
            raise ValueError("no image frames or directory")

        if frames is not None:
            self._frames = list(frames)
            self._paths = None
        else:
            self._paths = sorted(Path(directory).glob(pattern))
            if not self._paths:
                raise FileNotFoundError(f"no images matching {pattern} in {directory}")
            self._frames = None

        self._n = len(self._frames if self._frames is not None else self._paths)
        self._i = 0
        self._loop = loop
        self._period = None if rate_hz in (None, 0) else 1.0 / rate_hz
        self._t0 = time.monotonic()

    def read(self):
        if self._i >= self._n:
            if not self._loop:
                return None
            self._i = 0

        if self._frames is not None:
            frame = self._frames[self._i]
        else:
            frame = cv2.imread(str(self._paths[self._i]), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise OSError(f"could not read {self._paths[self._i]}")

        # Pace to a requested rate so a synthetic run can imitate a real one.
        if self._period is not None:
            target = self._t0 + self._i * self._period
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)

        self._i += 1
        return time.monotonic(), frame


class Video(Capture):
    """
    Frames from a video file, timestamped from the container.

        Container timestamps rather than wall clock, so a run processed faster (or
        slower) than real time still reports the times the frames actually have.
        `writeup/two_channel.mp4` and `single_channel.mp4` are 240 fps and are the
        only genuinely high-rate footage available for a throughput test.
    """

    def __init__(self, path, grayscale=True, loop=False):
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"could not open {self.path}")
        self._gray = grayscale
        self._loop = loop
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._i = 0

    def read(self):
        ok, frame = self._cap.read()
        if not ok:
            if not self._loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                return None

        # CAP_PROP_POS_MSEC is unreliable on some builds; fall back to the index.
        ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1e3 if ms and ms > 0 else (self._i / self.fps if self.fps else self._i)
        self._i += 1

        if self._gray and frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return t, frame

    def close(self):
        self._cap.release()


class MonoCamera(Capture):
    """
    Live camera with a dedicated grabber thread and a drop-oldest slot.

        The thread only ever calls `cap.read()` and overwrites a single-frame slot.
        Holding just the newest frame is deliberate: for feedback control a stale
        frame is worse than no frame, so a queue that buffers backlog would be
        actively harmful.  Frames overwritten before being consumed are counted in
        `n_dropped` -- the number that tells you whether the estimator is keeping
        up.

        `cap.read()` releases the GIL inside OpenCV, so the grabber genuinely runs
        while the estimator works.
    """

    def __init__(
        self,
        index=0,
        width=None,
        height=None,
        fps=None,
        fourcc="MJPG",
        grayscale=True,
        backend=None,
    ):
        if backend is None:
            backend = default_backend()
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise OSError(f"could not open camera index {index}")

        # FOURCC before frame size: on V4L2 the driver otherwise renegotiates to
        # YUYV, whose bandwidth cannot sustain high rates over USB 2. macOS
        # AVFoundation ignores this and negotiates on its own.
        if fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._gray = grayscale
        self._lock = threading.Lock()
        self._slot = None
        self._new = threading.Event()
        self._stop = threading.Event()
        self.n_grabbed = 0
        self.n_dropped = 0

        self._thread = threading.Thread(
            target=self._grab_loop, name="grabber", daemon=True
        )
        self._thread.start()

    @property
    def actual(self):
        """
        What the driver actually gave us, which is rarely what was asked.
        """

        return {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self._cap.get(cv2.CAP_PROP_FPS)),
        }

    def _grab_loop(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.001)
                continue
            t = time.monotonic()
            if self._gray and frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with self._lock:
                if self._slot is not None:
                    self.n_dropped += 1  # consumer never saw the previous one
                self._slot = (t, frame)
                self.n_grabbed += 1
            self._new.set()

    def read(self, timeout=2.0):
        """
        Newest frame, waiting up to ``timeout`` seconds. ``None`` on timeout.
        """

        if not self._new.wait(timeout):
            return None
        with self._lock:
            item = self._slot
            self._slot = None
            self._new.clear()
        return item

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


class StereoCamera(Capture):
    """
    Two sources read as one, with the timestamp skew measured rather than assumed.

        ``read()`` returns ``(t, [frame_a, frame_b])`` where ``t`` is the **mean**
        capture time -- the instant the pair is closest to describing.  Taking the
        first camera's stamp instead would bias every pose by half the skew in a
        fixed direction, which is exactly the kind of error that survives filtering.

        **Sync is the honest weak point of a software-triggered pair.**  Without a
        hardware trigger the two cameras free-run and their frames land wherever
        they land.  So the skew is recorded on every read, and `skew_stats` reports
        it -- because the number that matters is not whether sync is perfect but
        whether it is small next to the motion.  At 420 fps one frame of skew is
        2.4 ms; the robot moves at 15-22 mm/s in hover, so that is about 0.05 mm,
        comfortably inside the budget.  During a 1.4 m/s^2 climb it is worse, and
        `filter.PoseFilter.predict_ahead` is the software remedy: advance the earlier
        view to the later one's timestamp.  A hardware trigger remains the right
        answer, per docs/pose_localization_project_context.md section 5 stage 3.

        ``max_skew_s`` drops pairs that are too far apart rather than silently
        fusing them, since a stale view produces a confident wrong triangulation and
        a dropped frame produces a gap that is visible in the log.
    """

    def __init__(self, sources, max_skew_s=None, timeout=2.0):
        self.sources = list(sources)
        if len(self.sources) < 2:
            raise ValueError("StereoCamera needs at least two sources")
        self.max_skew_s = max_skew_s
        self.timeout = timeout
        self.n_read = 0
        self.n_skew_dropped = 0
        self._skews = []

    def read(self):
        items = []
        for s in self.sources:
            item = s.read() if not isinstance(s, MonoCamera) else s.read(self.timeout)
            if item is None:
                return None
            items.append(item)

        stamps = [t for t, _ in items]
        skew = max(stamps) - min(stamps)
        self._skews.append(skew)
        self.n_read += 1

        if self.max_skew_s is not None and skew > self.max_skew_s:
            self.n_skew_dropped += 1
            return self.read()

        return float(np.mean(stamps)), [f for _, f in items]

    def skew_stats(self):
        """
        Measured capture skew, in milliseconds. The number to put in a report.
        """

        if not self._skews:
            return {}
        s = np.asarray(self._skews) * 1e3
        return {
            "n": int(len(s)),
            "median_ms": float(np.median(s)),
            "p95_ms": float(np.percentile(s, 95)),
            "max_ms": float(s.max()),
            "dropped": self.n_skew_dropped,
        }

    def close(self):
        for s in self.sources:
            s.close()


def open_stereo(specs, max_skew_s=None, **kw):
    """
    Build a `StereoCamera` from two CLI-friendly source strings.
    """

    return StereoCamera([open_source(s, **kw) for s in specs], max_skew_s=max_skew_s)


def default_backend():
    """
    The capture backend for this platform.

        Defined here because `MonoCamera` is where a backend is actually opened;
        `elp.py` imports it rather than re-deriving the same two-way choice.
    """

    return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_V4L2


def measure_fps(source, n=120):
    """
    Time real reads. The only honest frame-rate number.

        Mirrors the helper in `visual_servo/servo.ipynb`, whose README makes the
        point that `CAP_PROP_FPS` reports what was requested, not what arrives.
    """

    t0 = time.monotonic()
    got = 0
    for _ in range(n):
        if source.read() is None:
            break
        got += 1
    dt = time.monotonic() - t0
    return (got / dt) if dt > 0 else 0.0, got


def open_source(spec, **kw):
    """
    Build a source from a CLI-friendly string.

        ``"camera"`` / ``"camera:1"`` for a device index, otherwise a path to a
        video file or a directory of images.
    """

    s = str(spec)
    if s.startswith("camera"):
        _, _, idx = s.partition(":")
        return MonoCamera(index=int(idx) if idx else 0, **kw)

    path = Path(s)
    if path.is_dir():
        return Image(
            directory=path,
            **{k: v for k, v in kw.items() if k in ("pattern", "rate_hz", "loop")},
        )
    if not path.exists():
        raise FileNotFoundError(s)
    return Video(path, **{k: v for k, v in kw.items() if k in ("grayscale", "loop")})
