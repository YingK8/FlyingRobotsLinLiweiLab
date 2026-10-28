#!/usr/bin/env python3
"""
Record a flight: one video per camera, plus the timing that makes them a pair.

This is the *flight* recorder, not the calibration one. Calibration deliberately has no
video path -- `calib/calibrate.py` photographs a board that is standing still, and the
argument for that is `calib/theory.md` section 16. A flight cannot be asked to hold still,
so here everything is written and the pose pipeline picks it apart afterwards.

Two consequences follow from that, and they are the whole design:

* **Lossy is fine here, and only here.** Nothing detects a sub-pixel corner in this footage.
  The pose pipeline segments a bright rim against a dark scene, recorded with H.264 or MPEG-4 Part 2. Compression quality still needs
  checking on the real target. Calibration stores lossless PNG photographs.
* **The two cameras do not fire together**, so frame *i* of ``A.mp4`` and ``B.mp4`` are a
  few milliseconds apart. `frames.csv` records each camera's own capture time, and
  `stereo.fuse` uses them to move both views to a common instant (`pose/theory.md`
  section 17). Without that file the videos are just two videos.

Encoding runs on its own thread behind a bounded queue; dropped pairs remain in
frames.csv with written=0. See camera/theory.md section 1.8 for portable capture.

Each take is its own **flight**: SPACE starts one, SPACE stops it, and it lands in a dated
folder of its own with a directory per camera. Takes are not comparable -- a different trim,
a different day -- so they must not share a directory.

    results/flights/2026-08-25_133327/A/A.mp4
                                     /B/B.mp4
                                     /frames.csv   what makes the two a timed pair
                                     /meta.json    cameras, mode, fps, frames, drops

    python record.py                     # SPACE starts and stops, q quits
    python record.py --indices 0         # one camera
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

from controller.camera import identify
from controller.camera import sources

DEFAULT_DIR = HERE.parents[1] / "results" / "flights"
QUEUE_DEPTH = 64         # bounded: an encoder that falls behind must drop and say so,
                         # not grow until the machine swaps


# ---- one folder per flight ----------------------------------------------------------
def _skew(src):
    """Capture-skew stats, or {} for a source that does not track them."""

    return src.skew_stats() if hasattr(src, "skew_stats") else {}


def take_suffix(note):
    """``_<design><hz>`` parsed from a campaign note ("half ring 90Hz x10 ..."), else ''.

    The operator reads the design and frequency off the folder name in Finder, so the
    campaign name goes into the directory itself, not only into meta.json (2026-09-21)."""

    m = re.search(r"\b(half|whole|no)\s*ring\s*(\d+)\s*Hz", str(note or ""), re.I)
    return f"_{m.group(1).lower()}{m.group(2)}" if m else ""


def new_flight(root=DEFAULT_DIR, tags="AB", suffix=""):
    """A dated folder for one take, with a directory per camera.
    ``root/YYYY-mm-dd_HHMMSS[<suffix>]`` -- the suffix is `take_suffix` of the note.

    One take is one flight, and takes are not comparable: a different trim, a different
    board, a different day.
    """

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(root) / f"{stamp}{suffix}"
    for k in range(1, 100):                 # two takes inside one second must not merge
        if not out.exists():
            break
        out = Path(root) / f"{stamp}_{k}{suffix}"
    for tag in tags:
        (out / tag).mkdir(parents=True, exist_ok=True)
    return out


class FlightWriter:
    """
    Write a stereo take from frames someone else is already reading.

        `record` owns its camera. The control loop owns *its* camera, and two owners
        of one USB camera is not a thing -- so a flight cannot be filmed by running
        both. This takes the frames the loop has already read and writes exactly what
        `record` writes: one mp4 per camera, `frames.csv`, `meta.json`. That sameness
        is the point -- `open_recording`, `read_index` and `live_viz.from_recording`
        then replay a control run with no idea it was not shot by `record`.

        Encoding runs on its own thread behind a bounded queue, as in `record`: a
        control loop must never block on an encoder, so a queue that fills drops the
        frame and counts it rather than stalling the flight.

        `frames.csv` is written **as the frames arrive**, not at the end. The caller
        here is a generator's `finally`, which runs under `GeneratorExit` and can be
        cut short by a signal or interpreter shutdown; buffering the index until then
        risks 4 GB of video that nothing can turn back into a timed stereo pair. A row
        per frame costs nothing and survives a kill -9.
    """

    def __init__(self, out_dir=DEFAULT_DIR, tags="AB", fps=120.0, meta=None, codec="auto"):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video fps must be positive and finite")
        if codec not in ("auto", "avc1", "mp4v"):
            raise ValueError("codec must be auto, avc1 or mp4v")
        self.codec_request, self.codec = codec, None
        self._finish = threading.Event()
        self._closed = False
        self.dir = new_flight(out_dir, tags, take_suffix((meta or {}).get("note")))
        self.tags, self.fps, self.meta = tags, float(fps), dict(meta or {})
        self.n, self.dropped, self.errors = 0, 0, 0
        self._t_first = self._t_last = None      # for fps_measured, see close()
        self.writers, self.size = None, None
        self._released = False   # set by the encoder thread once it finalises
        # Header up front, rows as they land: see the class docstring.
        self._csv = open(self.dir / "frames.csv", "w", buffering=1)
        # `written` is 0 for a frame the queue dropped: its row is kept (the capture time is
        # real) but the mp4 has no frame for it. Without the flag, pairing mp4 frame i with
        # row i put every later frame at an earlier row's time -- 91 of 125 design-B takes,
        # seconds of error by the cut (2026-09-13). `read_index` drops the unwritten rows.
        self._csv.write("index,t_capture,skew_s,"
                        + ",".join(f"t_{t.lower()}" for t in self.tags) + ",written\n")
        self._work = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._run, name="encode", daemon=True)
        self._thread.start()

    def _run(self):
        try:
            while True:
                try:
                    job = self._work.get(timeout=0.1)
                except queue.Empty:
                    if self._finish.is_set():
                        break
                    continue
                try:
                    for w, f in job:
                        # BGR works with the Windows MP4 encoders as well as macOS.
                        w.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f)
                except Exception as e:
                    self.errors += 1
                    if self.errors == 1:
                        print(f"encoder error (take requires validation): {e}")
                finally:
                    self._work.task_done()
        finally:
            # Only this thread releases writers, after its queue has drained.
            for w in self.writers or []:
                try:
                    w.release()
                except Exception:
                    self.errors += 1
            self._released = True

    def _open_writers(self, size):
        candidates = (["mp4v", "avc1"] if sys.platform == "win32" else ["avc1", "mp4v"])
        if self.codec_request != "auto":
            candidates = [self.codec_request]
        paths = [self.dir / tag / f"{tag}.mp4" for tag in self.tags]
        for codec in candidates:
            writers = []
            try:
                for path in paths:
                    writers.append(cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                                   self.fps, size, True))
                if all(w.isOpened() for w in writers):
                    self.codec = codec
                    return writers
            except cv2.error:
                pass
            for w in writers:
                w.release()
            for path in paths:
                path.unlink(missing_ok=True)
        raise OSError(f"no working MP4 writer among {candidates}; run the notebook codec check")

    def add(self, t, frames, stamps=None, skew=0.0):
        """Queue a stereo read without waiting for space; reject invalid frames/encoders."""

        if self._closed:
            raise RuntimeError("recorder is already closed")
        if len(frames) != len(self.tags) or not frames:
            raise ValueError("one frame per camera is required")
        h, w = frames[0].shape[:2]
        if any(f.shape[:2] != (h, w) for f in frames):
            raise ValueError("camera frame dimensions differ")
        if self.size is not None and self.size != [w, h]:
            raise ValueError("frame dimensions changed during recording")
        if self.writers is None:
            self.size = [w, h]
            self.writers = self._open_writers((w, h))
        written = 1
        try:
            self._work.put_nowait(list(zip(self.writers, [f.copy() for f in frames])))
        except queue.Full:
            self.dropped += 1
            written = 0
        if self._t_first is None:
            self._t_first = t
        self._t_last = t
        st = stamps or (t,) * len(frames)
        self._csv.write(f"{self.n},{t:.6f},{skew:.6f},"
                        + ",".join(f"{x:.6f}" for x in st) + f",{written}\n")
        self.n += 1

    def measured_fps(self):
        """Frames per second this take actually achieved, or 0.0 for a take too short."""

        span = (self._t_last - self._t_first) if self._t_first is not None else 0.0
        return (self.n - 1) / span if self.n > 1 and span > 0 else 0.0

    def close(self, stats=None):
        """Drain and finalise before claiming success; repeated close is harmless."""
        if self._closed:
            return self.dir
        self._closed = True
        self._csv.close()
        metadata = {**self.meta, "mode": self.size, "fps": self.fps,
                    "fps_semantics": "constant_rate_playback_only",
                    "timestamp_semantics": "host_read_completion_not_exposure",
                    "codec": self.codec, "fps_measured": round(self.measured_fps(), 2),
                    "n_frames": self.n, "dropped": self.dropped,
                    "encoder_errors": self.errors, "encoder_finalized": False,
                    "skew": stats or {}, "created": datetime.now().isoformat(timespec="seconds")}
        path = self.dir / "meta.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._finish.set()
        self._thread.join(timeout=60.0)
        if self._thread.is_alive() or not self._released:
            # Do not race release() against a still-running native encoder.
            raise RuntimeError(f"encoder did not finalise; incomplete take: {self.dir}")
        metadata.update(encoder_errors=self.errors, encoder_finalized=True)
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"  {self.n} capture pairs -> {self.dir}; {self.dropped} queue drops; "
              f"capture {self.measured_fps():.1f} Hz, playback {self.fps:g} FPS ({self.codec})")
        return self.dir


def flights(root=DEFAULT_DIR):
    """Every flight folder under ``root``, oldest first."""

    return sorted(d for d in Path(root).glob("[0-9]*-[0-9]*") if d.is_dir())


def latest_flight(root=DEFAULT_DIR):
    """The most recent flight under ``root``, or ``root`` itself if it holds video."""

    found = flights(root)
    return found[-1] if found else Path(root)


def record(out_dir=DEFAULT_DIR, indices=None, width=1280, height=800, fps=120.0,
           rotate180=True, max_skew_s=None, preview=True, start=False, note=None,
           backend=None, codec="auto", seconds=None):
    """Shared CLI/notebook recorder. Timed cells or Interrupt both close the take.

    fps is requested from the camera AND used as the container playback rate; neither
    establishes actual exposure times. Preserve frames.csv for analysis (theory.md section 1.8).
    No skew rejection by default: keep every consumed pair and measure its skew.
    """
    if codec not in ("auto", "avc1", "mp4v"):
        raise ValueError("codec must be auto, avc1 or mp4v")
    if seconds is not None and (not np.isfinite(seconds) or seconds <= 0):
        raise ValueError("seconds must be positive and finite")
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("positive width, height and fps required")
    idx = (identify.elp_indices(backend=backend) if indices is None
           else [indices] if isinstance(indices, int) else list(indices))
    if len(idx) not in (1, 2) or len(set(idx)) != len(idx):
        raise ValueError("choose one or two distinct camera indices")
    tags = "AB"[:len(idx)]
    options = dict(width=width, height=height, fps=fps, grayscale=True,
                   rotate180=rotate180, backend=backend)
    src = (sources.open_source(f"camera:{idx[0]}", **options) if len(idx) == 1 else
           sources.open_stereo([f"camera:{i}" for i in idx], max_skew_s=max_skew_s, **options))
    cameras = getattr(src, "sources", [src])
    fw, sink, recording, done = None, None, False, []
    meta = {"camera_indices": idx, "rotate180": bool(rotate180),
            "requested": {"width": width, "height": height, "fps": fps, "fourcc": "MJPG"}}
    if note:
        meta["note"] = str(note)
    t0 = time.monotonic()
    try:
        meta["cameras"] = [cam.actual for cam in cameras]
        print("Driver-reported modes (FPS is not a measurement):", meta["cameras"])
        sink = sources.Sink("flight recorder").open() if preview else None
        if sink is not None and sink.inline and not start and seconds is None:
            raise ValueError("notebook preview needs seconds; recording needs start=True")
        if start:
            fw, recording = FlightWriter(out_dir, tags, fps, meta, codec=codec), True
            print(f"recording -> {fw.dir}; Interrupt once to stop early, then wait for finalisation")
        while seconds is None or time.monotonic() - t0 < seconds:
            item = src.read()
            if item is None:
                raise OSError("camera stopped delivering frames; take is partial, inspect before use")
            t, payload = item
            frames = list(payload) if isinstance(payload, (list, tuple)) else [payload]
            if any(f.shape[:2] != (height, width) for f in frames):
                raise OSError(f"requested {width}x{height}, received {[f.shape for f in frames]}")
            if recording:
                fw.add(t, frames, getattr(src, "last_stamps", None), getattr(src, "last_skew", 0.0))
            if sink is not None and sink.due():
                views = []
                for tag, f in zip(tags, frames):
                    view = f.copy() if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                    cv2.putText(view, tag, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                    views.append(view)
                view = np.hstack(views)
                text = (f"REC {time.monotonic()-t0:.1f}s {fw.n} pairs" if recording else "PREVIEW")
                text += " | Interrupt to finish" if sink.inline else " | SPACE record/stop; q quit"
                cv2.putText(view, text, (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (0, 0, 255) if recording else (255, 255, 255), 1)
                key = sink.show(view)
                if key == ord("q"):
                    break
                if key == ord(" "):
                    if recording:
                        done.append(fw.close(_skew(src)))
                        fw = None
                    else:
                        fw = FlightWriter(out_dir, tags, fps, meta, codec=codec)
                    recording = not recording
    except KeyboardInterrupt:
        print("Interrupted; finalising files. Wait for this cell to finish.")
    except BaseException as exc:
        if fw is not None:
            fw.meta["capture_error"] = str(exc)
        raise
    finally:
        # Release cameras even if encoder shutdown or GUI cleanup fails.
        try:
            src.close()
        finally:
            try:
                if sink is not None:
                    sink.close()
            finally:
                if fw is not None:
                    fw.meta["capture_counters"] = [
                        {"grabbed": c.n_grabbed, "overwritten_before_consumption": c.n_dropped}
                        for c in cameras]
                    done.append(fw.close(_skew(src)))
    return done


def read_index(rec_dir):
    """``(stamps, skews)`` from ``frames.csv``, or ``(None, None)`` when it is missing.

    ``stamps`` is one row per frame IN THE MP4 and one column per camera, so row i is
    video frame i. Without it the videos can still be replayed, but every pair has to be
    assumed simultaneous, which is the assumption `pose/theory.md` section 17 exists to
    remove.

    Takes written before the `written` column (2026-09-13) keep a row for every DROPPED
    frame too, and cannot be filtered here: for them row i is not frame i once anything
    was dropped (`meta.json` `dropped` > 0), and their frame times need repairing.
    """

    path = Path(rec_dir) / "frames.csv"
    if not path.exists():
        return None, None
    # comment="#" also reads the older takes, which prefixed a "# skew_n, 1082" line.
    df = pd.read_csv(path, comment="#")
    if "written" in df.columns:
        df = df[df["written"] == 1]
    if df.empty:
        return None, None
    per_cam = [c for c in df.columns if c.startswith("t_") and c != "t_capture"]
    return df[per_cam].to_numpy(float), df["skew_s"].to_numpy(float)


def open_recording(rec_dir):
    """``(captures, stamps)`` for one flight: one `cv2.VideoCapture` per camera.

    ``rec_dir`` may be the flights root, in which case the most recent flight is opened.
    """

    rec_dir = latest_flight(rec_dir)
    # `<tag>/<tag>.mp4` only -- what FlightWriter writes. Anything ELSE dropped in a view
    # directory (an overlay render, a trimmed clip) used to be opened as an extra camera,
    # silently: `disc_axis` then read view 1 as the overlay of view 0 and every stereo
    # number downstream came from a pair that was never a pair.
    # ponytail: the flat fallback below cannot apply this rule -- it has no tag to match --
    # so keep renders out of a flat take dir.
    videos = sorted(v for v in rec_dir.glob("*/*.mp4") if v.stem == v.parent.name) \
        or sorted(rec_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no video in {rec_dir}")
    stamps, _ = read_index(rec_dir)
    if stamps is None:
        print(f"{rec_dir}: no frames.csv, so the two views are assumed simultaneous")
    return [cv2.VideoCapture(str(v)) for v in videos], stamps


def _self_check(tmp=None):
    """A take must come back playable. This is the 2026-09-01 regression, in 20 lines.

    Eleven of thirteen takes that day had no moov atom: `close()` released the writers
    from the caller while the encoder thread was still draining into them. Nothing
    noticed until the pose pipeline could not open the files, so the check runs the
    whole FlightWriter round trip and asserts the artefacts, not the code path.
    """

    import shutil
    import tempfile

    root = Path(tmp or tempfile.mkdtemp(prefix="flightwriter-"))
    fw = FlightWriter(root, tags="AB", fps=30.0, meta={"source": "_self_check"})
    frames = [np.zeros((64, 80), np.uint8), np.zeros((64, 80), np.uint8)]
    for i in range(90):                      # long enough to outrun the 64-deep queue
        frames[0][:] = frames[1][:] = i * 2
        fw.add(i / 30.0, frames, (i / 30.0, i / 30.0 + 0.002), 0.002)
    out = fw.close({"n": 90})

    assert fw._released, "encoder thread never finalised"
    for tag in "AB":
        v = out / tag / f"{tag}.mp4"
        assert v.exists(), v
        assert b"moov" in v.read_bytes(), f"{v} has no moov atom -- unplayable"
    stamps, skews = read_index(out)
    kept = 90 - fw.dropped
    assert stamps is not None and stamps.shape == (kept, 2), None if stamps is None else stamps.shape
    assert len(skews) == kept, len(skews)
    rows = (out / "frames.csv").read_text().splitlines()
    assert rows[0].endswith(",written") and len(rows) == 91, (rows[0], len(rows))
    assert sum(int(r.rsplit(",", 1)[1]) for r in rows[1:]) == kept
    # the reader keeps written rows only, whatever the queue happened to do above
    fake = root / "fake"
    fake.mkdir()
    (fake / "frames.csv").write_text("index,t_capture,skew_s,t_a,t_b,written\n"
                                     "0,0.0,0,0.0,0.0,1\n1,0.1,0,0.1,0.1,0\n2,0.2,0,0.2,0.2,1\n")
    fs, _ = read_index(fake)
    assert fs.shape == (2, 2) and list(fs[:, 0]) == [0.0, 0.2], fs
    meta = json.loads((out / "meta.json").read_text())
    assert meta["fps"] == 30.0 and meta["fps_measured"] > 0, meta
    caps, _ = open_recording(out)
    for c in caps:
        assert c.isOpened(), "written mp4 will not reopen"
        c.release()
    if tmp is None:
        shutil.rmtree(root, ignore_errors=True)
    print(f"record: self-check passed (90 frames, both mp4s finalised{'' if fw.dropped == 0 else f', {fw.dropped} dropped'})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_DIR)
    p.add_argument("--indices", nargs="+", type=int, default=None,
                   help="default: whichever indices the two ELPs hold right now")
    p.add_argument("--mode", default="1280x800")
    p.add_argument("--fps", type=float, default=120.0)
    p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "avfoundation", "v4l2"])
    p.add_argument("--codec", default="auto", choices=["auto", "avc1", "mp4v"])
    p.add_argument("--seconds", type=float, help="end this preview/recording after this many seconds")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--note", default=None,
                   help="free text into meta.json, e.g. \"align C 100Hz coil A C shut down\"")
    p.add_argument("--start", action="store_true",
                   help="roll from the first frame instead of waiting for SPACE")
    p.add_argument("--self-check", action="store_true", help="no camera needed")
    a = p.parse_args(argv)
    if a.self_check:
        _self_check()
        return 0
    w, h = (int(v) for v in a.mode.lower().split("x"))
    record(a.out, a.indices, width=w, height=h, fps=a.fps, rotate180=not a.no_flip,
           start=a.start, note=a.note, backend=a.backend, codec=a.codec, seconds=a.seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
