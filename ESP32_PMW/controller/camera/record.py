#!/usr/bin/env python3
"""
Record a flight: one video per camera, plus the timing that makes them a pair.

This is the *flight* recorder, not the calibration one. Calibration deliberately has no
video path -- `calib/calibrate.py` photographs a board that is standing still, and the
argument for that is `calib/theory.md` section 16. A flight cannot be asked to hold still,
so here everything is written and the pose pipeline picks it apart afterwards.

Two consequences follow from that, and they are the whole design:

* **Lossy is fine here, and only here.** Nothing detects a sub-pixel corner in this footage.
  The pose pipeline segments a bright rim against a dark scene, which survives H.264 at a
  sane bitrate. Calibration is the opposite case and pays for FFV1.
* **The two cameras do not fire together**, so frame *i* of ``A.mp4`` and ``B.mp4`` are a
  few milliseconds apart. `frames.csv` records each camera's own capture time, and
  `stereo.fuse` uses them to move both views to a common instant (`pose/theory.md`
  section 17). Without that file the videos are just two videos.

Encoding runs on its own thread: H.264 costs a few milliseconds a frame and the read loop
is otherwise idle waiting for the next pair.

Each take is its own **flight**: SPACE starts one, SPACE stops it, and it lands in a dated
folder of its own with a directory per camera. Takes are not comparable -- a different trim,
a different day -- and writing them together used to concatenate them into one video with
no record of where one ended.

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
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import identify  # noqa: E402
import sources  # noqa: E402

DEFAULT_DIR = HERE.parents[1] / "results" / "flights"
FOURCC = "avc1"          # H.264 in an .mp4. See the module docstring for why lossy is
                         # acceptable in this file and nowhere near calibration.
QUEUE_DEPTH = 64         # bounded: an encoder that falls behind must drop and say so,
                         # not grow until the machine swaps


# ---- one folder per flight ----------------------------------------------------------
def new_flight(root=DEFAULT_DIR, tags="AB"):
    """A dated folder for one take, with a directory per camera. ``root/YYYY-mm-dd_HHMMSS``.

    One take is one flight, and takes are not comparable: a different trim, a different
    board, a different day. Writing them into one directory used to concatenate them into
    a single video with no record of where one ended.
    """

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = Path(root) / stamp
    for k in range(1, 100):                 # two takes inside one second must not merge
        if not out.exists():
            break
        out = Path(root) / f"{stamp}_{k}"
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

    def __init__(self, out_dir=DEFAULT_DIR, tags="AB", fps=60.0, meta=None):
        self.dir = new_flight(out_dir, tags)
        self.tags, self.fps, self.meta = tags, float(fps), dict(meta or {})
        self.n, self.dropped, self.errors = 0, 0, 0
        self.writers, self.size = None, None
        # Header up front, rows as they land: see the class docstring.
        self._csv = open(self.dir / "frames.csv", "w", buffering=1)
        self._csv.write("index,t_capture,skew_s,"
                        + ",".join(f"t_{t.lower()}" for t in self.tags) + "\n")
        self._work = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._run, name="encode", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            job = self._work.get()
            try:
                if job is None:
                    return
                for w, f in job:
                    w.write(f)
            except Exception as e:  # noqa: BLE001
                # One bad frame must not kill the encoder. A dead encoder used to be
                # invisible until `close` waited forever for a `task_done` that was
                # never coming, and the take lost its meta.json.
                self.errors += 1
                if self.errors == 1:
                    print(f"  encoder error (frames will be missing): {e}")
            finally:
                self._work.task_done()

    def add(self, t, frames, stamps=None, skew=0.0):
        """One stereo read. Never raises and never blocks: a flight outranks its film."""

        if self.writers is None:
            h, w = frames[0].shape[:2]
            self.size = [w, h]
            self.writers = [cv2.VideoWriter(str(self.dir / tag / f"{tag}.mp4"),
                                            cv2.VideoWriter_fourcc(*FOURCC), self.fps,
                                            (w, h), False) for tag in self.tags]
            if not all(x.isOpened() for x in self.writers):
                raise OSError(f"no {FOURCC} writer on this build")
        try:
            self._work.put_nowait(list(zip(self.writers, frames)))
        except queue.Full:
            self.dropped += 1
        st = stamps or (t,) * len(frames)
        self._csv.write(f"{self.n},{t:.6f},{skew:.6f},"
                        + ",".join(f"{x:.6f}" for x in st) + "\n")
        self.n += 1

    def close(self, stats=None):
        """
        Finalise the take: what it means first, then flush the encoder.

            Order matters, and it is the opposite of the obvious one. This runs from a
            generator's `finally`, which can be executing while a `SystemExit` is
            already propagating -- measured, not theorised -- so anything after a slow
            step may simply never happen. `meta.json` is a few bytes and is what turns
            two mp4s into a take, so it is written *before* the encoder flush and the
            thread join that can be cut short.
        """

        self._csv.close()
        (self.dir / "meta.json").write_text(json.dumps(
            {**self.meta, "mode": self.size, "fps": self.fps, "n_frames": self.n,
             "dropped": self.dropped, "encoder_errors": self.errors,
             "skew": stats or {},
             "created": datetime.now().isoformat(timespec="seconds")}, indent=2))

        # Never `queue.join()`: it waits on a counter only the worker decrements, so a
        # worker that died hangs the caller. Sentinel plus a bounded thread join.
        try:
            self._work.put(None, timeout=5.0)
        except queue.Full:
            pass
        self._thread.join(timeout=30.0)
        for w in self.writers or []:
            w.release()
        if self._thread.is_alive():
            print("  encoder did not finish; the tail of the video may be missing")
        print(f"  {self.n} frame(s) -> {self.dir}"
              + (f", {self.dropped} dropped" if self.dropped else ""))
        return self.dir


def flights(root=DEFAULT_DIR):
    """Every flight folder under ``root``, oldest first."""

    return sorted(d for d in Path(root).glob("[0-9]*-[0-9]*") if d.is_dir())


def latest_flight(root=DEFAULT_DIR):
    """The most recent flight under ``root``, or ``root`` itself if it holds video."""

    found = flights(root)
    return found[-1] if found else Path(root)


def record(out_dir=DEFAULT_DIR, indices=None, width=1280, height=800, fps=60.0,
           rotate180=True, max_skew_s=None, preview=True):
    """Live preview; SPACE starts and stops recording, q quits. Returns the directory.

    ``max_skew_s`` is ``None`` on purpose. Re-reading until a pair lands close together is
    the calibration trick, and it costs seven frames out of eight; a flight is recorded
    once and cannot be re-shot, so every pair is kept and the skew is written down instead.
    """

    out_dir = Path(out_dir)
    # None means 'the ELPs, as of now'; see identify.elp_indices.
    idx = (identify.elp_indices() if indices is None
           else [indices] if isinstance(indices, int) else list(indices))
    tags = "AB"[:len(idx)]

    src = (sources.open_source(f"camera:{idx[0]}", width=width, height=height,
                               grayscale=True, rotate180=rotate180) if len(idx) == 1 else
           sources.open_stereo([f"camera:{i}" for i in idx], max_skew_s=max_skew_s,
                               width=width, height=height, grayscale=True,
                               rotate180=rotate180))

    work = queue.Queue(maxsize=QUEUE_DEPTH)
    dropped = [0]

    def _writer():
        while True:
            job = work.get()
            try:
                if job is None:
                    return
                for w, f in job:
                    w.write(f)
            finally:
                work.task_done()      # so close_flight can wait for the encoder to drain

    thread = threading.Thread(target=_writer, name="encode", daemon=True)
    thread.start()

    def close_flight(flight, writers, rows, n):
        """Finish one take: flush the encoder, close the files, write what they mean."""

        work.join()                             # every queued frame written before release
        for w in writers or []:
            w.release()
        if flight is None:
            return
        write_index(flight, tags, rows, src.skew_stats() if hasattr(src, "skew_stats") else {})
        (flight / "meta.json").write_text(json.dumps(
            {"camera_indices": idx, "mode": [width, height], "fps": fps,
             "rotate180": bool(rotate180), "n_frames": n, "dropped": dropped[0],
             "created": datetime.now().isoformat(timespec="seconds")}, indent=2))
        print(f"  {n} frame(s) -> {flight}")

    flight, writers, rows, recording, t0, n = None, None, [], False, 0.0, 0
    done = []
    sink = sources.Sink("flight recorder").open() if preview else None
    try:
      try:
        while True:
            item = src.read()
            if item is None:
                print("source ended")
                break
            t, payload = item
            frames = list(payload) if isinstance(payload, (list, tuple)) else [payload]

            if recording:
                if writers is None:
                    h, w = frames[0].shape[:2]
                    flight = new_flight(out_dir, tags)
                    writers = [cv2.VideoWriter(str(flight / tag / f"{tag}.mp4"),
                                               cv2.VideoWriter_fourcc(*FOURCC), fps,
                                               (w, h), False) for tag in tags]
                    if not all(x.isOpened() for x in writers):
                        raise OSError(f"no {FOURCC} writer on this build")
                try:
                    work.put_nowait(list(zip(writers, frames)))
                except queue.Full:
                    dropped[0] += 1
                stamps = getattr(src, "last_stamps", None) or (t,) * len(frames)
                rows.append((n, t, getattr(src, "last_skew", 0.0), stamps))
                n += 1

            if preview:
                view = np.hstack([f if f.ndim == 3 else
                                  cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in frames])
                cv2.putText(view, (f"REC {t - t0:5.1f}s  {n} frames" if recording
                                   else f"{n} frames   SPACE = record, q = quit"),
                            (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if recording else (255, 255, 255), 2, cv2.LINE_AA)
                key = sink.show(view)
                if key == ord("q"):
                    break
                if key == ord(" "):
                    recording = not recording
                    t0 = t
                    if recording:
                        rows, n = [], 0
                        print("recording")
                    else:                       # each take is its own flight folder
                        print(f"stopped at {n} frames")
                        close_flight(flight, writers, rows, n)
                        done.append(flight)
                        flight, writers = None, None
      except KeyboardInterrupt:
        # A cell has no q; interrupting must still close the flight cleanly below.
        print("\ninterrupted")
    finally:
        if sink is not None:
            sink.close()
        work.put(None)
        thread.join(timeout=30.0)
        if writers is not None:                 # quit while still rolling
            close_flight(flight, writers, rows, n)
            done.append(flight)
        stats = src.skew_stats() if hasattr(src, "skew_stats") else {}
        src.close()

    print(f"\n{len(done)} flight(s) in {out_dir}")
    if dropped[0]:
        print(f"  {dropped[0]} dropped: the encoder could not keep up")
    if stats:
        print(f"  capture skew: {stats}")
    return done


def write_index(out_dir, tags, rows, stats=None):
    """``frames.csv``: what turns two videos back into a timed stereo pair."""

    path = Path(out_dir) / "frames.csv"
    with open(path, "w") as fh:
        for k, v in (stats or {}).items():
            fh.write(f"# skew_{k}, {v}\n")
        fh.write("index,t_capture,skew_s,"
                 + ",".join(f"t_{tag.lower()}" for tag in tags) + "\n")
        for i, t, skew, stamps in rows:
            fh.write(f"{i},{t:.6f},{skew:.6f},"
                     + ",".join(f"{x:.6f}" for x in stamps) + "\n")
    return path


def read_index(rec_dir):
    """``(stamps, skews)`` from `write_index`, or ``(None, None)`` when it is missing.

    ``stamps`` is one row per frame and one column per camera. Without it the videos can
    still be replayed, but every pair has to be assumed simultaneous, which is the
    assumption `pose/theory.md` section 17 exists to remove.
    """

    path = Path(rec_dir) / "frames.csv"
    if not path.exists():
        return None, None
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    if len(lines) < 2:
        return None, None
    cols = lines[0].split(",")
    per_cam = [c for c in cols if c.startswith("t_") and c != "t_capture"]
    rows = [l.split(",") for l in lines[1:]]
    stamps = np.array([[float(r[cols.index(c)]) for c in per_cam] for r in rows])
    skews = np.array([float(r[cols.index("skew_s")]) for r in rows])
    return stamps, skews


def open_recording(rec_dir):
    """``(captures, stamps)`` for one flight: one `cv2.VideoCapture` per camera.

    ``rec_dir`` may be the flights root, in which case the most recent flight is opened.
    """

    rec_dir = latest_flight(rec_dir)
    videos = sorted(rec_dir.glob("*/*.mp4")) or sorted(rec_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no video in {rec_dir}")
    stamps, _ = read_index(rec_dir)
    return [cv2.VideoCapture(str(v)) for v in videos], stamps


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_DIR)
    p.add_argument("--indices", nargs="+", type=int, default=None,
                   help="default: whichever indices the two ELPs hold right now")
    p.add_argument("--mode", default="1280x800")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--no-flip", action="store_true")
    a = p.parse_args(argv)
    w, h = (int(v) for v in a.mode.lower().split("x"))
    record(a.out, a.indices, width=w, height=h, fps=a.fps, rotate180=not a.no_flip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
