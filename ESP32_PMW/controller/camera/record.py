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
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

from controller.camera import identify
from controller.camera import sources

DEFAULT_DIR = HERE.parents[1] / "results" / "flights"
FOURCC = "avc1"  # H.264 in an .mp4. See the module docstring for why lossy is
# acceptable in this file and nowhere near calibration.
QUEUE_DEPTH = 64
CAP_FPS_CEILING = {
    (1280, 800): 121.4,
    (1280, 720): 121.2,
    (1024, 768): 120.3,
    (800, 600): 98.8,
    (640, 480): 209.9,
    (640, 400): 271.3,
    (320, 240): 421.7,
    (160, 120): 285.4,
}
CAP_FPS_MAX_REQUEST = 1000.0


def _cap_request(cap_fps, width, height):
    """The rate to ASK the sensor for, in fps, or None to ask for nothing.

    See `CAP_FPS_CEILING` and ``record``'s ``cap_fps``.
    """

    if cap_fps is None or str(cap_fps).strip().lower() == "max":
        return CAP_FPS_CEILING.get((int(width), int(height)), CAP_FPS_MAX_REQUEST)
    value = float(cap_fps)
    return value if value > 0 else None


def _warn_playback_rate(take):
    """Say so when the mp4's declared rate is far from what was actually captured.

    `fps` in meta.json is what the file says; `fps_measured` is what happened. A wide gap
    means the video does not play at true speed. That is cosmetic for this analysis, which
    times everything from `frames.csv`, but it is exactly the kind of quiet inconsistency
    that misleads a later reader, so it is said out loud at close.
    """

    try:
        m = json.loads((Path(take) / "meta.json").read_text())
        declared = float(m.get("fps") or 0.0)
        measured = float(m.get("fps_measured") or 0.0)
    except Exception:
        return
    if declared > 0 and measured > 0 and abs(measured - declared) / declared > 0.10:
        print(
            f"  !! {Path(take).name}: the mp4 declares {declared:g} fps but {measured:.1f} "
            f"fps was captured ({measured / declared:.2f}x). It will not play at true "
            f"speed; `fps_measured` in meta.json is the rate to trust."
        )


def _writer_fps(requested, declared, granted):
    """The rate to declare on the mp4, from the request, the old default, and the grant."""

    good = [float(g) for g in granted if g and float(g) > 1.0]
    if requested is None:
        return float(declared)
    return min(good) if good else float(requested)


# ---- one folder per flight ----------------------------------------------------------
def _skew(src):
    """Capture-skew stats, or {} for a source that does not track them."""

    return src.skew_stats() if hasattr(src, "skew_stats") else {}


def take_suffix(note):
    """``_<design><hz>`` parsed from a campaign note ("half ring 90Hz x10 ..."), else ''.

    The operator reads the design and frequency off the folder name in Finder, so the
    campaign name goes into the directory itself, not only into meta.json (2026-09-21).
    """

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
    for k in range(1, 100):  # two takes inside one second must not merge
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

    def __init__(self, out_dir=DEFAULT_DIR, tags="AB", fps=120.0, meta=None):
        self.dir = new_flight(out_dir, tags, take_suffix((meta or {}).get("note")))
        self.tags, self.fps, self.meta = tags, float(fps), dict(meta or {})
        self.n, self.dropped, self.errors = 0, 0, 0
        self._t_first = self._t_last = None  # for fps_measured, see close()
        self.writers, self.size = None, None
        self._released = False  # set by the encoder thread once it finalises
        # Header up front, rows as they land: see the class docstring.
        self._csv = open(self.dir / "frames.csv", "w", buffering=1)
        # `written` is 0 for a frame the queue dropped: its row is kept (the capture time is
        # real) but the mp4 has no frame for it. Without the flag, pairing mp4 frame i with
        # row i put every later frame at an earlier row's time -- 91 of 125 design-B takes,
        # seconds of error by the cut (2026-09-13). `read_index` drops the unwritten rows.
        self._csv.write(
            "index,t_capture,skew_s,"
            + ",".join(f"t_{t.lower()}" for t in self.tags)
            + ",written\n"
        )
        self._work = queue.Queue(maxsize=QUEUE_DEPTH)
        self._thread = threading.Thread(target=self._run, name="encode", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            job = self._work.get()
            try:
                if job is None:
                    # Finalise HERE, on the thread that wrote every frame. close() used to
                    # release from the caller's thread after a 30 s join timeout, so on a
                    # long take the queue was still draining and release() ran CONCURRENTLY
                    # with write() -- which leaves mdat unpatched and no moov, i.e. an
                    # unplayable file. Measured 2026-09-01: 11 of 13 takes were lost that
                    # way, and the survivors were the short ones.
                    for w in self.writers or []:
                        try:
                            w.release()
                        except Exception as e:  # noqa: BLE001
                            self.errors += 1
                            print(f"  release failed (file unplayable): {e}")
                    self._released = True
                    return
                for w, f in job:
                    w.write(f)
            except Exception as e:  # noqa: BLE001
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
            self.writers = [
                cv2.VideoWriter(
                    str(self.dir / tag / f"{tag}.mp4"),
                    cv2.VideoWriter_fourcc(*FOURCC),
                    self.fps,
                    (w, h),
                    False,
                )
                for tag in self.tags
            ]
            if not all(x.isOpened() for x in self.writers):
                raise OSError(f"no {FOURCC} writer on this build")
        written = 1
        try:
            self._work.put_nowait(list(zip(self.writers, frames)))
        except queue.Full:
            self.dropped += 1
            written = 0
        if self._t_first is None:
            self._t_first = t
        self._t_last = t
        st = stamps or (t,) * len(frames)
        self._csv.write(
            f"{self.n},{t:.6f},{skew:.6f},"
            + ",".join(f"{x:.6f}" for x in st)
            + f",{written}\n"
        )
        self.n += 1

    def measured_fps(self):
        """Frames per second this take actually achieved, or 0.0 for a take too short."""

        span = (self._t_last - self._t_first) if self._t_first is not None else 0.0
        return (self.n - 1) / span if self.n > 1 and span > 0 else 0.0

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
        (self.dir / "meta.json").write_text(
            json.dumps(
                {
                    **self.meta,
                    "mode": self.size,
                    "fps": self.fps,
                    # What the take ACTUALLY ran at. `fps` above is the mp4 header, fixed before
                    # the first frame arrived; this is measured, and the two disagreeing is
                    # normal -- the camera is not asked for a rate and delivers its mode's max.
                    "fps_measured": round(self.measured_fps(), 2),
                    "n_frames": self.n,
                    "dropped": self.dropped,
                    "encoder_errors": self.errors,
                    "skew": stats or {},
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )

        # Never `queue.join()`: it waits on a counter only the worker decrements, so a
        # worker that died hangs the caller. Sentinel plus a bounded thread join.
        try:
            self._work.put(None, timeout=5.0)
        except queue.Full:
            pass
        # Generous: the encoder has to drain the whole queue before it can finalise, and
        # cutting it short is what corrupted the file rather than merely truncating it.
        self._thread.join(timeout=120.0)
        if not self._released:
            # The worker died or never saw the sentinel. Release here as a last resort --
            # it is not safe if the worker is still writing, but an unreleased writer is
            # certainly unplayable, so this can only improve matters.
            print("  encoder did not finalise; releasing from the caller as a fallback")
            for w in self.writers or []:
                w.release()
        bad = [
            w.name
            for w in sorted(self.dir.glob("*/*.mp4"))
            if b"moov" not in w.read_bytes()[-1 << 20 :]
        ]
        if bad:
            print(f"  UNPLAYABLE, no moov atom: {', '.join(bad)} -- this take is lost")
        print(
            f"  {self.n} frame(s) -> {self.dir}"
            + (f", {self.dropped} dropped" if self.dropped else "")
        )
        return self.dir


def flights(root=DEFAULT_DIR):
    """Every flight folder under ``root``, oldest first."""

    return sorted(d for d in Path(root).glob("[0-9]*-[0-9]*") if d.is_dir())


def latest_flight(root=DEFAULT_DIR):
    """The most recent flight under ``root``, or ``root`` itself if it holds video."""

    found = flights(root)
    return found[-1] if found else Path(root)


def record(
    out_dir=DEFAULT_DIR,
    indices=None,
    width=1280,
    height=800,
    fps=120.0,
    rotate180=True,
    max_skew_s=None,
    preview=True,
    start=False,
    note=None,
    cap_fps=None,
):
    """Live preview; SPACE starts and stops recording, q quits. Returns the directory.

    ``note`` is free text written into ``meta.json`` verbatim -- the operator's own
    record of what this take was measuring (e.g. "150 Hz hold, ch0 carrier 30%"),
    which the folder timestamp cannot say.

    ``start=True`` rolls from the first frame and needs no key, which is the only way to
    shoot from a notebook cell: `sources.Sink.show` returns -1 inline, so SPACE never
    arrives and the window path's start/stop is unreachable there. Stop by interrupting
    the kernel -- the take is closed in the `finally` either way.

    ``max_skew_s`` is ``None`` on purpose. Re-reading until a pair lands close together is
    the calibration trick, and it costs seven frames out of eight; a flight is recorded
    once and cannot be re-shot, so every pair is kept and the skew is written down instead.

    ``cap_fps`` REQUESTS a capture rate from the sensor, which ``fps`` alone never did: that
    argument only declares the mp4's rate, and `open_stereo` was called with no rate at all,
    so the sensor ran at its own default and every take before 2026-09-24 was consumer-bound
    at ~185-192 fps. **Unset now asks for the mode's measured ceiling** (`CAP_FPS_CEILING`) --
    271.3 fps at 640x400, a different number at every other mode -- so the default is already
    the maximum the mode has been measured to give. Pass a number to override it, or ``0`` to
    request nothing. What the driver GRANTS is read back from `Source.actual` and used as the
    writer's rate, so the file plays at true speed; a grant that does not match what is
    finally delivered is reported at close, and `meta.json`'s ``fps_measured`` is the number
    the analysis should trust.

    Resolution is the floor on all of this, not a preference. The trace is a disc outline,
    and 640x400 puts ~130 px across it, while 320x240 is a CROP in which the rotor overflows
    the frame -- the ring fit lands on a 367 px ellipse in a 320 px image (`camera/theory.md`
    1.3). So spend the rate at a mode that resolves the rotor; do not chase 320x240's 420 fps.

    ``preview=False`` drops the window. `np.hstack` of the two frames, a `putText` and a
    window blit run on every frame, which is worth roughly a tenth of the rate: the same
    cameras at the same mode reach ~207-210 fps in `tilt_sweep.run`, which has no window.
    """

    out_dir = Path(out_dir)
    # None means 'the ELPs, as of now'; see identify.elp_indices.
    idx = (
        identify.elp_indices()
        if indices is None
        else [indices] if isinstance(indices, int) else list(indices)
    )
    tags = "AB"[: len(idx)]

    request = _cap_request(cap_fps, width, height)

    src = (
        sources.open_source(
            f"camera:{idx[0]}",
            width=width,
            height=height,
            grayscale=True,
            rotate180=rotate180,
            fps=request,
        )
        if len(idx) == 1
        else sources.open_stereo(
            [f"camera:{i}" for i in idx],
            max_skew_s=max_skew_s,
            width=width,
            height=height,
            grayscale=True,
            rotate180=rotate180,
            fps=request,
        )
    )

    granted = (
        [
            s.actual.get("fps", 0.0)
            for s in getattr(src, "sources", [src])
            if hasattr(s, "actual")
        ]
        if request is not None
        else []
    )
    if request is not None:
        print(
            f"capture: asked for {request:g} fps, driver reports "
            f"{', '.join(f'{g:g}' for g in granted) or 'nothing'}"
        )
    write_fps = _writer_fps(request, fps, granted)

    meta = {
        "camera_indices": idx,
        "rotate180": bool(rotate180),
        "cap_fps_requested": request,
        "cap_fps_granted": granted or None,
    }
    if note:
        meta["note"] = str(note)

    fw, recording, t0 = None, False, 0.0
    done = []
    if start:
        fw, recording = FlightWriter(out_dir, tags, write_fps, meta), True
        print(f"recording -> {fw.dir}   (interrupt the kernel to stop)")
    sink = sources.Sink("flight recorder").open() if preview else None
    try:
        try:
            while True:
                item = src.read()
                if item is None:
                    print("source ended")
                    break
                t, payload = item
                frames = (
                    list(payload) if isinstance(payload, (list, tuple)) else [payload]
                )

                if recording:
                    fw.add(
                        t,
                        frames,
                        getattr(src, "last_stamps", None),
                        getattr(src, "last_skew", 0.0),
                    )

                if preview:
                    view = np.hstack(
                        [
                            f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
                            for f in frames
                        ]
                    )
                    n = fw.n if fw else 0
                    cv2.putText(
                        view,
                        (
                            f"REC {t - t0:5.1f}s  {n} frames"
                            if recording
                            else f"{n} frames   SPACE = record, q = quit"
                        ),
                        (10, view.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255) if recording else (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    key = sink.show(view)
                    if key == ord("q"):
                        break
                    if key == ord(" "):
                        recording = not recording
                        t0 = t
                        if recording:  # each take is its own flight folder
                            fw = FlightWriter(out_dir, tags, write_fps, meta)
                            print("recording")
                        else:
                            print(f"stopped at {fw.n} frames")
                            done.append(fw.close(_skew(src)))
                            fw = None
        except KeyboardInterrupt:
            # A cell has no q; interrupting must still close the flight cleanly below.
            print("\ninterrupted")
    finally:
        if sink is not None:
            sink.close()
        if fw is not None:  # quit while still rolling
            done.append(fw.close(_skew(src)))
        stats = _skew(src)
        src.close()

    print(f"\n{len(done)} flight(s) in {out_dir}")
    if stats:
        print(f"  capture skew: {stats}")
    for take in done:
        _warn_playback_rate(take)
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
    videos = sorted(
        v for v in rec_dir.glob("*/*.mp4") if v.stem == v.parent.name
    ) or sorted(rec_dir.glob("*.mp4"))
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
    for i in range(90):  # long enough to outrun the 64-deep queue
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
    assert stamps is not None and stamps.shape == (kept, 2), (
        None if stamps is None else stamps.shape
    )
    assert len(skews) == kept, len(skews)
    rows = (out / "frames.csv").read_text().splitlines()
    assert rows[0].endswith(",written") and len(rows) == 91, (rows[0], len(rows))
    assert sum(int(r.rsplit(",", 1)[1]) for r in rows[1:]) == kept
    # the reader keeps written rows only, whatever the queue happened to do above
    fake = root / "fake"
    fake.mkdir()
    (fake / "frames.csv").write_text(
        "index,t_capture,skew_s,t_a,t_b,written\n"
        "0,0.0,0,0.0,0.0,1\n1,0.1,0,0.1,0.1,0\n2,0.2,0,0.2,0.2,1\n"
    )
    fs, _ = read_index(fake)
    assert fs.shape == (2, 2) and list(fs[:, 0]) == [0.0, 0.2], fs
    meta = json.loads((out / "meta.json").read_text())
    assert meta["fps"] == 30.0 and meta["fps_measured"] > 0, meta
    # The mp4's declared rate must follow what the driver GRANTED, not what was asked.
    # Writing a 271 fps take at 240 plays it 13% slow; writing it at the old 120, twice
    # too fast; and an unrequested rate must leave the caller's declared value untouched.
    assert _writer_fps(None, 120.0, []) == 120.0
    assert _writer_fps(None, 120.0, [271.0]) == 120.0
    assert _writer_fps(240.0, 120.0, [271.3, 271.0]) == 271.0
    assert _writer_fps(240.0, 120.0, []) == 240.0
    assert (
        _writer_fps(240.0, 120.0, [0.0, 0.0]) == 240.0
    )  # implausible grant is no grant
    # Unset and 'max' both ask for the mode's measured ceiling, and it is PER MODE: 271.3 is
    # right at 640x400, while 1280x800 tops out at 121.4 and asking 271 there is meaningless.
    assert _cap_request(None, 640, 400) == 271.3
    assert _cap_request("max", 640, 400) == 271.3
    assert _cap_request(None, 1280, 800) == 121.4
    assert _cap_request(None, 111, 222) == CAP_FPS_MAX_REQUEST
    assert _cap_request(240, 640, 400) == 240.0
    assert _cap_request(0, 640, 400) is None  # the escape hatch
    assert _writer_fps(_cap_request(None, 640, 400), 120.0, [271.3]) == 271.3
    caps, _ = open_recording(out)
    for c in caps:
        assert c.isOpened(), "written mp4 will not reopen"
        c.release()
    if tmp is None:
        shutil.rmtree(root, ignore_errors=True)
    print(
        f"record: self-check passed (90 frames, both mp4s finalised{'' if fw.dropped == 0 else f', {fw.dropped} dropped'})"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=DEFAULT_DIR)
    p.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=None,
        help="default: whichever indices the two ELPs hold right now",
    )
    p.add_argument("--mode", default="1280x800")
    p.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="the mp4's declared rate, used when no --cap-fps is given",
    )
    p.add_argument(
        "--cap-fps",
        default=None,
        help="request this capture rate from the sensor, or 'max'; unset asks for the "
        "mode's measured ceiling (271.3 fps at 640x400), and 0 asks for nothing. What the "
        "driver grants becomes the mp4's rate. Do not use 320x240 to buy rate: it is a "
        "crop and the rotor overflows it",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="no window; the per-frame hstack and blit cost about a tenth of "
        "the rate",
    )
    p.add_argument("--no-flip", action="store_true")
    p.add_argument(
        "--note",
        default=None,
        help='free text into meta.json, e.g. "align C 100Hz coil A C shut down"',
    )
    p.add_argument(
        "--start",
        action="store_true",
        help="roll from the first frame instead of waiting for SPACE",
    )
    p.add_argument("--self-check", action="store_true", help="no camera needed")
    a = p.parse_args(argv)
    if a.self_check:
        _self_check()
        return 0
    w, h = (int(v) for v in a.mode.lower().split("x"))
    record(
        a.out,
        a.indices,
        width=w,
        height=h,
        fps=a.fps,
        rotate180=not a.no_flip,
        preview=not a.no_preview,
        start=a.start,
        note=a.note,
        cap_fps=a.cap_fps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
