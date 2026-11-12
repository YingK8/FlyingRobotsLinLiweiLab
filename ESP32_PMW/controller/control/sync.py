#!/usr/bin/env python3
"""Line up a run's three artefacts: serial traffic, video frames, controller rows.

A tilt experiment produces three files written by three different things, and the
question asked of them is always the same one: *the robot pitched over here -- what was
the firmware saying, and which video frame is that?* This answers it.

THE SYNC IS FREE, AND THAT IS THE DESIGN
----------------------------------------
`fly()` drives the coils and reads the cameras in one process, so there is no clock to
estimate. Everything is stamped with `time.monotonic()` at the moment it happened:

    frames.csv    t_capture, t_a, t_b   MonoCamera's grabber thread stamps each frame
    hover_run.log [t] -> cmd            what the host sent
                  [t] <- line           what the firmware said
                  [t] <> csv t=0        the run CSV's origin, written once

The run CSV is the odd one out: its `t` column is relative to that origin, because a
column of 9-digit absolute times at 500 Hz is a lot of bytes to say one number. The `<>`
marker is what converts it, so nothing here has to guess an offset or cross-correlate.

**Do not fit an offset between these clocks.** They are the same clock. If an alignment
here looks wrong the cause is a missing marker (a run from before 2026-09-01) or the
wrong flight folder, not drift -- and a fitted offset would hide both.

THE FIRMWARE'S OWN CLOCK IS SEPARATE, AND IS NOT USED FOR ALIGNMENT
------------------------------------------------------------------
`driveTelemetry` prints `t=<millis>` from the ESP32's clock at 2 Hz. That is a *device*
timebase: it starts at the board's boot, not the host's, and no one has measured its
drift against the host. It is parsed here as `t_device_ms` because it is the only way to
see a serial line that arrived late -- host stamp minus device stamp jumping means the
USB buffer backed up -- but alignment to video always uses the host stamp, which is
taken at the instant the line was read.

    uv run python controller/control/sync.py                    # self-check
    uv run python controller/control/sync.py results/takeoff/20260901_163255.csv
"""

from __future__ import annotations

import re
import sys
from bisect import bisect_left
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: `[123.456] -> seq=go` / `[123.456] <- state=2 ...` / `[123.456] <> csv t=0`
LINE_RE = re.compile(r"^\[(\d+\.\d+)\]\s+(->|<-|<>)\s*(.*)$")
#: the ESP32's own millis(), from driveTelemetry's shared 2 Hz line
DEV_MS_RE = re.compile(r"\bt=(\d+)\b")


def read_log(path):
    """``(entries, t0)`` from a run log. ``t0`` is the run CSV's origin, or None.

    ``entries`` is ``[(t_monotonic, direction, text, t_device_ms)]``, direction being
    ``->`` sent or ``<-`` received. Unstamped lines are dropped rather than guessed at:
    a line with no clock cannot be aligned to anything, and inventing one for it would
    put a confident wrong marker on the video.
    """

    entries, t0 = [], None
    for raw in Path(path).read_text(errors="replace").splitlines():
        m = LINE_RE.match(raw)
        if not m:
            continue
        t, direction, text = float(m.group(1)), m.group(2), m.group(3)
        if direction == "<>":
            t0 = t
            continue
        dev = DEV_MS_RE.search(text) if direction == "<-" else None
        entries.append((t, direction, text, int(dev.group(1)) if dev else None))
    return entries, t0


def read_frames(flight_dir):
    """``[(t_capture, index)]`` for one flight, from `camera/record.py`'s `frames.csv`."""

    from controller.camera import record

    stamps, _ = record.read_index(flight_dir)
    if stamps is None:
        return []
    # Column 0 is camera A. The per-camera stamps differ by the capture skew, which is
    # the thing `frames.csv` exists to record; A is the reference everywhere else too.
    return [(float(row[0]), i) for i, row in enumerate(stamps)]


def find_flight(entries, root=None):
    """The flight folder whose frames overlap this log, or None.

    Matched on the monotonic clock rather than on the folder's wall-clock name. Both
    would usually work, but only one of them is exact: `time.monotonic()` is unique
    within a boot, whereas two takes can share a second and a run can straddle one.
    """

    from controller.camera import record

    if not entries:
        return None
    lo, hi = entries[0][0], entries[-1][0]
    best, best_n = None, 0
    for d in record.flights(root or record.DEFAULT_DIR):
        f = read_frames(d)
        if not f:
            continue
        n = sum(1 for t, _ in f if lo <= t <= hi)
        if n > best_n:
            best, best_n = d, n
    return best


class Run:
    """One attempt, with its three files reduced to a single clock."""

    def __init__(self, csv_path=None, log_path=None, flight_dir=None, root=None):
        self.csv_path = Path(csv_path) if csv_path else None
        self.log_path = Path(log_path) if log_path else ROOT / "hover_run.log"
        self.entries, self.t0 = read_log(self.log_path)
        self.flight = Path(flight_dir) if flight_dir else find_flight(self.entries, root)
        self.frames = read_frames(self.flight) if self.flight else []
        self._ft = [t for t, _ in self.frames]

    # ---- the two questions ----------------------------------------------------------
    def frame_at(self, t, csv_time=False):
        """Nearest video frame to monotonic ``t``. ``(index, t_frame, dt_s)`` or None.

        ``csv_time=True`` takes ``t`` from the run CSV's `t` column instead, which is
        relative to the `<>` origin.
        """

        if not self._ft:
            return None
        t = self.to_monotonic(t) if csv_time else t
        i = bisect_left(self._ft, t)
        cand = [j for j in (i - 1, i) if 0 <= j < len(self._ft)]
        j = min(cand, key=lambda j: abs(self._ft[j] - t))
        return self.frames[j][1], self._ft[j], self._ft[j] - t

    def serial(self, t_from, t_to, csv_time=False):
        """Every stamped serial line in a window, oldest first."""

        a, b = (self.to_monotonic(t_from), self.to_monotonic(t_to)) if csv_time \
            else (t_from, t_to)
        return [e for e in self.entries if a <= e[0] <= b]

    # ---- clock plumbing -------------------------------------------------------------
    def to_monotonic(self, t_csv):
        """Run-CSV time -> the monotonic clock the video and the serial log share."""

        if self.t0 is None:
            raise ValueError(
                f"{self.log_path} has no '<> csv t=0' marker, so the CSV's relative time "
                "cannot be placed on the video's clock. Runs before 2026-09-01 have none; "
                "pass monotonic times instead of csv_time=True.")
        return self.t0 + float(t_csv)

    def to_csv_time(self, t_mono):
        """The inverse, for reading a serial event off against the controller's rows."""

        if self.t0 is None:
            raise ValueError(f"{self.log_path} has no '<> csv t=0' marker")
        return float(t_mono) - self.t0

    def summary(self):
        sent = sum(1 for e in self.entries if e[1] == "->")
        got = len(self.entries) - sent
        span = (self.entries[-1][0] - self.entries[0][0]) if self.entries else 0.0
        out = [f"log      {self.log_path}",
               f"         {sent} sent, {got} received, {span:.1f} s span",
               f"         csv origin {'t0=%.3f' % self.t0 if self.t0 else 'MISSING'}"]
        if self.flight and self.frames:
            lo, hi = self._ft[0], self._ft[-1]
            # A run whose serial and video barely overlap is the failure this catches:
            # usually the wrong flight folder, occasionally a take that stopped early.
            cover = sum(1 for t in self._ft
                        if self.entries and self.entries[0][0] <= t <= self.entries[-1][0])
            out += [f"video    {self.flight}",
                    f"         {len(self.frames)} frames, {hi - lo:.1f} s, "
                    f"{cover} inside the log's span"]
        else:
            out += ["video    no flight folder overlaps this log"]
        return "\n".join(out)


def report(csv_path=None, log_path=None, flight_dir=None):
    """Print the alignment, plus the serial traffic around the biggest tilt in the CSV."""

    run = Run(csv_path, log_path, flight_dir)
    print(run.summary())
    if csv_path is None or run.t0 is None or not run.frames:
        return run

    import numpy as np

    from controller.control import takeoff_report

    d = takeoff_report.load(csv_path)
    tilt = d.get("tilt_deg")
    if tilt is None or not np.isfinite(tilt).any():
        print("\nno tilt column in this CSV")
        return run
    k = int(np.nanargmax(tilt))
    t = float(d["t"][k])
    hit = run.frame_at(t, csv_time=True)
    print(f"\nlargest tilt {tilt[k]:.2f} deg at csv t={t:.3f}s")
    print(f"  video frame {hit[0]} of {len(run.frames)} (off by {hit[2] * 1e3:+.1f} ms)")
    print("  serial around it:")
    for e in run.serial(t - 0.25, t + 0.25, csv_time=True):
        print(f"    [{run.to_csv_time(e[0]):+.3f}] {e[1]} {e[2]}")
    return run


def _self_check():
    """Round-trip a synthetic log and flight through the aligner.

    Asserts the alignment, not the parsing: a frame index that is off by one is the
    failure this catches, and it is invisible in a log that merely parses.
    """

    import shutil
    import tempfile

    import numpy as np

    from controller.camera import record

    tmp = Path(tempfile.mkdtemp(prefix="sync-"))
    try:
        # A flight at a plausible 227 Hz, starting a second into the run.
        fw = record.FlightWriter(tmp, tags="AB", fps=227.0)
        base, dt = 1000.0, 1.0 / 227.0
        f = [np.zeros((32, 40), np.uint8), np.zeros((32, 40), np.uint8)]
        for i in range(300):
            t = base + 1.0 + i * dt
            fw.add(t, f, (t, t + 0.002), 0.002)
        flight = fw.close()

        log = tmp / "run.log"
        log.write_text(
            f"[{base + 0.500:.3f}] -> seq=go\n"
            f"[{base + 0.900:.3f}] <> csv t=0\n"
            f"[{base + 1.500:.3f}] <- t=12345 freq=210.0 | I[A]: 1,1,1,1\n"
            "a line with no stamp at all\n"
            f"[{base + 2.000:.3f}] -> stop\n")

        run = Run(log_path=log, flight_dir=flight)
        assert len(run.entries) == 3, run.entries          # the unstamped line is dropped
        assert run.t0 == base + 0.900, run.t0
        assert run.entries[1][3] == 12345, run.entries[1]  # device millis parsed

        # csv t=0.6 is base+1.5, which is frame 0.5/dt into a take starting at base+1.0
        idx, t_f, off = run.frame_at(0.6, csv_time=True)
        assert idx == round(0.5 / dt), (idx, round(0.5 / dt))
        assert abs(off) <= dt, off
        assert abs(run.to_csv_time(run.to_monotonic(1.234)) - 1.234) < 1e-9

        got = run.serial(0.5, 0.7, csv_time=True)
        assert len(got) == 1 and got[0][1] == "<-", got

        # find_flight must pick this take out by clock overlap alone.
        assert find_flight(run.entries, root=tmp) == flight

        # A log with no marker must refuse, not silently align to zero.
        (tmp / "old.log").write_text(f"[{base:.3f}] -> seq=go\n")
        try:
            Run(log_path=tmp / "old.log", flight_dir=flight).to_monotonic(0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("a log with no origin must refuse to convert")

        print(f"sync: self-check passed ({len(run.frames)} frames, frame {idx} at csv t=0.6)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report(*sys.argv[1:])
    else:
        _self_check()
