#!/usr/bin/env python3
"""Real-time coil current-sense visualiser: a 4x4 dot matrix, split into four
2x2 quadrants (top-left=A, top-right=B, bottom-left=C, bottom-right=D). Each
quadrant's 4 dots move together, fading dark grey -> yellow as that channel's
CS voltage goes 0 -> 500 mV.

Reads the "I[A]: A=.. B=.. C=.. D=.." telemetry fragment (telemetry.h,
printCurrentAndDuty -- shared by main_current_pid.cpp, main_experiment.cpp,
main_pi_profile.cpp), and converts amps -> CS-pin mV using the per-channel
VNH5019 SENS gain (A/V) -- same values as src/constants.h's SENS[], so
there is one source of truth for that calibration on both sides.

`CoilCurrentMatrixView` is the reusable piece: any script that already owns
a live serial connection (tools/run_experiment.py) constructs one, then
calls `.feed_line(line)` + `.refresh()` on each line it reads, right next to
where it already prints/logs that line. This runs the visualiser in-process
-- no separate port (which would conflict with the owning script) and no
tail-following a log file (which lags). tools/experiment_tui.py's matrix
pane reuses this module's channel/quadrant layout and amps_to_mv()
directly, with curses color pairs instead of matplotlib.

This module's own CLI (below) is for standalone/offline use only -- e.g.
replaying an already-saved log after the fact, or watching a port that
genuinely has no other reader:
  --port PORT      open the serial port directly (SerialComm, byte-by-byte
                    reads). Fine for a port nothing else has open, but do NOT
                    use this alongside another script that already owns the
                    port -- byte-by-byte reads silently drop bytes under
                    fast-bursting telemetry (confirmed on hardware), and two
                    processes can't share one serial device.
  --log-file FILE   tail-follow a log file as it's being written.

Usage:
  # offline replay of a saved log:
  uv run python tools/coil_current_matrix.py --log-file state_space_cw_2A.log
  # live from a port nothing else has open:
  uv run python tools/coil_current_matrix.py --port /dev/ttyUSB0
"""
import argparse
import os
import re
import sys

import matplotlib.pyplot as plt

for _backend in ("QtAgg", "TkAgg", "GTK4Agg", "GTK3Agg", "MacOSX"):
    try:
        plt.switch_backend(_backend)  # raises if the backend's deps are missing
        break
    except Exception:
        continue
else:
    sys.exit("no interactive matplotlib backend available -- install one of: "
              "PyQt5/PySide6 (pip install pyqt5) or Tk (apt install python3-tk)")
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_comm import SerialComm

# VNH5019 CS gain, A/V, per channel -- matches src/constants.h's SENS[].
DEFAULT_SENS = {"A": 15.26, "B": 15.28, "C": 15.57, "D": 15.34}

CHANNELS = ("A", "B", "C", "D")
MV_MAX = 500.0
DOT_GREY = (0.25, 0.25, 0.25)
DOT_YELLOW = (1.0, 0.87, 0.0)
BG_GREY = "#c8c8c8"

_LINE_RE = re.compile(
    r"I\[A\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)"
)

# quadrant origin (col, row) in the 4x4 grid, row 0 = top
_QUADRANT_ORIGIN = {"A": (0, 2), "B": (2, 2), "C": (0, 0), "D": (2, 0)}


def amps_to_mv(channel: str, amps: float) -> float:
    """Invert C=diag(1/SENS): CS-pin volts = amps / SENS_i (A/V)."""
    return 1000.0 * amps / DEFAULT_SENS[channel]


def dot_color(mv: float):
    frac = max(0.0, min(1.0, mv / MV_MAX))
    return tuple(g + frac * (y - g) for g, y in zip(DOT_GREY, DOT_YELLOW))


def build_figure():
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor(BG_GREY)
    ax.set_facecolor(BG_GREY)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    dots = {}
    labels = {}
    for ch, (cx, cy) in _QUADRANT_ORIGIN.items():
        for dx in range(2):
            for dy in range(2):
                x, y = cx + dx, cy + dy
                circ = Circle((x, y), 0.32, color=DOT_GREY, zorder=2)
                ax.add_patch(circ)
                dots.setdefault(ch, []).append(circ)
        lx, ly = cx + 0.5, cy + 0.5
        labels[ch] = ax.text(lx, ly, f"{ch}\n0mV", ha="center", va="center",
                              fontsize=9, color="white", zorder=3,
                              fontweight="bold")
    fig.tight_layout()
    return fig, ax, dots, labels


class CoilCurrentMatrixView:
    """Non-blocking, embeddable version of the dot-matrix display. Construct
    once, then call feed_line() + refresh() per line from a script's own
    serial-read loop; close() when that loop exits."""

    def __init__(self):
        plt.ion()
        self.fig, self.ax, self._dots, self._labels = build_figure()
        self.mv = {ch: 0.0 for ch in CHANNELS}
        self.fig.show()

    def feed_line(self, line: str) -> bool:
        """Parse one telemetry line, updating internal state. Returns True
        if the line matched (i.e. there's something new to refresh())."""
        m = _LINE_RE.search(line)
        if not m:
            return False
        for ch, raw in zip(CHANNELS, m.groups()):
            self.mv[ch] = max(0.0, amps_to_mv(ch, float(raw)))
        return True

    def refresh(self):
        """Push current state to the dots/labels and pump the GUI event loop
        without blocking (caller's read loop keeps driving the timing)."""
        for ch in CHANNELS:
            color = dot_color(self.mv[ch])
            for circ in self._dots[ch]:
                circ.set_color(color)
            self._labels[ch].set_text(f"{ch}\n{self.mv[ch]:.0f}mV")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close(self.fig)


class LogFollower:
    """tail -f a growing log file; new_lines() returns whatever complete
    lines have appeared since the last call (possibly none)."""

    def __init__(self, path: str):
        self._f = open(path, "r")

    def new_lines(self):
        lines = self._f.readlines()
        return [l.rstrip("\n") for l in lines]

    def close(self):
        self._f.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", default=None,
                      help="read live from this serial port (see docstring caveats)")
    src.add_argument("--log-file", default=None,
                      help="tail-follow this log file instead of the serial port")
    p.add_argument("--baud", type=int, default=115200,
                    help="only used with --port")
    p.add_argument("--interval-ms", type=int, default=50,
                    help="redraw poll interval (default 50ms)")
    args = p.parse_args()

    if args.log_file:
        source = LogFollower(args.log_file)
        get_lines = source.new_lines
    else:
        link = SerialComm(port=args.port, baud=args.baud)
        source = link

        def get_lines():
            lines = []
            while True:
                line = link.handle_serial_comm()
                if line is None:
                    break
                lines.append(line)
            return lines

    view = CoilCurrentMatrixView()
    view.fig.canvas.mpl_connect("close_event", lambda _e: sys.exit(0))

    try:
        while True:
            for line in get_lines():
                view.feed_line(line)
            view.refresh()
            plt.pause(args.interval_ms / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        view.close()


if __name__ == "__main__":
    main()
