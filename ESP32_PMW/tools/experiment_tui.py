#!/usr/bin/env python3
"""Curses TUI for tools/run_experiment.py: two-column layout (telemetry
stream on the left, an ASCII rendition of the coil-current matrix on the
right) with a bottom status/control bar. Reuses coil_current_matrix.py's
channel/quadrant layout and amps<->mV conversion for the matrix pane -- only
the drawing backend (curses color pairs instead of matplotlib) differs,
since this renders inside the terminal run_experiment.py is already running
in, not a separate window.

Not meant to be imported by anything except tools/run_experiment.py, which
owns the SerialSession and passes it in already open/reset.

Keybindings:
  r  run/stop toggle -- for current_pid/pi_profile (still WAITING-gated),
     sends the start command; for --fw experiment (autonomous, no WAITING),
     just arms recording. Always turns recording on if it wasn't already;
     while running, sends the e-stop.
  s  immediate e-stop, any state.
  w  toggle recording on/off without affecting run state.
  :  command-line mode -- type an arbitrary command (kp=2.5, dir=cw, ...)
     for current_pid/pi_profile live tuning, Enter to send.
  q  quit (sends e-stop first).
"""
from __future__ import annotations

import curses
import os
import re
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coil_current_matrix import CHANNELS, MV_MAX, _QUADRANT_ORIGIN, amps_to_mv

_TELEMETRY_RE = re.compile(
    r"I\[A\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)"
    r"(?:\s*\|\s*duty\[%\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+))?"
)

WAITING_MARKER = "waiting for start"
ESTOP_MARKER = "estop ->"
RUNNING_MARKERS = ("-> ramping", "-> running experiment")
ARMING_MARKER = "arming"

# Phase/running detection has two paths: one-time transition banners (fast,
# but a single line -- confirmed on hardware to occasionally get corrupted
# or dropped by a brownout right as coils energize, the exact moment a
# banner like "START -> running experiment" is printed) and the telemetry
# lines below (repeat every tick, so a single lost line self-heals within
# ~1 tick instead of permanently desyncing the TUI's running/phase state).
_PHASE_NUM_RE = re.compile(r"\bphase=(\d+)\b")
_LABEL_TELEMETRY_RE = re.compile(r"\bt=\d+\s+label=")
# Shared enum ordering across main_current_pid.cpp/main_pi_profile.cpp
# (see lib/ExperimentPhase: both insert WAITING right after ARMING).
_PID_PHASE_NAMES = {0: "ARMING", 1: "WAITING", 2: "RAMP_UP", 3: "HOLD",
                    4: "ENDING", 5: "STOPPED", 6: "PROFILE_LOAD_FAILED"}
_PID_RUNNING_PHASES = {2, 3, 4}  # RAMP_UP, HOLD, ENDING

_PAIR_GREY = 1
_PAIR_YELLOW_DIM = 2
_PAIR_YELLOW_MID = 3
_PAIR_YELLOW_HOT = 4
_PAIR_RED = 5
_PAIR_GREEN = 6


def _color_bucket(frac: float) -> int:
    """0..1 current fraction -> curses color-pair index (grey -> yellow)."""
    if frac < 0.05:
        return _PAIR_GREY
    if frac < 0.35:
        return _PAIR_YELLOW_DIM
    if frac < 0.7:
        return _PAIR_YELLOW_MID
    return _PAIR_YELLOW_HOT


def run(session, fw, out_dir, log_f, t0, needs_start, done_markers, pending_commands,
        auto_start=False) -> bool:
    """Blocking. Returns True once a done/e-stop banner was seen, False if
    the user quit first. Always sends the e-stop command on the way out.
    needs_start: False for --fw experiment (autonomous, no WAITING gate)."""
    result = {"seen_done": False}
    curses.wrapper(_main, session, fw, out_dir, log_f, t0, needs_start, done_markers,
                    pending_commands, auto_start, result)
    return result["seen_done"]


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_PAIR_GREY, curses.COLOR_WHITE, -1)
    curses.init_pair(_PAIR_YELLOW_DIM, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_YELLOW_MID, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_YELLOW_HOT, curses.COLOR_YELLOW, -1)
    curses.init_pair(_PAIR_RED, curses.COLOR_RED, -1)
    curses.init_pair(_PAIR_GREEN, curses.COLOR_GREEN, -1)


def _main(stdscr, session, fw, out_dir, log_f, t0, needs_start, done_markers,
          pending_commands, auto_start, result):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(30)
    _init_colors()

    mv = {ch: 0.0 for ch in CHANNELS}
    duty = {ch: 0.0 for ch in CHANNELS}
    log_lines = deque(maxlen=1000)
    phase = "?"
    trigger_seen = False
    sent_pending = False
    running = False
    recording = False
    rec_file = None
    rec_index = 0
    done = False
    cmd_mode = False
    cmd_buf = ""
    quit_requested = False

    def send(cmd):
        session.send(cmd)
        log_lines.append(f"  -> sent {cmd!r}")

    def start_recording():
        nonlocal recording, rec_file, rec_index
        rec_index += 1
        path = os.path.join(out_dir, f"recording_{rec_index}.csv")
        rec_file = open(path, "w")
        rec_file.write("t_s,phase,A,B,C,D,dutyA,dutyB,dutyC,dutyD\n")
        recording = True
        log_lines.append(f"  -> recording -> {os.path.basename(path)}")

    def stop_recording():
        nonlocal recording, rec_file
        if rec_file:
            rec_file.close()
            rec_file = None
        recording = False

    try:
        while True:
            ch_in = stdscr.getch()
            if cmd_mode:
                if ch_in in (curses.KEY_ENTER, 10, 13):
                    if cmd_buf.strip():
                        send(cmd_buf.strip())
                    cmd_buf = ""
                    cmd_mode = False
                elif ch_in == 27:  # ESC
                    cmd_buf = ""
                    cmd_mode = False
                elif ch_in in (curses.KEY_BACKSPACE, 127, 8):
                    cmd_buf = cmd_buf[:-1]
                elif 32 <= ch_in < 127:
                    cmd_buf += chr(ch_in)
            elif ch_in == ord('q'):
                quit_requested = True
            elif ch_in == ord('s'):
                send("s")
            elif ch_in == ord('r'):
                if not running:
                    if needs_start:
                        send("start")
                    if not recording:
                        start_recording()
                else:
                    send("s")
            elif ch_in == ord('w'):
                if recording:
                    stop_recording()
                else:
                    start_recording()
            elif ch_in == ord(':'):
                cmd_mode = True
                cmd_buf = ""

            if quit_requested:
                break

            line = session.readline()
            if line:
                log_lines.append(line)
                log_f.write(f"{time.time() - t0:7.2f}s  {line}\n")
                log_f.flush()

                low = line.lower()
                if WAITING_MARKER in low:
                    phase = "WAITING"
                    running = False
                    if needs_start and not trigger_seen:
                        trigger_seen = True
                        if not sent_pending:
                            for cmd in pending_commands:
                                send(cmd)
                            sent_pending = True
                        if auto_start:
                            send("start")
                elif ESTOP_MARKER in low:
                    phase = "STOPPED"
                    running = False
                    done = True
                elif any(m in low for m in RUNNING_MARKERS):
                    phase = "RUNNING"
                    running = True
                elif ARMING_MARKER in low:
                    phase = "ARMING"
                    running = False
                    if not needs_start and not trigger_seen:
                        # main_experiment.cpp is autonomous -- no WAITING
                        # banner to hang pending_commands off of, and no
                        # "start" to send (it begins on its own).
                        trigger_seen = True
                        if not sent_pending:
                            for cmd in pending_commands:
                                send(cmd)
                            sent_pending = True
                elif any(m in low for m in done_markers):
                    phase = "DONE"
                    running = False
                    done = True

                # Authoritative/self-healing: telemetry repeats every tick,
                # so even if the one-time transition banner above was lost,
                # this re-syncs running/phase within ~1 tick instead of
                # leaving them permanently stuck on stale state.
                m_phase = _PHASE_NUM_RE.search(line)
                if m_phase:
                    num = int(m_phase.group(1))
                    if num in _PID_PHASE_NAMES:
                        phase = _PID_PHASE_NAMES[num]
                        running = num in _PID_RUNNING_PHASES
                        if phase in ("STOPPED", "PROFILE_LOAD_FAILED"):
                            done = True
                elif fw == "experiment" and _LABEL_TELEMETRY_RE.search(line):
                    # main_experiment.cpp only emits this telemetry format
                    # while RUNNING -- seeing it is proof enough on its own.
                    phase = "RUNNING"
                    running = True

                m = _TELEMETRY_RE.search(line)
                if m:
                    vals = m.groups()
                    for i, chn in enumerate(CHANNELS):
                        mv[chn] = max(0.0, amps_to_mv(chn, float(vals[i])))
                        if vals[4 + i] is not None:
                            duty[chn] = float(vals[4 + i])
                    if recording and rec_file:
                        rec_file.write(
                            f"{time.time() - t0:.3f},{phase},"
                            f"{vals[0]},{vals[1]},{vals[2]},{vals[3]},"
                            f"{duty['A']},{duty['B']},{duty['C']},{duty['D']}\n")
                        rec_file.flush()

            if done and auto_start:
                break

            _draw(stdscr, log_lines, mv, phase, fw, t0, recording, running,
                  rec_index, cmd_mode, cmd_buf)
    finally:
        # SAFETY: always e-stop on the way out, quit or done, matching every
        # other driver's invariant -- never leave coils energized just
        # because this session ended.
        try:
            session.send("s")
        except Exception:
            pass
        stop_recording()
        result["seen_done"] = done


def _draw(stdscr, log_lines, mv, phase, fw, t0, recording, running,
          rec_index, cmd_mode, cmd_buf):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    if h < 10 or w < 40:
        try:
            stdscr.addstr(0, 0, "terminal too small")
        except curses.error:
            pass
        stdscr.refresh()
        return

    body_h = h - 3  # reserve 3 lines for the status/control bar
    left_w = w // 2
    right_x0 = left_w + 2
    right_w = w - right_x0

    for y in range(body_h):
        try:
            stdscr.addch(y, left_w, curses.ACS_VLINE)
        except curses.error:
            pass

    visible = list(log_lines)[-body_h:]
    for i, line in enumerate(visible):
        try:
            stdscr.addstr(i, 0, line[:max(0, left_w - 1)])
        except curses.error:
            pass

    _draw_matrix(stdscr, body_h, right_x0, right_w, mv)

    rec_attr = curses.color_pair(_PAIR_RED) | curses.A_BOLD if recording else curses.color_pair(_PAIR_GREY)
    run_attr = curses.color_pair(_PAIR_GREEN) | curses.A_BOLD if running else curses.color_pair(_PAIR_RED)
    elapsed = time.time() - t0
    try:
        stdscr.addstr(body_h, 0, "-" * w)
        stdscr.addstr(body_h + 1, 0, "REC ")
        stdscr.addstr(body_h + 1, 4, "●", rec_attr)
        stdscr.addstr(body_h + 1, 6, " RUN ")
        stdscr.addstr(body_h + 1, 11, "●", run_attr)
        status = (f"  fw={fw} phase={phase} t={elapsed:6.1f}s"
                  + (f"  rec=recording_{rec_index}.csv" if recording else ""))
        stdscr.addstr(body_h + 1, 13, status[:max(0, w - 14)])
        if cmd_mode:
            stdscr.addstr(body_h + 2, 0, f":{cmd_buf}"[:max(0, w - 1)])
        else:
            stdscr.addstr(body_h + 2, 0,
                          "r=run/stop  s=E-STOP  w=toggle-recording  :cmd  q=quit"[:max(0, w - 1)])
    except curses.error:
        pass
    stdscr.refresh()


def _draw_matrix(stdscr, body_h, x0, width, mv):
    if width < 20 or body_h < 6:
        return
    cell_w = width // 2
    cell_h = body_h // 2
    for ch in CHANNELS:
        cx, cy = _QUADRANT_ORIGIN[ch]  # row0=top convention (see coil_current_matrix.py)
        r = 0 if cy == 2 else 1
        c = 0 if cx == 0 else 1
        frac = max(0.0, min(1.0, mv[ch] / MV_MAX))
        bucket = _color_bucket(frac)
        attr = curses.color_pair(bucket) | (curses.A_BOLD if bucket >= _PAIR_YELLOW_MID else curses.A_NORMAL)
        y = r * cell_h + cell_h // 2
        x = x0 + c * cell_w + max(0, cell_w // 2 - 6)
        label = f"{ch} {mv[ch]:4.0f}mV"
        try:
            stdscr.addstr(y, x, "●●●●", attr)
            stdscr.addstr(y + 1, x, label, attr)
        except curses.error:
            pass
