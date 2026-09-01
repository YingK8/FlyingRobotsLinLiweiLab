"""
Non-blocking, newline-framed serial link. Python mirror of lib/SerialComm on the ESP32.

Call handle_serial_comm() once per tick; it never blocks. Framing is ASCII lines terminated
by \n, \r, or \r\n, matching the firmware protocol.

Every path that drives the coils goes through SerialComm, because SerialComm is what stamps
the energised seconds into the coil thermal model. See CLAUDE.md "Safety".
"""

from __future__ import annotations

import re
import sys
import time
from typing import NamedTuple

import serial
from serial.tools import list_ports


def find_port(explicit=None):
    """First USB-serial port pyserial can see. pyserial knows the per-OS device names."""

    if explicit:
        return explicit
    hits = [p.device for p in list_ports.comports() if p.vid is not None]
    if not hits:
        sys.exit("no USB serial device found -- pass port explicitly")
    return sorted(hits)[0]


# driveTelemetry prints this at 2 Hz in every state:
#   "t=17644 freq=0.0 | I[A]: A=0.00 B=0.00 C=0.00 D=0.00 | duty[%]: A=0.0 ... | trip=0"
# freq= and state= are anchored to a line start or a space/pipe, so the rejection message
# "!freq=163.00" does not read as a frequency the coils are running at.
_FREQ_RE = re.compile(r"(?:^|[ |])freq=(-?\d+(?:\.\d+)?)")
_STATE_RE = re.compile(r"(?:^|[ |])state=(\d)")
_I_RE = re.compile(r"I\[A\]:\s*(.*?)\s*\|")
_DUTY_RE = re.compile(r"duty\[%\]:\s*(.*?)(?:\||$)")
_VAL_RE = re.compile(r"[A-D]=(-?[\d.]+)")


class Telemetry(NamedTuple):
    """One parsed telemetry line. A field is None or empty when the line did not carry it."""

    state: int | None = None
    freq: float | None = None
    amps: tuple[float, ...] = ()
    duty: tuple[float, ...] = ()


def _values(line, head):
    m = head.search(line)
    return tuple(float(v) for v in _VAL_RE.findall(m.group(1))) if m else ()


def parse_telemetry(line: str) -> Telemetry:
    """Pull whatever this line carries. Lines arrive truncated and interleaved."""

    state = _STATE_RE.search(line)
    freq = _FREQ_RE.search(line)
    return Telemetry(
        state=int(state.group(1)) if state else None,
        freq=float(freq.group(1)) if freq else None,
        amps=_values(line, _I_RE),
        duty=_values(line, _DUTY_RE),
    )


class SerialComm:
    MAX_LINE_LEN = 128  # overflow guard against garbage

    #: Host link speed. MUST match `src/constants.h`'s `SERIAL_BAUD` and
    #: `platformio.ini`'s `monitor_speed` -- there is no handshake and no ack, so a
    #: mismatch is silent rather than an error: the firmware never parses a command and
    #: the coils hold their last value. Raised from 115200 on 2026-09-01 because 29
    #: bytes took 2.5 ms on the wire there, longer than a 500 Hz control period. See
    #: `theory.md` 19.13.
    BAUD = 921600

    def __init__(self, port=None, baud=BAUD):
        self.ser = serial.Serial()
        self.ser.port = find_port(port)
        self.ser.baudrate = baud
        self.ser.timeout = 0  # non-blocking reads
        self.ser.dtr = False  # keep IO0 high -> app boot, not bootloader/reset
        self.ser.rts = False
        self.ser.open()
        self._rxbuf = ""
        self._drive_since = None
        self._drive_s = 0.0

    def reset_device(self, pulse_s=0.15):
        """EN-pulse reset via RTS."""

        self.ser.reset_input_buffer()
        self.ser.rts = True
        time.sleep(pulse_s)
        self.ser.rts = False

    def handle_serial_comm(self, outgoing=""):
        """
        Non-blocking. Sends `outgoing` (if any) plus '\n'. Returns the first complete
        incoming line, or None if none finished yet. If multiple lines are waiting, only
        one is returned per call -- the rest stay buffered for the next call.
        """

        if outgoing:
            self._note_drive(outgoing)
            self.ser.write((outgoing + "\n").encode())

        # One read for everything waiting, not one syscall per byte. `drain()` calls this
        # until it returns None, so at 200 Hz a 120-byte telemetry line cost 120 syscalls
        # in whichever tick it landed in -- fine against a 5 ms budget, not against 2 ms.
        if self.ser.in_waiting:
            self._rxbuf += self.ser.read(self.ser.in_waiting).decode(
                "utf-8", errors="replace")
        while True:
            hits = [j for j in (self._rxbuf.find("\r"), self._rxbuf.find("\n")) if j >= 0]
            if not hits:
                break
            i = min(hits)
            line, self._rxbuf = self._rxbuf[:i], self._rxbuf[i + 1:]
            # Blank lines are the other half of a CRLF, or an idle newline. Skipped
            # rather than returned, because `drain()` stops on None and would read ""
            # as a live line -- and one telemetry line is CRLF-terminated, so this is
            # the common case, not the edge one.
            if line:
                return line
        if len(self._rxbuf) > self.MAX_LINE_LEN:
            self._rxbuf = ""    # overflow guard: garbage, not a line we lost
        return None

    def _note_drive(self, cmd):
        """Start the clock on anything that energises, stop it on `stop`/`land`.

        `seq=clear` and `seq=ramp:` only queue tasks, so they stay out. `seq=go` is what
        starts the ramp turning current into heat.

        Any new command that energises belongs in this list, or its heat goes unaccounted
        and the model reads cold while the coils are not.
        """

        c = cmd.strip().lower()
        if c.startswith(("seq=go", "freq=", "mag=", "throttle=")):
            if self._drive_since is None:
                self._drive_since = time.monotonic()
        elif c in ("stop", "land") and self._drive_since is not None:
            self._drive_s += time.monotonic() - self._drive_since
            self._drive_since = None

    def energised_s(self):
        """Seconds of drive so far, including an interval still open."""

        extra = 0.0 if self._drive_since is None else time.monotonic() - self._drive_since
        return self._drive_s + extra

    def close(self):
        secs = self.energised_s()
        self.ser.close()
        if secs > 0.5:
            try:
                from ai.thermal import coil_thermal
            except ImportError:
                print(f"[coils] THERMAL MODEL MISSING -- {secs:.0f}s of drive NOT tracked.\n"
                      f"[coils] restore ai/thermal/coil_thermal.py before the next run.")
                return
            t = coil_thermal.add_energised(secs)
            print(f"[coils] {secs:.0f}s energised this session -> ~{t:.0f}C estimated")


def demo():
    line = ("t=17644 freq=0.0 | I[A]: A=0.00 B=0.00 C=0.00 D=0.00 | "
            "duty[%]: A=0.0 B=0.0 C=0.0 D=0.0 | spread=0.000 bal=0 trip=0")
    t = parse_telemetry(line)
    assert t.amps == (0.0, 0.0, 0.0, 0.0), t
    assert t.duty == (0.0, 0.0, 0.0, 0.0), t
    assert t.freq == 0.0 and t.state is None, t

    live = parse_telemetry(line.replace("A=0.00 B=0.00", "A=0.27 B=0.28")
                               .replace("duty[%]: A=0.0", "duty[%]: A=100.0"))
    assert live.amps[0] == 0.27, live
    # Current at zero with the bridges still driving is NOT off.
    assert max(live.duty) == 100.0, live

    assert parse_telemetry("garbage") == Telemetry(), "never read values out of noise"
    assert parse_telemetry("!freq=163.00 rejected").freq is None, "a rejection is not a rate"
    assert parse_telemetry("state=2 freq=210.0").state == 2

    # The RX framer, against a fake port. It reads everything waiting in one syscall
    # now instead of one per byte, so the buffer it scans can hold several lines, half a
    # line, or a CRLF split across two reads -- all of which the byte-at-a-time version
    # could not produce and this one can.
    class _FakePort:
        def __init__(self, chunks):
            self.chunks, self.n_reads = list(chunks), 0

        @property
        def in_waiting(self):
            return len(self.chunks[0]) if self.chunks else 0

        def read(self, _n):
            self.n_reads += 1
            return self.chunks.pop(0)

    def _drain(chunks):
        """Every line the framer yields, pumping once per arriving chunk.

            One `drain()` stops at the first None -- a half-received line waits for the
            next tick rather than being returned truncated -- so a chunk boundary needs
            a fresh pump, which is exactly what the next control tick would do.
        """

        c = SerialComm.__new__(SerialComm)      # no port, no open()
        c.ser, c._rxbuf = _FakePort(chunks), ""
        out = []
        while True:
            while (ln := c.handle_serial_comm()) is not None:
                out.append(ln)
            if not c.ser.chunks:
                return out, c._rxbuf, c.ser.n_reads

    got, rest, reads = _drain([b"state=2 freq=210.0\r\n"])
    assert got == ["state=2 freq=210.0"] and rest == "", (got, rest)
    assert reads <= 2, f"one line should not cost {reads} reads"

    # Several lines in one read, which is what a 921600 link delivers into one tick.
    got, rest, _ = _drain([b"a=1\r\nb=2\r\nc=3\r\n"])
    assert got == ["a=1", "b=2", "c=3"], got

    # A partial line stays buffered rather than being returned truncated.
    got, rest, _ = _drain([b"state=2 fr"])
    assert got == [] and rest == "state=2 fr", (got, rest)

    # ...and completes on the next read, including a CRLF split across the boundary.
    got, rest, _ = _drain([b"state=2 fr", b"eq=210.0\r", b"\nnext=1\n"])
    assert got == ["state=2 freq=210.0", "next=1"] and rest == "", (got, rest)

    # Overflow guard: garbage with no newline is dropped, not grown without bound.
    got, rest, _ = _drain([b"x" * (SerialComm.MAX_LINE_LEN + 10)])
    assert got == [] and rest == "", (got, rest)

    print("link: parses zero, live and rejected telemetry; rejects noise; "
          "framer handles split, batched and overflowing reads\n  ok")


if __name__ == "__main__":
    demo()
