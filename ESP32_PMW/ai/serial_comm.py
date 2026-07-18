"""Non-blocking, newline-framed serial link -- Python mirror of the ESP32
SerialComm library (lib/SerialComm). Call handle_serial_comm() once per tick;
it never blocks.

Framing: ASCII lines terminated by \\n, \\r, or \\r\\n -- matches the existing
firmware protocol (main_tilt.cpp) and PC-side tools (run_experiment.py,
trigger_reset_log.py), which this module's find_port()/reset pulse are
lifted from.
"""
from __future__ import annotations

import glob
import sys
import time

import serial


def find_port(explicit=None):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found -- pass port explicitly")


class SerialComm:
    MAX_LINE_LEN = 128  # overflow guard against garbage

    def __init__(self, port=None, baud=115200):
        self.ser = serial.Serial()
        self.ser.port = find_port(port)
        self.ser.baudrate = baud
        self.ser.timeout = 0  # non-blocking reads
        self.ser.dtr = False  # keep IO0 high -> app boot, not bootloader/reset
        self.ser.rts = False
        self.ser.open()
        self._rxbuf = ""

    def reset_device(self, pulse_s=0.15):
        """EN-pulse reset via RTS -- same timing as trigger_reset_log.py."""
        self.ser.reset_input_buffer()
        self.ser.rts = True
        time.sleep(pulse_s)
        self.ser.rts = False

    def handle_serial_comm(self, outgoing=""):
        """Non-blocking. Sends `outgoing` (if any) plus '\\n'. Returns the
        first complete incoming line, or None if none finished yet. If
        multiple lines are waiting, only one is returned per call -- the
        rest stay buffered for the next call."""
        if outgoing:
            self.ser.write((outgoing + "\n").encode())

        # Read one byte at a time (mirrors the C++ side's available()/read()
        # loop) so that once a line completes and we return, whatever's left
        # unread simply stays in the OS/UART buffer for the next call --
        # nothing already pulled off the wire is ever discarded.
        while self.ser.in_waiting:
            c = self.ser.read(1).decode("utf-8", errors="replace")
            if c in "\r\n":
                if self._rxbuf:
                    line, self._rxbuf = self._rxbuf, ""
                    return line
            else:
                self._rxbuf += c
                if len(self._rxbuf) > self.MAX_LINE_LEN:
                    self._rxbuf = ""
        return None

    def close(self):
        self.ser.close()
