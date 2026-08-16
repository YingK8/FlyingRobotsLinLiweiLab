#!/usr/bin/env python3
"""Self-check for serial_comm.SerialComm's line-framing, using pyserial's
built-in loop:// loopback -- no hardware, no mocking library needed.
"""
import sys

import serial

from serial_comm import SerialComm


def make_comm():
    comm = SerialComm.__new__(SerialComm)
    comm.ser = serial.serial_for_url("loop://", timeout=0)
    comm._rxbuf = ""
    return comm


def test_basic_line():
    comm = make_comm()
    comm.ser.write(b"hello\n")
    assert comm.handle_serial_comm() == "hello"
    assert comm.handle_serial_comm() is None


def test_crlf_and_bare_cr():
    comm = make_comm()
    comm.ser.write(b"a\r\nb\rc\n")
    assert comm.handle_serial_comm() == "a"
    assert comm.handle_serial_comm() == "b"
    assert comm.handle_serial_comm() == "c"
    assert comm.handle_serial_comm() is None


def test_split_across_calls():
    comm = make_comm()
    comm.ser.write(b"par")
    assert comm.handle_serial_comm() is None
    comm.ser.write(b"tial\n")
    assert comm.handle_serial_comm() == "partial"


def test_only_first_line_per_call():
    comm = make_comm()
    comm.ser.write(b"one\ntwo\n")
    assert comm.handle_serial_comm() == "one"
    assert comm.handle_serial_comm() == "two"


def test_overflow_guard():
    comm = make_comm()
    comm.ser.write(b"x" * (SerialComm.MAX_LINE_LEN + 1) + b"\n")
    comm.ser.write(b"ok\n")
    assert comm.handle_serial_comm() == "ok"


def test_outgoing_is_sent():
    # loop:// reflects writes straight back into the same read buffer, so the
    # outgoing bytes get drained by this same call -- verify via round trip
    # that '\n' was appended and the content is unchanged.
    comm = make_comm()
    assert comm.handle_serial_comm("ping") == "ping"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: PASS")
    print("all tests PASS")
    sys.exit(0)
