#!/usr/bin/env python3
"""Python mirror of `lib/DriveFrame/src/drive_frame.h` -- the binary DRIVE/ACK wire format.

The firmware and the native controller both include the C++ header, so THAT is the format.
This exists for the host-side tooling that is not the flight binary: tests, the serial
log reader, and anything that wants to see what went out without running C++.

**A mirror is a liability unless it is held to its original.** `demo()` therefore decodes
byte vectors emitted by the C++ encoder itself (`lib/DriveFrame/test_drive_frame.cpp`'s
sibling generator), rather than round-tripping only against itself -- a self-consistent
mirror that disagrees with the firmware is worse than no mirror, because it looks correct.

    uv run python controller/control/drive_frame.py

See `control/theory.md` 26.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

SYNC_DRIVE, SYNC_ACK = 0xA5, 0xA6
TYPE_DRIVE, TYPE_ACK = 0x01, 0x02
NCH = 4
DRIVE_LEN, ACK_LEN = 26, 11

#: Must equal the header's. A mismatch here is silent on the wire and shows up as a
#: mis-scaled command, so `demo()` checks the scaling against the C++ vectors.
AMP_SCALE = 655.35
PHASE_SCALE = 182.04444


class Drive(NamedTuple):
    seq: int
    amp_pct: tuple            # 4, 0..100
    phase_deg: tuple          # 4, 0..360
    freq_hz: float            # 0 = DC idle


class Ack(NamedTuple):
    seq_echo: int
    n_rx: int
    n_crc: int
    state: int


def crc16(b: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor."""

    c = 0xFFFF
    for x in b:
        c ^= x << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
    return c


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _f32(x):
    """Round to float32, the way the C++ does every step of this arithmetic.

    Not pedantry. The header computes `(uint16_t)(pct * AMP_SCALE + 0.5f)` in float, and
    at pct = 50 that is 50.0f * 655.35f = 32767.4988, +0.5 = 32767.9988, truncating to
    **32767**. The same expression in Python's float64 is 32767.5000000000011 + 0.5 =
    32768.0000000000011, truncating to **32768** -- one LSB apart, on a round number a
    host is very likely to send. Caught by `demo()` against the C++ vectors on the first
    run of this file.

    The consequence is small (0.0015 % of duty) and the principle is not: a mirror that
    is nearly byte-identical is a mirror that will one day disagree about something that
    matters, in a format with no per-frame acknowledgement to catch it. Same lesson as
    `pose/theory.md` 21.2, where a float32 ulp moved the pose solve by 0.4 mm.
    """

    return float(struct.unpack("<f", struct.pack("<f", x))[0])


def encode(d: Drive) -> bytes:
    """One DRIVE frame. Values are clamped HERE, so the wire never carries nonsense."""

    amps = [int(_f32(_f32(_clamp(a, 0.0, 100.0) * _f32(AMP_SCALE)) + 0.5))
            for a in d.amp_pct]
    phs = []
    for p in d.phase_deg:
        v = int(_f32(_f32((p % 360.0) * _f32(PHASE_SCALE)) + 0.5))
        phs.append(0 if v > 65535 else v)      # 359.9973+ rounds up to 65536, i.e. 0 deg
    body = struct.pack("<BH4H4HI", TYPE_DRIVE, d.seq & 0xFFFF, *amps, *phs,
                       int(_f32(_f32(_clamp(d.freq_hz, 0.0, 1000.0) * 1000.0) + 0.5)))
    # The CRC covers everything but the sync byte: sync is what resynchronisation keys on
    # and so cannot also be the thing being checked.
    return bytes([SYNC_DRIVE]) + body + struct.pack("<H", crc16(body))


def decode(b: bytes) -> Drive | None:
    """A DRIVE frame, or None if this is not one or the CRC fails."""

    if len(b) != DRIVE_LEN or b[0] != SYNC_DRIVE or b[1] != TYPE_DRIVE:
        return None
    if struct.unpack_from("<H", b, 24)[0] != crc16(b[1:24]):
        return None
    seq = struct.unpack_from("<H", b, 2)[0]
    amps = struct.unpack_from("<4H", b, 4)
    phs = struct.unpack_from("<4H", b, 12)
    f = struct.unpack_from("<I", b, 20)[0]
    return Drive(seq, tuple(a / AMP_SCALE for a in amps),
                 tuple(p / PHASE_SCALE for p in phs), f / 1000.0)


def encode_ack(a: Ack) -> bytes:
    body = struct.pack("<BHHHB", TYPE_ACK, a.seq_echo & 0xFFFF, a.n_rx & 0xFFFF,
                       a.n_crc & 0xFFFF, a.state & 0xFF)
    return bytes([SYNC_ACK]) + body + struct.pack("<H", crc16(body))


def decode_ack(b: bytes) -> Ack | None:
    if len(b) != ACK_LEN or b[0] != SYNC_ACK or b[1] != TYPE_ACK:
        return None
    if struct.unpack_from("<H", b, 9)[0] != crc16(b[1:9]):
        return None
    return Ack(*struct.unpack_from("<HHHB", b, 2))


#: Emitted by `lib/DriveFrame/test_drive_frame.cpp`'s generator, i.e. by the C++ encoder
#: the firmware actually runs. Regenerate if the format ever changes -- and if it changes
#: without these changing, one side has drifted and `demo()` says so.
CPP_VECTORS = {
    (0, (0, 0, 0, 0), (0, 0, 0, 0), 0.0):
        "a5 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 5d a3",
    (1, (100, 100, 100, 100), (0, 90, 180, 270), 210.0):
        "a5 01 01 00 ff ff ff ff ff ff ff ff 00 00 00 40 00 80 00 c0 50 34 03 00 63 40",
    (40000, (0.0, 33.333, 99.999, 100), (0, 90, 179.9, 359.99), 210.123):
        "a5 01 40 9c 00 00 55 55 fe ff ff ff 00 00 00 40 ee 7f fe ff cb 34 03 00 72 24",
    (65535, (12.5, 25, 37.5, 50), (45.5, 135.25, 225.125, 315), 101.001):
        "a5 01 ff ff 00 20 00 40 00 60 ff 7f 5b 20 2e 60 17 a0 00 e0 89 8a 01 00 4b cc",
}
CPP_ACK = "a6 02 d2 04 88 13 03 00 02 20 40"


def demo():
    # 1. THE THING THAT MATTERS: byte-for-byte agreement with the C++ encoder. Not a
    #    round trip through this file, which would pass just as well if both halves of
    #    this module were wrong in the same way.
    for (seq, a, p, f), hexs in CPP_VECTORS.items():
        want = bytes.fromhex(hexs.replace(" ", ""))
        got = encode(Drive(seq, tuple(a), tuple(p), f))
        assert got == want, f"seq {seq}:\n  py  {got.hex(' ')}\n  cpp {want.hex(' ')}"
        d = decode(want)
        assert d.seq == seq, d
        for i in range(NCH):
            assert abs(d.amp_pct[i] - a[i]) < 0.002, (i, d.amp_pct[i], a[i])
            e = abs(d.phase_deg[i] - p[i])
            assert e < 0.01 or abs(e - 360.0) < 0.01, (i, d.phase_deg[i], p[i])
        assert abs(d.freq_hz - f) < 0.001, (d.freq_hz, f)
    assert encode_ack(Ack(1234, 5000, 3, 2)) == bytes.fromhex(CPP_ACK.replace(" ", ""))
    assert decode_ack(bytes.fromhex(CPP_ACK.replace(" ", ""))) == Ack(1234, 5000, 3, 2)

    # 2. Clamping happens at the sender. A host bug cannot put 150 % on the wire.
    d = decode(encode(Drive(0, (150.0, -20.0, 0, 0), (-90.0, 0, 0, 0), -5.0)))
    assert abs(d.amp_pct[0] - 100.0) < 0.01 and abs(d.amp_pct[1]) < 0.01, d.amp_pct
    assert abs(d.phase_deg[0] - 270.0) < 0.01, d.phase_deg
    assert d.freq_hz == 0.0, d.freq_hz

    # 3. Every single-bit flip in the payload is caught. The link is unacknowledged per
    #    frame, so a corrupt frame that decodes is a coil command nobody sent.
    good = encode(Drive(7, (10, 20, 30, 40), (0, 90, 180, 270), 150.0))
    for i in range(1, DRIVE_LEN):
        for b in range(8):
            bad = bytearray(good)
            bad[i] ^= 1 << b
            assert decode(bytes(bad)) is None, f"undetected flip at byte {i} bit {b}"

    # 4. Not-a-frame is rejected rather than misread. A truncated frame is the common
    #    case at 921600 and must never decode as a shorter one.
    assert decode(good[:-1]) is None
    assert decode(good + b"\x00") is None
    assert decode(b"\xa6" + good[1:]) is None, "an ACK is not a DRIVE"
    assert decode_ack(encode(Drive(0, (0,) * 4, (0,) * 4, 0.0))) is None

    # 5. Every ASCII command the host sends is 7-bit, which is the property that lets the
    #    two framings share one stream. If this ever fails, the framer's rule that a
    #    high-bit byte at a line boundary means "frame" has stopped being safe.
    for cmd in ("seq=clear", "seq=ramp:2:210:30000:1:2", "seq=go", "stop", "land", "hover",
                "throttle=100", "az=315", "mag=0.300", "freq=210.00", "duty=100:80:100:90",
                "phase=0:90:180:270", "probe=150:1500"):
        assert all(ord(c) < 0x80 for c in cmd), cmd

    print(f"drive_frame: {len(CPP_VECTORS)} vectors byte-identical to the C++ encoder; "
          f"clamped at the sender;\n  all {(DRIVE_LEN - 1) * 8} single-bit flips caught; "
          f"truncation and cross-type rejected; ASCII stays 7-bit\n  ok")


if __name__ == "__main__":
    demo()
