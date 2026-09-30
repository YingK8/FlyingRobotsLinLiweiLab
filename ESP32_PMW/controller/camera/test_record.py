#!/usr/bin/env python3
"""One flight per take, in its own dated folder, and both frames.csv schemas still read.

No camera needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from controller.camera.record import flights, latest_flight, new_flight, read_index

HEADER = "index,t_capture,skew_s,t_a,t_b"


def _rows(frames):
    return [f"{i},{i / 30:.6f},0.001000,{i / 30:.6f},{i / 30 + 1e-3:.6f}"
            for i in range(frames)]


def _fake_flight(root, frames=5, legacy=False):
    """A flight folder with the files `open_recording` and `read_index` look for.

    `legacy` writes the pre-2026-08-30 frames.csv, which prefixed skew stats as comment
    lines. 47 of the 55 takes on disk are that shape, so reading it is not optional.
    """

    f = new_flight(root)
    for tag in "AB":
        (f / tag / f"{tag}.mp4").write_bytes(b"")
    head = [f"# skew_n, {frames}", f"# skew_p95, 0.002"] if legacy else []
    (f / "frames.csv").write_text("\n".join(head + [HEADER] + _rows(frames)) + "\n")
    return f


def test_flights_are_separate():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        assert flights(root) == [], "an empty root has no flights"
        assert latest_flight(root) == root, "with no flights, the root is the recording"

        a, b = _fake_flight(root), _fake_flight(root, frames=7)
        assert a != b, "two takes in the same second must not share a folder"
        assert flights(root) == sorted([a, b]), "both takes are listed, oldest first"
        assert latest_flight(root) == max(a, b)

        stamps, skews = read_index(b)
        assert stamps.shape == (7, 2) and len(skews) == 7
        assert {p.name for p in b.iterdir()} == {"A", "B", "frames.csv"}
        assert (b / "A" / "A.mp4").exists() and (b / "B" / "B.mp4").exists()


def test_legacy_index_still_reads():
    """The takes already on disk must survive the writer being unified."""

    with tempfile.TemporaryDirectory() as tmp:
        old = _fake_flight(Path(tmp), frames=6, legacy=True)
        stamps, skews = read_index(old)
        assert stamps is not None, "a commented frames.csv must not read as missing"
        assert stamps.shape == (6, 2), stamps.shape
        assert len(skews) == 6


def test_missing_index_is_not_an_error():
    """8 takes on disk were aborted before frames.csv landed. Readers skip, never crash."""

    with tempfile.TemporaryDirectory() as tmp:
        f = new_flight(Path(tmp))
        assert read_index(f) == (None, None)


if __name__ == "__main__":
    test_flights_are_separate()
    test_legacy_index_still_reads()
    test_missing_index_is_not_an_error()
    print("record: one flight per take; both frames.csv schemas read; "
          "an aborted take skips\n  ok")
