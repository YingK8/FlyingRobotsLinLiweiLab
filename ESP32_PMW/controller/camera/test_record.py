#!/usr/bin/env python3
"""One flight per take, in its own dated folder. No camera needed."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
from record import flights, latest_flight, new_flight, read_index, write_index


def _fake_flight(root, frames=5):
    """A flight folder with the files `open_recording` and `read_index` look for."""

    f = new_flight(root)
    for tag in "AB":
        (f / tag / f"{tag}.mp4").write_bytes(b"")
    write_index(f, "AB", [(i, i / 30, 1e-3, (i / 30, i / 30 + 1e-3)) for i in range(frames)])
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


if __name__ == "__main__":
    test_flights_are_separate()
    print("test_flights_are_separate: one flight per take, in its own dated folder\n  ok")
