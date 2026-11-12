"""Ramp files: `LLM/ramps/<name>.json`, the coil frequency profile one run drives.

    {"note": "why this ramp exists",
     "segments": [{"from_hz": 2, "to_hz": 210, "seconds": 30, "shape": "ease", "k": 2}]}

shape: "linear", "ease" (k>=1), "poly" (f ~ t^k) or "exp". k defaults to 1. Segments run
back to back and must join (`controller/control/ramp.check`, 55 s cap). The host sends
them as `seq=ramp` lines and `stop` at the end; nothing here touches serial.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from controller.control import ramp
from LLM.record.io import read_json

FOLDER = Path(__file__).resolve().parents[1] / "ramps"
SHAPES = {"linear": ramp.POLY, "poly": ramp.POLY, "ease": ramp.EASE, "exp": ramp.EXP}


def ramp_path(name):
    """A ramp name from LLM/ramps, or a path to a ramp JSON."""
    path = Path(name)
    return path if path.suffix == ".json" else FOLDER / f"{name}.json"


def run_ramp_path(run):
    """The ramp a run drove. Runs before 2026-11-02 stored it as pattern.json."""
    run = Path(run)
    return run / "ramp.json" if (run / "ramp.json").exists() else run / "pattern.json"


def load_ramp(path):
    """Ramp file -> validated ((from_hz, to_hz, seconds, mode, k), ...) for controller.control.ramp."""
    data = read_json(path)
    if "frequency_segments" in data:            # old pattern.json in an existing run folder
        segs = tuple(tuple(s) for s in data["frequency_segments"])
    else:
        segs = tuple((s["from_hz"], s["to_hz"], s["seconds"], SHAPES[s["shape"]], s.get("k", 1))
                     for s in data["segments"])
    for f0, f1, seconds, mode, k in segs:
        if not all(math.isfinite(v) for v in (f0, f1, seconds, k)) or min(f0, f1) <= 0:
            raise ValueError("frequencies must be positive and finite")
        # seq=ramp sends whole milliseconds; a fraction would be rounded off silently.
        if abs(seconds * 1000 - round(seconds * 1000)) > 1e-6:
            raise ValueError("segment seconds must be whole milliseconds")
        if mode == ramp.EASE and k < 1 or not 0 < k <= 50:
            raise ValueError("k must be in (0, 50], and >= 1 for ease")
    ramp.check(segs)
    return segs


def planned_frequency(segs, sample_hz=100):
    """The commanded frequency vs time, for plots and processed/command_planned.csv."""
    end = ramp.duration_s(segs)
    t = np.unique(np.r_[np.arange(0, end, 1 / sample_hz), end, np.cumsum([s[2] for s in segs])])
    return pd.DataFrame({"t_s": t, "frequency_hz": [ramp.frequency_at(segs, float(x)) for x in t],
                         "provenance": "planned"})


if __name__ == "__main__":
    import json, tempfile
    for p in sorted(FOLDER.glob("*.json")):
        print(f"{p.stem}: {ramp.label(load_ramp(p))}")
    assert planned_frequency(((2, 50, 1, ramp.POLY, 1),)).frequency_hz.iloc[-1] == 50
    for bad in ([2, 50, 1.0001, 0, 1], [0, 50, 1, 0, 1], [2, 50, 1, 9, 1]):
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump({"frequency_segments": [bad]}, f)
            load_ramp(f.name)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    print("ramps: OK")
