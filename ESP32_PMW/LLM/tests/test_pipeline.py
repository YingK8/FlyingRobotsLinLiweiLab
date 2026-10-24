"""Integration checks use synthetic videos and known geometry, not hardware."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from LLM.demo import make_demo
from LLM.io import read_json, write_json
from LLM.pattern import example_pattern, sample_pattern, validate_pattern
from LLM.pipeline import add_clock, process_run, synchronize
from controller.control import ramp


def _reject(call):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("expected rejection")


def run():
    plan = example_pattern()
    values = sample_pattern(plan)
    assert values.frequency_hz.iloc[0] == 2 and values.frequency_hz.iloc[-1] == 170
    assert ramp.seq_lines(plan["frequency_segments"])[-1] == "seq=go"
    assert ramp.frequency_at([[0, 100, 1, ramp.EASE, 2]], .25) == 10
    assert abs(ramp.frequency_at([[0, 100, 1, ramp.EXP, 2]], .5) - 100/(np.e+1)) < 1e-10
    assert ramp.frequency_at([[0, 100, 1, ramp.POLY, 2]], .5) == 25
    for key, value in (("trigger", "takeoff"), ("frequency_segments", [[2, 100, float("nan"), 0, 1]]),
                       ("frequency_segments", [[2, 100, .0001, 0, 1]])):
        invalid = dict(plan); invalid[key] = value
        _reject(lambda: validate_pattern(invalid))
    with tempfile.TemporaryDirectory(prefix="llm-pipeline-test-") as tmp:
        run_dir = Path(tmp) / "run"
        quality = make_demo(run_dir)
        assert quality["valid_position_frames"] == 240
        pose = pd.read_csv(run_dir / "processed/pose.csv")
        assert pose.coordinate_frame.eq("experiment_world").all()
        assert np.allclose(pose.nx**2 + pose.ny**2 + pose.nz**2, 1)
        events = read_json(run_dir / "processed/events.json")
        assert 1.04 < events["takeoff"]["onset_s"] < 1.15
        a = pd.read_csv(run_dir / "export/camera_A_frame_map.csv")
        b = pd.read_csv(run_dir / "export/camera_B_frame_map.csv")
        assert np.allclose(a.t_s, b.t_s) and a.image_valid.all() and b.image_valid.all()
        _reject(lambda: make_demo(run_dir))
        # Re-associating optical events invalidates an earlier electrical mapping.
        synchronize(run_dir, [20, 20, 24, 24], [20, 20, 24, 24], [1, 4])
        assert "esp" not in read_json(run_dir / "processed/sync.json")["clocks"]
        synchronize(run_dir, [20, 20, 24, 24], [20, 20, 24, 24], [0, 3])
        add_clock(run_dir, "esp", [10, 13], [0, 3], "synthetic test anchors")
        applied = {"t_device_s": 10, "frequency_hz": 2, "provenance": "device_applied"}
        applied.update({f"duty_{c}_pct": 20 for c in "ABCD"})
        applied.update({f"phase_{c}_deg": i*90 for i, c in enumerate("ABCD")})
        pd.DataFrame([applied]).to_csv(run_dir / "raw/applied_commands.csv", index=False)
        process_run(run_dir, clips=False)
        cropped = pd.read_csv(run_dir / "export/command_applied.csv")
        assert len(cropped) == 1 and cropped.context_before_window.iloc[0] and cropped.t_s.iloc[0] == 0
        (run_dir / "raw/applied_commands.csv").unlink()
        # Missing optional evidence on a rerun must never resurrect old curves.
        (run_dir / "raw/telemetry.jsonl").unlink()
        (run_dir / "pattern.json").unlink()
        quality = process_run(run_dir, clips=False)
        assert quality["current_status"] == "missing"
        assert not (run_dir / "export/current.csv").exists()
        assert not (run_dir / "processed/command_planned.csv").exists()
        assert not (run_dir / "export/command_applied.csv").exists()
        assert not (run_dir / "export/camera_A.mp4").exists()
        assert not (run_dir / "export/INCOMPLETE.txt").exists()
        # A failed re-sync invalidates the old clocks rather than mixing generations.
        _reject(lambda: synchronize(run_dir, [0, 0, 4, 4], [0, 0, 4, 4], [0, 3]))
        assert not (run_dir / "processed/sync.json").exists()
        synchronize(run_dir, [20, 20, 24, 24], [20, 20, 24, 24], [0, 3])
        cfg = read_json(run_dir / "config.json"); cfg["led"]["threshold"] = 150
        write_json(run_dir / "config.json", cfg)
        _reject(lambda: process_run(run_dir, clips=False))
    print("pipeline: geometry, pattern, clock provenance, takeoff, clips and rerun checks passed")


if __name__ == "__main__":
    run()
