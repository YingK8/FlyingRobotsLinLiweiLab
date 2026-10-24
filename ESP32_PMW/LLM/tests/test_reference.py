"""Offline reference construction from known stereo geometry."""

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from LLM.reference import build_world_from_reference
from LLM.tests.test_capture_pose import _capture, _expect_error, _pixels, _rig
from LLM.world import validate_world_frame


def run():
    with tempfile.TemporaryDirectory(prefix="llm-reference-") as directory:
        root = Path(directory)
        capture = root / "capture"
        _capture(capture)
        calibration = root / "rig.json"
        K, transforms = _rig(calibration)
        physical = {"bearing": np.array([5., 4., 200.]),
                    "plumb_bottom": np.array([10., 20., 200.]),
                    "plumb_top": np.array([10., -30., 200.])}
        reference = {"schema_version": 1, "frame_index": 2, "note": "Synthetic stationary plumb line"}
        for name, point in physical.items():
            pixels = _pixels(point, K, transforms)
            reference[name] = {"a": pixels[:2], "b": pixels[2:]}
        path = root / "reference.json"
        path.write_text(json.dumps(reference))
        frame = build_world_from_reference(capture, path, calibration)
        origin, rotation = validate_world_frame(frame)
        np.testing.assert_allclose(origin, physical["bearing"], atol=1e-9)
        np.testing.assert_allclose(rotation[:, 2], [0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(rotation[:, 0], [1, 0, 0], atol=1e-9)
        assert frame["evidence"]["frame_index"] == 2
        assert frame["evidence"]["encoded_frame_index"] == 1
        assert np.isclose(frame["evidence"]["plumb_length_mm"], 50)
        assert frame["evidence"]["triangulated_points"]["bearing"]["reprojection_rms_px"] < 1e-9

        dropped = dict(reference, frame_index=1)
        _expect_error(lambda: build_world_from_reference(capture, dropped, calibration), "not present")
        same_point = dict(reference, plumb_top=reference["plumb_bottom"])
        _expect_error(lambda: build_world_from_reference(capture, same_point, calibration), "distinct")
        malformed = dict(reference, bearing={"a": [1, 2, 3], "b": [4, 5, 6]})
        _expect_error(lambda: build_world_from_reference(capture, malformed, calibration), "finite")
        mismatch = copy.deepcopy(reference)
        mismatch["plumb_top"]["b"][1] += 20
        _expect_error(lambda: build_world_from_reference(capture, mismatch, calibration), "reprojection_error")
        _expect_error(lambda: build_world_from_reference(capture, reference, calibration, max_skew_s=.0001), "skew")
        vertical_hint = dict(reference, x_hint_camera=[0, -1, 0])
        _expect_error(lambda: build_world_from_reference(capture, vertical_hint, calibration), "parallel")
    print("LLM reference self-checks passed (offline synthetic points)")


if __name__ == "__main__":
    run()
