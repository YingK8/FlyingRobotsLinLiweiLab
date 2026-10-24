"""Offline self-checks: encoded row mapping and physical stereo geometry."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from LLM.capture import import_flight, load_flight
from LLM.pose import triangulate_annotations


def _expect_error(function, message):
    try:
        function()
    except (ValueError, FileExistsError) as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected error containing {message}")


def _capture(path, *, wrong_count=False):
    path.mkdir(parents=True)
    for tag in "AB":
        (path / tag).mkdir()
        writer = cv2.VideoWriter(str(path / tag / f"{tag}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 50, (320, 240))
        if not writer.isOpened():
            raise RuntimeError("Self-check needs OpenCV mp4v encoding")
        try:
            for i in range(2 if wrong_count and tag == "B" else 3):
                writer.write(np.full((240, 320, 3), 40 + i * 30, dtype=np.uint8))
        finally:
            writer.release()
    pd.DataFrame({"index": [0, 1, 2, 3], "t_capture": [10, 10.02, 10.04, 10.06],
                  "t_a": [10, 10.02, 10.04, 10.06],
                  "t_b": [10.001, 10.021, 10.041, 10.061],
                  "skew_s": [.001] * 4, "written": [1, 0, 1, 1]}).to_csv(path / "frames.csv", index=False)
    (path / "meta.json").write_text(json.dumps({"n_frames": 4, "dropped": 1, "encoder_errors": 0,
                                                "mode": [320, 240], "rotate180": False}))


def _rig(path):
    intrinsic = np.array([[200., 0, 160.], [0, 200., 120.], [0, 0, 1.]])
    ta, tb = np.eye(4), np.eye(4)
    tb[0, 3] = 40
    data = {"cameras": [{"name": name, "K": intrinsic.tolist(), "dist": [0.] * 5,
                         "T_world_cam": transform.tolist()} for name, transform in zip("AB", (ta, tb))],
            "image_size": [320, 240], "world_frame": "camera_A", "rotate180": False}
    path.write_text(json.dumps(data))
    return intrinsic, (ta, tb)


def _pixels(point, intrinsic, transforms):
    result = []
    for transform in transforms:
        local = transform[:3, :3].T @ (point - transform[:3, 3])
        projected = intrinsic @ local
        result.extend((projected[:2] / projected[2]).tolist())
    return result


def run_tests():
    with tempfile.TemporaryDirectory(prefix="llm-capture-pose-") as directory:
        tmp = Path(directory)
        source = tmp / "source"
        _capture(source)
        flight = load_flight(source)
        np.testing.assert_array_equal(flight["frame_indices"], [0, 2, 3])
        np.testing.assert_allclose(flight["times_a"], [10, 10.04, 10.06])
        assert flight["metadata"]["dropped_capture_rows"] == 1
        imported = import_flight(source, tmp / "run")
        assert imported["sha256"]["frames.csv"]
        assert (source / "frames.csv").read_bytes() == (tmp / "run/raw/camera/frames.csv").read_bytes()
        _expect_error(lambda: import_flight(source, tmp / "run"), "already exists")

        wrong = tmp / "wrong"
        _capture(wrong, wrong_count=True)
        _expect_error(lambda: load_flight(wrong), "decoded frames")
        rows = pd.read_csv(wrong / "frames.csv").drop(columns="written")
        rows.to_csv(wrong / "frames.csv", index=False)
        _expect_error(lambda: load_flight(wrong), "Legacy index")

        calibration = tmp / "rig.json"
        intrinsic, transforms = _rig(calibration)
        center = np.array([5., 4., 200.])
        tip = center + np.array([0., -10., 2.])
        coordinates = _pixels(center, intrinsic, transforms)
        tip_coordinates = _pixels(tip, intrinsic, transforms)
        ann = tmp / "annotations.csv"
        columns = ["frame_index", "ax_px", "ay_px", "bx_px", "by_px",
                   "a_tip_x_px", "a_tip_y_px", "b_tip_x_px", "b_tip_y_px"]
        pd.DataFrame([[0, *coordinates, *tip_coordinates],
                      [2, *coordinates, *([np.nan] * 4)]], columns=columns).to_csv(ann, index=False)
        pose = triangulate_annotations(tmp / "run", ann, calibration)
        assert len(pose) == 3
        assert pose.loc[0, "position_valid"] and pose.loc[0, "orientation_valid"]
        np.testing.assert_allclose(pose.loc[0, ["x_mm", "y_mm", "z_mm"]].to_numpy(float), center, atol=1e-9)
        np.testing.assert_allclose(pose.loc[0, ["nx", "ny", "nz"]].to_numpy(float), (tip - center) / np.linalg.norm(tip - center), atol=1e-9)
        assert pose.loc[1, "position_valid"] and not pose.loc[1, "orientation_valid"]
        assert not pose.loc[2, "position_valid"]
        assert pose.loc[2, "position_reason"] == "missing_annotation"

        # An annotation error inconsistent with epipolar geometry must not yield a
        # valid 3-D point even when the linear solver returns a finite vector.
        bad = pd.read_csv(ann)
        bad.loc[0, "by_px"] += 20
        bad.to_csv(ann, index=False)
        invalid = triangulate_annotations(tmp / "run", ann, calibration)
        assert not invalid.loc[0, "position_valid"]
        assert invalid.loc[0, "position_reason"] == "reprojection_error"

        # Clock disagreement invalidates position instead of mixing two instants.
        skewed = triangulate_annotations(tmp / "run", ann, calibration, max_skew_s=0.0001)
        assert not skewed["position_valid"].any()
        assert (skewed["position_reason"] == "stereo_skew_exceeds_limit").all()

        bad_calib = json.loads(calibration.read_text())
        bad_calib["image_size"] = [640, 480]
        calibration.write_text(json.dumps(bad_calib))
        _expect_error(lambda: triangulate_annotations(tmp / "run", ann, calibration), "resolution")
        _rig(calibration)
        bad_calib = json.loads(calibration.read_text())
        del bad_calib["rotate180"]
        calibration.write_text(json.dumps(bad_calib))
        _expect_error(lambda: triangulate_annotations(tmp / "run", ann, calibration), "preprocessing_confirmed")
        triangulate_annotations(tmp / "run", ann, calibration, preprocessing_confirmed=True)
    print("LLM capture/pose self-checks passed (offline synthetic data; no cameras or coils used)")


if __name__ == "__main__":
    run_tests()
