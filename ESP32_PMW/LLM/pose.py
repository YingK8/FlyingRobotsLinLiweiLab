"""Quality-gated stereo reconstruction of annotated matching physical points.

This first adapter accepts manual annotations or a detector's exported pixel
coordinates. It does not certify the existing rim estimator for the horizontal
camera geometry. Annotate the same physical point in both views, not unrelated
ellipse centres. The optional tip must be a visible, directed axis endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from controller.calib.rig import StereoRig
from LLM.capture import load_flight


CENTRE_COLUMNS = ("ax_px", "ay_px", "bx_px", "by_px")
TIP_COLUMNS = ("a_tip_x_px", "a_tip_y_px", "b_tip_x_px", "b_tip_y_px")


def _validated_rig(path, capture, preprocessing_confirmed):
    data = json.loads(Path(path).read_text())
    rig = StereoRig.load(path)
    if len(rig) != 2:
        raise ValueError("Stereo reconstruction requires exactly two calibrated cameras")
    image_size = data.get("image_size")
    if image_size is None:
        raise ValueError("Calibration must declare image_size=[width,height]")
    for tag in "AB":
        if capture["metadata"]["decoded_image_sizes"][tag] != list(image_size):
            raise ValueError(f"Camera {tag} resolution does not match calibration; recalibrate or explicitly resample upstream")
    for camera in rig.cameras:
        T = camera.T_world_cam
        if (not np.isfinite(T).all() or not np.isfinite(camera.K).all()
                or (camera.dist is not None and not np.isfinite(camera.dist).all())
                or not np.allclose(T[3], [0, 0, 0, 1])
                or not np.allclose(camera.R.T @ camera.R, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(camera.R), 1, atol=1e-6)
                or camera.K[0, 0] <= 0 or camera.K[1, 1] <= 0):
            raise ValueError("Calibration contains invalid camera matrices")
    if np.linalg.norm(rig.a.position - rig.b.position) < 1e-6:
        raise ValueError("Stereo baseline is zero")
    rotation_known = "rotate180" in data and "rotate180" in capture["metadata"]
    if rotation_known and data["rotate180"] != capture["metadata"]["rotate180"]:
        raise ValueError("Recording rotation does not match calibration")
    if not rotation_known and not preprocessing_confirmed:
        raise ValueError("Calibration/recording lack matching rotate180 metadata; confirm camera A/B order, rotation and crop with preprocessing_confirmed=True")
    return rig, data


def _point(uv_a, uv_b, rig, image_size, max_reprojection_px, min_ray_angle_deg):
    points = np.asarray([uv_a, uv_b], dtype=float)
    if not np.isfinite(points).all():
        return None, np.nan, np.nan, "missing_point"
    if np.any(points < 0) or np.any(points[:, 0] >= image_size[0]) or np.any(points[:, 1] >= image_size[1]):
        return None, np.nan, np.nan, "point_outside_image"
    try:
        uv = [cv2.undistortPoints(p.reshape(1, 1, 2), c.K, c.dist).reshape(2)
              for p, c in zip(points, rig.cameras)]
        rays = [c.R @ np.r_[p, 1.0] for p, c in zip(uv, rig.cameras)]
        rays = [r / np.linalg.norm(r) for r in rays]
        angle = float(np.degrees(np.arccos(np.clip(abs(rays[0] @ rays[1]), 0, 1))))
        if angle < min_ray_angle_deg:
            return None, np.nan, angle, "weak_triangulation_angle"
        extrinsics = [np.linalg.inv(c.T_world_cam)[:3] for c in rig.cameras]
        homogeneous = cv2.triangulatePoints(*extrinsics, uv[0].reshape(2, 1), uv[1].reshape(2, 1)).ravel()
        if not np.isfinite(homogeneous).all() or abs(homogeneous[3]) < 1e-12:
            return None, np.nan, angle, "triangulation_at_infinity"
        point = homogeneous[:3] / homogeneous[3]
        errors = []
        for observation, cam in zip(points, rig.cameras):
            local = cam.R.T @ (point - cam.position)
            if local[2] <= 0:
                return None, np.nan, angle, "point_behind_camera"
            projected = cv2.projectPoints(local.reshape(1, 3), np.zeros(3), np.zeros(3), cam.K, cam.dist)[0].reshape(2)
            errors.append(np.linalg.norm(projected - observation))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        if max(errors) > max_reprojection_px:
            return None, rms, angle, "reprojection_error"
        return point, rms, angle, "ok"
    except cv2.error:
        return None, np.nan, np.nan, "opencv_geometry_error"


def triangulate_annotations(run_dir, annotations_csv, calibration_path, *,
                            max_skew_s=0.01, max_reprojection_px=2.0,
                            min_ray_angle_deg=1.0, min_axis_length_mm=0.1,
                            preprocessing_confirmed=False):
    """Return every encoded frame, with separate position/orientation validity.

    Required annotation columns: ``frame_index,ax_px,ay_px,bx_px,by_px``.
    Optional: ``a_tip_x_px,a_tip_y_px,b_tip_x_px,b_tip_y_px``. ``frame_index``
    means the ORIGINAL capture index, not the index after dropped frames vanish.

    All output XYZ and unit directions are in camera A coordinates, in mm. Gravity
    rebasing and experiment-clock mapping happen later. Threshold defaults are
    engineering starting points, not camera accuracy measurements. No interpolation
    or carried-forward pose is performed across missing data.
    """
    thresholds = [max_skew_s, max_reprojection_px, min_ray_angle_deg, min_axis_length_mm]
    if not np.isfinite(thresholds).all() or any(x <= 0 for x in thresholds):
        raise ValueError("Pose quality thresholds must be finite and positive")
    if min_ray_angle_deg >= 90:
        raise ValueError("min_ray_angle_deg must be below 90 degrees")
    capture = load_flight(run_dir)
    rig, calibration = _validated_rig(calibration_path, capture, preprocessing_confirmed)
    annotations = pd.read_csv(annotations_csv)
    required = {"frame_index", *CENTRE_COLUMNS}
    if not required.issubset(annotations.columns):
        raise ValueError(f"Annotations require {sorted(required)}")
    indices = pd.to_numeric(annotations["frame_index"], errors="raise").to_numpy(float)
    if not np.isfinite(indices).all() or np.any(indices != np.floor(indices)) or annotations["frame_index"].duplicated().any():
        raise ValueError("Annotation frame_index must contain unique integers")
    if not set(indices).issubset(set(capture["frame_indices"])):
        raise ValueError("Annotations reference frames not present in the encoded videos")
    annotations["frame_index"] = indices.astype(np.int64)
    tip_present = [name in annotations for name in TIP_COLUMNS]
    if any(tip_present) and not all(tip_present):
        raise ValueError("Provide all four axis-tip pixel columns or none")
    for name in (*CENTRE_COLUMNS, *(TIP_COLUMNS if all(tip_present) else ())):
        annotations[name] = pd.to_numeric(annotations[name], errors="raise")
    annotations = annotations.set_index("frame_index")
    output = []
    for index, ta, tb in zip(capture["frame_indices"], capture["times_a"], capture["times_b"]):
        row = {
            "frame_index": int(index), "t_source_a_s": float(ta), "t_source_b_s": float(tb),
            "x_mm": np.nan, "y_mm": np.nan, "z_mm": np.nan,
            "nx": np.nan, "ny": np.nan, "nz": np.nan,
            "position_valid": False, "orientation_valid": False,
            "stereo_skew_s": abs(float(tb - ta)), "fit_rms_px": np.nan,
            "ray_angle_deg": np.nan, "axis_length_mm": np.nan, "axis_tip_rms_px": np.nan,
            "position_reason": "missing_annotation", "orientation_reason": "position_invalid",
            "pose_method": "annotated_correspondences", "coordinate_frame": "camera_A",
        }
        if row["stereo_skew_s"] > max_skew_s:
            row["position_reason"] = "stereo_skew_exceeds_limit"
        elif index in annotations.index:
            observation = annotations.loc[index]
            center, rms, angle, reason = _point(
                observation[list(CENTRE_COLUMNS[:2])], observation[list(CENTRE_COLUMNS[2:])],
                rig, calibration["image_size"], max_reprojection_px, min_ray_angle_deg)
            row.update(fit_rms_px=rms, ray_angle_deg=angle, position_reason=reason)
            if center is not None:
                local = rig.a.R.T @ (center - rig.a.position)
                row.update(zip(("x_mm", "y_mm", "z_mm"), map(float, local)))
                row["position_valid"] = True
                row["orientation_reason"] = "no_axis_tip_annotation"
                if all(tip_present):
                    tip, tip_rms, _, tip_reason = _point(
                        observation[list(TIP_COLUMNS[:2])], observation[list(TIP_COLUMNS[2:])],
                        rig, calibration["image_size"], max_reprojection_px, min_ray_angle_deg)
                    row["axis_tip_rms_px"] = tip_rms
                    row["orientation_reason"] = f"axis_tip_{tip_reason}"
                    if tip is not None:
                        direction = rig.a.R.T @ (tip - center)
                        length = float(np.linalg.norm(direction))
                        row["axis_length_mm"] = length
                        if length >= min_axis_length_mm:
                            row.update(zip(("nx", "ny", "nz"), map(float, direction / length)))
                            row.update(orientation_valid=True, orientation_reason="ok")
                        else:
                            row["orientation_reason"] = "axis_too_short"
        output.append(row)
    result = pd.DataFrame(output)
    result.attrs.update(
        coordinate_frame="camera_A", units="mm", annotations_source=str(Path(annotations_csv).resolve()),
        calibration_source=str(Path(calibration_path).resolve()),
        preprocessing_confirmed=bool(preprocessing_confirmed),
        quality_thresholds=dict(zip(("max_skew_s", "max_reprojection_px", "min_ray_angle_deg", "min_axis_length_mm"), thresholds)),
        metric_scale_measured=calibration.get("square_mm_measured"),
        calibration_world_frame=calibration.get("world_frame", "unspecified"),
        limitation="No temporal motion compensation; annotations must identify the same physical point. Axis roll is not observed.",
    )
    return result


def extract_pose(run_dir, calibration_path, *, annotations_csv=None, **kwargs):
    """Explicit annotation adapter; automatic horizontal-view detection is pending."""
    if annotations_csv is None:
        raise ValueError("Provide annotations_csv with matching stereo points; automatic robot detection is not validated for this setup")
    return triangulate_annotations(run_dir, annotations_csv, calibration_path, **kwargs)


if __name__ == "__main__":
    from LLM.tests.test_capture_pose import run_tests
    run_tests()
