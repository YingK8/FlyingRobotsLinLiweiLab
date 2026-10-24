"""An explicit gravity-aligned experiment frame, separate from robot zeroing.

All input positions and directions belong to camera A's calibrated frame.
This module does not infer gravity from the robot or from a tabletop.
"""
from __future__ import annotations

import numpy as np


def _vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return vector


def _valid_flags(values, name):
    """Accept CSV booleans without interpreting the string 'False' as true."""
    flags = []
    for value in values:
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            flags.append(False)
        elif isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            flags.append(value.strip().lower() in {"true", "1"})
        elif isinstance(value, (bool, np.bool_, int, float, np.number)) and value in (0, 1):
            flags.append(bool(value))
        else:
            raise ValueError(f"{name} must contain explicit boolean or 0/1 values")
    return np.asarray(flags, dtype=bool)


def build_world_frame(origin, up, x_hint=(1, 0, 0)):
    """Build a right-handed frame using measured upward gravity and an azimuth hint.

    ``origin`` is a fixed bearing reference in mm; ``up`` points up the plumb
    line. By default, the camera A right axis supplies only the azimuth.
    A nearly vertical hint is rejected rather than silently changing axes.
    """
    origin = _vector(origin, "origin")
    z = _vector(up, "up")
    hint = _vector(x_hint, "x_hint")
    lengths = np.array([np.linalg.norm(z), np.linalg.norm(hint)])
    if not np.isfinite(lengths).all() or np.any(lengths < 1e-12):
        raise ValueError("up and x_hint must have finite nonzero lengths")
    z = z / lengths[0]
    hint = hint / lengths[1]
    x = hint - np.dot(hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        raise ValueError("x_hint is nearly parallel to up; choose and record another hint")
    x /= np.linalg.norm(x)
    rotation = np.column_stack((x, np.cross(z, x), z))
    return {
        "schema_version": 1,
        "units": "mm",
        "source_frame": "camera_A",
        "origin_camera_mm": origin.tolist(),
        "R_camera_from_world": rotation.tolist(),
        "up_camera": z.tolist(),
        "x_hint_camera": hint.tolist(),
        "definition": "plumb-up z; x_hint projected perpendicular to z; y=z cross x",
    }


def validate_world_frame(frame):
    """Return the origin and proper rotation, refusing implicit frame repairs."""
    if frame.get("schema_version") != 1 or frame.get("units") != "mm":
        raise ValueError("world frame requires schema_version=1 and units='mm'")
    if frame.get("source_frame", "camera_A") != "camera_A":
        raise ValueError("world frame must be expressed in camera_A coordinates")
    if not {"origin_camera_mm", "R_camera_from_world"}.issubset(frame):
        raise ValueError("world frame is missing its origin or rotation")
    origin = _vector(frame["origin_camera_mm"], "origin_camera_mm")
    rotation = np.asarray(frame["R_camera_from_world"], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0)):
        raise ValueError("R_camera_from_world must be an orthonormal right-handed rotation")
    return origin, rotation


def transform_pose_dataframe(df, frame):
    """Transform position and axis independently; invalid estimates remain invalid.

    Valid axes are normalised, but their sign is preserved. Their two reported
    lean angles are not an estimate of rotation around the robot's own axis.
    """
    origin, rotation = validate_world_frame(frame)
    position_cols = ["x_mm", "y_mm", "z_mm"]
    axis_cols = ["nx", "ny", "nz"]
    required = position_cols + axis_cols + ["position_valid", "orientation_valid"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"pose is missing required columns: {sorted(missing)}")
    result = df.copy()
    if "coordinate_frame" in df and not df.coordinate_frame.eq("camera_A").all():
        raise ValueError("input pose must be in camera_A; refusing a second world transform")
    result["source_coordinate_frame"] = "camera_A"
    result["coordinate_frame"] = "experiment_world"
    positions = df[position_cols].to_numpy(dtype=float)
    axes = df[axis_cols].to_numpy(dtype=float)
    pos_valid = _valid_flags(df.position_valid, "position_valid") & np.isfinite(positions).all(axis=1)
    ori_valid = _valid_flags(df.orientation_valid, "orientation_valid") & np.isfinite(axes).all(axis=1)
    lengths = np.linalg.norm(axes, axis=1)
    ori_valid &= np.isfinite(lengths) & (lengths > 1e-12)
    positions = (positions - origin) @ rotation
    pos_valid &= np.isfinite(positions).all(axis=1)
    transformed_axes = np.full_like(axes, np.nan)
    transformed_axes[ori_valid] = (axes[ori_valid] / lengths[ori_valid, None]) @ rotation
    positions[~pos_valid] = np.nan
    result[position_cols] = positions
    result[axis_cols] = transformed_axes
    result["position_valid"], result["orientation_valid"] = pos_valid, ori_valid
    result["alpha_x_deg"] = np.degrees(np.arctan2(transformed_axes[:, 0], transformed_axes[:, 2]))
    result["alpha_y_deg"] = np.degrees(np.arctan2(transformed_axes[:, 1], transformed_axes[:, 2]))
    return result


def _self_check():
    frame = build_world_frame([10, 20, 30], [0.2, -0.8, 0.5])
    origin, rotation = validate_world_frame(frame)
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.allclose(rotation.T @ (origin + rotation[:, 2] - origin), [0, 0, 1])
    print("world frame self-check passed")


if __name__ == "__main__":
    _self_check()
