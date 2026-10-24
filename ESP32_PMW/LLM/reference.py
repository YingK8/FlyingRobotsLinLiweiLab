"""Establish a fixed bearing origin and gravity frame from stereo pixel points.

The operator identifies the same physical bearing point and both ends of a
stationary plumb line in one recorded stereo pair. The tabletop is not used as
a horizontal reference. See controller/calib/theory.md for stereo calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from LLM.capture import load_flight
from LLM.pose import _point, _validated_rig
from LLM.world import build_world_frame


def build_world_from_reference(run_dir, reference_json, calibration_path,
                               preprocessing_confirmed=False, *, max_skew_s=0.01,
                               max_reprojection_px=2.0, min_ray_angle_deg=1.0):
    """Return a world-frame dictionary; do not write or replace any evidence.

    ``reference_json`` is a path or dictionary containing ``schema_version=1``,
    the original ``frame_index`` and ``bearing``, ``plumb_bottom``, ``plumb_top``.
    Each point contains ``a: [x_px,y_px]`` and ``b: [x_px,y_px]``. An optional
    ``x_hint_camera`` chooses azimuth; its default is camera A's right direction.

    Top/bottom labels determine the sign of gravity and must be checked by the
    operator. This verifies geometry, not whether the plumb line was stationary
    or whether the annotations identify the intended physical points.
    """
    reference = dict(reference_json) if isinstance(reference_json, dict) else json.loads(Path(reference_json).read_text())
    if reference.get("schema_version") != 1:
        raise ValueError("Reference requires schema_version=1")
    index = reference.get("frame_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("Reference frame_index must be a nonnegative capture-index integer")
    limits = [max_skew_s, max_reprojection_px, min_ray_angle_deg]
    if not np.isfinite(limits).all() or any(x <= 0 for x in limits) or min_ray_angle_deg >= 90:
        raise ValueError("Reference quality limits must be finite and positive; ray angle must be below 90 degrees")
    capture = load_flight(run_dir)
    rig, calibration = _validated_rig(calibration_path, capture, preprocessing_confirmed)
    found = np.flatnonzero(capture["frame_indices"] == index)
    if len(found) != 1:
        raise ValueError("Reference frame_index is not present in the encoded videos")
    encoded_index = int(found[0])
    ta, tb = (float(capture[f"times_{tag}"][encoded_index]) for tag in "ab")
    if abs(ta - tb) > max_skew_s:
        raise ValueError("Reference stereo skew exceeds max_skew_s")
    points, diagnostics, pixels = {}, {}, {}
    for name in ("bearing", "plumb_bottom", "plumb_top"):
        entry = reference.get(name)
        if not isinstance(entry, dict) or not {"a", "b"}.issubset(entry):
            raise ValueError(f"Reference {name} requires camera a/b pixel coordinates")
        uv = np.asarray([entry["a"], entry["b"]], dtype=float)
        if uv.shape != (2, 2) or not np.isfinite(uv).all():
            raise ValueError(f"Reference {name} needs two finite [x_px,y_px] pairs")
        point, rms, angle, reason = _point(
            uv[0], uv[1], rig, calibration["image_size"], max_reprojection_px, min_ray_angle_deg)
        if point is None:
            raise ValueError(f"Reference {name} failed triangulation: {reason}")
        local = rig.a.R.T @ (point - rig.a.position)
        points[name] = local
        pixels[name] = {"a": uv[0].tolist(), "b": uv[1].tolist()}
        diagnostics[name] = {"camera_A_mm": local.tolist(), "reprojection_rms_px": rms,
                             "ray_angle_deg": angle}
    up = points["plumb_top"] - points["plumb_bottom"]
    length = float(np.linalg.norm(up))
    if not np.isfinite(length) or length <= 1e-6:
        raise ValueError("Plumb top and bottom must be distinct physical points")
    frame = build_world_frame(points["bearing"], up, reference.get("x_hint_camera", [1, 0, 0]))
    frame["evidence"] = {
        "method": "stereo_static_reference_points",
        "frame_index": index, "encoded_frame_index": encoded_index,
        "t_source_a_s": ta, "t_source_b_s": tb, "stereo_skew_s": abs(ta - tb),
        "reference_pixels": pixels, "triangulated_points": diagnostics,
        "plumb_length_mm": length, "note": str(reference.get("note", "")),
        "source_file": None if isinstance(reference_json, dict) else str(Path(reference_json).resolve()),
        "calibration_source": str(Path(calibration_path).resolve()),
        "metric_scale_measured": calibration.get("square_mm_measured"),
        "preprocessing_confirmed": bool(preprocessing_confirmed),
        "quality_thresholds": {"max_skew_s": float(max_skew_s),
                               "max_reprojection_px": float(max_reprojection_px),
                               "min_ray_angle_deg": float(min_ray_angle_deg)},
        "operator_assumptions": "Same physical points in both views; plumb line stationary; top/bottom correctly labelled; bearing point fixed.",
    }
    return frame


if __name__ == "__main__":
    from LLM.tests.test_reference import run
    run()
