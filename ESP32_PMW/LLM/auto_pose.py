"""Experimental per-frame disc-centre detections for the stereo annotation path.

Reuses ``DiscStereoEstimator``'s stateless ``segment_disc`` component (see
controller/pose/theory.md 20), not its permissive 3D answer. An ellipse centre
is a silhouette proxy: in perspective it need not project the physical rotor
centre. Geometric consistency alone does not establish position accuracy.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from LLM.capture import load_flight
from LLM.pose import _validated_rig, TIP_COLUMNS


def _fresh_centre(frame, segmenter, plate, config):
    """Return only evidence freshly segmented from this frame, never past pixels."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    segment = segmenter(gray, plate=plate, thresh=config["threshold"],
                        low=config["low_threshold"], min_area=config["min_area_px"],
                        max_area=config["max_area_px"])
    if segment is None:
        return None, np.nan, "no_current_frame_segmentation"
    mask = segment.mask
    if mask is None or np.shape(mask) != gray.shape or not np.any(mask):
        return None, np.nan, "missing_current_frame_mask"
    try:
        centre = np.asarray(segment.ellipse[0], dtype=float)
        axes = np.asarray(segment.ellipse[1], dtype=float)
        rms, area = float(segment.fit_rms_px), float(segment.area_px)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None, np.nan, "invalid_segmentation_geometry"
    if centre.shape != (2,) or axes.shape != (2,) or not np.isfinite([*centre, *axes, rms, area]).all():
        return None, np.nan, "nonfinite_segmentation_geometry"
    h, w = gray.shape
    if not (0 <= centre[0] < w and 0 <= centre[1] < h) or min(axes) <= 0 or rms < 0:
        return None, rms, "invalid_ellipse_bounds"
    if not config["min_area_px"] <= area <= config["max_area_px"]:
        return None, rms, "area_outside_limits"
    if min(axes) / max(axes) < config["min_axis_ratio"]:
        return None, rms, "ellipse_too_edge_on"
    if rms > config["max_fit_rms_px"] or rms / max(axes) > config["max_fit_rms_rel"]:
        return None, rms, "ellipse_fit_residual"
    return centre, rms, "experimental_ellipse_center_proxy"


def generate_annotations(run_dir, calibration_path, *, preprocessing_confirmed=False,
                         background_paths=None, threshold=None, low_threshold=None,
                         min_area_px=800., max_area_px=40000., max_fit_rms_px=3.,
                         max_fit_rms_rel=.05, min_axis_ratio=.03, max_frames=None):
    """Return capture-indexed pixel detections; absent views stay NaN.

    Both videos are validated by ``load_flight`` first, and their saved pixel
    geometry must match the calibration. ``background_paths`` may explicitly map
    A/B to empty-scene grayscale images; no cached or automatically inferred plate
    is loaded. Threshold defaults come from the existing disc segmenter; the
    additional quality limits are starting guesses, not validated uncertainty.

    No shaft endpoints are fabricated. The existing ``find_mast`` returns a
    component centroid, not a corresponding material point in both views, so its
    output cannot be inserted as stereo tip annotations without further work.
    """
    from controller.pose.disc_pose import segment_disc

    limits = [min_area_px, max_area_px, max_fit_rms_px, max_fit_rms_rel, min_axis_ratio]
    if not np.isfinite(limits).all() or min(limits) <= 0:
        raise ValueError("Automatic detection quality limits must be finite and positive")
    if min_area_px > max_area_px or min_axis_ratio > 1:
        raise ValueError("Invalid area limits or min_axis_ratio")
    for name, value in (("threshold", threshold), ("low_threshold", low_threshold)):
        if value is not None and (not np.isfinite(value) or not 0 <= value <= 255):
            raise ValueError(f"{name} must be in 0..255 or None")
    if threshold is not None and low_threshold is not None and low_threshold >= threshold:
        raise ValueError("low_threshold must be smaller than threshold")
    if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
        raise ValueError("max_frames must be a positive integer or None")
    capture = load_flight(run_dir)
    _validated_rig(calibration_path, capture, preprocessing_confirmed)
    config = dict(threshold=threshold, low_threshold=low_threshold, min_area_px=float(min_area_px),
                  max_area_px=float(max_area_px), max_fit_rms_px=float(max_fit_rms_px),
                  max_fit_rms_rel=float(max_fit_rms_rel), min_axis_ratio=float(min_axis_ratio))
    backgrounds = {}
    paths = dict(background_paths or {})
    if set(paths) - set("AB"):
        raise ValueError("background_paths keys must be A and/or B")
    for tag, path in paths.items():
        plate = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        size = capture["metadata"]["decoded_image_sizes"][tag]
        if plate is None or list(plate.shape[::-1]) != size:
            raise ValueError(f"Background {tag} cannot be decoded or has wrong dimensions")
        backgrounds[tag] = plate
    decoders, output = [], []
    limit = len(capture["frame_indices"]) if max_frames is None else min(max_frames, len(capture["frame_indices"]))
    try:
        decoders = [cv2.VideoCapture(capture[f"video_{tag.lower()}"]) for tag in "AB"]
        if any(not decoder.isOpened() for decoder in decoders):
            raise ValueError("Cannot reopen validated stereo videos")
        for encoded_index, capture_index in enumerate(capture["frame_indices"][:limit]):
            row = {"frame_index": int(capture_index), "encoded_frame_index": encoded_index,
                   "annotation_source": "experimental_ellipse_center_proxy",
                   "orientation_reason": "axis_endpoint_unobserved"}
            row.update({name: np.nan for name in TIP_COLUMNS})
            for tag, decoder in zip("AB", decoders):
                ok, frame = decoder.read()
                if not ok:
                    raise ValueError(f"Camera {tag} decode ended early at encoded frame {encoded_index}")
                centre, rms, reason = _fresh_centre(frame, segment_disc, backgrounds.get(tag), config)
                prefix = tag.lower()
                row.update({f"{prefix}x_px": np.nan if centre is None else float(centre[0]),
                            f"{prefix}y_px": np.nan if centre is None else float(centre[1]),
                            f"{prefix}_detection_valid": centre is not None,
                            f"{prefix}_fit_rms_px": rms, f"{prefix}_detection_reason": reason})
            output.append(row)
        if max_frames is None and any(decoder.read()[0] for decoder in decoders):
            raise ValueError("Video frame count changed since validation")
    finally:
        for decoder in decoders:
            decoder.release()
    result = pd.DataFrame(output)
    result.attrs.update(
        detector="controller.pose.disc_pose.segment_disc", config=config,
        calibration_source=str(Path(calibration_path).resolve()),
        background_sources={tag: str(Path(path).resolve()) for tag, path in paths.items()},
        source="experimental_ellipse_center_proxy", orientation_observed=False,
        limitation="Ellipse centres need not be projections of the same physical point; validate against visible markers. No calibrated accuracy or axis estimate.",
    )
    return result


if __name__ == "__main__":
    from LLM.tests.test_auto_pose import run
    run()
