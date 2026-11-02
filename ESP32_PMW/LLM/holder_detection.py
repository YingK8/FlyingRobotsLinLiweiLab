"""Image-only detector for the holder's four-quadrant material intersections.

These are custom checker targets, not encoded tags. A returned candidate has no
CAD identity. ``detect_labeled_centers`` only associates a user-supplied named
ROI when it contains exactly one candidate. All thresholds below are initial
offline settings, not measured camera/print performance or confidence levels.
"""
from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def _gray(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("Holder image must be a uint8 grayscale or BGR array")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError("Holder image must be grayscale or three-channel BGR")
    if not image.size:
        raise ValueError("Holder image is empty")
    return np.ascontiguousarray(image)


def _region(roi, width, height):
    if roi is None:
        return 0, 0, width, height
    values = np.asarray(roi)
    if (values.shape != (4,) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()):
        raise ValueError("Holder ROI must be four integer pixels [x, y, width, height]")
    x, y, w, h = (int(v) for v in values)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise ValueError("Holder ROI is outside the image or has non-positive size")
    return x, y, w, h


_ANGLE_COUNT = 96
_ANGLES = np.arange(_ANGLE_COUNT) * (2 * np.pi / _ANGLE_COUNT)
_RING_DIRECTIONS = np.column_stack((np.cos(_ANGLES), np.sin(_ANGLES)))


def _ring_evidence(gray, point, radius, min_contrast):
    xy = point + radius * _RING_DIRECTIONS
    if (np.any(xy < 1) or np.any(xy[:, 0] > gray.shape[1] - 2)
            or np.any(xy[:, 1] > gray.shape[0] - 2)):
        return None
    values = cv2.remap(gray, xy[:, 0].astype(np.float32).reshape(1, -1),
                       xy[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel().astype(float)
    values = (np.roll(values, 1) + 2 * values + np.roll(values, -1)) / 4
    low, high = np.percentile(values, [10, 90])
    contrast = float(high - low)
    if contrast < min_contrast:
        return None
    bright = values > (low + high) / 2
    transitions = np.flatnonzero(bright != np.roll(bright, 1))
    if len(transitions) != 4:
        return None
    widths = np.diff(np.r_[transitions, transitions[0] + _ANGLE_COUNT])
    # Two intersecting projected boundaries have opposite rays; a white square's
    # outer corner or a dot has only two transitions. These permissive angular
    # limits allow oblique views but do not claim to cover grazing incidence.
    if widths.min() < 6 or widths.max() > 42:
        return None
    opposite_error = np.max(np.abs(transitions[2:] - transitions[:2] - _ANGLE_COUNT / 2))
    if opposite_error > 5:
        return None
    agreement = float(np.mean(bright == np.roll(bright, _ANGLE_COUNT // 2)))
    if agreement < .85:
        return None
    return {"contrast": contrast, "opposite_agreement": agreement,
            "transitions": transitions, "radius_px": float(radius)}


def detect_checker_centers(image, *, roi=None, min_contrast=35.0,
                           radii_px=(3.0, 5.0, 8.0, 12.0), min_radius_support=2,
                           max_candidates=500, min_distance_px=5.0):
    """Return unlabeled checker-center candidates in original-image pixels.

    Candidate corners are refined with OpenCV ``cornerSubPix`` and must show
    four alternating light/dark sectors on at least ``min_radius_support``
    concentric rings. Local opposite boundary rays must agree; this supports
    rotated and moderately foreshortened targets. Occlusion, very small targets,
    glare or grazing views may cause missing detections and are never filled in.

    ``roi`` is an optional integer [x, y, width, height] search rectangle. Ring
    support is required inside this ROI; include a margin around the target.
    ``quality`` is a diagnostic score, NOT a probability or physical accuracy.
    Any other checker in the scene can also pass: there is no ID decoding here.
    """
    gray = _gray(image)
    x, y, w, h = _region(roi, gray.shape[1], gray.shape[0])
    try:
        radii = np.asarray(radii_px, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("radii_px must contain distinct positive finite radii") from exc
    if (radii.ndim != 1 or not np.isfinite(radii).all() or np.any(radii < 2)
            or len(np.unique(radii)) != len(radii) or len(radii) < 2):
        raise ValueError("radii_px must contain at least two distinct finite radii >= 2 pixels")
    radii = np.sort(radii)
    if (isinstance(min_radius_support, (bool, np.bool_))
            or not isinstance(min_radius_support, (int, np.integer))
            or not 2 <= min_radius_support <= len(radii)):
        raise ValueError("min_radius_support must be an integer from 2 to the number of radii")
    if (isinstance(max_candidates, (bool, np.bool_))
            or not isinstance(max_candidates, (int, np.integer))
            or not 1 <= max_candidates <= 10000):
        raise ValueError("max_candidates must be an integer from 1 to 10000")
    if not np.isfinite(min_contrast) or not 0 < min_contrast <= 255:
        raise ValueError("min_contrast must be finite and in (0, 255]")
    if not np.isfinite(min_distance_px) or min_distance_px < 1:
        raise ValueError("min_distance_px must be finite and at least one pixel")
    gray = np.ascontiguousarray(gray[y:y+h, x:x+w])
    if min(gray.shape) < max(12, 2 * radii[min_radius_support-1] + 3):
        return []
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=int(max_candidates),
                                      qualityLevel=.01, minDistance=2, blockSize=3)
    if corners is None:
        return []
    # A small window avoids pulling a compact target center toward outer corners.
    corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                              (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001))
    candidates = []
    for point in corners.reshape(-1, 2):
        support = [e for radius in radii
                   if (e := _ring_evidence(gray, point, radius, min_contrast)) is not None]
        if len(support) < min_radius_support:
            continue
        # A distant edge/noise pattern must not independently supply the second
        # ring. Transition directions must remain stable across accepted scales.
        best = []
        for anchor in support:
            group = []
            for evidence in support:
                distances = np.abs(anchor["transitions"][:, None] - evidence["transitions"][None, :])
                distances = np.minimum(distances, _ANGLE_COUNT - distances)
                if np.max(np.min(distances, axis=1)) <= 4:
                    group.append(evidence)
            if len(group) > len(best):
                best = group
        if len(best) < min_radius_support:
            continue
        contrast = float(np.median([s["contrast"] for s in best]))
        agreement = float(np.mean([s["opposite_agreement"] for s in best]))
        candidates.append({"center_px": [float(point[0] + x), float(point[1] + y)],
                           "contrast": contrast,
                           "quality": float(contrast / 255 * agreement * len(best) / len(radii)),
                           "opposite_agreement": agreement,
                           "radii_px": [s["radius_px"] for s in best]})
    # Refinement can converge multiple seeds to one saddle. Keep the best one;
    # final reading order makes candidate indices reproducible for review.
    candidates.sort(key=lambda c: (-c["quality"], c["center_px"][1], c["center_px"][0]))
    selected = []
    for candidate in candidates:
        if all(np.linalg.norm(np.subtract(candidate["center_px"], other["center_px"])) >= min_distance_px
               for other in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda c: (c["center_px"][1], c["center_px"][0]))


def detect_labeled_centers(image, regions, **detector_options):
    """Refine explicit label→ROI associations without guessing ambiguous IDs.

    Each result retains every candidate. A missing or ambiguous ROI has no
    ``center_px``. Overlapping ROIs that select the same center are ambiguous
    for *both* labels. Label names are validated against CAD by the caller.
    """
    if not isinstance(regions, Mapping):
        raise ValueError("regions must map CAD labels to ROI [x, y, width, height]")
    if "roi" in detector_options:
        raise ValueError("Pass per-label ROIs in regions, not detector_options")
    result = {}
    for label, roi in regions.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Region labels must be non-empty strings")
        if roi is None:
            raise ValueError("A labeled region requires an explicit ROI")
        candidates = detect_checker_centers(image, roi=roi, **detector_options)
        status = "detected" if len(candidates) == 1 else "missing" if not candidates else "ambiguous"
        result[label] = {"status": status, "candidates": candidates}
        if status == "detected":
            result[label]["center_px"] = candidates[0]["center_px"]
    accepted = [(label, r["center_px"]) for label, r in result.items() if r["status"] == "detected"]
    for i, (label, center) in enumerate(accepted):
        for other_label, other_center in accepted[i+1:]:
            if np.linalg.norm(np.subtract(center, other_center)) < 3:
                for name in (label, other_label):
                    result[name]["status"] = "ambiguous"
                    result[name]["reason"] = "same_center_assigned_to_multiple_labels"
                    result[name].pop("center_px", None)
    return result


def _self_check():
    import unittest
    from LLM.tests import test_holder_detection
    outcome = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromModule(test_holder_detection))
    if not outcome.wasSuccessful():
        raise AssertionError("holder checker detector checks failed")


if __name__ == "__main__":
    _self_check()
