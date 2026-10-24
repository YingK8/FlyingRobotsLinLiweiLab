"""Offline LED detection and explicit source-to-experiment clock mappings.

Existing camera timestamps are host capture receipts, not exposure timestamps.
The intervals below bracket observations on that source clock; they are not
calibrated bounds on the physical LED switching time.
"""

from pathlib import Path

import numpy as np


def _vector(values, name, *, increasing=False):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional sequence")
    if increasing and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def detect_led_trace(brightness, source_times_s, config=None):
    """Detect rising edges from ROI means; exposed for deterministic offline checks.

    Thresholds are initial guesses on grayscale values 0..255 and must be checked
    against each recording. Hysteresis requires an observed dark state to rearm.
    A short bright spike cannot confirm a rising edge.
    """
    cfg = {"threshold": 160., "low_threshold": 80., "min_on_frames": 2,
           "timing_kind": "host_capture_receipt"}
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"unknown LED config fields: {sorted(unknown)}")
        cfg.update(config)
    low, high = float(cfg["low_threshold"]), float(cfg["threshold"])
    count = cfg["min_on_frames"]
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 255:
        raise ValueError("LED thresholds must satisfy 0 <= low_threshold < threshold <= 255")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("min_on_frames must be a positive integer")
    values = _vector(brightness, "brightness")
    times = _vector(source_times_s, "source_times_s", increasing=True)
    if len(values) != len(times):
        raise ValueError("brightness and timestamps must have exactly the same length")
    if np.any((values < 0) | (values > 255)):
        raise ValueError("brightness must be in grayscale range 0..255")
    edges, last_dark, candidate, armed = [], None, None, False
    saw_unconfirmed = False
    for index, value in enumerate(values):
        if value <= low:
            last_dark, candidate, armed = index, None, True
            continue
        if not armed:
            continue
        if value < high:
            candidate = None
            continue
        saw_unconfirmed = True
        if candidate is None:
            candidate = index
        if index - candidate + 1 >= count:
            lower, upper = float(times[last_dark]), float(times[candidate])
            edges.append({"source_time_s": (lower + upper) / 2,
                          "lower_s": lower, "upper_s": upper,
                          "frame_index": int(candidate),
                          "confirmation_frame_index": int(index),
                          "confirmation_source_time_s": float(times[index])})
            armed, candidate = False, None
    if edges:
        status, reason = "detected", "dark-to-bright observations confirmed by consecutive bright frames"
    elif last_dark is None:
        status, reason = "uncertain", "no dark observation; LED may already be on or ROI/thresholds may be wrong"
    elif saw_unconfirmed:
        status, reason = "uncertain", "bright candidates were too short to confirm"
    else:
        status, reason = "not_detected", "no confirmed dark-to-bright transition"
    warnings = []
    if values[0] > low:
        warnings.append("recording did not begin dark; an earlier rising edge cannot be recovered")
    return {"status": status, "edges": edges, "reason": reason, "warnings": warnings,
            "timing_kind": cfg["timing_kind"], "config": cfg,
            "interval_meaning": "bracket of source timestamp observations; exposure and transport latency uncalibrated",
            "trace": [{"frame_index": i, "source_time_s": float(t), "brightness": float(v)}
                      for i, (t, v) in enumerate(zip(times, values))]}


def detect_led(video_path, source_times_s, roi, config=None):
    """Read every decoded frame and check its one-to-one timestamp correspondence.

    ``roi`` is [x, y, width, height] in the saved video's pixel coordinates.
    Missing timestamps or an unmatched video/frame table are errors, never filled
    from a nominal FPS. Times must already exclude recorder-dropped frames.
    """
    import cv2

    times = _vector(source_times_s, "source_times_s", increasing=True)
    if len(roi) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in roi):
        raise ValueError("roi must contain four integers [x, y, width, height]")
    x, y, width, height = map(int, roi)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("roi origin must be nonnegative and dimensions positive")
    capture = cv2.VideoCapture(str(Path(video_path)))
    values = []
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot decode LED video: {video_path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if x + width > w or y + height > h:
                raise ValueError(f"roi {list(roi)} exceeds decoded frame dimensions {w}x{h}")
            crop = frame[y:y + height, x:x + width]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            values.append(float(np.mean(gray)))
    finally:
        capture.release()
    if len(values) != len(times):
        raise ValueError(f"decoded {len(values)} frames but received {len(times)} timestamps; alignment refused")
    result = detect_led_trace(values, times, config)
    result.update({"video_path": str(Path(video_path)), "roi": [x, y, width, height],
                   "decoded_frames": len(values)})
    return result


def fit_clock(source_anchors, experiment_anchors, source_name="unknown"):
    """Fit t_experiment = scale * t_source + offset from corresponding events.

    One event only identifies offset. Multiple events estimate average clock
    scale; the estimate still includes LED-detection and timestamp uncertainty.
    In particular, zero residual with two anchors is not an accuracy measurement.
    """
    source = _vector(source_anchors, "source_anchors", increasing=True)
    target = _vector(experiment_anchors, "experiment_anchors", increasing=True)
    if len(source) != len(target):
        raise ValueError("source and experiment anchor counts must match")
    if len(source) == 1:
        scale, method = 1., "single_anchor_offset"
    else:
        source_delta, target_delta = source - source.mean(), target - target.mean()
        scale = float(np.dot(source_delta, target_delta) / np.dot(source_delta, source_delta))
        method = "affine"
    offset = float(target.mean() - scale * source.mean())
    if not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("clock mapping must have finite positive scale and finite offset")
    residuals = target - (scale * source + offset)
    return {"source_name": str(source_name), "scale": scale, "offset_s": offset,
            "method": method, "drift_measured": len(source) > 1,
            "residual_max_s": float(np.abs(residuals).max()),
            "anchors": [{"source_time_s": float(s), "experiment_time_s": float(t),
                         "residual_s": float(r)} for s, t, r in zip(source, target, residuals)],
            "limitation": ("clock drift is unmeasured" if len(source) == 1 else
                           "average scale estimated from events; residual is not an optical synchronization bound")}


def map_times(times, clock):
    """Map source timestamps without changing the original array."""
    source = np.asarray(times, dtype=float)
    scale, offset = float(clock["scale"]), float(clock["offset_s"])
    if not np.isfinite(source).all() or not np.isfinite([scale, offset]).all() or scale <= 0:
        raise ValueError("timestamps and clock parameters must be finite, with positive scale")
    return source * scale + offset


def _self_check():
    from LLM.tests.test_sync import run
    run()


if __name__ == "__main__":
    _self_check()
