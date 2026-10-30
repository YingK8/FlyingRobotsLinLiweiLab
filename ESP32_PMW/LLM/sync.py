"""Offline LED detection and explicit source-to-experiment clock mappings.

Existing camera timestamps are host capture receipts, not exposure timestamps.
The intervals below bracket observations on that source clock; they are not
calibrated bounds on the physical LED switching time.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EPSILON_S = 1e-9


@dataclass
class LevelRun:
    level: bool
    starts_at: int
    ends_at: int


def _vector(values, name, *, increasing=False):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional sequence")
    if increasing and np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def _runs_held_for(levels, times, minimum_duration_s):
    runs = []
    for index, level in enumerate(levels):
        if level is None:
            continue
        if runs and runs[-1].level == level:
            runs[-1].ends_at = index
        else:
            runs.append(LevelRun(level, index, index))
    position = 0
    while position < len(runs):
        held_s = times[runs[position].ends_at] - times[runs[position].starts_at]
        if len(runs) == 1 or held_s + _EPSILON_S >= minimum_duration_s:
            position += 1
            continue
        if position == 0:
            del runs[0]
            continue
        if position + 1 == len(runs):
            del runs[-1]
            position -= 1
            continue
        runs[position - 1].ends_at = runs[position + 1].ends_at
        del runs[position:position + 2]
        position -= 1
    return runs


def detect_led_trace(brightness, source_times_s, config=None):
    """Level transitions from ROI means; `min_on_s` defaults to one measured frame."""
    settings = {"threshold": 160., "low_threshold": 80., "min_on_s": None,
                "timing_kind": "host_capture_receipt"}
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError(f"unknown LED config fields: {sorted(unknown)}")
        settings.update(config)
    low_threshold, high_threshold = float(settings["low_threshold"]), float(settings["threshold"])
    if not np.isfinite([low_threshold, high_threshold]).all() or not 0 <= low_threshold < high_threshold <= 255:
        raise ValueError("LED thresholds must satisfy 0 <= low_threshold < threshold <= 255")
    values = _vector(brightness, "brightness")
    times = _vector(source_times_s, "source_times_s", increasing=True)
    if len(values) != len(times):
        raise ValueError("brightness and timestamps must have exactly the same length")
    if np.any((values < 0) | (values > 255)):
        raise ValueError("brightness must be in grayscale range 0..255")
    frame_period_s = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    min_on_s = frame_period_s if settings["min_on_s"] is None else float(settings["min_on_s"])
    if not np.isfinite(min_on_s) or min_on_s <= 0:
        raise ValueError("min_on_s must be a positive duration; pass one when the "
                         "recording is too short to measure a frame interval")
    settings["min_on_s"], settings["frame_period_s"] = min_on_s, frame_period_s

    levels, current_level = [], None
    for value in values:
        if current_level is not True and value >= high_threshold:
            current_level = True
        elif current_level is not False and value <= low_threshold:
            current_level = False
        levels.append(current_level)

    runs = _runs_held_for(levels, times, min_on_s)
    transitions = []
    for before, after in zip(runs, runs[1:]):
        last_time_before = float(times[before.ends_at])
        first_time_after = float(times[after.starts_at])
        confirmation_index = next(
            (index for index in range(after.starts_at, after.ends_at + 1)
             if times[index] - times[after.starts_at] + _EPSILON_S >= min_on_s),
            after.ends_at)
        transitions.append({
            "level": bool(after.level),
            "source_time_s": (last_time_before + first_time_after) / 2,
            "lower_s": last_time_before, "upper_s": first_time_after,
            "frame_index": after.starts_at,
            "confirmation_frame_index": int(confirmation_index),
            "confirmation_source_time_s": float(times[confirmation_index]),
        })
    edges = [transition for transition in transitions if transition["level"]]
    saw_bright_sample = bool(np.any(values >= high_threshold))
    if transitions:
        status, reason = "detected", "level changes held for at least min_on_s"
    elif not runs or all(run.level for run in runs):
        status, reason = "uncertain", "no dark observation; LED may already be on or ROI/thresholds may be wrong"
    elif saw_bright_sample:
        status, reason = "uncertain", "bright candidates were too short to confirm"
    else:
        status, reason = "not_detected", "no confirmed level transition"
    warnings = []
    if levels[0] is True:
        warnings.append("recording did not begin dark; an earlier rising edge cannot be recovered")
    return {"status": status, "edges": edges, "transitions": transitions,
            "reason": reason, "warnings": warnings,
            "timing_kind": settings["timing_kind"], "config": settings,
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


def decode_markers(trace, on_s, gap_s, tol_rel=0.25):
    """START/STOP pulse pairs decoded against the timing the firmware declared."""
    for name, value in (("on_s", on_s), ("gap_s", gap_s)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive duration")
    if not np.isfinite(tol_rel) or not 0 < tol_rel < 1:
        raise ValueError("tol_rel must be in (0, 1)")
    frame_period_s = float(trace["config"].get("frame_period_s", 0.0))
    tolerance_s = max(tol_rel * min(on_s, gap_s), 2 * frame_period_s)
    transitions = trace["transitions"]
    markers = []
    position = 0
    while position + 2 < len(transitions):
        three_edges = transitions[position:position + 3]
        levels = [bool(edge["level"]) for edge in three_edges]
        edge_times = [float(edge["source_time_s"]) for edge in three_edges]
        kind = ("start" if levels == [True, False, True] else
                "stop" if levels == [False, True, False] else None)
        measured_on_s, measured_gap_s = edge_times[1] - edge_times[0], edge_times[2] - edge_times[1]
        matches = (abs(measured_on_s - on_s) <= tolerance_s
                   and abs(measured_gap_s - gap_s) <= tolerance_s)
        if kind and matches:
            markers.append({"kind": kind, "first_source_time_s": edge_times[0],
                            "edges_s": edge_times, "on_s": measured_on_s,
                            "gap_s": measured_gap_s, "tolerance_s": tolerance_s})
            position += 3
        else:
            position += 1
    return {"markers": markers, "tolerance_s": tolerance_s, "frame_period_s": frame_period_s,
            "declared": {"on_s": float(on_s), "gap_s": float(gap_s)}}


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
