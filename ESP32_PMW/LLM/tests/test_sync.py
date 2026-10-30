"""Plain-assert self-checks; no physical-camera timing claims."""

import tempfile
from pathlib import Path

import numpy as np

from LLM.sync import decode_markers, detect_led, detect_led_trace, fit_clock, map_times


def _reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid input was not rejected")


def test_edges():
    times = np.arange(12) / 100
    result = detect_led_trace([0, 0, 190, 100, 210, 220, 180, 0, 100, 240, 240, 230], times)
    assert result["status"] == "detected"
    assert [e["frame_index"] for e in result["edges"]] == [2, 9]
    assert result["edges"][0]["lower_s"] == .01
    assert result["edges"][0]["upper_s"] == .02
    assert result["edges"][0]["confirmation_frame_index"] == 3
    assert [t["level"] for t in result["transitions"]] == [True, False, True]
    assert detect_led_trace([255, 255, 255], [0, 1, 2])["status"] == "uncertain"
    assert detect_led_trace([0, 0, 0], [0, 1, 2])["status"] == "not_detected"
    assert detect_led_trace([0, 255, 0], [0, 1, 2])["status"] == "uncertain"
    late = detect_led_trace([255, 255, 0, 255, 255], [0, 1, 2, 3, 4])
    assert not late["edges"] and late["status"] == "uncertain" and late["warnings"]
    assert detect_led_trace([0, 255, 0, 0], [0, .01, .02, .03])["status"] == "uncertain"
    assert not detect_led_trace([0, 255, 0, 0], [0, .01, .02, .03])["edges"]
    assert detect_led_trace([0, 0, 255, 255, 0, 0], np.arange(6) / 100)["edges"]
    _reject(lambda: detect_led_trace([0, 200], [0, 0]))
    _reject(lambda: detect_led_trace([0, 200], [0, np.nan]))
    _reject(lambda: detect_led_trace([0, 200], [0]))
    _reject(lambda: detect_led_trace([0, 200], [0, 1], {"min_on_frames": 1.5}))
    _reject(lambda: detect_led_trace([0, 200], [0, 1], {"min_on_s": 0}))
    _reject(lambda: detect_led_trace([0, 200], [0, 1], {"low_threshold": 200}))


def _marker_trace(on_ms=30, gap_ms=30, fps=120.0, stop_frame=None, glint=False, hold_gap=None):
    """Synthetic ROI means for a START marker, optionally followed by a STOP."""
    dt = 1 / fps
    frames = 60 if stop_frame is not None else 40
    times = 100 + np.arange(frames) * dt
    period = on_ms / 1e3

    def level(t):
        if glint:
            return 0 < t < dt          # one bright sample, then dark again
        if t < 6 * dt:
            return 0
        if t < 6 * dt + period:
            return 255
        if t < 6 * dt + period + (hold_gap or gap_ms) / 1e3:
            return 0
        if stop_frame is None or t < stop_frame * dt:
            return 255
        if t < stop_frame * dt + period:
            return 0
        if t < stop_frame * dt + period + gap_ms / 1e3:
            return 255
        return 0

    values = [level(t - 100) for t in times]
    return detect_led_trace(values, times), times


def test_markers():
    trace, _ = _marker_trace()
    decoded = decode_markers(trace, .03, .03)
    assert [m["kind"] for m in decoded["markers"]] == ["start"]
    assert abs(decoded["markers"][0]["on_s"] - .03) < .02
    assert decoded["tolerance_s"] >= 2 / 120

    glint, _ = _marker_trace(glint=True)
    assert not decode_markers(glint, .03, .03)["markers"]
    wrong, _ = _marker_trace(hold_gap=120)
    assert not decode_markers(wrong, .03, .03)["markers"]
    assert not decode_markers(detect_led_trace([255] * 40, np.arange(40) / 120), .03, .03)["markers"]

    both, times = _marker_trace(stop_frame=40)
    decoded = decode_markers(both, .03, .03)
    assert [m["kind"] for m in decoded["markers"]] == ["start", "stop"]
    assert decoded["markers"][1]["first_source_time_s"] > decoded["markers"][0]["first_source_time_s"]

    _reject(lambda: decode_markers(both, 0, .03))
    _reject(lambda: decode_markers(both, .03, np.nan))
    _reject(lambda: decode_markers(both, .03, .03, 0))
    _reject(lambda: decode_markers(both, .03, .03, 1.5))


def test_clocks():
    one = fit_clock([15.2], [0], "A")
    assert not one["drift_measured"] and one["method"] == "single_anchor_offset"
    assert np.allclose(map_times([15.2, 16.2], one), [0, 1])
    two = fit_clock([100, 110], [0, 10.001], "B")
    assert two["drift_measured"] and two["method"] == "affine"
    assert np.isclose(two["scale"], 1.0001)
    assert np.allclose(map_times([100, 105, 110], two), [0, 5.0005, 10.001])
    many = fit_clock([0, 1, 2], [0, 1, 2.1])
    assert many["residual_max_s"] > 0
    for source, target in [([], []), ([0, 0], [0, 1]), ([0, 1], [1, 0]), ([0, np.inf], [0, 1]), ([0], [0, 1])]:
        _reject(lambda: fit_clock(source, target))
    _reject(lambda: map_times([np.nan], one))


def test_video_correspondence():
    import cv2

    with tempfile.TemporaryDirectory(prefix="llm_sync_") as temp:
        path = Path(temp) / "led.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (32, 24))
        assert writer.isOpened(), "MJPG encoder unavailable; video test not exercised"
        try:
            for value in [0, 0, 255, 255, 255]:
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        result = detect_led(path, np.arange(5) / 20, [2, 2, 10, 10])
        assert result["status"] == "detected" and result["edges"][0]["frame_index"] == 2
        assert result["decoded_frames"] == 5
        _reject(lambda: detect_led(path, np.arange(4) / 20, [2, 2, 10, 10]))
        _reject(lambda: detect_led(path, np.arange(5) / 20, [30, 20, 10, 10]))
        _reject(lambda: detect_led(path, np.arange(5) / 20, [0., 0, 10, 10]))


def run():
    test_edges()
    test_markers()
    test_clocks()
    test_video_correspondence()
    print("LLM sync: duration-gated LED levels, START/STOP marker decoding, clock mapping,"
          " and decoded-frame correspondence checks passed")


if __name__ == "__main__":
    run()
