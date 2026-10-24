"""Plain-assert self-checks; no physical-camera timing claims."""

import tempfile
from pathlib import Path

import numpy as np

from LLM.sync import detect_led, detect_led_trace, fit_clock, map_times


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
    assert [e["frame_index"] for e in result["edges"]] == [4, 9]
    assert result["edges"][0]["lower_s"] == .01
    assert result["edges"][0]["upper_s"] == .04
    assert result["edges"][0]["confirmation_frame_index"] == 5
    assert detect_led_trace([255, 255, 255], [0, 1, 2])["status"] == "uncertain"
    assert detect_led_trace([0, 0, 0], [0, 1, 2])["status"] == "not_detected"
    assert detect_led_trace([0, 255, 0], [0, 1, 2])["status"] == "uncertain"
    late = detect_led_trace([255, 255, 0, 255, 255], [0, 1, 2, 3, 4])
    assert len(late["edges"]) == 1 and late["warnings"]
    _reject(lambda: detect_led_trace([0, 200], [0, 0]))
    _reject(lambda: detect_led_trace([0, 200], [0, np.nan]))
    _reject(lambda: detect_led_trace([0, 200], [0]))
    _reject(lambda: detect_led_trace([0, 200], [0, 1], {"min_on_frames": 1.5}))
    _reject(lambda: detect_led_trace([0, 200], [0, 1], {"low_threshold": 200}))


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
    test_clocks()
    test_video_correspondence()
    print("LLM sync: LED hysteresis, clock mapping, and decoded-frame correspondence checks passed")


if __name__ == "__main__":
    run()
