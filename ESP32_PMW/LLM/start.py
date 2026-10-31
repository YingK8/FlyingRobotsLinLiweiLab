"""Associate one recorded flight START with optical markers, never host-send time."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from LLM.io import invalidate_report, sha256, write_json
from LLM.pipeline import add_clock, synchronize
from LLM.telemetry import parse_log

_START = re.compile(
    r"SYNC_START v=2 led0=([0-9]+) led1=([0-9]+) drive0=([0-9]+) drive1=([0-9]+) "
    r"on_us=([0-9]+) gap_us=([0-9]+)")
_STOP = re.compile(r"SYNC_STOP v=2 t0=([0-9]+) t1=([0-9]+) on_us=([0-9]+) gap_us=([0-9]+)")


def read_start(path):
    """One READY, one START and at most one STOP; everything else is refused."""
    ready, start, stop = False, None, None
    previous_host = -math.inf
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(raw)
            line, host = row["line"], row["host_receive_s"]
            if (not isinstance(line, str) or isinstance(host, bool)
                    or not isinstance(host, (int, float)) or not math.isfinite(host)):
                raise ValueError("invalid RX record")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise ValueError(f"invalid raw log record at line {line_no}") from exc
        if host < previous_host:
            raise ValueError("host receive time reversed; split recordings")
        previous_host = host
        if line.startswith("SYNC_READY"):
            if line != "SYNC_READY v=1" or ready:
                raise ValueError("unsupported READY or multiple boots; use one boot/START per run")
            ready = True
        elif line.startswith("SYNC_START"):
            match = _START.fullmatch(line)
            if not ready or start is not None or match is None:
                raise ValueError("missing READY, duplicate START, malformed START, or a "
                                 "SYNC_START v=1 log from firmware that no longer exists")
            values = [int(value) for value in match.groups()]
            if values[:4] != sorted(values[:4]) or values[3] >= (2**32) * 1000:
                raise ValueError("invalid START order or unsupported millis() wrap")
            if min(values[4:]) <= 0:
                raise ValueError("START marker timing must be positive")
            start = dict(zip(("led0_us", "led1_us", "drive0_us", "drive1_us",
                              "on_us", "gap_us"), values))
            start.update(raw_line_number=line_no, host_receive_s=host)
        elif line.startswith("SYNC_STOP"):
            match = _STOP.fullmatch(line)
            if start is None or stop is not None or match is None:
                raise ValueError("STOP before START, duplicate STOP, or malformed STOP")
            values = [int(value) for value in match.groups()]
            if values[:2] != sorted(values[:2]) or values[1] >= (2**32) * 1000:
                raise ValueError("invalid STOP order or unsupported millis() wrap")
            if values[0] < start["drive1_us"]:
                raise ValueError("STOP precedes the drive start")
            if values[2:] != [start["on_us"], start["gap_us"]]:
                raise ValueError("STOP marker timing disagrees with START; mixed recordings")
            stop = dict(zip(("t0_us", "t1_us", "on_us", "gap_us"), values))
            stop.update(raw_line_number=line_no, host_receive_s=host)
    if start is None:
        raise ValueError("no device SYNC_START v=2 in raw log; do not infer it from seq=go or host time")
    current = parse_log(path)
    if current.attrs["parse_errors"]:
        raise ValueError("malformed telemetry; inspect the raw log before START association")
    for row in current.itertuples():
        if ((row.raw_line_number < start["raw_line_number"]
             and row.t_device_s > start["led0_us"] / 1e6)
                or (row.raw_line_number > start["raw_line_number"]
                    and row.t_device_s + .001 < start["drive1_us"] / 1e6)):
            raise ValueError("telemetry clock disagrees with START; possible reset or mixed recordings")
    return {"start": start, "stop": stop}


def led_roi(video_path, times, seconds=5.0):
    import cv2
    from controller.pose.scene import locate_led
    capture, frames = cv2.VideoCapture(str(video_path)), []
    while len(frames) < len(times) and times[len(frames)] - times[0] <= seconds:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    circle = locate_led(frames, times[:len(frames)])
    if circle is None:
        raise ValueError(f"no Hamming LED code in the first {seconds:g} s of {video_path}; pass the ROI by hand")
    cx, cy, r = circle
    height, width = frames[0].shape
    x, y = max(int(cx - r), 0), max(int(cy - r), 0)
    return [x, y, min(int(2 * r + 1), width - x), min(int(2 * r + 1), height - y)]


def synchronize_start(run, roi_a=None, roi_b=None):
    """Two optical markers per camera + one device event -> all three clock mappings."""
    run = Path(run)
    invalidate_report(run, "START association changed or failed; rerun sync-start then process.")
    sync_path = run / "processed/sync.json"
    sync_path.unlink(missing_ok=True)
    device_events = read_start(run / "raw/telemetry.jsonl")
    if roi_a is None or roi_b is None:
        from LLM.capture import load_flight
        flight = load_flight(run)
        roi_a, roi_b = (roi or led_roi(flight[f"video_{tag}"], flight[f"times_{tag}"])
                        for roi, tag in ((roi_a, "a"), (roi_b, "b")))
    start, stop = device_events["start"], device_events["stop"]
    reference_s = (start["led0_us"] + start["led1_us"]) / 2e6
    stop_experiment_s = None if stop is None else (stop["t0_us"] + stop["t1_us"]) / 2e6 - reference_s
    result = synchronize(run, roi_a, roi_b, markers={
        "on_s": start["on_us"] / 1e6, "gap_s": start["gap_us"] / 1e6,
        "stop_experiment_s": stop_experiment_s})
    try:
        clock = add_clock(run, "esp", [reference_s], [0.0],
                          f"raw/telemetry.jsonl line {start['raw_line_number']}: LED GPIO call midpoint; optical/electrical delay unvalidated")
        result["clocks"]["esp"] = clock
        result["input_sha256"]["raw/telemetry.jsonl"] = sha256(run / "raw/telemetry.jsonl")
        result["device_start"] = start | {
            "led_reference_device_s": reference_s,
            "drive_apply_interval_t_s": [start[key] / 1e6 - reference_s
                                         for key in ("drive0_us", "drive1_us")],
            "measurement_kind": "software_call_brackets_not_measured_current_onset",
            "drift_measured": False,
        }
        if stop is not None:
            result["device_stop"] = stop | {
                "stop_reference_device_t_s": stop_experiment_s,
                "measurement_kind": "software_call_brackets_not_measured_current_onset"}
        result["association"] = (
            "operator selected START LED ROIs; the firmware-declared pulse pair was decoded "
            "in both cameras and matched to the sole device START"
            + (" and STOP" if stop is not None else ""))
        write_json(sync_path, result)
    except BaseException:
        sync_path.unlink(missing_ok=True)
        raise
    return result
