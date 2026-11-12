"""Associate one recorded flight START with optical markers, never host-send time."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from LLM.record.io import invalidate_report, read_json, sha256, write_json
from LLM.analyse.pipeline import add_clock, synchronize
from LLM.record.telemetry import parse_log

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


def find_led(video_path, times, start_host_s, on_s, gap_s, before_s=1.0, after_s=0.5,
             threshold=40):
    """[x, y, w, h] of the spot that blinks the START pulse pair near the device START.

    Window: camera and serial host stamps are both `time.monotonic()` in one process, and
    the START line is received just after the LED pulses, so look back `before_s`. Any
    pixel that changes on at least three frame steps is a candidate (on-off-on); a moving
    robot also qualifies, so a component is kept only if its mean trace is dark, then the
    firmware-declared pair (`sync.decode_markers`, the test the sync applies later), then
    held on -- three transitions and no more.
    """
    import cv2
    from LLM.analyse.led import decode_markers, detect_led_trace
    times = np.asarray(times, float)
    first, last = np.searchsorted(times, [start_host_s - before_s, start_host_s + after_s])
    capture, frames = cv2.VideoCapture(str(video_path)), []
    for _ in range(first):
        capture.grab()
    for _ in range(first, last):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(frames) < 4:
        raise ValueError(f"{video_path}: no frames around START; pass the LED ROI by hand")
    toggles = np.zeros(frames[0].shape, np.uint16)
    for previous, current in zip(frames, frames[1:]):
        toggles += cv2.threshold(cv2.absdiff(current, previous), threshold, 1, cv2.THRESH_BINARY)[1]
    count, _, stats, _ = cv2.connectedComponentsWithStats((toggles >= 3).astype(np.uint8))
    window_times = times[first:first + len(frames)]
    for label in 1 + np.argsort(-stats[1:, cv2.CC_STAT_AREA]):
        # No padding: tight boxes keep the most contrast in the ROI mean, though
        # `detect_led_trace` now sets its thresholds from the trace (2026-11-06).
        x0, y0, w, h = map(int, stats[label, :4])
        x1, y1 = x0 + w, y0 + h
        trace = detect_led_trace([float(f[y0:y1, x0:x1].mean()) for f in frames], window_times)
        # Exactly dark -> on-off-on -> held on: a periodic blinker (a spinning rotor) can
        # match one pulse pair by chance, but not with nothing else in the window.
        markers = decode_markers(trace, on_s, gap_s)["markers"]
        if len(trace["transitions"]) == 3 and [m["kind"] for m in markers] == ["start"]:
            return [x0, y0, x1 - x0, y1 - y0]
    raise ValueError(f"{video_path}: no blinking spot decodes the START pulse pair; pass the LED ROI by hand")


#: Host receipt minus LED change, sync_check_..._201924 and _202040 against the
#: serial-envelope clock below. One definition, shared with the controllers.
from controller.control.constants import CAMERA_LATENCY_S


def serial_offset(rows):
    """host_monotonic - device_clock, from telemetry lines ``t=<ms>``: their host receipt
    is late by a positive, jittery USB + polling delay, so the lower envelope (2nd
    percentile) is the offset. ~1 ms, measured the same way CAMERA_LATENCY_S was."""
    lag = [row["host_receive_s"] - int(m.group(1)) / 1e3 for row in rows
           if (m := re.match(r"t=(\d+) ", row["line"]))]
    if len(lag) < 5:
        raise ValueError(f"{len(lag)} telemetry lines; the host-clock sync needs >= 5")
    return float(np.percentile(lag, 2)), len(lag), float(np.median(lag) - np.percentile(lag, 2))


def synchronize_host(run, camera_latency_s=CAMERA_LATENCY_S):
    """All three clocks onto experiment time (t = 0 at the device START) with no LED.

    The ESP32 is mapped to the host through its telemetry stamps (`serial_offset`); the
    cameras are already on the host clock, less CAMERA_LATENCY_S. Writes the same
    processed/sync.json the LED path does. Expected error ~4 ms after the latency
    correction (the spread it was measured with), drift unmeasured; never an optical
    association -- the operator chose this over the LED on 2026-11-05.
    """
    from LLM.analyse.led import fit_clock
    run = Path(run)
    invalidate_report(run, "Host-clock sync changed or failed; rerun sync then process.")
    (run / "processed/sync.json").unlink(missing_ok=True)
    rows = [json.loads(raw) for raw in (run / "raw/telemetry.jsonl").read_text().splitlines()]
    events = read_start(run / "raw/telemetry.jsonl")
    start, stop = events["start"], events["stop"]
    offset, n_lines, median_lag = serial_offset(rows)
    reference_s = (start["led0_us"] + start["led1_us"]) / 2e6
    host_at_start = reference_s + offset
    clocks = {f"camera_{tag}": fit_clock([host_at_start + camera_latency_s], [0.0], f"camera_{tag}")
              for tag in "AB"}
    clocks["esp"] = fit_clock([reference_s], [0.0], "esp")
    clocks["esp"]["evidence"] = (f"raw/telemetry.jsonl line {start['raw_line_number']}: START GPIO-call midpoint")
    result = {"schema_version": 1, "method": "host_clock", "clocks": clocks,
              "association": ("host monotonic clock: ESP32 via the lower envelope of telemetry receipt "
                              f"({n_lines} lines, median lag {median_lag * 1e3:.1f} ms above it); cameras via "
                              f"frame receipt minus {camera_latency_s * 1e3:.0f} ms (CAMERA_LATENCY_S)"),
              "serial_offset_s": offset, "camera_latency_s": camera_latency_s,
              "expected_error_s": 0.004, "hardware_sync_validated": False,
              "led_config": read_json(run / "config.json").get("led"),
              "input_sha256": {name: sha256(run / name) for name in
                               ("raw/camera/frames.csv", "raw/camera/A/A.mp4", "raw/camera/B/B.mp4",
                                "raw/telemetry.jsonl")},
              "device_start": start | {"led_reference_device_s": reference_s,
                                       "drive_apply_interval_t_s": [start[k] / 1e6 - reference_s
                                                                    for k in ("drive0_us", "drive1_us")],
                                       "measurement_kind": "software_call_brackets_not_measured_current_onset"}}
    if stop is not None:
        result["device_stop"] = stop | {"stop_reference_device_t_s": (stop["t0_us"] + stop["t1_us"]) / 2e6 - reference_s}
    write_json(run / "processed/sync.json", result)
    return result


def synchronize_start(run, roi_a=None, roi_b=None):
    """Two optical markers per camera + one device event -> all three clock mappings."""
    run = Path(run)
    invalidate_report(run, "START association changed or failed; rerun sync-start then process.")
    sync_path = run / "processed/sync.json"
    sync_path.unlink(missing_ok=True)
    device_events = read_start(run / "raw/telemetry.jsonl")
    roi_source = "operator-given" if roi_a is not None and roi_b is not None else "auto-found (find_led)"
    if roi_a is None or roi_b is None:
        from LLM.record.takes import load_flight
        flight = load_flight(run)
        found = device_events["start"]
        roi_a, roi_b = (roi or find_led(flight[f"video_{tag}"], flight[f"times_{tag}"],
                                        found["host_receive_s"], found["on_us"] / 1e6, found["gap_us"] / 1e6)
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
            f"{roi_source} START LED ROIs; the firmware-declared pulse pair was decoded "
            "in both cameras and matched to the sole device START"
            + (" and STOP" if stop is not None else ""))
        write_json(sync_path, result)
    except BaseException:
        sync_path.unlink(missing_ok=True)
        raise
    return result
