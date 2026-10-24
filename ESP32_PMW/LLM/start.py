"""Associate one recorded flight START with optical edges, never host-send time."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from LLM.io import invalidate_report, sha256, write_json
from LLM.pipeline import add_clock, synchronize
from LLM.telemetry import parse_log

_START = re.compile(
    r"SYNC_START v=1 led0=([0-9]+) led1=([0-9]+) drive0=([0-9]+) drive1=([0-9]+)")


def read_start(path):
    """Require exactly one ready boot and one complete event in the raw RX log.

    Firmware timestamps use esp_timer_get_time() microseconds. Existing t= current
    telemetry uses millis(), the same ESP timer divided by 1000 (before wrap).
    Resets/re-arms, malformed events and time reversal require a separate run.
    """
    ready, event = False, None
    previous_host = -math.inf
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            row = json.loads(raw)
            line = row["line"]
            host = row["host_receive_s"]
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
            if not ready or event is not None or match is None:
                raise ValueError("missing READY, duplicate START, or malformed START")
            values = [int(value) for value in match.groups()]
            if values != sorted(values) or values[-1] >= (2**32) * 1000:
                raise ValueError("invalid START order or unsupported millis() wrap")
            event = dict(zip(("led0_us", "led1_us", "drive0_us", "drive1_us"), values))
            event.update(raw_line_number=line_no, host_receive_s=host)
    if event is None:
        raise ValueError("no device SYNC_START v=1 in raw log; do not infer it from seq=go or host time")
    # Validates device resets even when the log contains no second READY banner.
    current = parse_log(path)
    if current.attrs["parse_errors"]:
        raise ValueError("malformed telemetry; inspect the raw log before START association")
    for row in current.itertuples():
        if ((row.raw_line_number < event["raw_line_number"]
             and row.t_device_s > event["led0_us"] / 1e6)
                or (row.raw_line_number > event["raw_line_number"]
                    and row.t_device_s + .001 < event["drive1_us"] / 1e6)):
            raise ValueError("telemetry clock disagrees with START; possible reset or mixed recordings")
    return event


def synchronize_start(run, roi_a, roi_b):
    """One optical rise per camera + one device event -> all three clock mappings."""
    run = Path(run)
    invalidate_report(run, "START association changed or failed; rerun sync-start then process.")
    sync_path = run / "processed/sync.json"
    sync_path.unlink(missing_ok=True)
    event = read_start(run / "raw/telemetry.jsonl")
    reference_s = (event["led0_us"] + event["led1_us"]) / 2e6
    result = synchronize(run, roi_a, roi_b, [0.0])
    try:
        clock = add_clock(run, "esp", [reference_s], [0.0],
                          f"raw/telemetry.jsonl line {event['raw_line_number']}: LED GPIO call midpoint; optical/electrical delay unvalidated")
        result["clocks"]["esp"] = clock
        result["input_sha256"]["raw/telemetry.jsonl"] = sha256(run / "raw/telemetry.jsonl")
        result["device_start"] = event | {
            "led_reference_device_s": reference_s,
            "drive_apply_interval_t_s": [event[key] / 1e6 - reference_s
                                         for key in ("drive0_us", "drive1_us")],
            "measurement_kind": "software_call_brackets_not_measured_current_onset",
            "drift_measured": False,
        }
        result["association"] = "operator selected START LED ROIs; one optical rise per camera matched to the sole device START"
        write_json(sync_path, result)
    except BaseException:
        sync_path.unlink(missing_ok=True)
        raise
    return result
