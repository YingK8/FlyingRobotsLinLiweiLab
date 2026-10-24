"""Preserve raw serial lines; parse existing low-rate CS estimates without inventing phase."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import pandas as pd

from controller.control.link import SerialComm

_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


def _finite_number(token):
    if not _NUMBER.fullmatch(token):
        raise ValueError(f"malformed numeric token: {token!r}")
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite numeric token: {token!r}")
    return value


def _sample(line):
    stamps = re.findall(r"(?:^|\s)t=([^\s|]*)(?=[\s|]|$)", line)
    if len(stamps) != 1 or not re.fullmatch(r"[0-9]+", stamps[0]):
        raise ValueError("current sample requires one complete integer t= timestamp")
    try:
        device_s = int(stamps[0]) / 1000
    except (ValueError, OverflowError) as exc:
        raise ValueError("device timestamp is outside the supported numeric range") from exc
    if not math.isfinite(device_s):
        raise ValueError("nonfinite device timestamp")
    sections = re.findall(r"I\[A\]:\s*([^|]*)", line)
    if len(sections) != 1:
        raise ValueError("current sample requires exactly one I[A]: section")
    values = {}
    tokens = sections[0].split()
    if len(tokens) != 4:
        raise ValueError("current section must contain exactly four channel tokens")
    for token in tokens:
        match = re.fullmatch(r"([A-D])=(.+)", token)
        if match is None or match[1] in values:
            raise ValueError("current labels must be unique A/B/C/D tokens")
        values[match[1]] = _finite_number(match[2])
    if set(values) != set("ABCD"):
        raise ValueError("current labels must cover A/B/C/D")
    frequencies = re.findall(r"(?:^|\s)freq=([^\s|]*)(?=[\s|]|$)", line)
    if len(frequencies) > 1:
        raise ValueError("current sample contains multiple frequency fields")
    frequency = _finite_number(frequencies[0]) if frequencies else None
    return device_s, frequency, values


def record_telemetry(path, port, duration_s):
    """Passive receiver for a port not already owned by the controller. No reset/send.

    An active controller must tee its existing serial stream instead: a second
    serial reader can steal lines. Opening a USB adapter still needs a bench check.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        link = SerialComm(port=port)
        try:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                line = link.handle_serial_comm()
                if line is not None:
                    f.write(json.dumps({"host_receive_s": time.monotonic(), "line": line}) + "\n")
                    f.flush()
                else:
                    time.sleep(.002)
        finally:
            link.close()


def parse_log(path):
    """Parse complete samples without altering raw evidence.

    Malformed lines are skipped with ``DataFrame.attrs['parse_errors']`` entries
    containing their 1-based line number and reason. Decreasing host time and
    duplicate/decreasing device sample time abort because one clock mapping can
    no longer be trusted. Ordinary non-telemetry lines are retained in the raw
    file but do not count as parse errors.
    """
    rows, errors = [], []
    previous_device_s, previous_host_s = -1, -math.inf
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        try:
            entry = json.loads(raw)
            if not isinstance(entry, dict) or not isinstance(entry.get("line"), str):
                raise ValueError("record requires a string line field")
            if isinstance(entry.get("host_receive_s"), bool):
                raise ValueError("receive timestamp must be numeric, not boolean")
            host = float(entry["host_receive_s"])
            if not math.isfinite(host):
                raise ValueError("nonfinite receive timestamp")
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            errors.append({"line": line_no, "reason": str(exc)})
            continue
        if host < previous_host_s:
            raise ValueError(f"host receive time decreased at line {line_no}; split recordings before alignment")
        previous_host_s = host
        line = entry["line"]
        if "I[A]" not in line:
            continue
        try:
            device_s, frequency, values = _sample(line)
        except ValueError as exc:
            errors.append({"line": line_no, "reason": str(exc)})
            continue
        if device_s <= previous_device_s:
            raise ValueError(f"device time reset/duplicate/wrap at line {line_no}: split recordings before alignment")
        previous_device_s = device_s
        row = {"t_device_s": device_s, "host_receive_s": host,
               "frequency_reported_hz": frequency,
               "measurement_kind": "filtered_current_sense_estimate",
               "raw_line_number": line_no}
        row.update({f"current_{c}_A": value for c, value in values.items()})
        rows.append(row)
    columns = ["t_device_s", "host_receive_s", "frequency_reported_hz", "measurement_kind", "raw_line_number"] + [f"current_{c}_A" for c in "ABCD"]
    result = pd.DataFrame(rows, columns=columns)
    result.attrs["parse_errors"] = errors
    return result


if __name__ == "__main__":
    from LLM.tests.test_telemetry import run
    run()
