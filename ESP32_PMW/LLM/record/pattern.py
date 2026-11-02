"""Planned frequency/duty/phase patterns. This module never sends serial data."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from controller.control import ramp

CHANNELS = "ABCD"


def example_pattern():
    return {"schema_version": 1, "name": "example_only_rise_then_fall",
            "trigger": "time", "frequency_segments": [[2, 180, 2, 1, 2], [180, 210, 0.5, 0, 1], [210, 170, 0.5, 0, 1]],
            "channel_updates": [{"t_s": 0, "duty_pct": [20, 20, 20, 20], "phase_deg": [0, 90, 180, 270]}],
            "note": "Schema/demo values only; not a characterized flight recipe. Time-driven, not takeoff-triggered."}


def validate_pattern(pattern):
    if pattern.get("schema_version") != 1 or pattern.get("trigger") != "time":
        raise ValueError("v1 supports timed patterns only; online takeoff-triggered execution is not implemented")
    segments = pattern["frequency_segments"]
    if not segments:
        raise ValueError("frequency_segments is empty")
    for seg in segments:
        if len(seg) != 5 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in seg):
            raise ValueError("each frequency segment must have five finite numeric values")
        f0, f1, seconds, mode, k = seg
        if min(f0, f1) <= 0 or seconds < .001 or abs(seconds * 1000 - round(seconds * 1000)) > 1e-6:
            raise ValueError("frequency must be positive and durations must be whole milliseconds")
        if mode == ramp.POLY and k <= 0 or mode == ramp.EASE and k < 1 or abs(k) > 50:
            raise ValueError("unsupported shape: POLY k>0, EASE k>=1, all |k|<=50")
    ramp.check(segments)
    updates = pattern.get("channel_updates", [])
    if not updates and pattern.get("channel_settings_status") == "firmware_defaults_unrecorded":
        return pattern
    if not updates or updates[0].get("t_s") != 0:
        raise ValueError("channel_updates must define all four duties/phases at t_s=0")
    last = -1.0
    for update in updates:
        t = update["t_s"]
        if not math.isfinite(t) or not last < t <= ramp.duration_s(segments):
            raise ValueError("channel update times must increase and be within the pattern")
        for key in ("duty_pct", "phase_deg"):
            vals = update[key]
            if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
                raise ValueError(f"{key} requires four finite values in A/B/C/D order")
        if not all(0 <= v <= 100 for v in update["duty_pct"]):
            raise ValueError("duty_pct must be in [0,100]")
        last = t
    return pattern


def sample_pattern(pattern, sample_hz=100):
    validate_pattern(pattern)
    if not math.isfinite(sample_hz) or not 0 < sample_hz <= 10000:
        raise ValueError("sample_hz must be in (0,10000]")
    segs, updates = pattern["frequency_segments"], pattern["channel_updates"]
    end = ramp.duration_s(segs)
    times = np.unique(np.r_[np.arange(0, end, 1/sample_hz), end,
                            np.cumsum([s[2] for s in segs]), [u["t_s"] for u in updates]])
    rows, update_index = [], 0
    for t in times:
        while update_index + 1 < len(updates) and updates[update_index + 1]["t_s"] <= t:
            update_index += 1
        u = updates[update_index] if updates else {"duty_pct": [np.nan]*4, "phase_deg": [np.nan]*4}
        row = {"t_s": float(t), "frequency_hz": ramp.frequency_at(segs, float(t)), "provenance": "planned"}
        for i, ch in enumerate(CHANNELS):
            row[f"duty_{ch}_pct"] = u["duty_pct"][i]
            row[f"phase_{ch}_deg"] = u["phase_deg"][i]
        rows.append(row)
    return pd.DataFrame(rows)
