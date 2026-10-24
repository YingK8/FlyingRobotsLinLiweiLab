"""Offline takeoff inference from valid, gravity-aligned stereo centre height.

Thresholds below are starting guesses, not bench-validated robot parameters.
The event is a sustained centre-height threshold crossing, not a measurement
of physical bearing clearance. Keep the original recording for review.
"""
from __future__ import annotations

import numpy as np

from .world import _valid_flags

DEFAULT_CONFIG = {
    "baseline_start_s": 0.0,
    "baseline_end_s": 0.2,
    "min_height_mm": 1.0,
    "noise_sigma_factor": 5.0,
    "min_upward_speed_mm_s": 1.0,
    "sustain_s": 0.05,
    "max_gap_s": 0.05,
    "min_baseline_samples": 5,
    "end_s": None,
}


def _config(config):
    supplied = dict(config or {})
    unknown = set(supplied) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown takeoff configuration: {sorted(unknown)}")
    cfg = {**DEFAULT_CONFIG, **supplied}
    for key, value in cfg.items():
        if key == "end_s" and value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be finite and numeric") from exc
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(number):
            raise ValueError(f"{key} must be finite and numeric")
        cfg[key] = number
    if cfg["baseline_end_s"] <= cfg["baseline_start_s"]:
        raise ValueError("baseline_end_s must follow baseline_start_s")
    for key in ("min_height_mm", "sustain_s", "max_gap_s"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("noise_sigma_factor", "min_upward_speed_mm_s"):
        if cfg[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    count = cfg["min_baseline_samples"]
    if count < 2 or count != int(count):
        raise ValueError("min_baseline_samples must be an integer >= 2")
    cfg["min_baseline_samples"] = int(count)
    if cfg["end_s"] is not None and cfg["end_s"] <= cfg["baseline_end_s"]:
        raise ValueError("end_s must follow the baseline interval")
    return cfg


def detect_takeoff(df, config=None):
    """Return JSON-safe evidence for a detected, absent, or uncertain event.

    ``onset_s`` is the first observed above-threshold sample; its interval with
    the preceding sample is retained. It is not backdated to physical release.
    Invalid observations and large time gaps reset a candidate. Clock times
    must be finite and strictly increasing; this function never sorts them.
    """
    cfg = _config(config)
    missing = {"t_s", "z_mm", "position_valid"} - set(df.columns)
    if missing:
        raise ValueError(f"takeoff is missing required columns: {sorted(missing)}")
    t = df.t_s.to_numpy(dtype=float)
    z = df.z_mm.to_numpy(dtype=float)
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("t_s must be finite and strictly increasing")
    valid = _valid_flags(df.position_valid, "position_valid") & np.isfinite(z)
    result = {
        "schema_version": 1, "status": "uncertain", "onset_s": None,
        "confirmed_s": None, "onset_interval_s": None,
        "method": "sustained_center_height_proxy", "config": cfg,
        "physical_bearing_clearance_measured": False,
        "reason": "insufficient_baseline", "baseline": None, "threshold": None,
    }
    baseline_mask = (t >= cfg["baseline_start_s"]) & (t <= cfg["baseline_end_s"]) & valid
    baseline = z[baseline_mask]
    if len(baseline) < cfg["min_baseline_samples"]:
        result["baseline"] = {"n_valid": int(len(baseline))}
        return result
    center = float(np.median(baseline))
    sigma = float(1.4826 * np.median(np.abs(baseline - center)))
    height = max(cfg["min_height_mm"], cfg["noise_sigma_factor"] * sigma)
    threshold = center + height
    result["baseline"] = {"n_valid": int(len(baseline)), "median_z_mm": center,
                          "sigma_mad_mm": sigma, "assumption": "robot seated throughout interval"}
    result["threshold"] = {"rise_mm": float(height), "z_mm": float(threshold)}
    selected = np.flatnonzero((t > cfg["baseline_end_s"])
                              & ((t <= cfg["end_s"]) if cfg["end_s"] is not None else True))
    if len(selected) == 0:
        result["reason"] = "no_observations_after_baseline"
        return result
    previous = int(np.flatnonzero(t <= cfg["baseline_end_s"])[-1])
    previous = previous if valid[previous] else None
    candidate = None
    uncertain = False
    gap_count = 0
    for index in selected:
        continuous = previous is not None and t[index] - t[previous] <= cfg["max_gap_s"] + 1e-12
        if not valid[index] or not continuous:
            uncertain = True
            gap_count += 1
            candidate = None
        if not valid[index]:
            previous = None
            continue
        above = z[index] > threshold
        if not above:
            candidate = None
        elif candidate is None and continuous and z[previous] <= threshold:
            speed = (z[index] - z[previous]) / (t[index] - t[previous])
            if speed >= cfg["min_upward_speed_mm_s"]:
                candidate = (int(index), int(previous), float(speed))
        elif candidate is None:
            uncertain = True
        if candidate is not None and t[index] - t[candidate[0]] + 1e-12 >= cfg["sustain_s"]:
            start, before, entry_speed = candidate
            window = np.arange(before, index + 1)
            slope = float(np.polyfit(t[window] - t[before], z[window], 1)[0])
            if slope >= cfg["min_upward_speed_mm_s"]:
                result.update(status="detected", reason="sustained_height_and_upward_trend",
                              onset_s=float(t[start]), confirmed_s=float(t[index]),
                              onset_interval_s=[float(t[before]), float(t[start])],
                              entry_speed_mm_s=entry_speed, trend_speed_mm_s=slope,
                              earlier_observation_gaps=bool(gap_count), gap_count=gap_count)
                return result
        previous = int(index)
    if candidate is not None:
        uncertain = True
    if cfg["end_s"] is not None and cfg["end_s"] - t[selected[-1]] > cfg["max_gap_s"]:
        uncertain = True
    result.update(status="uncertain" if uncertain else "no_takeoff", gap_count=gap_count,
                  reason="missing_data_or_unconfirmed_height_rise" if uncertain else "no_sustained_height_rise")
    return result


def _self_check():
    import pandas as pd
    t = np.arange(101) / 100
    frame = pd.DataFrame({"t_s": t, "z_mm": np.maximum(t - 0.4, 0) * 20,
                          "position_valid": True})
    result = detect_takeoff(frame)
    assert result["status"] == "detected"
    assert result["confirmed_s"] >= result["onset_s"] + 0.05 - 1e-12
    print("takeoff self-check passed")


if __name__ == "__main__":
    _self_check()
