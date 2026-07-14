#!/usr/bin/env python3
"""Offline schema/sanity check for a RatioCurrentController JSON profile
(src/PiProfile.h's schema, task_sequences/pi_profile_*.json) -- BEFORE it
ever gets staged onto the device. Mirrors this project's established
"validate offline before hardware" pattern (tools/validate_duty_allocator.py,
tools/validate_freq_gain_model.py). Pure Python, no ArduinoJson/native-build
involved -- see test/test_ratio_controller/'s own scope note for why the
control LAW is tested natively in C++ while the JSON schema is validated
here instead.

Usage:
  uv run python tools/validate_pi_profile.py task_sequences/pi_profile_tilt.json
"""
from __future__ import annotations

import argparse
import json
import sys

CHANNELS = ("A", "B", "C", "D")
VALID_MODES = ("shared_constraint", "independent")

# name -> (required, sane range) for scalar fields with defaults in
# src/PiProfile.cpp -- kept in sync with that file's fallback values.
SCALAR_FIELDS = {
    "ramp_pct_per_ms": (False, (0.0, 10.0)),
    "duty_min": (False, (0.0, 100.0)),
    "duty_max": (False, (0.0, 100.0)),
    "i_max_a": (False, (0.0, 50.0)),
    "overcurrent_backoff_pct": (False, (0.0, 100.0)),
    "min_switch_margin_a": (False, (0.0, 10.0)),
    "nominal_tick_ms": (False, (0.01, 1000.0)),
    "magnitude_settle_tol_a": (False, (0.0, 10.0)),
}
GAIN_FIELDS = {
    "kp": (0.0, 100.0),
    "ki": (0.0, 100.0),
    "kd": (0.0, 100.0),
}


def validate(profile: dict, path: str) -> list[str]:
    """Returns a list of error strings (empty if valid)."""
    errors = []

    mode = profile.get("mode")
    if mode not in VALID_MODES:
        errors.append(f'"mode"={mode!r} must be one of {VALID_MODES}')

    ratios = profile.get("ratios")
    if not isinstance(ratios, dict):
        errors.append('"ratios" must be an object')
    else:
        for ch in CHANNELS:
            if ch not in ratios:
                errors.append(f'"ratios" missing channel "{ch}"')
                continue
            v = ratios[ch]
            if not isinstance(v, (int, float)) or v <= 0:
                errors.append(f'"ratios.{ch}"={v!r} must be a positive number')
        extra = set(ratios) - set(CHANNELS)
        if extra:
            errors.append(f'"ratios" has unrecognized channel(s) {sorted(extra)} '
                           f"(expected only {CHANNELS})")

    gains = profile.get("gains", {})
    if not isinstance(gains, dict):
        errors.append('"gains" must be an object if present')
    else:
        for name, (lo, hi) in GAIN_FIELDS.items():
            if name not in gains:
                continue  # optional, PiProfile.cpp defaults it
            v = gains[name]
            if not isinstance(v, (int, float)) or not (lo <= v <= hi):
                errors.append(f'"gains.{name}"={v!r} must be in [{lo}, {hi}]')

    for name, (required, (lo, hi)) in SCALAR_FIELDS.items():
        if name not in profile:
            if required:
                errors.append(f'missing required field "{name}"')
            continue
        v = profile[name]
        if not isinstance(v, (int, float)) or not (lo <= v <= hi):
            errors.append(f'"{name}"={v!r} must be in [{lo}, {hi}]')

    duty_min = profile.get("duty_min", 5.0)
    duty_max = profile.get("duty_max", 100.0)
    if isinstance(duty_min, (int, float)) and isinstance(duty_max, (int, float)):
        if duty_min >= duty_max:
            errors.append(f'"duty_min"={duty_min} must be < "duty_max"={duty_max}')

    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profiles", nargs="+", help="one or more pi_profile*.json files")
    args = ap.parse_args()

    any_failed = False
    for path in args.profiles:
        with open(path) as f:
            profile = json.load(f)
        errors = validate(profile, path)
        if errors:
            any_failed = True
            print(f"[FAIL] {path}")
            for e in errors:
                print(f"    - {e}")
        else:
            name = profile.get("name", "(unnamed)")
            mode = profile["mode"]
            print(f"[PASS] {path} (name={name!r}, mode={mode})")

    if any_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
