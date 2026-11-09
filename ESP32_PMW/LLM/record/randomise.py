"""Per-run randomised coil duty, and the run's full parameter record.

Each run draws one duty per channel, independently, so that run-to-run conditions the
operator does not control (coil temperature, rotor seating, lighting, battery of the day)
are not confounded with the drive: the duty is assigned by a recorded random draw, never
chosen. Duty is `duty=A:B:C:D` (firmware `cmdDuty`): a per-channel multiplier, in percent,
on the carrier duty the ramp commands (`collective * duty/100`). 100 is the unrandomised
drive.

Distribution: d = 100 - X with X ~ Exponential(mean `scale_pct`), redrawn until
d >= `floor_pct`. Most mass sits near full drive, with a tail toward weak channels.
scale 15 / floor 30 are the 2026-11-06 defaults, chosen for the request, NOT tuned:
P(d < 85) = 37 %, P(d < 70) = 13.5 %, P(redraw) = e^(-70/15) = 0.9 %.

`run_parameters.json` is written BEFORE START, into raw/, and is never overwritten: it is
the record of what was planned and sent. What the device applied is its telemetry
`duty[%]` (processed/current.csv `duty_applied_*_pct`); what flowed is the CS current.
Keep the three separate: duty is not current.

    .venv/bin/python -m LLM.record.randomise      # self-check
"""
import hashlib
import json
import platform
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CHANNELS = "ABCD"            # firmware channels 0..3, src/constants.h A/B/C/D blocks
DEFAULT = {"scale_pct": 15.0, "floor_pct": 30.0}


def draw_duty(rng, scale_pct=DEFAULT["scale_pct"], floor_pct=DEFAULT["floor_pct"]):
    """Four independent duties in percent, rounded to 0.1 (the value sent is the value kept)."""
    if not (scale_pct > 0 and 0 <= floor_pct < 100):
        raise ValueError("need scale_pct > 0 and 0 <= floor_pct < 100")
    out = []
    while len(out) < len(CHANNELS):
        d = round(100.0 - float(rng.exponential(scale_pct)), 1)
        if d >= floor_pct:
            out.append(d)
    return out


def duty_command(duty):
    return "duty=" + ":".join(f"{d:.1f}" for d in duty)


def _git():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=root.parent,
                               capture_output=True, text=True, timeout=5).stdout
        return {"commit": commit or None, "dirty_files": len(dirty.splitlines())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty_files": None}


def write_parameters(run, *, duty="random", dist=None, seed=None, ramp=None, camera=None,
                     calibration=None, world=None, leds=None, note="", design=None, switch_hz=None):
    """Plan the run's duty and write raw/run_parameters.json; returns the duty command
    (None when duty is None). duty: "random", four explicit percentages, or None.
    design: a `record/design.plan` (per-parameter mode, range, step, drawn value, seed),
    recorded whole; the duty values then come from it. switch_hz: 100 % below this
    planned frequency, the duty above it (Bench.run)."""
    from LLM.record.io import sha256
    path = Path(run) / "raw/run_parameters.json"
    if path.exists():
        raise ValueError(f"{path} exists: one parameter record per run, never overwritten")
    dist = {**DEFAULT, **(dist or {})}
    if duty == "random":
        seed = secrets.randbits(63) if seed is None else int(seed)
        values = draw_duty(np.random.default_rng(seed), dist["scale_pct"], dist["floor_pct"])
        mode = "random"
    elif duty is None:
        values, mode, seed = None, "none", None
    else:
        values, seed = [round(float(d), 1) for d in duty], None
        mode = "design" if design else "fixed"
        if len(values) != 4 or not all(0 <= d <= 100 for d in values):
            raise ValueError("duty needs four percentages in 0..100")
    command = None if values is None else duty_command(values)
    ramp_path = Path(run) / "ramp.json"
    record = {
        "schema_version": 1,
        "written": "before START; planned and host-sent values, not device-applied or measured",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        # host time.monotonic() is what frames.csv / telemetry stamp; this pair maps it to UTC
        "time_anchor": {"unix_s": time.time(), "monotonic_s": time.monotonic(),
                        "meaning": "unix = monotonic + (unix_s - monotonic_s); same host, same boot"},
        "duty": {"mode": mode, "channels": list(CHANNELS),
                 "percent": None if values is None else dict(zip(CHANNELS, values)),
                 "command": command, "seed": seed,
                 "generator": "numpy.random.default_rng(seed) (PCG64); exponential, rejection below floor",
                 "distribution": ({"form": "100 - Exponential(mean=scale_pct), redrawn below floor_pct",
                                   **dist} if mode == "random" else
                                  "per channel: see design.parameters.duty_*_pct" if mode == "design" else None),
                 "meaning": "per-channel multiplier on the ramp's carrier duty (collective*duty/100); "
                            + ("sent once in IDLE before seq=go; reset clears it" if switch_hz is None or command is None
                               else f"duty=100 on all coils sent in IDLE before seq=go; this duty sent by the host "
                                    f"when the PLANNED frequency reaches {switch_hz:g} Hz "
                                    "(capture meta host_duty_switch_send_complete_s); reset clears it"),
                 "switch_at_planned_hz": None if command is None else switch_hz,
                 "device_applied": "telemetry duty[%] -> processed/current.csv duty_applied_*_pct"},
        "ramp": ({"file": "ramp.json", "sha256": sha256(ramp_path),
                  "segments": json.loads(ramp_path.read_text())["segments"], "name": ramp}
                 if ramp_path.exists() else {"name": ramp}),
        "camera": camera,
        "calibration": ({"path": str(Path(calibration).resolve()),
                         "sha256": sha256(calibration) if Path(calibration).exists() else None,
                         "exists": Path(calibration).exists()} if calibration else None),
        "world_frame_sha256": None if world is None else
        hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest(),
        "design": design,
        "led_rois_xywh": leds,
        "note_at_start": note,
        "software": {"git": _git(), "python": sys.version.split()[0], "numpy": np.__version__,
                     "platform": platform.platform(), "host": platform.node()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return command


def _self_check():
    import tempfile
    rng = np.random.default_rng(0)
    d = np.array([draw_duty(rng) for _ in range(20000)]).ravel()
    assert d.min() >= 30 and d.max() <= 100
    drop = 100 - d
    # truncated exponential: the mean drop sits just under the scale, the median at ln2*scale
    assert 13.5 < drop.mean() < 15.0, drop.mean()
    assert abs(np.median(drop) - 15 * np.log(2)) < 0.4, np.median(drop)
    assert abs((d < 70).mean() - (np.exp(-2) - np.exp(-70 / 15)) / (1 - np.exp(-70 / 15))) < 0.01
    assert draw_duty(np.random.default_rng(7)) == draw_duty(np.random.default_rng(7))
    assert duty_command([100, 92.5, 71.0, 30.04]) == "duty=100.0:92.5:71.0:30.0"
    with tempfile.TemporaryDirectory() as tmp:
        cmd = write_parameters(tmp, seed=42, ramp="x", note="n")
        rec = json.loads((Path(tmp) / "raw/run_parameters.json").read_text())
        assert cmd == rec["duty"]["command"] and rec["duty"]["seed"] == 42
        assert list(rec["duty"]["percent"].values()) == draw_duty(np.random.default_rng(42))
        try:
            write_parameters(tmp)
            raise AssertionError("an existing parameter record must not be overwritten")
        except ValueError:
            pass
    with tempfile.TemporaryDirectory() as tmp:
        assert write_parameters(tmp, duty=None) is None
    print("randomise: truncated-exponential duty draws, seed reproducibility, write-once record ok")


if __name__ == "__main__":
    _self_check()
