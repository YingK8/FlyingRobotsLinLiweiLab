#!/usr/bin/env python3
"""Shared telemetry parsing + tuning-metric computations for
main_current_pid.cpp serial logs, used by both tools/plot_pid_log.py
(plotting + report) and tools/pid_autotune.py (machine-parsable METRICS line,
scored across trials) so there's one parser/metrics implementation, not two.
Telemetry line format (~2 Hz): "t=12345 phase=3 freq=142.1 | I[A]: A=4.98
B=4.91 C=5.03 D=5.10 | duty[%]: A=55.0 B=52.0 C=54.0 D=50.0 | spread=0.190
dir=1 kp=2.20 ki=0.10 kd=0.15"; the trailing dir=/kp=/ki=/kd= fields are
optional (older logs predate runtime-tunable gains), parse_log() still works
without them.
"""
from __future__ import annotations

import re

import numpy as np

STATES = {0: "ARMING", 1: "RAMP_UP", 2: "HOLD", 3: "ENDING", 4: "STOPPED"}
HOLD_PHASE = 2
RAMP_UP_PHASE = 1
RESONANCE_BAND_HZ = (90.0, 190.0)  # L/R-corner resonance region, see project memory

_CORE = re.compile(
    r"phase=(\d+)\s+freq=([\d.]+)\s+\|\s+"
    r"I\[A\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)\s+\|\s+"
    r"duty\[%\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)\s+\|\s+"
    r"spread=([\d.]+)"
)
_TRAILER = re.compile(
    r"dir=(\d+)\s+kp=([\-\d.]+)\s+ki=([\-\d.]+)\s+kd=([\-\d.]+)"
    r"(?:\s+ramp=([\-\d.]+))?"  # optional -- older logs predate ramp=
)
_TIME_PREFIX = re.compile(r"\s*([\d.]+)s\s")  # trigger_reset_log.py's timestamp prefix

_EXP_CORE = re.compile(
    r"t=(\d+)\s+label=(\S+)\s+\|\s+"
    r"I\[A\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)\s+\|\s+"
    r"duty\[%\]:\s+A=([\-\d.]+)\s+B=([\-\d.]+)\s+C=([\-\d.]+)\s+D=([\-\d.]+)"
)

_METRICS_LINE = re.compile(
    r"METRICS\s+hold_spread=([\w.\-]+)\s+ss_err=([\w.\-]+)\s+"
    r"settle_s=([\w.\-]+)\s+resonance_peak=([\w.\-]+)"
)


def parse_log(path: str) -> dict:
    """Parse a main_current_pid.cpp serial log into a dict of numpy arrays
    (t, state, freq, i_a..i_d, d_a..d_d, spread, and dir/kp/ki/kd if every
    parsed line carried them). Raises SystemExit if no telemetry found."""
    rows = []
    t_rel = []
    trailer_rows = []
    for line in open(path):
        m = _CORE.search(line)
        if not m:
            continue
        tm = _TIME_PREFIX.match(line)
        t_rel.append(float(tm[1]) if tm else len(rows))
        rows.append(tuple(m.groups()))
        tm2 = _TRAILER.search(line)
        trailer_rows.append(tuple(tm2.groups()) if tm2 else None)

    if not rows:
        raise SystemExit(
            f"no main_current_pid.cpp telemetry lines found in {path} "
            "(expected 'phase=.. freq=.. | I[A]: ... | duty[%]: ... | spread=..')"
        )

    data = {
        "t": np.array(t_rel),
        "state": np.array([int(r[0]) for r in rows]),
        "freq": np.array([float(r[1]) for r in rows]),
        "i_a": np.array([float(r[2]) for r in rows]),
        "i_b": np.array([float(r[3]) for r in rows]),
        "i_c": np.array([float(r[4]) for r in rows]),
        "i_d": np.array([float(r[5]) for r in rows]),
        "d_a": np.array([float(r[6]) for r in rows]),
        "d_b": np.array([float(r[7]) for r in rows]),
        "d_c": np.array([float(r[8]) for r in rows]),
        "d_d": np.array([float(r[9]) for r in rows]),
        "spread": np.array([float(r[10]) for r in rows]),
    }
    if all(trailer_rows):
        data["dir"] = np.array([int(r[0]) for r in trailer_rows])
        data["kp"] = np.array([float(r[1]) for r in trailer_rows])
        data["ki"] = np.array([float(r[2]) for r in trailer_rows])
        data["kd"] = np.array([float(r[3]) for r in trailer_rows])
        if all(r[4] is not None for r in trailer_rows):
            data["ramp"] = np.array([float(r[4]) for r in trailer_rows])
    return data


def parse_experiment_log(path: str) -> dict:
    """Parse a main_experiment.cpp serial log ("t=.. label=.. | I[A]: ... |
    duty[%]: ...", both the on-label-change print and the periodic print) into
    a dict of numpy arrays (t, label [object array of str], i_a..i_d, d_a..d_d).
    Raises SystemExit if no telemetry found."""
    rows = []
    for line in open(path):
        m = _EXP_CORE.search(line)
        if not m:
            continue
        rows.append(m.groups())

    if not rows:
        raise SystemExit(
            f"no main_experiment.cpp telemetry lines found in {path} "
            "(expected 't=.. label=.. | I[A]: ... | duty[%]: ...')"
        )

    return {
        "t": np.array([int(r[0]) for r in rows]),
        "label": np.array([r[1] for r in rows], dtype=object),
        "i_a": np.array([float(r[2]) for r in rows]),
        "i_b": np.array([float(r[3]) for r in rows]),
        "i_c": np.array([float(r[4]) for r in rows]),
        "i_d": np.array([float(r[5]) for r in rows]),
        "d_a": np.array([float(r[6]) for r in rows]),
        "d_b": np.array([float(r[7]) for r in rows]),
        "d_c": np.array([float(r[8]) for r in rows]),
        "d_d": np.array([float(r[9]) for r in rows]),
    }


def _first_settle_time(hold_t, below, settle_hold_s: float):
    """First hold_t[i] (relative) where `below` stays True for a full
    settle_hold_s window starting at i, confirmed against real data (not
    just "ran out of samples") -- returns hold_t[i]-hold_t[0], or None."""
    n = len(hold_t)
    for i in range(n):
        if not below[i]:
            continue
        window_end = hold_t[i] + settle_hold_s
        end_idx = int(np.searchsorted(hold_t, window_end, side="right")) - 1
        if end_idx >= n or hold_t[end_idx] < window_end:
            continue  # data doesn't reach far enough to confirm this window
        if below[i:end_idx + 1].all():
            return float(hold_t[i] - hold_t[0])
    return None


def compute_metrics(data: dict, spread_limit: float = 0.1,
                     settle_hold_s: float = 2.0) -> dict:
    """Tuning metrics from parsed telemetry; any that can't be computed (e.g.
    no HOLD samples) is None.
    hold_spread     max spread during the full HOLD phase
    ss_err          |mean(i_meas) - i_min_target| over the last 10s of HOLD;
                    i_min_target is the settled-window mean of the argmin
                    channel, derived from the log, not hardcoded
    settle_s        first HOLD-relative time spread stays continuously below
                    spread_limit for >= settle_hold_s, confirmed against data
    resonance_peak  max spread during RAMP_UP in the 90-190Hz L/R-corner
                    band, the historically hardest failure mode a HOLD-only
                    metric would miss
    resonance_peak_freq_hz  the drive frequency at which resonance_peak
                    occurred -- used by validate_resonance_model.py to check
                    the fitted RLC model's predicted resonance location
                    against where the real controller actually struggled
    """
    t, state, freq, spread = data["t"], data["state"], data["freq"], data["spread"]
    currents = np.stack([data["i_a"], data["i_b"], data["i_c"], data["i_d"]], axis=1)

    metrics = {"hold_spread": None, "ss_err": None, "settle_s": None,
               "resonance_peak": None, "resonance_peak_freq_hz": None}

    hold_mask = state == HOLD_PHASE
    if hold_mask.any():
        metrics["hold_spread"] = float(spread[hold_mask].max())

        hold_t = t[hold_mask]
        hold_spread = spread[hold_mask]
        settled_mask = hold_mask & (t >= hold_t[-1] - 10.0)
        if settled_mask.any():
            settled_currents = currents[settled_mask]
            per_channel_mean = settled_currents.mean(axis=0)
            i_min_target = float(per_channel_mean.min())
            overall_mean = float(settled_currents.mean())
            metrics["ss_err"] = abs(overall_mean - i_min_target)

        below = hold_spread < spread_limit
        metrics["settle_s"] = _first_settle_time(hold_t, below, settle_hold_s)

    lo, hi = RESONANCE_BAND_HZ
    ramp_mask = (state == RAMP_UP_PHASE) & (freq >= lo) & (freq <= hi)
    if ramp_mask.any():
        ramp_spread = spread[ramp_mask]
        metrics["resonance_peak"] = float(ramp_spread.max())
        metrics["resonance_peak_freq_hz"] = float(freq[ramp_mask][np.argmax(ramp_spread)])

    return metrics


def format_metrics_line(metrics: dict) -> str:
    """Machine-parsable one-line summary, e.g.:
      METRICS hold_spread=0.090 ss_err=0.042 settle_s=12.3 resonance_peak=0.310
    Missing metrics print as 'nan' (parse_metrics_line round-trips them)."""
    def fmt(v, prec):
        return f"{v:.{prec}f}" if v is not None else "nan"

    return (f"METRICS hold_spread={fmt(metrics['hold_spread'], 3)} "
            f"ss_err={fmt(metrics['ss_err'], 3)} "
            f"settle_s={fmt(metrics['settle_s'], 1)} "
            f"resonance_peak={fmt(metrics['resonance_peak'], 3)}")


def parse_metrics_line(text: str) -> dict | None:
    """Inverse of format_metrics_line -- parses a METRICS line (or any text
    containing one) back into a dict of floats (nan for missing), or None
    if no METRICS line is present."""
    m = _METRICS_LINE.search(text)
    if not m:
        return None
    return {
        "hold_spread": float(m.group(1)),
        "ss_err": float(m.group(2)),
        "settle_s": float(m.group(3)),
        "resonance_peak": float(m.group(4)),
    }
