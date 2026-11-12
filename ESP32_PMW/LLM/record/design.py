"""Run design: what each run's takeoff parameters are, and how they move between runs.

Each parameter has a mode:

    fixed   the value, every run
    random  drawn per run: uniform in [lo, hi]; duty is 100 - Exponential(scale), redrawn
            below lo (record/randomise.py) so most runs sit near full drive
    step    the value this run; after a KEPT run (Continue) it moves by `step` (+ or -).
            A discarded run repeats its value. Stepping past [lo, hi] is refused, not
            clamped: a clamp would silently repeat the end value as if it were new.

The ramp is a cubic from 1 Hz to `f_top` at a given MEAN rate (Hz/s), so its duration is
(f_top - 1) / rate -- the same ramp velocity whatever the top (operator, 2026-11-06) --
then a hold at f_top. Durations go out as whole milliseconds; `ramp.check` holds the
55 s cap. The design and every drawn value are recorded in raw/run_parameters.json.

    .venv/bin/python -m LLM.record.design      # self-check
"""
import json
import secrets
from pathlib import Path

import numpy as np

from LLM.record.randomise import CHANNELS, DEFAULT as DUTY_DEFAULT, draw_duty

MODES = ("fixed", "random", "step")
F_START_HZ = 1.0
# value, lo, hi, step, mode. Defaults: the 2026-11-06 cubic_1to230Hz_23.6s_hold10s ramp
# (mean 9.7 Hz/s is cubic_1to195Hz_20s's) and the per-coil random duty. Bounds are
# operator-editable guards, not measured limits: f_top above ~195 Hz is unflown here.
PARAMS = {
    "f_top_hz":    {"label": "Top frequency (Hz)", "value": 230.0, "lo": 100.0, "hi": 260.0, "step": 5.0, "mode": "fixed"},
    "rate_hz_s":   {"label": "Mean ramp rate (Hz/s)", "value": 9.7, "lo": 4.0, "hi": 20.0, "step": 0.5, "mode": "fixed"},
    "hold_s":      {"label": "Hold at top (s)", "value": 10.0, "lo": 0.0, "hi": 20.0, "step": 1.0, "mode": "fixed"},
    **{f"duty_{c}_pct": {"label": f"Duty coil {c} (%)", "value": 100.0, "lo": DUTY_DEFAULT["floor_pct"],
                         "hi": 100.0, "step": 5.0, "mode": "random"} for c in CHANNELS},
    # All coils 100 % on the way up, the duty above only from here (operator 2026-11-06:
    # capture and spin-up at full drive). Planned host-time frequency, not measured
    # takeoff; capped at f_top by Bench.run. 200 Hz is the operator's figure.
    # 0 (or anything at/below the ramp's 1 Hz start): the duty from START, no 100 % phase.
    "duty_switch_hz": {"label": "Duty switch: 100 % below (Hz; 0 = from START)", "value": 200.0, "lo": 0.0, "hi": 260.0,
                       "step": 5.0, "mode": "fixed"},
}


def plan(specs, *, seed=None, duty_scale_pct=DUTY_DEFAULT["scale_pct"]):
    """specs: {name: {mode, value, lo, hi, step}} -> the run's values and their record."""
    seed = secrets.randbits(63) if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    out = {"seed": seed, "generator": "numpy.random.default_rng(seed) (PCG64)", "duty_scale_pct": duty_scale_pct,
           "parameters": {}}
    for name in PARAMS:                      # fixed order: same seed -> same draws
        s = specs[name]
        if s["mode"] not in MODES:
            raise ValueError(f"{name}: mode {s['mode']!r} not in {MODES}")
        if not s["lo"] <= s["hi"]:
            raise ValueError(f"{name}: min {s['lo']} > max {s['hi']}")
        if s["mode"] == "random":
            if name.startswith("duty_"):
                value = draw_duty(rng, duty_scale_pct, s["lo"])[0]   # one channel's draw
                rule = f"100 - Exponential(mean {duty_scale_pct}), redrawn below {s['lo']}"
            else:
                value = round(float(rng.uniform(s["lo"], s["hi"])), 3)
                rule = f"uniform [{s['lo']}, {s['hi']}]"
        else:
            value = float(s["value"])
            if not s["lo"] <= value <= s["hi"]:
                raise ValueError(f"{name}: {value} is outside [{s['lo']}, {s['hi']}]")
            rule = "fixed" if s["mode"] == "fixed" else f"step {s['step']:+g} after each kept run"
        out["parameters"][name] = {**{k: s[k] for k in ("mode", "lo", "hi", "step")}, "value": value, "rule": rule}
    return out


def segments(f_top_hz, rate_hz_s, hold_s):
    """The ramp as ramp-file segments; durations in whole milliseconds."""
    if f_top_hz <= F_START_HZ or rate_hz_s <= 0 or hold_s < 0:
        raise ValueError("need f_top > 1 Hz, rate > 0, hold >= 0")
    ramp_s = round((f_top_hz - F_START_HZ) / rate_hz_s, 3)
    segs = [{"from_hz": F_START_HZ, "to_hz": f_top_hz, "seconds": ramp_s, "shape": "poly", "k": 3.0}]
    if hold_s > 0:
        segs.append({"from_hz": f_top_hz, "to_hz": f_top_hz, "seconds": round(hold_s, 3), "shape": "linear"})
    return segs


def write_ramp(path, design):
    """The run's ramp file from a plan; validated by the same loader the run uses."""
    from LLM.record.ramps import load_ramp
    p = {k: v["value"] for k, v in design["parameters"].items()}
    segs = segments(p["f_top_hz"], p["rate_hz_s"], p["hold_s"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"note": f"Generated by LLM/record/design.py, seed {design['seed']}: cubic 1 -> "
                                        f"{p['f_top_hz']:g} Hz at mean {p['rate_hz_s']:g} Hz/s, hold {p['hold_s']:g} s.",
                                "segments": segs}, indent=2))
    load_ramp(path)                         # raises (e.g. over the 55 s cap) before anything drives
    return path


def duty_values(design):
    return [design["parameters"][f"duty_{c}_pct"]["value"] for c in CHANNELS]


def advance(specs):
    """After a kept run: step-mode values move by their step. Returns (specs, messages);
    a step that would leave [lo, hi] is not taken and is reported."""
    new, notes = {k: dict(v) for k, v in specs.items()}, []
    for name, s in new.items():
        if s["mode"] != "step":
            continue
        nxt = round(s["value"] + s["step"], 6)
        if s["lo"] <= nxt <= s["hi"]:
            s["value"] = nxt
        else:
            notes.append(f"{PARAMS[name]['label']}: next {nxt:g} is outside [{s['lo']:g}, {s['hi']:g}]; held at {s['value']:g}")
    return new, notes


def _self_check():
    import tempfile
    specs = {k: dict(v) for k, v in PARAMS.items()}
    a, b = plan(specs, seed=3), plan(specs, seed=3)
    assert a == b, "same seed, same design"
    assert a["parameters"]["f_top_hz"]["value"] == 230.0
    assert a["parameters"]["duty_switch_hz"]["value"] == 200.0
    duties = duty_values(a)
    assert all(30 <= d <= 100 for d in duties) and len(set(duties)) > 1, duties
    # uniform random stays in range; fixed outside its range is refused
    specs["hold_s"].update(mode="random", lo=2.0, hi=4.0)
    assert all(2 <= plan(specs, seed=s)["parameters"]["hold_s"]["value"] <= 4 for s in range(50))
    bad = {k: dict(v) for k, v in PARAMS.items()}
    bad["f_top_hz"]["value"] = 300.0
    try:
        plan(bad)
        raise AssertionError("a fixed value outside [lo, hi] must be refused")
    except ValueError:
        pass
    # step moves after a kept run, and refuses to leave the range
    st = {k: dict(v) for k, v in PARAMS.items()}
    st["f_top_hz"].update(mode="step", value=250.0, step=5.0)
    st, notes = advance(st)
    assert st["f_top_hz"]["value"] == 255.0 and not notes
    st, _ = advance(st)
    st, notes = advance(st)
    assert st["f_top_hz"]["value"] == 260.0 and notes, (st["f_top_hz"], notes)
    # same ramp velocity: 230 Hz at 9.7 Hz/s is 23.608 s, then the hold
    s = segments(230.0, 9.7, 10.0)
    assert abs(s[0]["seconds"] - 229 / 9.7) < 1e-3 and s[1]["seconds"] == 10.0
    assert len(segments(195.0, 9.7, 0.0)) == 1
    with tempfile.TemporaryDirectory() as tmp:
        write_ramp(Path(tmp) / "r.json", plan({k: dict(v) for k, v in PARAMS.items()}, seed=1))
        long = {k: dict(v) for k, v in PARAMS.items()}
        long["rate_hz_s"]["value"], long["hold_s"]["value"] = 4.0, 20.0     # 57 s + 20 s
        try:
            write_ramp(Path(tmp) / "x.json", plan(long, seed=1))
            raise AssertionError("over the 55 s cap must be refused")
        except ValueError:
            pass
    print("design: fixed/random/step plans, seed reproducibility, ramp velocity, step bounds, 55 s cap ok")


if __name__ == "__main__":
    _self_check()
