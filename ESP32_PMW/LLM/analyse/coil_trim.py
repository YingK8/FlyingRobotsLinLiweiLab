"""Frequency-scheduled per-coil duty trim, measured from runs flown at 100 % duty.

The four coils do not resonate at the same frequency, so their currents part ways with
frequency, and not by a constant ratio: 2026-11-15 (12 runs, 1 -> 195..250 Hz at 100 %),
A is the STRONGEST coil at 80-100 Hz (+6-11 % of the mean) and the WEAKEST from ~140 Hz
(-19..-29 %), with B and C ~15 % high there. A fixed trim cannot fix that; one per
frequency band can. `controller/control/theory.md` 22.9.

Duty can only go DOWN from 100 %, so in each band the weakest coil stays at 100 % and
the others are trimmed to match it: balance is bought with field strength, and
``field_fraction`` in the schedule says how much (weakest / mean current). This trims
AMPLITUDE only. Each coil's current phase also moves with its own resonance
(`phase-claim-is-stale-and-analog`: theta = atan(Q(f/f0 - f0/f))), and nothing here
corrects that.

    .venv/bin/python -m LLM.analyse.coil_trim                 # from every 100 % run in LLM/runs
    .venv/bin/python -m LLM.analyse.coil_trim RUN [RUN ...]
    .venv/bin/python -m LLM.analyse.coil_trim --self-check

Writes LLM/trims/coil_trim_<stamp>.json; never touches a run folder.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
TRIMS = PROJECT / "LLM/trims"
COILS = "ABCD"
BIN_HZ = 20.0          # one trim per 20 Hz band: the ratios move ~0.05-0.1 per band above 60 Hz
F_FROM_HZ = 60.0       # below this the CS spread is noise (IQR 0.19-0.35 at <40 Hz): no trim
MIN_CURRENT_A = 0.15   # a sample with any coil under this is CS noise, not a ratio (a guess)
MIN_RUNS = 3           # a band measured in fewer runs is not trimmed: it holds the band below
FLOOR_PCT = 50.0       # never trim a coil below this; a guess -- deeper means a broken coil, not a trim


def samples(runs):
    """Rows (run, f, current A..D) from runs' processed/current.csv where every coil was
    commanded 100 % and every current is above MIN_CURRENT_A."""
    out = []
    for run in map(Path, runs):
        f = run / "processed/current.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        cur = [f"current_{c}_A" for c in COILS]
        duty = [f"duty_applied_{c}_pct" for c in COILS]
        d = d[(d.frequency_reported_hz > 0) & (d[duty].min(axis=1) >= 99.5) & (d[cur].min(axis=1) > MIN_CURRENT_A)]
        if len(d):
            out.append(pd.DataFrame({"run": run.name, "f": d.frequency_reported_hz.values,
                                     **{c: d[f"current_{c}_A"].values for c in COILS}}))
    if not out:
        raise ValueError("no samples at 100 % duty in the given runs")
    return pd.concat(out, ignore_index=True)


def schedule(s):
    """Per BIN_HZ band from F_FROM_HZ: median of each coil's current over the four-coil mean,
    then trim = 100 * weakest / own. Returns the knots [{hz, duty_pct[4], field_fraction,
    n, runs}]; a band seen in < MIN_RUNS runs repeats the band below."""
    rel = s[list(COILS)].div(s[list(COILS)].mean(axis=1), axis=0)
    rel["run"], rel["band"] = s.run, (s.f // BIN_HZ * BIN_HZ)
    knots, last = [], None
    for band, g in rel[rel.band >= F_FROM_HZ].groupby("band"):
        if g.run.nunique() < MIN_RUNS:
            if last is not None:
                knots.append(dict(last, hz=float(band), n=int(len(g)), runs=int(g.run.nunique()), held=True))
            continue
        r = g[list(COILS)].median().values
        duty = 100.0 * r.min() / r
        if duty.min() < FLOOR_PCT:
            raise ValueError(f"{band:g} Hz band: a coil needs {duty.min():.0f} % (< {FLOOR_PCT:g}); check the coil, "
                             "do not trim it")
        last = dict(hz=float(band), duty_pct=[round(float(v), 1) for v in duty],
                    field_fraction=round(float(r.min()), 3), n=int(len(g)), runs=int(g.run.nunique()), held=False)
        knots.append(last)
    return knots


def build(runs, out_dir=TRIMS):
    s = samples(runs)
    knots = schedule(s)
    out = dict(kind="coil_trim", created=datetime.now().isoformat(timespec="seconds"),
               method="weakest coil 100 %, others * weakest/own; median ratio to the 4-coil mean per band",
               bin_hz=BIN_HZ, f_from_hz=F_FROM_HZ, below_first_knot="100 % all coils",
               runs=sorted(s.run.unique().tolist()), n_samples=len(s), knots=knots,
               caveats=["amplitude only: per-coil current PHASE is not corrected",
                        "measured on the CS estimate at ~2 Hz telemetry during ramps",
                        "duty is assumed proportional to coil current (linear RLC)"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"coil_trim_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2))
    return path, out


def commands(trim, f_top_hz):
    """[(hz, 'duty=a:b:c:d')] for the knots below the ramp's top, for run_panel."""
    from LLM.record.randomise import duty_command
    return [(k["hz"], duty_command(k["duty_pct"])) for k in trim["knots"] if k["hz"] < f_top_hz]


def runs_at_full_duty(root=PROJECT / "LLM/runs"):
    return sorted(p for p in Path(root).glob("*_2026*") if (p / "processed/current.csv").exists())


def _self_check():
    import tempfile
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as tmp:
        runs = []
        for k in range(4):
            run = Path(tmp) / f"sync_check_2026111{k}_000000_x"
            (run / "processed").mkdir(parents=True)
            f = np.linspace(1, 250, 60)
            gain = np.where(f[:, None] < 120, [1.1, 1.0, 1.0, 0.9], [0.75, 1.15, 1.15, 1.0])
            cur = 2.0 * gain * (1 + rng.normal(0, 0.02, (len(f), 4)))
            pd.DataFrame({"frequency_reported_hz": f, **{f"current_{c}_A": cur[:, i] for i, c in enumerate(COILS)},
                          **{f"duty_applied_{c}_pct": 100.0 for c in COILS}}).to_csv(run / "processed/current.csv")
            runs.append(run)
        path, out = build(runs, Path(tmp) / "trims")
        hi = next(k for k in out["knots"] if k["hz"] == 200.0)
        assert hi["duty_pct"][0] == 100.0 and abs(hi["duty_pct"][1] - 100 * 0.75 / 1.15) < 4, hi
        lo = next(k for k in out["knots"] if k["hz"] == 80.0)
        assert lo["duty_pct"][3] == 100.0 and lo["duty_pct"][0] < 85, lo
        assert all(k["hz"] >= F_FROM_HZ for k in out["knots"])
        cmds = commands(out, 210.0)
        assert cmds[-1][0] == 200.0 and cmds[0][1].startswith("duty=")
        assert json.loads(path.read_text())["knots"] == out["knots"]
    print("coil_trim: weakest coil kept at 100 %, others trimmed per band; commands below the top only")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        _self_check()
    else:
        path, out = build(sys.argv[1:] or runs_at_full_duty())
        print(f"{path}  ({len(out['runs'])} runs, {out['n_samples']} samples)")
        print(" hz    A      B      C      D    field  runs")
        for k in out["knots"]:
            print(f"{k['hz']:4.0f} " + " ".join(f"{d:6.1f}" for d in k["duty_pct"])
                  + f"   {k['field_fraction']:.2f}  {k['runs']}" + ("  (held)" if k["held"] else ""))
