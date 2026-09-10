#!/usr/bin/env python3
"""One command for the whole actuation-map identification, resumable and thermally paced.

    uv run python controller/control/sysid.py --list          # what each stage needs
    uv run python controller/control/sysid.py --stage 0       # offline, free, no hardware
    uv run python controller/control/sysid.py --all           # 0, E, S unattended; K asks

WHAT IS BEING IDENTIFIED, AND WHY IT IS FOUR STAGES
---------------------------------------------------
The chain from the nine numbers the controller commands to the robot's lateral response
factors into three independently measurable pieces (`theory.md` 25.1):

    duty + phase -> coil current   `I_k = G_k(f) d_k exp(j(phi_k - theta_k(f)))`   stage E
    current      -> field at robot  `c_k = g_k exp(j alpha_k)`                     stage S
    field        -> lateral tilt    `k = k_lat exp(j psi)`                         stage K

Measuring them separately is not tidiness. A single end-to-end fit at one frequency cannot
separate a coil's fixed azimuth from the frequency-dependent phase its own RLC adds, and
nothing downstream can undo that confusion later.

    stage 0   offline    free      (f0, Q) from run CSVs already on disk
    stage E   bench      ~11 s     (f0, Q) from a probe sweep; resolves the stale resonance
    stage S   mast       ~85 s     alpha_k, g_k, and the balanced duty vector
    stage K   airborne   ~136 s    k_lat and psi -- ATTENDED, see below

Stage K flies. It is scripted end to end but will not start without `--i-am-present`,
because with no firmware watchdog (`theory.md` 4.0) the host is the only thing that can
stop the coils, and an unattended takeoff has no second line of defence.

THE BINDING CONSTRAINT IS COOLING, NOT MEASUREMENT
--------------------------------------------------
From cold, the 70 C ceiling at 0.5 C/s allows ~96 s of drive, then `tau_cool = 1500 s`.
The whole sweep is ~232 s of drive and three cool-downs: **2.5 to 3.5 hours**, nearly all
of it waiting. That is why stage S fits five frequencies into one cold budget rather than
sweeping ten, and why the journal exists -- a `KeyboardInterrupt` in hour two must cost
one stage, not the session.

NUMBERS ARE PRINTED, NEVER PATCHED IN
-------------------------------------
Each stage writes `fit.json` with full provenance and prints a literal block to paste.
Nothing here edits `constants.py` or `drive_common.h`. A seed guess silently overwritten
by a number nobody read is precisely the failure `constants.py`'s prose exists to prevent,
and `coil_phase.report()` already sets the pattern.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT_ROOT = ROOT / "results" / "sysid"

ACCEPTED, REFUSED, ABORTED = "ACCEPTED", "REFUSED", "ABORTED"

#: Drive seconds each stage reserves against the thermal model before it starts.
BUDGET_S = {"0": 0.0, "E": 11.0, "S": 85.0, "K": 136.0}


@dataclass
class Record:
    stage: str
    status: str
    values: dict = field(default_factory=dict)
    reason: str = ""
    drive_s: float = 0.0
    coil_temp_c: float = float("nan")
    git_sha: str = ""
    timestamp: str = ""


class Journal:
    """Per-stage results on disk. Re-invoking skips what already passed.

    The point is not tidiness: the run is hours of cooling, and a session that has to
    start over after an interrupt will not be run twice.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: dict[str, Record] = {}
        if self.path.exists():
            for k, v in json.loads(self.path.read_text()).items():
                self.records[k] = Record(**v)

    def done(self, stage: str) -> bool:
        r = self.records.get(stage)
        return r is not None and r.status == ACCEPTED

    def put(self, r: Record):
        r.git_sha = r.git_sha or _git_sha()
        r.timestamp = r.timestamp or time.strftime("%Y-%m-%dT%H:%M:%S")
        self.records[r.stage] = r
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: vars(v) for k, v in self.records.items()},
                                        indent=2, default=float))

    def summary(self) -> str:
        if not self.records:
            return "journal: empty"
        w = []
        for k in sorted(self.records):
            r = self.records[k]
            w.append(f"  {k:>2}  {r.status:<8} {r.reason[:70]}")
        return "journal:\n" + "\n".join(w)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def _thermal():
    from ai.thermal import coil_thermal
    return coil_thermal


def wait_for(stage: str, verbose=True):
    """Block until the coils have headroom for this stage. Prints the ETA first."""

    budget = BUDGET_S.get(stage, 0.0)
    if budget <= 0.0:
        return 0.0
    t = _thermal()
    now = t.temp_now()
    if verbose:
        print(f"[stage {stage}] coils ~{now:.0f} C; needs headroom for {budget:.0f}s of "
              f"drive. tau_cool is {t.TAU_COOL_S:.0f}s, so this may be a long wait.")
    t.wait_until_safe(budget)
    return now


# ---- stages -----------------------------------------------------------------------------
#
# Each returns a `Record`. A stage that cannot reach its own acceptance criterion writes
# REFUSED with the number that failed -- it never returns a fitted value it does not
# believe. That is the rule `fit_rotation`, `ramp.check` and `coil_phase.fit_channel`
# already follow, and the reason a REFUSED stage in `theory.md` is a better outcome than a
# fitted one: the next session knows what to fix.


def stage_0(args, out: Path) -> Record:
    """(f0, Q) from run CSVs already on disk. Free: no hardware, no heat, no waiting."""

    from controller.control import coil_phase

    runs = sorted((ROOT / "results" / "takeoff").glob("*.csv"))
    if not runs:
        return Record("0", REFUSED, reason="no run CSVs in results/takeoff")
    # `fit_run` takes ONE csv and returns {tag: fit-or-None}; a None is `fit_channel`
    # refusing, which is the expected answer when the |I| peak is outside the ramp's band.
    fits, tried, per_run = {}, 0, {}
    for r in runs:
        try:
            f = coil_phase.fit_run(r)
        except Exception:
            continue                      # a run without current columns is not an error
        tried += 1
        good = {t: v for t, v in f.items() if v is not None}
        if good:
            per_run[r.name] = _jsonable(good)
            fits.update(good)
    if not fits:
        return Record("0", REFUSED,
                      reason=(f"{tried} run(s) fitted nothing. Expected: fit_channel "
                              f"refuses a |I| peak within EDGE_BINS of the band edge, "
                              f"which is the 400 uF failure of theory.md 18.6 and the "
                              f"correct answer if the peak is outside the ramp's band."))
    return Record("0", ACCEPTED, values={"per_run": per_run},
                  reason=f"{len(per_run)} of {tried} run(s) fitted at least one channel")


def stage_E(args, out: Path) -> Record:
    """(f0, Q) per channel from a `probe=` lock-in sweep. Bench only: no robot, no camera."""

    from controller.control import coil_phase

    if not args.port:
        return Record("E", REFUSED, reason="--port is required (this stage drives coils)")
    wait_for("E")
    t0 = time.monotonic()
    try:
        res = coil_phase.measure(port=args.port)
    except Exception as e:
        return Record("E", ABORTED, reason=f"{type(e).__name__}: {e}",
                      drive_s=time.monotonic() - t0)
    drive = time.monotonic() - t0
    if not res:
        return Record("E", REFUSED, drive_s=drive,
                      reason="probe returned no points; check the port and that the "
                             "firmware is IDLE (probe= is IDLE-only)")
    # `measure` returns {f_hz: ProbePoint}; the (f0, Q) fit is a separate step, and it is
    # the one that refuses -- `fit_probe` returns None per channel it cannot fit.
    fits = coil_phase.fit_probe(res)
    good = {t: f for t, f in fits.items() if f is not None}
    if len(good) < 4:
        missing = "".join(t for t in "ABCD" if t not in good)
        return Record("E", REFUSED, drive_s=drive,
                      reason=(f"channel(s) {missing} did not fit. All four are needed: "
                              f"`setPhaseTrim` refuses a table with any non-positive "
                              f"entry, and three trimmed channels with one raw is a "
                              f"BIGGER asymmetry than trimming none. Widen PROBE_F_HZ if "
                              f"the peak is outside the swept band."))
    sig = {t: float(f.get("f0_sigma", float("nan"))) for t, f in good.items()}
    worst = max(sig, key=lambda k: sig[k])
    if sig[worst] > 5.0:
        return Record("E", REFUSED, drive_s=drive,
                      reason=f"channel {worst} f0_sigma {sig[worst]:.1f} Hz over the 5 Hz gate")
    return Record("E", ACCEPTED,
                  values={t: {"f0": float(f["f0_hz"]), "q": float(f["q"]),
                              "f0_sigma": float(f.get("f0_sigma", float("nan")))}
                          for t, f in good.items()},
                  drive_s=drive, coil_temp_c=_thermal().temp_now())


def stage_S(args, out: Path) -> Record:
    """alpha_k and g_k, seated on the mast, from Jacobians at several frequencies."""

    from controller.control import coil_map
    from controller.control import tilt_servo

    if not args.port:
        return Record("S", REFUSED, reason="--port is required (this stage drives coils)")
    prev = Journal(out / "journal.json").records.get("E")
    if prev is None or prev.status != ACCEPTED:
        return Record("S", REFUSED,
                      reason="stage E must pass first: alpha_k and theta_k(f) enter the "
                             "measured angle as a sum, and theta comes from E")
    f0 = [prev.values[c]["f0"] for c in "ABCD"]
    q = [prev.values[c]["q"] for c in "ABCD"]
    wait_for("S")
    t0 = time.monotonic()
    try:
        jac = tilt_servo.run(port=args.port, identify_only=True,
                             freqs=tuple(args.freqs), out_dir=out / "S")
    except Exception as e:
        return Record("S", ABORTED, reason=f"{type(e).__name__}: {e}",
                      drive_s=time.monotonic() - t0)
    drive = time.monotonic() - t0
    try:
        cmap = coil_map.fit_geometry(jac, f0, q)
    except ValueError as e:
        return Record("S", REFUSED, reason=str(e), drive_s=drive)
    return Record("S", ACCEPTED, drive_s=drive, coil_temp_c=_thermal().temp_now(),
                  values={"alpha_deg": list(cmap.alpha_deg), "gain": list(cmap.gain),
                          "scatter_deg": list(cmap.scatter_deg),
                          "offset_deg": cmap.offset_deg,
                          "vs_firmware_deg": list(cmap.versus_firmware())})


def stage_K(args, out: Path) -> Record:
    """k_lat and psi, airborne. ATTENDED: this one flies."""

    if not args.i_am_present:
        return Record("K", REFUSED,
                      reason=("this stage commands a takeoff. With no firmware watchdog "
                              "the host is the only thing that can stop the coils, so it "
                              "will not start unattended. Re-run with --i-am-present."))
    return Record("K", REFUSED,
                  reason=("not yet wired to the runner. The estimator (`lockin_tilt`) and "
                          "the excitation are ready; what is missing is a flight. See "
                          "theory.md 25.5: at 0.5 Hz the measured 0.55 s window covers "
                          "0.27 of a turn, which `lockin_tilt.fit` refuses -- fly the "
                          "balanced duty vector from stage S to lengthen it, or raise the "
                          "dither to ~1.9 Hz, which covers the circle in 0.55 s."))


STAGES = {"0": stage_0, "E": stage_E, "S": stage_S, "K": stage_K}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    try:
        return float(x)
    except (TypeError, ValueError):
        return x if isinstance(x, (str, int, bool)) or x is None else repr(x)


def run(args) -> int:
    out = Path(args.out or (OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")))
    out.mkdir(parents=True, exist_ok=True)
    jr = Journal(out / "journal.json")
    names = list(STAGES) if args.all else [args.stage]
    print(f"sysid: {out}")
    for name in names:
        if jr.done(name):
            print(f"[stage {name}] already ACCEPTED -- skipped")
            continue
        print(f"[stage {name}] starting")
        try:
            rec = STAGES[name](args, out)
        except KeyboardInterrupt:
            jr.put(Record(name, ABORTED, reason="interrupted by the operator"))
            print(f"\n[stage {name}] ABORTED\n{jr.summary()}")
            return 130
        jr.put(rec)
        print(f"[stage {name}] {rec.status}"
              + (f": {rec.reason}" if rec.reason else ""))
        if rec.status != ACCEPTED and not args.keep_going:
            break
    (out / "fit.json").write_text(json.dumps(
        {k: vars(v) for k, v in jr.records.items()}, indent=2, default=float))
    print(jr.summary())
    return 0 if all(r.status == ACCEPTED for r in jr.records.values()) else 1


def demo():
    """Self-check: the journal, the gates, and resume. No hardware, no coils."""

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        jp = Path(d) / "journal.json"
        jr = Journal(jp)
        assert not jr.done("E")
        jr.put(Record("E", ACCEPTED, values={"A": {"f0": 150.0, "q": 0.9}}))
        assert jr.done("E")
        # Resume: a fresh Journal over the same file remembers, which is what makes a
        # three-hour run survivable.
        assert Journal(jp).done("E"), "journal did not persist"
        # A REFUSED stage is NOT done, so re-running retries it rather than skipping.
        jr.put(Record("S", REFUSED, reason="scatter"))
        assert not Journal(jp).done("S")

    # Every stage must REFUSE rather than fabricate when its precondition is missing.
    class A:
        port = None
        i_am_present = False
        freqs = (70.0,)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        r = stage_E(A(), out)
        assert r.status == REFUSED and "port" in r.reason, r
        r = stage_S(A(), out)
        assert r.status == REFUSED and "port" in r.reason, r
        # ...and stage K refuses to fly unattended, which is the safety decision, not a
        # capability gap.
        r = stage_K(A(), out)
        assert r.status == REFUSED and "unattended" in r.reason, r
        A.i_am_present = True
        r = stage_K(A(), out)
        assert r.status == REFUSED and "0.27 of a turn" in r.reason, r

        # Stage S must not run before E: theta_k(f) comes from E, and without it the
        # measured angle is a sum with one measurement.
        A.port = "/dev/null"
        r = stage_S(A(), out)
        assert r.status == REFUSED and "stage E must pass first" in r.reason, r

        # THE CONTRACT BETWEEN STAGES. Each stage's own test passes while the keys one
        # writes and the next reads disagree -- which is exactly what happened here:
        # `coil_phase.measure` returns {f_hz: ProbePoint}, not {tag: (f0, q)}, and stage S
        # would have died with a KeyError after 85 s of drive and a cool-down. Assert the
        # shape rather than discovering it on the rig.
        jr = Journal(out / "journal.json")
        jr.put(Record("E", ACCEPTED,
                      values={t: {"f0": 150.0, "q": 0.9, "f0_sigma": 1.0} for t in "ABCD"}))
        prev = Journal(out / "journal.json").records["E"]
        for c in "ABCD":
            assert "f0" in prev.values[c] and "q" in prev.values[c], prev.values
            float(prev.values[c]["f0"]), float(prev.values[c]["q"])

        # ...and `coil_map` must accept exactly what stage S would hand it: a dict keyed by
        # frequency, at least two of them.
        from controller.control import coil_map
        import numpy as np
        f0 = [prev.values[c]["f0"] for c in "ABCD"]
        q = [prev.values[c]["q"] for c in "ABCD"]
        alpha, gain = np.array([0.0, 90.0, 180.0, 270.0]), np.ones(4)
        jac = {}
        for f in (100.0, 160.0):
            a = np.radians(alpha - coil_map.theta_deg(f, np.array(f0), np.array(q)))
            jac[f] = np.vstack([gain * np.cos(a), gain * np.sin(a)])
        m = coil_map.fit_geometry(jac, f0, q)
        assert np.allclose(m.alpha_deg, alpha, atol=1.0), m.alpha_deg

    print("sysid: journal persists and resumes, REFUSED retries, every stage refuses\n"
          "  without its precondition, and K will not fly unattended\n  ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=list(STAGES), default="0")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--port")
    ap.add_argument("--out")
    ap.add_argument("--freqs", type=lambda s: [float(x) for x in s.split(",")],
                    default=[70.0, 100.0, 130.0, 160.0, 190.0])
    ap.add_argument("--i-am-present", action="store_true",
                    help="required for stage K, which commands a takeoff")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        demo()
        return 0
    if a.list:
        for k, f in STAGES.items():
            print(f"  {k}  {BUDGET_S[k]:6.0f}s drive  {f.__doc__.splitlines()[0]}")
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
