#!/usr/bin/env python3
"""Automated PID-gain tuning driver for main_current_pid.cpp. Runs
coordinate-descent trials over {KP, KI, KD}: each trial shells out to
`tools/run_experiment.py --fw current_pid --auto-start --no-tui` with the
candidate gains, which resets the board, sends the gains once WAITING is
seen, sends start, lets the bounded run complete, and e-stops on the way
out (run_experiment.py owns all of that plumbing and its safety invariant
now -- this script no longer talks to the serial port directly). Gains
apply live in any phase now (see lib/ExperimentPhase's manual-start design),
so there's no more racing the old ~3s ARMING window before RAMP_UP locked
them out. Parses the resulting log via tools/pid_metrics.py and scores it.
Repeats a few rounds, one gain at a time, keeping whatever scored best.

Usage:
  uv run python tools/pid_autotune.py [--port /dev/ttyUSB0] [--rounds 2]
      [--run-seconds 45] [--out-dir tools/autotune_runs]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pid_metrics import compute_metrics, format_metrics_line, parse_log


def run_trial(port, kp, ki, kd, ramp, run_seconds, trial_dir) -> str:
    """Run one bounded trial via run_experiment.py, logging into trial_dir.
    Returns the serial log path to parse."""
    os.makedirs(trial_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(HERE, "run_experiment.py"),
           "--fw", "current_pid", "--skip-build", "--auto-start", "--no-tui",
           "--kp", str(kp), "--ki", str(ki), "--kd", str(kd), "--ramp", str(ramp),
           "--out-dir", trial_dir]
    if port:
        cmd += ["--port", port]
    result = subprocess.run(cmd, timeout=run_seconds + 30)
    if result.returncode != 0:
        print(f"  WARNING: run_experiment.py exited {result.returncode} -- "
              "trial may be incomplete, treat this trial as suspect")
    return os.path.join(trial_dir, "serial.log")


def score(metrics: dict) -> float:
    """Lower is better. Simple deterministic weighted sum -- no generic
    optimizer. A missing metric (nan, e.g. the run never reached HOLD) scores
    as a large fixed penalty so a broken trial never looks artificially good."""
    def val(x, penalty):
        return x if (x is not None and x == x) else penalty  # x==x is False for nan

    return (val(metrics["hold_spread"], 10.0)
            + 0.5 * val(metrics["resonance_peak"], 10.0)
            + 0.1 * val(metrics["settle_s"], 30.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None)
    ap.add_argument("--kp0", type=float, default=2.2)
    ap.add_argument("--ki0", type=float, default=0.10)
    ap.add_argument("--kd0", type=float, default=0.15)
    ap.add_argument("--ramp0", type=float, default=0.05,
                     help="seed for MIN_RAMP_PCT_PER_MS (default: %(default)s)")
    ap.add_argument("--gains-file", default=None,
                     help="tools/model_gains.json from tools/compute_model_gains.py -- "
                          "if given, its 'recommended' block overrides --kp0/--ki0/--kd0 "
                          "with model-informed seeds instead of the arbitrary defaults")
    ap.add_argument("--rounds", type=int, default=2,
                     help="coordinate-descent rounds over {KP,KI,KD} (default: %(default)s)")
    ap.add_argument("--run-seconds", type=float, default=45.0,
                     help="how long to let each trial run + log -- must exceed "
                          "ARM_MS + ramp_duration_ms + HOLD_MS + margin (default: %(default)s)")
    ap.add_argument("--out-dir",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_runs"))
    args = ap.parse_args()

    if args.gains_file:
        with open(args.gains_file) as f:
            rec = json.load(f)["recommended"]
        args.kp0, args.ki0, args.kd0 = rec["KP"], rec["KI"], rec["KD"]
        print(f"seeding from {args.gains_file}: kp0={args.kp0:.3f} "
              f"ki0={args.ki0:.4f} kd0={args.kd0:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.csv")

    gains = {"kp": args.kp0, "ki": args.ki0, "kd": args.kd0, "ramp": args.ramp0}
    state = {"trial_n": 0, "best_score": None}

    with open(results_path, "w") as rf:
        rf.write("trial,kp,ki,kd,ramp,hold_spread,ss_err,settle_s,resonance_peak,score\n")

        def try_gains(candidate: dict) -> float:
            state["trial_n"] += 1
            n = state["trial_n"]
            trial_dir = os.path.join(args.out_dir, f"trial_{n:03d}")
            print(f"[{n}] kp={candidate['kp']:.3f} ki={candidate['ki']:.3f} "
                  f"kd={candidate['kd']:.3f} ramp={candidate['ramp']:.4f}")
            log_path = run_trial(args.port, candidate["kp"], candidate["ki"],
                                  candidate["kd"], candidate["ramp"], args.run_seconds, trial_dir)
            data = parse_log(log_path)
            metrics = compute_metrics(data)
            s = score(metrics)
            is_best = state["best_score"] is None or s < state["best_score"]
            rf.write(f"{n},{candidate['kp']},{candidate['ki']},{candidate['kd']},"
                      f"{candidate['ramp']},{metrics['hold_spread']},"
                      f"{metrics['ss_err']},{metrics['settle_s']},"
                      f"{metrics['resonance_peak']},{s}\n")
            rf.flush()
            print(f"  {format_metrics_line(metrics)} -> score={s:.4f}"
                  f"{'  ** new best **' if is_best else ''}")
            return s

        state["best_score"] = try_gains(gains)
        print(f"baseline score={state['best_score']:.4f}")

        for _ in range(args.rounds):
            for key in ("kp", "ki", "kd", "ramp"):
                base = gains[key]
                for factor in (0.7, 1.0, 1.3):
                    candidate = dict(gains)
                    candidate[key] = base * factor
                    s = try_gains(candidate)
                    if s < state["best_score"]:
                        state["best_score"] = s
                        gains = candidate
                        print(f"  -> new best gains: {gains}")

    print(f"\nbest gains: {gains}  score={state['best_score']:.4f}")
    print(f"results: {results_path}")


if __name__ == "__main__":
    main()
