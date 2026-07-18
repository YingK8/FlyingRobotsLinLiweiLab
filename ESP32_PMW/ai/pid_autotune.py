#!/usr/bin/env python3
"""Automated PID-gain tuning driver for main_current_pid.cpp. Runs
coordinate-descent trials over {KP, KI, KD}: EN-pulse reset the board, set
gains via dir=/kp=/ki=/kd= over ai/serial_comm.py's SerialComm (must land
during the ~3s ARMING window before RAMP_UP locks gain changes out, see
main_current_pid.cpp's dispatchCommand; used directly rather than shelling
out to trigger_reset_log.py, since a trial needs several sequenced commands
plus draining telemetry for the run's duration), let the bounded run
complete, parse the log via ai/pid_metrics.py, and score it. Repeats a
few rounds, one gain at a time, keeping whatever scored best. SAFETY: every
trial's serial session sends 's' (e-stop) in a finally block on every exit
path, independently re-implementing trigger_reset_log.py's invariant since
this script doesn't call that one.

Usage:
  uv run python ai/pid_autotune.py [--port /dev/ttyUSB0] [--rounds 2]
      [--run-seconds 45] [--out-dir ai/autotune_runs]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serial_comm import SerialComm
from pid_metrics import compute_metrics, format_metrics_line, parse_log

# Must land within the ~3s ARMING window before RAMP_UP locks gains out --
# main_current_pid.cpp: Serial.begin()+delay(1000), then ARM_MS=3000ms.
BOOT_BANNER_TIMEOUT_S = 6.0
ARMED_BANNER_TIMEOUT_S = 6.0
POLL_S = 0.02


def run_trial(port, kp, ki, kd, ramp, run_seconds, out_path):
    """Reset the board, set gains, let one bounded run complete, log
    telemetry to out_path. Returns True if the ARMED banner was seen (gains
    believed applied in time), False otherwise (results still logged, but
    the caller should treat the trial as suspect)."""
    comm = SerialComm(port=port)
    gains_applied = False
    try:
        comm.reset_device()
        t0 = time.time()
        with open(out_path, "w") as f:
            def log(line):
                if line:
                    f.write(f"{time.time() - t0:7.2f}s  {line}\n")

            saw_boot = False
            deadline = time.time() + BOOT_BANNER_TIMEOUT_S
            while time.time() < deadline:
                line = comm.handle_serial_comm()
                if line is None:
                    time.sleep(POLL_S)
                    continue
                log(line)
                if "arming" in line.lower():
                    saw_boot = True
                    break
            if not saw_boot:
                print(f"  WARNING: no boot banner within {BOOT_BANNER_TIMEOUT_S}s "
                      "-- sending gains anyway, best-effort")

            for cmd in (f"kp={kp}", f"ki={ki}", f"kd={kd}", f"ramp={ramp}"):
                log(comm.handle_serial_comm(cmd))
                time.sleep(0.15)

            deadline = time.time() + ARMED_BANNER_TIMEOUT_S
            while time.time() < deadline:
                line = comm.handle_serial_comm()
                if line is None:
                    time.sleep(POLL_S)
                    continue
                log(line)
                if "armed" in line.lower():
                    gains_applied = True
                    break
            if not gains_applied:
                print("  WARNING: no ARMED banner seen -- gains may not have "
                      "landed before RAMP_UP; treat this trial as suspect")

            while time.time() - t0 < run_seconds:
                line = comm.handle_serial_comm()
                if line is None:
                    time.sleep(POLL_S)
                    continue
                log(line)
    finally:
        # SAFETY: always e-stop before exiting -- mirrors trigger_reset_log.py's
        # mandatory invariant (independently re-implemented here since this
        # script drives the serial port directly rather than calling that one).
        try:
            comm.handle_serial_comm("s")
            time.sleep(0.3)
        except Exception as e:
            print(f"  WARNING: failed to send e-stop ({e}) -- verify coils "
                  "are off by hand")
        comm.close()
    return gains_applied


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
                     help="ai/model_gains.json from ai/compute_model_gains.py -- "
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
        rf.write("trial,kp,ki,kd,ramp,gains_applied,hold_spread,ss_err,settle_s,resonance_peak,score\n")

        def try_gains(candidate: dict) -> float:
            state["trial_n"] += 1
            n = state["trial_n"]
            out_path = os.path.join(args.out_dir, f"trial_{n:03d}.log")
            print(f"[{n}] kp={candidate['kp']:.3f} ki={candidate['ki']:.3f} "
                  f"kd={candidate['kd']:.3f} ramp={candidate['ramp']:.4f}")
            applied = run_trial(args.port, candidate["kp"], candidate["ki"],
                                 candidate["kd"], candidate["ramp"], args.run_seconds, out_path)
            data = parse_log(out_path)
            metrics = compute_metrics(data)
            s = score(metrics)
            is_best = state["best_score"] is None or s < state["best_score"]
            rf.write(f"{n},{candidate['kp']},{candidate['ki']},{candidate['kd']},"
                      f"{candidate['ramp']},{applied},{metrics['hold_spread']},"
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
