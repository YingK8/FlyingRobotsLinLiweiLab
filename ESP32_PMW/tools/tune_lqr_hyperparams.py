#!/usr/bin/env python3
"""Hardware-in-the-loop tuner for design_lqr_gains.py's (--q-spread, --r-duty)
cost weights, run independently per direction (CW/CCW) via bounded coordinate
descent with backtracking -- see the state-space plan
("PID-style derivative control + hardware-in-the-loop LQR gain tuner", Part B)
for the full design rationale.

Why coordinate descent + backtracking instead of full central-difference
gradient descent: each hardware trial costs ~40-50s (rebuild+reflash+full
autonomous run -- ARM_MS=3000 + ramp_duration_ms=20000 + HOLD_MS=5000, fixed
compile-time firmware constants, not shortened here). A full 2-param central-
difference gradient would cost 4+ trials/iteration before even stepping;
coordinate descent with backtracking is far cheaper and handles the observed
failure shape (CW flat, CCW has a sharp cliff at q_spread=100) more
gracefully -- one bad probe just triggers a shrink+retreat, not a corrupted
joint gradient.

Why q_spread/r_duty only (not q_track/q_integral/q_derivative): q_spread is
the main "equalize channels" knob and the one with an already-observed sharp,
direction-asymmetric failure mode. r_duty is design_lqr_gains.py's own
suggested fix for an edge-of-stable/saturating q_spread. q_track/q_integral
weren't implicated in the observed failure. q_derivative is deliberately
EXCLUDED from this automated search -- it's a new, uncharacterized failure
mode (noise-driven duty chatter) unlike q_spread/r_duty, which already have
hardware-observed good AND bad regimes; tune it manually first (see the plan's
Part A4) before ever considering automating it.

SAFETY: every trial is scored from a real hardware run at a FIXED
rmax=2.0 (not searched). A live per-line safety check during capture (current
> rmax*1.5, or spread > 2.0) trips an immediate e-stop, marks the trial
ABORTED-UNSAFE, and re-flashes the current best-known-good config before
continuing -- the board is never left sitting at a bad configuration between
trials. A guaranteed finalize() step (normal exit, exception, or
KeyboardInterrupt) re-flashes whichever is better -- the found best or the
known-good baseline -- for every direction touched, so this script can never
leave the board worse than it started.

Usage:
  uv run python tools/tune_lqr_hyperparams.py --directions cw ccw \
      --rmax 2.0 --max-trials-per-direction 15
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import serial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pid_metrics import parse_log

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(HERE, "..")
PIO = os.path.expanduser("~/.platformio/penv/bin/pio")
UPLOAD_PORT_DEFAULT = "/dev/ttyUSB0"
UPLOAD_SPEED = 115200

HOLD_PHASE = 2
BASELINE = {"q_track": 1.0, "q_spread": 50.0, "q_integral": 5.0, "q_derivative": 0.0, "r_duty": 0.1}
BOUNDS = {"q_spread": (10.0, 80.0), "r_duty": (0.03, 0.5)}
INITIAL_STEP = 1.5
MAX_BACKTRACK_ROUNDS = 3
MAX_CONSECUTIVE_ACCEPTS = 2
INFEASIBLE_COST = 1e6
SPREAD_TARGET = 0.4  # "max spread across the whole sequence" target -- see progress.md Section 5

MODEL_FOR = {"cw": "tools/state_space_model.json", "ccw": "tools/state_space_model_ccw.json"}
SUFFIX_FOR = {"cw": "_CW", "ccw": "_CCW"}
HEADER_FOR = {"cw": "src/lqr_gains_cw.h", "ccw": "src/lqr_gains_ccw.h"}
JSON_FOR = {"cw": "tools/lqr_gains_cw.json", "ccw": "tools/lqr_gains_ccw.json"}


def find_port(explicit=None):
    if explicit:
        return explicit
    for pat in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("no /dev/ttyUSB* or /dev/ttyACM* found -- pass --port")


def cost(data: dict, rmax: float, duty_sat_thresh: float = 98.0,
         overcurrent_margin: float = 1.15) -> tuple[float, dict]:
    """Lower is better. data is pid_metrics.parse_log()'s dict.

    Gates on run_max_spread (max spread across the ENTIRE run -- all
    phases/frequencies), not the old settled-HOLD-only mean_spread: a prior
    sweep (tools/tune_lqr_hyperparams_runs/20260712_164809/) optimized
    mean_spread and found "no improvement" over baseline, which in
    hindsight makes sense -- it never saw or scored the RAMP_UP resonance-
    crossing spikes (up to ~2.7A) that dominate this rig's actual worst-
    case performance (see progress.md Section 5). sat_frac/overcurrent_frac
    are still computed over the settled-HOLD window (duty saturation and
    overcurrent margin are meaningfully steady-state concepts)."""
    hold = data["state"] == HOLD_PHASE
    if not hold.any():
        return INFEASIBLE_COST, {"reason": "no HOLD samples"}

    hold_t = data["t"][hold]
    settled = hold & (data["t"] >= hold_t[-1] - 3.0)
    if not settled.any():
        settled = hold

    duties = np.stack([data[f"d_{c}"] for c in "abcd"], axis=1)[settled]
    currents = np.stack([data[f"i_{c}"] for c in "abcd"], axis=1)[settled]

    run_max_spread = float(data["spread"].max())
    sat_frac = float((duties >= duty_sat_thresh).mean())
    overcurrent_frac = float((currents.max(axis=1) > rmax * overcurrent_margin).mean())

    c = run_max_spread + 0.5 * sat_frac + 5.0 * overcurrent_frac
    return c, {"run_max_spread": run_max_spread, "sat_frac": sat_frac,
               "overcurrent_frac": overcurrent_frac}


def design_gains(direction: str, q_spread: float, r_duty: float) -> bool:
    """Runs design_lqr_gains.py for this direction. Returns True on success
    (stabilizing point, headers written), False if infeasible (non-
    stabilizing -- SystemExit inside the subprocess, nonzero returncode)."""
    cmd = [sys.executable, os.path.join(HERE, "design_lqr_gains.py"),
           "--model", os.path.join(REPO_ROOT, MODEL_FOR[direction]),
           "--q-track", str(BASELINE["q_track"]),
           "--q-spread", str(q_spread),
           "--q-integral", str(BASELINE["q_integral"]),
           "--q-derivative", str(BASELINE["q_derivative"]),
           "--r-duty", str(r_duty),
           "--out", os.path.join(REPO_ROOT, JSON_FOR[direction]),
           "--header", os.path.join(REPO_ROOT, HEADER_FOR[direction]),
           "--name-suffix", SUFFIX_FOR[direction]]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [design] infeasible/non-stabilizing: {result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr.strip()}")
        return False
    return True


def build_and_flash(upload_port: str) -> bool:
    env = dict(os.environ, PLATFORMIO_UPLOAD_SPEED=str(UPLOAD_SPEED))
    clean = subprocess.run([PIO, "run", "-e", "state_space", "-t", "clean"],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if clean.returncode != 0:
        print(f"    [build] clean failed: {clean.stderr.strip()[-500:]}")
        return False
    build = subprocess.run([PIO, "run", "-e", "state_space"],
                            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if build.returncode != 0:
        print(f"    [build] build failed: {build.stderr.strip()[-500:]}")
        return False
    upload = subprocess.run([PIO, "run", "-e", "state_space", "-t", "upload",
                              "--upload-port", upload_port],
                             cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if upload.returncode != 0:
        print(f"    [build] upload failed: {upload.stderr.strip()[-500:]}")
        return False
    return True


def capture(direction: str, rmax: float, upload_port: str, log_path: str,
            log_seconds: float = 33.0, live_current_margin: float = 2.0,
            live_spread_limit: float = 3.0) -> tuple[bool, str]:
    """Reset, send dir=/rmax=, log with a live safety abort. Returns
    (aborted_unsafe, reason).

    Both defaults raised from this script's original (1.5, 2.0) after the
    frequency-scaled feedforward fix (progress.md Section 5) was added:
    baseline gains now transiently reach ~2.0-2.7A spread and ~3.4A on a
    single channel during RAMP_UP's resonance crossing even in otherwise-
    healthy trials, well under the real hardware safety nets (I_MAX_A=12A
    hard trip in firmware, I_SOFT_LIMIT_A=10A soft backoff) but tighter
    than the original values were sized for. Without this the live abort
    fired on literally the baseline trial, stopping the search before any
    tuning happened at all -- these are soft, search-level guards, not the
    hardware's actual safety backstop, which is untouched."""
    s = serial.Serial()
    s.port = upload_port
    s.baudrate = 115200
    s.timeout = 0.5
    s.dtr = False
    s.rts = False
    s.open()
    s.reset_input_buffer()
    s.rts = True
    time.sleep(0.15)
    s.rts = False
    t0 = time.time()

    dir_sent = rmax_sent = False
    aborted, reason = False, ""
    try:
        with open(log_path, "w") as f:
            while time.time() - t0 < log_seconds:
                elapsed = time.time() - t0
                outgoing = ""
                if not dir_sent and elapsed >= 0.5:
                    outgoing, dir_sent = f"dir={direction}", True
                elif not rmax_sent and elapsed >= 1.0:
                    outgoing, rmax_sent = f"rmax={rmax}", True
                if outgoing:
                    s.write((outgoing + "\n").encode())
                line = s.readline()
                if not line:
                    continue
                text = line.decode("utf-8", errors="replace").rstrip()
                f.write(f"{elapsed:7.2f}s  {text}\n")

                # Live safety check: cheap regex-free scan for I[A] values.
                if "I[A]:" in text and "spread=" in text:
                    try:
                        i_part = text.split("I[A]:")[1].split("|")[0]
                        currents = [float(tok.split("=")[1]) for tok in i_part.split()]
                        spread_part = text.split("spread=")[1].split()[0]
                        spread_val = float(spread_part)
                        if max(currents) > rmax * live_current_margin:
                            aborted, reason = True, f"live overcurrent {max(currents):.2f}A > {rmax*live_current_margin:.2f}A"
                            break
                        if spread_val > live_spread_limit:
                            aborted, reason = True, f"live spread {spread_val:.2f}A > {live_spread_limit:.2f}A"
                            break
                    except (IndexError, ValueError):
                        pass
    finally:
        try:
            s.write(b"s\n")
            time.sleep(0.3)
        except Exception:
            pass
        s.close()

    return aborted, reason


def run_trial(direction: str, q_spread: float, r_duty: float, rmax: float,
              upload_port: str, run_dir: str, trial_num: int) -> dict:
    """Full per-trial procedure. Returns a result dict (always has 'cost')."""
    row = {"trial": trial_num, "direction": direction, "q_spread": q_spread,
           "r_duty": r_duty, "feasible": True, "aborted_unsafe": False,
           "cost": INFEASIBLE_COST, "run_max_spread": None, "sat_frac": None,
           "overcurrent_frac": None}

    lo, hi = BOUNDS["q_spread"]
    if not (lo <= q_spread <= hi):
        row["feasible"] = False
        row["note"] = "q_spread out of bounds"
        return row
    lo, hi = BOUNDS["r_duty"]
    if not (lo <= r_duty <= hi):
        row["feasible"] = False
        row["note"] = "r_duty out of bounds"
        return row

    if not design_gains(direction, q_spread, r_duty):
        row["feasible"] = False
        return row

    if not build_and_flash(upload_port):
        row["feasible"] = False
        row["note"] = "build/flash failed"
        return row

    log_path = os.path.join(run_dir, f"trial_{trial_num:03d}_{direction}.log")
    aborted, reason = capture(direction, rmax, upload_port, log_path)
    if aborted:
        row["aborted_unsafe"] = True
        row["note"] = reason
        return row

    data = parse_log(log_path)
    c, diag = cost(data, rmax)
    row["cost"] = c
    row.update(diag)
    return row


def finalize(direction: str, best: dict, rmax: float, upload_port: str) -> None:
    """Guaranteed final step: reflash whichever is better -- best found or
    baseline -- for this direction."""
    print(f"[finalize] {direction}: flashing final config q_spread={best['q_spread']}, r_duty={best['r_duty']}")
    ok = design_gains(direction, best["q_spread"], best["r_duty"])
    if not ok:
        print(f"[finalize] {direction}: WARNING -- final config unexpectedly infeasible, "
              "falling back to hardcoded baseline")
        design_gains(direction, BASELINE["q_spread"], BASELINE["r_duty"])
    build_and_flash(upload_port)


def tune_direction(direction: str, rmax: float, upload_port: str, run_dir: str,
                    max_trials: int, csv_rows: list) -> dict:
    trial_num_holder = [0]

    def trial(qs, rd):
        trial_num_holder[0] += 1
        n = trial_num_holder[0]
        print(f"  trial {n}: q_spread={qs:.2f} r_duty={rd:.3f} ...")
        row = run_trial(direction, qs, rd, rmax, upload_port, run_dir, n)
        csv_rows.append(row)
        status = ("INFEASIBLE" if not row["feasible"] else
                   "ABORTED-UNSAFE" if row["aborted_unsafe"] else
                   f"cost={row['cost']:.4f}")
        print(f"    -> {status}")
        return row

    best = {"q_spread": BASELINE["q_spread"], "r_duty": BASELINE["r_duty"]}
    best_cost = INFEASIBLE_COST
    baseline_cost = INFEASIBLE_COST
    best_run_max_spread = None
    try:
        base_row = trial(best["q_spread"], best["r_duty"])
        best_cost = base_row["cost"]
        baseline_cost = best_cost
        best_run_max_spread = base_row.get("run_max_spread")

        if base_row["aborted_unsafe"]:
            print(f"  [{direction}] WARNING: baseline itself aborted unsafe -- "
                  "stopping this direction's search immediately")
            return {"direction": direction, "best": best, "best_cost": best_cost,
                     "baseline_cost": baseline_cost, "improved": False,
                     "best_run_max_spread": best_run_max_spread}

        for param in ("q_spread", "r_duty"):
            if trial_num_holder[0] >= max_trials:
                break
            step = INITIAL_STEP
            for _round in range(MAX_BACKTRACK_ROUNDS):
                if trial_num_holder[0] >= max_trials:
                    break
                improved_this_round = False
                for sign in (1, -1):
                    if trial_num_holder[0] >= max_trials:
                        break
                    candidate = dict(best)
                    candidate[param] = best[param] * (step if sign > 0 else 1.0 / step)
                    row = trial(candidate["q_spread"], candidate["r_duty"])
                    if row["aborted_unsafe"]:
                        print(f"  [{direction}] unsafe trial -- reflashing best-known-good before continuing")
                        finalize(direction, best, rmax, upload_port)
                        continue
                    if not row["feasible"]:
                        continue
                    if row["cost"] < best_cost:
                        best, best_cost = candidate, row["cost"]
                        best_run_max_spread = row.get("run_max_spread")
                        improved_this_round = True
                        # momentum: one more step same direction/sign, capped
                        accepts = 1
                        while accepts < MAX_CONSECUTIVE_ACCEPTS and trial_num_holder[0] < max_trials:
                            candidate2 = dict(best)
                            candidate2[param] = best[param] * (step if sign > 0 else 1.0 / step)
                            row2 = trial(candidate2["q_spread"], candidate2["r_duty"])
                            if row2["aborted_unsafe"]:
                                finalize(direction, best, rmax, upload_port)
                                break
                            if row2["feasible"] and row2["cost"] < best_cost:
                                best, best_cost = candidate2, row2["cost"]
                                best_run_max_spread = row2.get("run_max_spread")
                                accepts += 1
                            else:
                                break
                        break  # done with this param's sign-exploration for this round
                if improved_this_round:
                    break  # move to next param
                step = 1 + (step - 1) * 0.5  # shrink and retry

        # polish: reconfirm best
        if trial_num_holder[0] < max_trials:
            reconfirm = trial(best["q_spread"], best["r_duty"])
            if not reconfirm["aborted_unsafe"] and reconfirm["feasible"]:
                if reconfirm["cost"] > best_cost * 1.2:
                    print(f"  [{direction}] reconfirm cost much worse ({reconfirm['cost']:.4f} vs "
                          f"{best_cost:.4f}) -- treating as noisy, keeping baseline if better")
                    if baseline_cost <= reconfirm["cost"]:
                        best, best_cost = {"q_spread": BASELINE["q_spread"], "r_duty": BASELINE["r_duty"]}, baseline_cost
                        best_run_max_spread = base_row.get("run_max_spread")
                else:
                    best_run_max_spread = reconfirm.get("run_max_spread")

        improved = best_cost < baseline_cost
        return {"direction": direction, "best": best, "best_cost": best_cost,
                "baseline_cost": baseline_cost, "improved": improved,
                "best_run_max_spread": best_run_max_spread}
    finally:
        # Guaranteed final step regardless of how this direction's search
        # ended (normal completion, an exception, or KeyboardInterrupt):
        # flash whichever is provably no worse -- the found best, or the
        # hardcoded baseline if best_cost never beat it.
        final_best = best if best_cost < baseline_cost else \
            {"q_spread": BASELINE["q_spread"], "r_duty": BASELINE["r_duty"]}
        finalize(direction, final_best, rmax, upload_port)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--directions", nargs="+", default=["cw", "ccw"], choices=["cw", "ccw"])
    ap.add_argument("--rmax", type=float, default=2.0)
    ap.add_argument("--port", default=UPLOAD_PORT_DEFAULT)
    ap.add_argument("--max-trials-per-direction", type=int, default=15)
    args = ap.parse_args()

    upload_port = find_port(args.port)
    ts = time.strftime("%Y%m%d_%H%M%S") if not os.environ.get("TUNE_TS") else os.environ["TUNE_TS"]
    run_dir = os.path.join(HERE, "tune_lqr_hyperparams_runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    print(f"run directory: {run_dir}")

    summary = {"directions": {}}
    for direction in args.directions:
        print(f"\n=== tuning {direction.upper()} ===")
        csv_rows = []
        try:
            result = tune_direction(direction, args.rmax, upload_port, run_dir,
                                     args.max_trials_per_direction, csv_rows)
        except KeyboardInterrupt:
            print(f"\n[{direction}] interrupted -- finalize() already ran via finally block")
            result = {"direction": direction, "interrupted": True}

        csv_path = os.path.join(run_dir, f"results_{direction}.csv")
        with open(csv_path, "w") as f:
            if csv_rows:
                keys = sorted({k for row in csv_rows for k in row.keys()})
                f.write(",".join(keys) + "\n")
                for row in csv_rows:
                    f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"wrote {csv_path}")
        summary["directions"][direction] = result

        if "best" in result:
            b = result["best"]
            if result["improved"]:
                print(f"{direction.upper()}: baseline cost={result['baseline_cost']:.4f} "
                      f"(q_spread=50, r_duty=0.1) -> best cost={result['best_cost']:.4f} "
                      f"(q_spread={b['q_spread']:.2f}, r_duty={b['r_duty']:.3f})")
            else:
                print(f"{direction.upper()}: no improvement found, reverted to baseline "
                      f"(cost={result['baseline_cost']:.4f})")

            rms = result.get("best_run_max_spread")
            if rms is not None:
                verdict = "PASS" if rms < SPREAD_TARGET else "FAIL"
                print(f"{direction.upper()}: run_max_spread={rms:.3f}A vs. "
                      f"{SPREAD_TARGET}A target -> {verdict}")
            else:
                print(f"{direction.upper()}: run_max_spread unavailable (no successful trial)")

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {summary_path}")
    print("Board is left flashed with the best-or-baseline config for the last direction tuned.")


if __name__ == "__main__":
    main()
