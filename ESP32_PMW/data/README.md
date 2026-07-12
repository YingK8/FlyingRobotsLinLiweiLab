# Experiment data — chronological index

Recorded captures (PicoScope CS traces), analysis figures, and ESP32 serial logs,
grouped by session. Scope channels are the fixed H-bridge frame; **firmware/coil
B and D are swapped relative to scope labels** (fw B → scope D, fw D → scope B).
CS = VNH5019 current sense = supply-sourced current only (not coil current).

## 2026-07-03_tilt-freq-sweeps/
First tilt frequency sweeps (1→210 Hz ease ramp) and forward/reverse tilt runs
(`tilt_test`, `tilt_test_reverse`), plus freq-gain/freq-response analyses of
`picoscope_stream_20260703_200227`. Outcome: direction-dependent supply
imbalance observed → triggered the coupling investigation.

## 2026-07-04a_coupling-matrix/
Solo/pairwise/ALL excitation sweeps (env:coupling_test, 190 Hz).
- `coupling_sweep_142357` — first sweep; `coupling_pairwise_143320` — the
  pairwise run behind the coupling matrix.
- `coupling_1supply_144748` — single-supply test; `coupling_1supply_origcfg_15*`
  — after reverting to the original AC/BD wiring (152739 is the analyzed one;
  152133/152323 were earlier takes).
Outcome: coupling is magnetic (not ground); strong pairs A–C and B–D, k≈0.24.

## 2026-07-04b_comp-calibration-cw/
Feedforward trim calibration, CW rotation (env:comp_test A/B captures).
- `comp_ab_162530` — v1 run that proved the CS→ADC taps are disconnected
  (on-board closed loop impossible).
- `comp_ff_iter1/iter2` — scope-in-the-loop iterations: equal-duty baseline
  spread max/min 1.88 → **1.07** with CW trims {1.337, 0.866, 0.794, 1.003}.
- `tilt_trimmed_verify` — trims in main_tilt at 210 Hz hold: spread 1.043.
- Figures: `comp_ff_balance.png`, `comp_ff_iter2_timeline.png`.

## 2026-07-04c_two-supply-recheck/
Boards moved back to two supplies; recalibration check. Baseline AND trimmed
matched single-supply within 1% per channel → supply topology irrelevant to the
(magnetic) redistribution; trims unchanged. `tilt_2supply_verify`: spread 1.063.

## 2026-07-04d_tilt-runs-cw/
User tilt-experiment runs with CW trims (`tilt_run1`, `tilt_run_4_july_5_32`).

## 2026-07-04e_recalibration-ccw/
Rotation reversed to CCW → coupling redistribution flips (sin Δφ is odd):
equal-duty baseline A went lowest→highest, spread 2.0. Iterated CCW trims
{0.839, 1.331, 0.982, 0.848}: spread 1.97 → **1.046** (`comp_ccw_iter0/1/2`).
`tilt_ccw_run` = full CCW tilt run (also shows AC-pair lower through the ramp).
Figures: `comp_ccw_balance.png` (CW vs CCW before/after),
`tilt_ccw_run_analysis.png` (per-coil + supply-pair timeline).

## 2026-07-04f_direction-disk-diagnosis/
Why the disk still tilted toward B/C:
- `ramp_freq_response_cw_ccw.png` + `tilt_cw_reversal` — the per-coil ramp peak
  stagger REARRANGES with rotation direction → drive/coupling-made, not coil
  hardware. Tilt itself stayed fixed in lab frame → geometric (field-per-amp).
- `tilt_ccw_nodisk` + `ramp_disk_vs_nodisk.png` — identical curves without the
  disk → disk is electrically invisible in CS; the ~130 Hz ramp peak is
  electrical (coil L/R corner), not disk mechanics.
- `tilt_ccw_sched` + `ramp_scheduled_trims.png` — frequency-scheduled trims:
  supply pairs track through spin-up (AC/BD 0.92 → 0.97+ below 150 Hz).
- `tilt_run_4_july_6_32` — user run in this period.

## 2026-07-04g_field-trim-sessions/
Interactive field-trim nulling attempts using the disk as sensor (serial
a+/a-/... then -/+ pair nudges), plus `tilt_cw_run` (last CW run, 20:23).
Serial logs contain the `field trim:` value lines from each session.
Not yet successful; open thread: geometric field asymmetry (B/C vs A/D),
suspect coil mounting height/tilt — calipers check pending.

## 2026-07-05_current-pid-manual-tuning/
Manual gain-tuning session for `main_current_pid.cpp`'s global min/max PI
(`current_pid_iter1`–`iter16`, EN-reset + `kp=/ki=/kd=/ramp=` between runs,
captured via `tools/trigger_reset_log.py`; telemetry PNGs from
`tools/plot_pid_log.py` for iter5, 7–14, 16). Converged to the gains recorded
in project memory (KP=2.2, KI=0.10, KD=0.15). Moved here (2026-07-11 cleanup)
from the repo root, where they'd accumulated as loose files.

## Removed as garbage (2026-07-05 cleanup)
- `tilt_ccw_verify.*` — capture recorded against the wrong firmware after a
  failed flash (PWMSequencer mid-refactor); serial log was empty.
- `comp_ff_iter1_rms.png` — auto-generated overview that was provably inverted
  (see workflow memory note; regenerate any overview with `tools/plot_rms.py`).

## Removed as garbage (2026-07-11 cleanup)
- `current_pid_iter15.log` — aborted trial: HOLD ended abruptly (skipped the
  ENDING ramp-down phase entirely) and the STOPPED-phase current reads spiked
  to implausible values (-12.75A/-12.58A, spread=12.752), consistent with an
  e-stop/ADC glitch rather than a valid data point.
