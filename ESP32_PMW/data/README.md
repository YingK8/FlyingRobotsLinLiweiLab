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
- `comp_ab_162530` — v1 run that proved the CS→ADC taps were disconnected at the
  time (on-board closed loop impossible). **Superseded:** the ADC pins were later
  rewired and each channel now reads its own coil — `SENS` in `src/drive_common.h`
  is per-channel calibrated and telemetry reports four independent currents. The CS
  pin still gives an unsigned magnitude sampled asynchronously at ~1 kHz, so it
  supports amplitude regulation but not commutation; see control/theory.md 17.2.
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

## 2026-07-14_tilt/
The largest session here (99 files). PI-balance and tilt development on the
2026-07-14 rig: `tilt_pi_fixed_run1/2`, `tilt_pi_governor_run3`,
`tilt_pi_balance_run4`, the `tilt_ratio_run1..10` sweep, `tilt_maxperf_15V_run6`
(+ its freq sweep), the `tilt_cw_*_run11..19` diagnostic series (coramp, low
signal, from-0Hz, v/f, 170 Hz, align), `spinup_torque_aware_run20/21`,
`dwell_staircase_run22`, `takeoff_220_run23/24`, plus `ceiling_run1/2`,
`coramp_fix_comparison.png` and `sync_model`. Each run is a `.csv` capture with
its `.log` serial companion and, where analyzed, a `.png`.

## current_pid_tuning/
Standalone current-PID tuning rig, `current_pid_iter1..N` — each iteration a
serial `.log` plus a `*_pid_telemetry.png`. Plotted by `ai/plots/plot_pid_log.py`.

## Loose captures (2026-07-19 onward)
Not yet grouped into a session directory:
- `cs_20260719_223458.*` and `cs_20260719_223543_part01..08.*` — a long
  current-sense recording split into parts (`*_cs.png` for the analyzed ones).
- `reset_test_20260719_230603.*` — reset-button / boot behaviour check.
- `takeoff_1.csv` — a single takeoff capture.
- `hover_zigzag_debug.{log,png}` — from the since-deleted `hover_zigzag` firmware.

## Removed as garbage (2026-07-05 cleanup)
- `tilt_ccw_verify.*` — capture recorded against the wrong firmware after a
  failed flash (PhaseSequencer mid-refactor); serial log was empty.
- `comp_ff_iter1_rms.png` — auto-generated overview that was provably inverted
  (see workflow memory note; regenerate any overview with `ai/plots/plot_rms.py`).
