#pragma once
// Tuning/timing constants for the state-space (LQID + coupling-aware
// feedforward + constrained allocator) controller -- kept out of the
// shared constants.h since main_current_pid.cpp/main_experiment.cpp don't
// need any of this.

#include "constants.h"

const float START_DUTY = 50.0f;
const float DUTY_MIN = 5.0f;
const float DUTY_MAX = 100.0f;
// Two distinct thresholds, not one, so the soft path actually gets a chance
// to act before the hard path ever fires: PwmActuator's hard trip (core 0,
// ~1ms ADC cadence) checks I_MAX_A and immediately latches off -- the true,
// never-exceed safety cap. CurrentBalanceController's soft backoff (core 1,
// CONTROL_TICK_MS=50ms cadence) checks the lower I_SOFT_LIMIT_A and gently
// ramps r_target down instead. If both used I_MAX_A, the ~50x-faster hard
// trip would always win the race and the soft path would be dead code --
// this was flagged as a real bug during review, not just a style issue.
// The margin (2A) matches the original code's own stated intent for
// R_MAX_A's default ("ceiling, kept below I_MAX_A for margin").
const float I_MAX_A = 12.0f;       // hard per-channel safety cap (PwmActuator, core 0)
const float I_SOFT_LIMIT_A = 10.0f; // soft backoff threshold (CurrentBalanceController, core 1)
const float OVERCURRENT_BACKOFF_A = 0.5f; // r step-back per tick when I_SOFT_LIMIT_A is exceeded

// DutyAllocator's actuation-effort weight. NOT the same number as
// design_lqr_gains.py's --r-duty default (0.1) -- that's a completely
// different cost formulation (offline LQR, Q terms scaled ~1-50) and
// reusing its value here was a real bug: the allocator's objective is
// ||Bd@u-x_target||^2 (residual in Amps, typically O(1-4) for the vector)
// + r_duty*||u-u_ff||^2 (duty deviation in PERCENTAGE POINTS, squared --
// even a 10-point deviation contributes 100*r_duty). At r_duty=0.1 that's
// 10, already dwarfing the entire current-tracking term -- confirmed on
// hardware as the cause of a real regression (duty barely moved away from
// feedforward despite a large, persistent current error). This value is
// picked so a ~30-point duty deviation contributes ~1 -- present as a tie-
// breaker/regularizer, not something that fights the primary objective.
const float ALLOCATOR_R_DUTY = 0.001f;

// The control law (CurrentBalanceController, core 1) must be evaluated at
// exactly this period -- it's the Ts the offline model (Ad/Bd, hence FF/K_x/
// K_d/K_z) was fit for. Not to be changed without full model re-identification.
const unsigned long CONTROL_TICK_MS = 50;
const float CONTROL_TS_S = CONTROL_TICK_MS / 1000.0f;

// Derivative-term noise filtering: raw ADC current has ~0.01A jitter, and a
// naive one-sample difference over CONTROL_TS_S would amplify that directly
// into duty chatter. Single-pole EMA on the derivative estimate, matching
// CurrentSense.h's own filtering convention. Tau chosen >= 3x
// CONTROL_TICK_MS so the filter meaningfully smooths without lagging the
// RAMP_UP-phase transient it's meant to help damp.
const float DERIV_FILTER_TAU_MS = 150.0f;
const float DERIV_ALPHA = CONTROL_TS_S / (DERIV_FILTER_TAU_MS / 1000.0f + CONTROL_TS_S);

// The feedforward/coupling model (Bd, hence FF/K_x/K_d/K_z) is built from a
// STATIC (DC, omega=0) step-response gain (tools/build_state_space_model.py)
// -- exactly correct only at each channel's LC resonance f0 (~153-156Hz per
// tools/rlc_fit.json, where the series R-L-C's impedance minimum |Z(f0)|=R
// makes the DC-resistance assumption momentarily true), wrong by a large,
// frequency-dependent factor elsewhere. Below this threshold the coils'
// true gain is much lower than the model assumes, so any error there is a
// low-frequency-regime artifact, not a real tracking failure -- integrating
// it winds up z_integral on a model that doesn't apply yet. Confirmed as
// the root cause of a real hardware bug: channel D pinned at DUTY_MAX for
// 13+s through RAMP_UP while its own error crossed from undershoot into
// sustained overshoot, because integral correction accumulated during the
// low-frequency region never unwound. Below this threshold the integral is
// held (not decayed -- it starts at 0 on ARMING->RAMP_UP and never
// accumulates until frequency clears the gate, which is equivalent to
// "decay to zero and hold" here since there's no pre-existing windup to
// unwind on a fresh run).
//
// See also FreqGainModel.h's freqGainCorrection() -- a complementary,
// independent fix that corrects the feedforward's own MAGNITUDE for the
// actual drive frequency (this gate alone only stops the integral from
// accumulating on top of a still-wrong feedforward; it does not fix the
// feedforward itself). See progress.md Section 5 for the full derivation.
const float FREQ_MODEL_VALID_HZ = 130.0f;

// freqGainCorrection()'s safety ceiling (see FreqGainModel.h) -- the true
// |Z(f)|/R ratio can reach ~550x near 1Hz (progress.md Section 5.4's
// table), which would be a useless, purely-saturating correction.
// tools/validate_freq_gain_model.py's closed-loop cross-check (Phase 3)
// found the raw per-channel RLC ratio (7-10x at 60Hz) is considerably more
// aggressive than what real applied duty implies is needed there -- most
// likely because a per-channel scalar correction doesn't capture how much
// of the compensation the allocator's cross-channel coupling (Bd's off-
// diagonal terms) already redistributes on its own. Started conservative
// (3.0x) for the first hardware pass rather than the raw model's value;
// tools/tune_lqr_hyperparams.py's automated search (Phase 6) is the
// intended place to refine this against real closed-loop cost, not this
// offline estimate.
const float FREQ_GAIN_CORRECTION_MAX = 3.0f;

// Frequency floor used inside freqGainCorrection()'s 1/(omega*C) term --
// avoids a division blowup as freq_hz -> 0. Matches PwmActuator's own
// START_FREQ (RAMP_UP never actually commands below this).
const float FREQ_GAIN_MIN_HZ = 1.0f;
