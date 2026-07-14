#pragma once
// Loads a small, flat JSON profile (ratios/mode/gains) into a
// RatioCurrentController::Config -- see task_sequences/pi_profile_*.json
// for examples. Deliberately separate from JsonPWMSequencer: that library
// loads open-loop actuator SCHEDULES; this loads closed-loop control
// PARAMETERS, a different concern with a much smaller, non-flat-array
// schema, so it doesn't reuse JsonPWMSequencer's method-call-array format.
//
// Schema:
//   {
//     "name": "tilt_demo",
//     "mode": "independent",            // or "shared_constraint"
//     "ratios": {"A": 1.0, "B": 0.5, "C": 0.3, "D": 0.8},
//     "gains": {"kp": 2.2, "ki": 0.10, "kd": 0.15},
//     "ramp_pct_per_ms": 0.05,
//     "duty_min": 5.0,
//     "duty_max": 100.0
//   }
// "i_max_a"/"overcurrent_backoff_pct"/"min_switch_margin_a"/
// "nominal_tick_ms" are optional, defaulting to main_current_pid.cpp's
// converged values (see loadPiProfile()'s implementation).

#include "RatioCurrentController.h"

// Parses `path` (a SPIFFS file, e.g. "/pi_profile.json") into `out`.
// Returns false (out left UNMODIFIED) on any missing file, parse error, or
// missing required field ("mode", "ratios" for all NUM_CHANNELS) -- mirrors
// main_experiment.cpp's "never runs on garbage" JSON-load-failure handling:
// callers must treat a false return as "stay latched off," not "use
// whatever's in out."
bool loadPiProfile(const char *path, RatioCurrentController::Config *out);
