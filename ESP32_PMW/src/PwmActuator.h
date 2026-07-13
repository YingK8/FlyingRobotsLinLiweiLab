#pragma once
// The low-level, timing-critical domain (core 0): ADC sampling, PWM/carrier
// stepping, the full bounded run state machine (ARMING->RAMP_UP->HOLD->
// ENDING->STOPPED), and the independent hard overcurrent trip. Applies
// whatever duty is currently in SharedMemory's command region every tick,
// indifferent to freshness -- no watchdog by design (see SharedMemory.h).
//
// This makes the bounded-run safety guarantee independent of core 1
// entirely: if the higher-level control law (core 1) hangs or crashes,
// this still runs the full state machine on its own schedule and safely
// latches coils off. Worst case, tracking quality degrades (duty frozen at
// whatever core 1 last wrote), but the run still terminates safely.

#include "constants.h"
#include "CurrentSense.h"
#include "PWMController.h"
#include "PWMSequencer.h"
#include "SharedMemory.h"

enum RunPhase { PHASE_ARMING = 0, PHASE_RAMP_UP = 1, PHASE_HOLD = 2, PHASE_ENDING = 3, PHASE_STOPPED = 4 };

class PwmActuator {
public:
  explicit PwmActuator(SharedMemory *shared);

  // One-time init (currentSense.seed(), initial controller/seq construction).
  // Must run on core 0, before run() is ever called -- NOT from Arduino's
  // setup() (which stays on core 1).
  void begin();

  // Call repeatedly in a tight loop from core 0's task.
  void run();

private:
  void reinitController(bool ccw);
  void allCoilsOff();
  void applyCommandedDuty(); // copy core 1's latest duty command onto the carriers
  void applyDirectionRequestIfSafe();
  bool checkOvercurrentTrip(); // returns true if it just tripped

  SharedMemory *_shared;
  PWMController *_controller;
  PWMSequencer *_seq;
  CurrentSense _currentSense;
  bool _directionIsCcw;

  RunPhase _phase;
  unsigned long _phase_start;
  unsigned long _last_ramp_ms;
  unsigned long _last_adc_us;
};
