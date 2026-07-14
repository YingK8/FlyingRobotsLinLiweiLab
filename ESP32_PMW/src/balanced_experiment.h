#pragma once

#include <Arduino.h>

// Shared firmware skeleton for the per-experiment mains (main_tilt.cpp,
// main_takeoff.cpp, ...). Each main just fills in an ExperimentConfig and calls
// experimentSetup()/experimentLoop(); this owns the drive stack
// (PhaseController + JSON sequencer + current-balance PI) and the
// ARM -> RUN -> DONE state machine.
//
// Design (see the approved plan): the JSON schedule owns commutation
// (frequency / phase / direction) and the per-step carrier LEVEL; the PI loop
// (CurrentBalanceController) owns the actual per-channel carrier duty, using the
// schedule's commanded carrier as its ceiling. So one tuned balance loop holds
// the four channel currents together (< 0.4 A spread) beneath whatever envelope
// any JSON commands.
//
//   piEnabled = true  -> lift/operational experiments (tilt, takeoff): the PI
//                        rebalances beneath the commanded ceiling.
//   piEnabled = false -> passthrough: the sequencer's commanded carrier is
//                        driven verbatim (ceiling characterization; system-ID
//                        sweeps like coupling/DC where forced balance would
//                        erase the very imbalance being measured).

struct ExperimentConfig {
  const char *jsonFile; // SPIFFS path, e.g. "/tilt.json"
  bool ccwDefault;      // boot direction; a JSON setDirection overrides it
  float driveFreq;      // non-zero boot frequency (begin(0) divides by zero)
  float iSafetyMax;     // hard per-channel overcurrent trip (A) -> latch off
  bool piEnabled;       // run the balance loop, or pass the schedule through

  // Explicit constructor (not a designated-initializer aggregate) so the thin
  // mains stay a one-liner and compile cleanly under gnu++11 as well as newer.
  ExperimentConfig(const char *json, bool ccw = true, float freq = 190.0f,
                   float iSafe = 10.0f, bool pi = true)
      : jsonFile(json), ccwDefault(ccw), driveFreq(freq), iSafetyMax(iSafe),
        piEnabled(pi) {}
};

// Call once from setup() and every loop() respectively.
void experimentSetup(const ExperimentConfig &cfg);
void experimentLoop();
