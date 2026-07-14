#pragma once

#include <Arduino.h>
#include "CurrentSense.h"
#include "PWMController.h"

// Shared phase-machine primitives for the main_*.cpp state machines
// (ARMING -> WAITING -> <running> -> STOPPED/DONE). Extracted because all
// three main_*.cpp files duplicated these ticks near-verbatim.
namespace ExperimentPhase {

// ARMING tick: coils off, fast-blink LED (arming/settling), recalibrate ADC
// zero (valid only with coils confirmed off, which this tick guarantees).
// Returns true once armMs has elapsed since phaseStart -- caller transitions
// to WAITING (never straight to a running phase; firmware never auto-starts
// off a timer, see ExperimentPhase::isStartCommand()).
inline bool armingTick(unsigned long now, unsigned long phaseStart, unsigned long armMs,
                        PWMController *controller, int numChannels,
                        CurrentSense &currentSense, int ledPin) {
  for (int i = 0; i < numChannels; i++) controller->setCarrierDutyCycle(i, 0.0f);
  digitalWrite(ledPin, (now / 150) & 1); // fast blink = arming/settling
  currentSense.recalibrateZero();        // coils confirmed off here
  return (now - phaseStart) >= armMs;
}

// Latched-off tick shared by every terminal/parked phase (WAITING, STOPPED,
// DONE, *_FAILED): coils off, slow heartbeat LED so the board still reads as
// alive without ever re-energizing on its own.
inline void latchedOffTick(unsigned long now, PWMController *controller,
                            int numChannels, int ledPin) {
  for (int i = 0; i < numChannels; i++) controller->setCarrierDutyCycle(i, 0.0f);
  digitalWrite(ledPin, (now / 1000) & 1);
}

// True for a trimmed, lowercased e-stop command. Callers should treat this
// as unconditional -- check it first, in every phase, unlike tuning commands
// that are only valid while parked.
inline bool isStopCommand(const String &cmd) {
  return cmd == "s" || cmd == "stop" || cmd == "estop";
}

// True for a trimmed, lowercased start command. Only meaningful while a
// phase machine is WAITING -- the caller is responsible for checking phase.
inline bool isStartCommand(const String &cmd) {
  return cmd == "r" || cmd == "run" || cmd == "start" || cmd == "go";
}

} // namespace ExperimentPhase
